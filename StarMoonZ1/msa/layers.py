"""
MSA 核心层：MemorySparseAttention / ChunkMeanPooler / MSABlock
==============================================================

MemorySparseAttention
----------------------
在标准 GQA 基础上增加：
  - 路由专用投影 qr_proj / kr_proj（仅 memory layer）
  - 文档级稀疏路由：用 query 路由键 QR 与记忆路由键 KR 做余弦相似度，
    选出 Top-k 文档/块，将其压缩 K/V 拼接到本地 K/V 之前参与注意力
  - 记忆前缀（已带文档级 RoPE）对查询完全可见；本地部分保持因果

ChunkMeanPooler
----------------
将长序列按 chunk_size 分块并做平均池化，得到压缩表示（MSA 的压缩记忆单元）。

MSABlock
--------
与标准 TransformerBlock 同构，但 memory layer 使用 MemorySparseAttention。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from StarMoonZ1.model.config import StarMoonZ1Config
from StarMoonZ1.model.model import RMSNorm, apply_rope


class ChunkMeanPooler(nn.Module):
    """Chunk-Mean Pooling 压缩器：按固定块大小分块平均池化"""

    def __init__(self, chunk_size: int = 128, pooling_type: str = "mean"):
        super().__init__()
        self.chunk_size = chunk_size
        self.pooling_type = pooling_type

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden_states: [B, T, H]
            attention_mask: [B, T] 1=有效 0=padding
        Returns:
            pooled:    [B, n_chunks, H]
            chunk_mask:[B, n_chunks] 有效块（含至少一个有效 token）
        """
        B, T, H = hidden_states.shape
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        pad = n_chunks * self.chunk_size - T
        if pad:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
        xc = hidden_states.view(B, n_chunks, self.chunk_size, H)
        if attention_mask is not None:
            m = attention_mask[:, :T]
            m = F.pad(m, (0, pad), value=0).view(B, n_chunks, self.chunk_size)
            masked = xc * m.unsqueeze(-1)
            denom = m.sum(-1, keepdim=True).clamp(min=1e-6)
            pooled = masked.sum(2) / denom
            chunk_mask = (m.sum(-1) > 0)
        else:
            pooled = xc.mean(2)
            chunk_mask = torch.ones(B, n_chunks, dtype=torch.bool, device=hidden_states.device)
        return pooled, chunk_mask


class MemorySparseAttention(nn.Module):
    """Memory Sparse Attention 层（标准 GQA + 文档级稀疏路由）"""

    def __init__(self, config: StarMoonZ1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_memory_layer = (
            config.memory_layers is not None and layer_idx in config.memory_layers
        )
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.sliding_window = config.sliding_window
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        # 路由专用投影（仅记忆层需要）
        if self.is_memory_layer:
            self.qr_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.kr_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)

        self.use_flash_attn = config.use_flash_attn
        self._flash_available = None
        if self.use_flash_attn:
            try:
                import flash_attn  # noqa: F401
                self._flash_available = True
            except ImportError:
                self._flash_available = False

    # ──────────────────────────────────────────
    # 路由
    # ──────────────────────────────────────────
    def compute_routing_scores(
        self,
        qr: Tensor,
        memory_kr: Tensor,
        memory_doc_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        核心公式: S_ij = max_t ( mean_h ( cos(QR_{t,h}, K̄R_{ij,h}) ) )
        Args:
            qr:         [B, H, T, d] 查询路由向量（未施加 RoPE，纯内容余弦）
            memory_kr:  [N_docs, N_chunks, n_kv_heads, d] 记忆路由键
            memory_doc_mask: [N_docs] 可选文档掩码
        Returns:
            doc_scores:   [B, N_docs] 每文档最高相关度
            chunk_scores: [B, N_docs, N_chunks] 块级分数
        """
        B, H, T, d = qr.shape
        N, C, KV, _ = memory_kr.shape
        qr_flat = qr.float().mean(dim=1)              # [B, T, d]
        kr_flat = memory_kr.float().mean(dim=2)       # [N, C, d]
        qr_n = F.normalize(qr_flat, dim=-1)
        kr_n = F.normalize(kr_flat, dim=-1)
        # sim[t, i, j] = cos(QR_t, KR_ij)
        sim = torch.einsum("btd,ncd->btnc", qr_n, kr_n)   # [B, T, N, C]
        chunk_scores = sim.max(dim=1).values              # [B, N, C] max over t
        doc_scores = chunk_scores.max(dim=2).values       # [B, N] max over chunks
        if memory_doc_mask is not None:
            doc_scores = doc_scores.masked_fill(
                ~memory_doc_mask.bool().to(doc_scores.device), float("-inf"))
        return doc_scores, chunk_scores

    def select_topk_memory(
        self,
        doc_scores: Tensor,
        memory_k: Tensor,
        memory_v: Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        选择 Top-k 记忆文档并拼接其压缩块。
        Returns:
            selected_k: [B, n_kv_heads, top_k*C, d]
            selected_v: [B, n_kv_heads, top_k*C, d]
            selected_indices: [B, top_k]
            selected_scores:  [B, top_k]
        """
        top_k = top_k or self.config.router_top_k
        B, N = doc_scores.shape
        k = min(top_k, N)
        topk = torch.topk(doc_scores, k, dim=-1)
        selected_scores = topk.values
        selected_indices = topk.indices                 # [B, k]
        idx = selected_indices.view(B, k, 1, 1, 1)     # [B,k,1,1,1]
        exp_k = memory_k.unsqueeze(0).expand(B, -1, -1, -1, -1)
        exp_v = memory_v.unsqueeze(0).expand(B, -1, -1, -1, -1)
        sk = torch.gather(exp_k, 1, idx.expand(B, k, memory_k.shape[1], memory_k.shape[2], memory_k.shape[3]))
        sv = torch.gather(exp_v, 1, idx.expand(B, k, memory_v.shape[1], memory_v.shape[2], memory_v.shape[3]))
        Bk, kk, C, KV, d = sk.shape
        selected_k = sk.reshape(B, kk * C, KV, d).transpose(1, 2)  # [B, KV, k*C, d]
        selected_v = sv.reshape(B, kk * C, KV, d).transpose(1, 2)
        return selected_k, selected_v, selected_indices, selected_scores

    # ──────────────────────────────────────────
    # 注意力核心
    # ──────────────────────────────────────────
    def _attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """缩放点积注意力（支持任意加性掩码，[B,H,T,L]）"""
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,T,L]
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, v)  # [B,H,T,d]

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        memory_k: Optional[Tensor] = None,
        memory_v: Optional[Tensor] = None,
        memory_kr: Optional[Tensor] = None,
        memory_doc_mask: Optional[Tensor] = None,
        use_memory: bool = True,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[dict]]:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        routing_info = None
        local_mask = attention_mask  # 本地 [B,1,T,T]（或 None / 2D）

        # ── 记忆路由 ──
        if self.is_memory_layer and use_memory and memory_k is not None:
            qr = self.qr_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            doc_scores, chunk_scores = self.compute_routing_scores(qr, memory_kr, memory_doc_mask)
            selected_k, selected_v, selected_indices, selected_scores = self.select_topk_memory(
                doc_scores, memory_k, memory_v, top_k=self.config.router_top_k)
            mem_len = selected_k.shape[2]
            # 记忆前缀（已带文档级 RoPE）拼到本地 K/V 之前
            k = torch.cat([selected_k, k], dim=2)
            v = torch.cat([selected_v, v], dim=2)
            # 构建组合掩码：记忆区全可见，本地区保留 local_mask
            combined = torch.zeros(B, 1, T, mem_len + T, device=hidden_states.device, dtype=hidden_states.dtype)
            if local_mask is None:
                # 纯因果（本地部分）
                causal = torch.triu(torch.full((T, T), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype), diagonal=1)
                combined[..., mem_len:] = causal.unsqueeze(0).unsqueeze(0)
            elif local_mask.dim() == 2:
                combined[..., mem_len:] = local_mask[:, None, None, :].to(hidden_states.dtype)
            else:  # 4D [B,1,T,T]
                combined[..., mem_len:] = local_mask
            attn_mask = combined
            routing_info = {
                "layer_idx": self.layer_idx,
                "doc_scores": doc_scores,
                "chunk_scores": chunk_scores,
                "selected_indices": selected_indices,
                "selected_scores": selected_scores,
                "mem_len": mem_len,
            }
        else:
            attn_mask = local_mask

        # ── 本地 KV 缓存 ──
        # 记忆只参与当步注意力，不写入缓存；缓存仅保存本地 KV 历史，
        # 避免跨解码步把记忆前缀重复累积（否则记忆会在每步被再次拼接并无限增长）
        if past_key_value is not None:
            local_k = k[:, :, mem_len:, :] if routing_info is not None else k
            local_v = v[:, :, mem_len:, :] if routing_info is not None else v
            cache_k = torch.cat([past_key_value[0], local_k], dim=2)
            cache_v = torch.cat([past_key_value[1], local_v], dim=2)
            if routing_info is not None:
                # 记忆前缀仍需拼到当步注意力 K/V 之前（不存入缓存）
                k_for_attn = torch.cat([k[:, :, :mem_len, :], cache_k], dim=2)
                v_for_attn = torch.cat([v[:, :, :mem_len, :], cache_v], dim=2)
            else:
                k_for_attn, v_for_attn = cache_k, cache_v
            new_kv = (cache_k, cache_v) if use_cache else None
            attn_mask = None  # 增量解码：当前唯一查询位于末尾，attends 全部 keys
        else:
            k_for_attn, v_for_attn = k, v
            new_kv = (k, v) if use_cache else None

        # ── GQA: 将 KV 头扩展到与 Q 头一致（含记忆前缀，flash / sdpa 均需要同头数）──
        if self.num_groups > 1:
            k_for_attn = (k_for_attn[:, :, None, :, :]
                          .expand(B, self.num_kv_heads, self.num_groups, -1, self.head_dim)
                          .reshape(B, self.num_heads, -1, self.head_dim))
            v_for_attn = (v_for_attn[:, :, None, :, :]
                          .expand(B, self.num_kv_heads, self.num_groups, -1, self.head_dim)
                          .reshape(B, self.num_heads, -1, self.head_dim))

        # ── 注意力计算 ──
        if self._flash_available and self.use_flash_attn and routing_info is None and attn_mask is None and self.sliding_window is None:
            from flash_attn import flash_attn_func  # type: ignore
            attn_output = flash_attn_func(
                q.transpose(1, 2).contiguous(),
                k_for_attn.transpose(1, 2).contiguous(),
                v_for_attn.transpose(1, 2).contiguous(),
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True,
            ).transpose(1, 2)
        else:
            attn_output = self._attention(q, k_for_attn, v_for_attn, attn_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_output), new_kv, routing_info


class MSABlock(nn.Module):
    """MSA Transformer Block：memory layer 用 MemorySparseAttention，其余用标准 GQA"""

    def __init__(self, layer_idx: int, config: StarMoonZ1Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_memory_layer = (
            config.memory_layers is not None and layer_idx in config.memory_layers
        )
        if self.is_memory_layer:
            self.self_attn = MemorySparseAttention(config, layer_idx)
        else:
            from StarMoonZ1.model.model import GroupedQueryAttention
            self.self_attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = None
        from StarMoonZ1.model.model import SwiGLU
        self.mlp = SwiGLU(config)
        self.gradient_checkpointing = False

    def _forward_impl(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor],
        past_key_value: Optional[Tuple[Tensor, Tensor]],
        use_cache: bool,
        memory_inputs: Optional[Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]],
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[dict]]:
        residual = hidden_states
        h = self.input_norm(hidden_states)
        if self.is_memory_layer and memory_inputs is not None:
            mk, mv, mkr, mdm = memory_inputs
            attn_out, kv, routing = self.self_attn(
                h, cos, sin, attention_mask, past_key_value, use_cache,
                memory_k=mk, memory_v=mv, memory_kr=mkr, memory_doc_mask=mdm, use_memory=True)
        else:
            attn_out, kv, routing = self.self_attn(
                h, cos, sin, attention_mask, past_key_value, use_cache)
        h = residual + attn_out
        residual = h
        h = self.post_attn_norm(h)
        return residual + self.mlp(h), kv, routing

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        memory_inputs: Optional[Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[dict]]:
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._forward_impl, hidden_states, cos, sin, attention_mask,
                past_key_value, use_cache, memory_inputs, use_reentrant=False)
        return self._forward_impl(
            hidden_states, cos, sin, attention_mask, past_key_value, use_cache, memory_inputs)
