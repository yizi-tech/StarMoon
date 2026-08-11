"""
MSA 模型：StarMoonZ1MSAModel / StarMoonZ1ForCausalLMWithMemory
==============================================================

在 StarMoon-z1 标准 Decoder 之上接入 MSA 长时记忆：
  - 仅 ``memory_layers`` 指定的层变为 MemorySparseAttention（稀疏路由）
  - 其余层保持标准 GQA，与现有模型权重完全兼容
  - 提供离线编码 ``encode_documents`` 构建记忆库，以及带记忆的生成 /
    Memory Interleave 多轮推理

权重迁移：从 StarMoon-z1 基础 checkpoint 加载时，记忆层多出的 qr_proj /
kr_proj 因无对应权重而随机初始化（strict=False），需经三阶段训练学会路由。
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from StarMoonZ1.model.config import StarMoonZ1Config
from StarMoonZ1.model.model import (
    RMSNorm, SwiGLU, apply_rope, precompute_rope_freqs, _build_weight_mapping,
)
from StarMoonZ1.msa.layers import MSABlock, ChunkMeanPooler
from StarMoonZ1.msa.memory_bank import MemoryBank, MemoryLayerBank


class StarMoonZ1MSAModel(nn.Module):
    def __init__(self, config: StarMoonZ1Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MSABlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        cos, sin = precompute_rope_freqs(
            config.head_dim, config.max_position_embeddings,
            config.rope_theta, config.rope_scaling)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.sliding_window = config.sliding_window
        self.chunk_pooler = ChunkMeanPooler(config.chunk_size)

    def get_rope(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        cos = self.rope_cos[:seq_len].to(device=device, dtype=self.token_embedding.weight.dtype)
        sin = self.rope_sin[:seq_len].to(device=device, dtype=self.token_embedding.weight.dtype)
        return cos, sin

    def _build_causal_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.sliding_window is not None:
            row = torch.arange(T, device=device).unsqueeze(1)
            col = torch.arange(T, device=device).unsqueeze(0)
            causal = col <= row
            window = col >= (row - self.sliding_window + 1)
            mask = torch.where(causal & window, torch.tensor(0.0, device=device, dtype=dtype),
                               torch.tensor(float("-inf"), device=device, dtype=dtype))
            return mask[None, None, :, :]
        return torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), diagonal=1)[None, None, :, :]

    def _truncate_kv_cache(self, past_key_values):
        max_len = self.sliding_window - 1
        out = []
        for k, v in past_key_values:
            if k.shape[2] > max_len:
                out.append((k[:, :, -max_len:, :], v[:, :, -max_len:, :]))
            else:
                out.append((k, v))
        return out

    # ──────────────────────────────────────────
    # Stage 1: 离线文档编码（构建记忆库）
    # ──────────────────────────────────────────
    @torch.no_grad()
    def encode_documents(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_compressed: bool = True,
    ) -> MemoryBank:
        """
        对文档集做前向传播，生成压缩记忆表示。

        Args:
            input_ids:     [N_docs, doc_len] 文档 token（建议 pad 到等长）
            attention_mask:[N_docs, doc_len] 1=有效 0=padding
            return_compressed: 是否返回压缩后的 KV/KR（否则返回原始隐藏状态）
        Returns:
            MemoryBank（per_layer 保存每层压缩表示）
        """
        N, T = input_ids.shape
        device = input_ids.device
        hidden = self.token_embedding(input_ids)
        # 文档级(Parallel) RoPE：每个文档位置从 0 开始
        doc_pos = torch.arange(T, device=device).unsqueeze(0).expand(N, -1)
        cos = self.rope_cos[doc_pos].to(device=device, dtype=hidden.dtype)
        sin = self.rope_sin[doc_pos].to(device=device, dtype=hidden.dtype)
        doc_mask = attention_mask if attention_mask is not None else torch.ones(N, T, dtype=torch.long, device=device)

        per_layer: Dict[int, MemoryLayerBank] = {}
        h = hidden
        for layer in self.layers:
            if layer.is_memory_layer:
                inp = layer.input_norm(h)
                # 该层注意力（无记忆，仅用于推进隐藏状态；因果，与生成一致）
                attn_out, _, _ = layer.self_attn(inp, cos, sin, None, None, False)
                h2 = h + attn_out
                h3 = layer.post_attn_norm(h2)
                out = h2 + layer.mlp(h3)
                # 投影 k/v/kr 并施加文档级 RoPE，按 KV 头维度分块池化 → 输出 5D [N, C, KV, d]
                KV = self.config.num_key_value_heads
                d = self.config.head_dim
                k = layer.self_attn.k_proj(inp).view(N, T, KV, d)
                v = layer.self_attn.v_proj(inp).view(N, T, KV, d)
                kr = layer.self_attn.kr_proj(inp).view(N, T, KV, d)
                k = apply_rope(k.transpose(1, 2), cos, sin).transpose(1, 2)  # [N, T, KV, d]
                kr = apply_rope(kr.transpose(1, 2), cos, sin).transpose(1, 2)
                # 池化时保持 KV 头：reshape 为 (N*KV, T, d) 后池化，再还原 5D
                NT = N * KV
                mask_flat = doc_mask.unsqueeze(1).expand(N, KV, T).reshape(NT, T)
                pk, chunk_mask = self.chunk_pooler(k.reshape(NT, T, d), mask_flat)  # [NT, C, d], [NT, C]
                pv, _ = self.chunk_pooler(v.reshape(NT, T, d), mask_flat)
                pkr, _ = self.chunk_pooler(kr.reshape(NT, T, d), mask_flat)
                Cc = pk.shape[1]
                pk = pk.reshape(N, KV, Cc, d).transpose(1, 2)    # [N, C, KV, d]
                pv = pv.reshape(N, KV, Cc, d).transpose(1, 2)
                pkr = pkr.reshape(N, KV, Cc, d).transpose(1, 2)
                chunk_mask = chunk_mask.reshape(N, KV, Cc).any(dim=1)  # [N, C]
                per_layer[layer.layer_idx] = MemoryLayerBank(pk, pv, pkr, chunk_mask)
                h = out
            else:
                out, _, _ = layer(h, cos, sin, None, None, False, None)
                h = out

        if attention_mask is not None:
            doc_lengths = attention_mask.sum(-1).to(torch.long)
            chunk_counts = ((doc_lengths + self.config.chunk_size - 1) // self.config.chunk_size)
        else:
            doc_lengths = torch.full((N,), T, dtype=torch.long, device=device)
            chunk_counts = torch.full((N,), (T + self.config.chunk_size - 1) // self.config.chunk_size, dtype=torch.long, device=device)

        meta = {"chunk_size": self.config.chunk_size, "router_top_k": self.config.router_top_k,
                "memory_layers": self.config.memory_layers}
        return MemoryBank(per_layer=per_layer, doc_lengths=doc_lengths, chunk_counts=chunk_counts, meta=meta)

    # ──────────────────────────────────────────
    # Stage 2+3: 在线前向（路由 + 生成）
    # ──────────────────────────────────────────
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[Tensor] = None,
        memory_bank: Optional[MemoryBank] = None,
        use_memory: bool = True,
        output_routing_info: bool = False,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape
        device = input_ids.device
        hidden = self.token_embedding(input_ids)
        if position_ids is not None:
            cos = self.rope_cos[position_ids].to(device=device, dtype=hidden.dtype)
            sin = self.rope_sin[position_ids].to(device=device, dtype=hidden.dtype)
        else:
            cos, sin = self.get_rope(T, device)
        if past_key_values is not None:
            if self.sliding_window is not None:
                past_key_values = self._truncate_kv_cache(past_key_values)
            attention_mask = None
        elif attention_mask is None:
            attention_mask = self._build_causal_mask(T, device, hidden.dtype)
        elif attention_mask.dim() == 2:
            causal = self._build_causal_mask(T, device, hidden.dtype)
            padding = torch.where(
                attention_mask[:, None, None, :].bool(),
                torch.zeros(1, dtype=hidden.dtype, device=device),
                torch.full((1,), float("-inf"), dtype=hidden.dtype, device=device))
            attention_mask = causal + padding

        new_pkv = [] if use_cache else None
        routings = [] if output_routing_info else None
        for i, layer in enumerate(self.layers):
            pk = past_key_values[i] if past_key_values is not None else None
            mem_inputs = None
            if layer.is_memory_layer and use_memory and memory_bank is not None and i in memory_bank.per_layer:
                lb = memory_bank.per_layer[i]
                mem_inputs = (lb.memory_k, lb.memory_v, lb.memory_kr, None)
            out, kv, routing = layer(hidden, cos, sin, attention_mask, pk, use_cache, mem_inputs)
            hidden = out
            if use_cache:
                new_pkv.append(kv)
            if output_routing_info:
                routings.append(routing)
        hidden = self.norm(hidden)
        res: Dict[str, Any] = {"last_hidden_state": hidden, "past_key_values": new_pkv}
        if output_routing_info:
            res["routing_infos"] = routings
        return res


class StarMoonZ1ForCausalLMWithMemory(nn.Module):
    """带 MSA 记忆的因果语言模型（训练 / 推理 / 生成）"""

    def __init__(self, config: StarMoonZ1Config):
        super().__init__()
        self.config = config
        self.model = StarMoonZ1MSAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.token_embedding.weight
        if config.depth_scale_init:
            self._apply_depth_scale_init()

    def _apply_depth_scale_init(self):
        n_layers = self.config.num_hidden_layers
        scale = 1.0 / math.sqrt(2.0 * n_layers)
        for layer in self.model.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, std=self.config.initializer_range * scale)
            nn.init.normal_(layer.mlp.down_proj.weight, std=self.config.initializer_range * scale)

    def gradient_checkpointing_enable(self):
        for layer in self.model.layers:
            layer.gradient_checkpointing = True

    def encode_documents(self, input_ids, attention_mask=None, return_compressed=True):
        return self.model.encode_documents(input_ids, attention_mask, return_compressed)

    # ──────────────────────────────────────────
    # 训练前向
    # ──────────────────────────────────────────
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[Tensor] = None,
        memory_bank: Optional[MemoryBank] = None,
        use_memory: bool = True,
        output_routing_info: bool = False,
    ) -> Dict[str, Any]:
        out = self.model(input_ids, attention_mask, past_key_values, use_cache,
                         position_ids, memory_bank, use_memory, output_routing_info)
        hidden = out["last_hidden_state"]
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size),
                                   shift_labels.view(-1), ignore_index=-100)
            if self.config.z_loss_coeff > 0 and self.training:
                loss = loss + self.config.z_loss_coeff * shift_logits.float().pow(2).mean()
            # 可选辅助路由损失：鼓励路由分数具有区分度（默认关闭）
            if self.config.msa_routing_loss_coeff > 0 and output_routing_info:
                routings = out.get("routing_infos") or []
                rloss = 0.0
                for r in routings:
                    if r is not None:
                        rloss = rloss + r["doc_scores"].float().pow(2).mean()
                loss = loss + self.config.msa_routing_loss_coeff * rloss
        res: Dict[str, Any] = {"logits": logits, "loss": loss, "past_key_values": out["past_key_values"]}
        if output_routing_info:
            res["routing_infos"] = out["routing_infos"]
        return res

    # ──────────────────────────────────────────
    # 采样辅助
    # ──────────────────────────────────────────
    @staticmethod
    def _sample(logits, gen, temperature, top_p, top_k, repetition_penalty):
        nl = logits[:, -1, :] / max(temperature, 1e-6)
        if repetition_penalty != 1.0:
            pen_mask = torch.zeros_like(nl)
            pen_mask.scatter_(1, gen, 1.0)
            scale = torch.where(nl > 0, 1.0 / repetition_penalty, repetition_penalty)
            nl = torch.where(pen_mask.bool(), nl * scale, nl)
        if top_k > 0:
            kv, _ = torch.topk(nl, top_k, dim=-1)
            nl[nl < kv[:, -1, None]] = float("-inf")
        if top_p < 1.0:
            sl, si = torch.sort(nl, descending=True, dim=-1)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            rm = cp > top_p
            rm[:, 1:] = rm[:, :-1].clone()
            rm[:, 0] = False
            nl[rm.scatter(1, si, rm)] = float("-inf")
        do_sample = temperature > 0
        if do_sample:
            return torch.multinomial(F.softmax(nl, dim=-1), 1)
        return nl.argmax(dim=-1, keepdim=True)

    # ──────────────────────────────────────────
    # 带记忆生成
    # ──────────────────────────────────────────
    @torch.no_grad()
    def generate_with_memory(
        self,
        input_ids: Tensor,
        memory_bank: Optional[MemoryBank] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 2,
        do_sample: bool = True,
        use_memory: bool = True,
    ) -> Tensor:
        was_training = self.training
        self.eval()
        pkv, gen = None, input_ids
        for _ in range(max_new_tokens):
            mi = gen[:, -1:] if pkv is not None else gen
            pos_ids = None
            if pkv is not None:
                pos_ids = torch.full((gen.shape[0], 1), gen.shape[1] - 1,
                                     dtype=torch.long, device=gen.device)
            o = self.forward(mi, None, None, pkv, True, position_ids=pos_ids,
                             memory_bank=memory_bank, use_memory=use_memory)
            pkv = o["past_key_values"]
            temp = temperature if do_sample else 0.0
            nt = self._sample(o["logits"], gen, temp, top_p, top_k, repetition_penalty)
            gen = torch.cat([gen, nt], dim=-1)
            if (nt == eos_token_id).all():
                break
        if was_training:
            self.train()
        return gen

    # ──────────────────────────────────────────
    # Memory Interleave 多轮推理
    # ──────────────────────────────────────────
    @staticmethod
    def _routing_to_docs(routing: Optional[dict], memory_bank: MemoryBank, tokenizer=None):
        if routing is None or memory_bank is None:
            return []
        idx = routing["selected_indices"]  # [B, k]
        scores = routing["selected_scores"]  # [B, k]
        b0 = idx[0].tolist()
        s0 = scores[0].tolist()
        docs = []
        for j, di in enumerate(b0):
            doc_id = memory_bank.doc_ids[di] if di < len(memory_bank.doc_ids) else f"doc#{di}"
            docs.append({"doc_id": doc_id, "score": float(s0[j])})
        return docs

    @torch.no_grad()
    def generate_with_interleave(
        self,
        input_ids: Tensor,
        memory_bank: MemoryBank,
        tokenizer=None,
        max_rounds: Optional[int] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 2,
        do_sample: bool = True,
        needs_more_fn=None,
    ) -> Tuple[Tensor, List[dict]]:
        """
        Memory Interleave：检索→生成→再检索→再生成的循环。

        说明：是否需要继续检索（needs_more）在生产中应依赖模型训练出的
        [NEED_MORE] 停止符；这里提供 ``needs_more_fn(partial_text, round)``
        回调供自定义，默认策略为「未到最大轮数即继续」。
        """
        max_rounds = max_rounds or self.config.max_interleave_rounds
        trace: List[dict] = []
        cur = input_ids
        for r in range(1, max_rounds + 1):
            # 取本轮路由信息（全量前向，不缓存，仅用于报告检索文档）
            rout = self.forward(cur, memory_bank=memory_bank, use_memory=True, output_routing_info=True)
            routing_infos = rout.get("routing_infos") or []
            last = next((x for x in reversed(routing_infos) if x is not None), None)
            retrieved = self._routing_to_docs(last, memory_bank, tokenizer)
            partial = self.generate_with_memory(
                cur, memory_bank=memory_bank, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty, eos_token_id=eos_token_id,
                do_sample=do_sample, use_memory=True)
            partial_text = tokenizer.decode(partial[0], skip_special_tokens=True) if tokenizer else None
            if needs_more_fn is not None:
                needs_more = bool(needs_more_fn(partial_text, r))
            else:
                needs_more = (r < max_rounds)
            trace.append({
                "round": r,
                "retrieved_docs": retrieved,
                "partial_response": partial_text,
                "needs_more_info": needs_more,
            })
            if not needs_more:
                break
            cur = partial
        return cur, trace

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    # ──────────────────────────────────────────
    # 权重迁移：从 StarMoon-z1 / MSA checkpoint 加载
    # ──────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype=torch.bfloat16,
                        device_map: str = "auto", use_flash_attn: bool = True):
        import os
        # 直接读取本地 config.json，保留 memory_layers 等 MSA 字段
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = StarMoonZ1Config(**cfg_dict)
        cfg.use_flash_attn = use_flash_attn
        m = cls(cfg).to(torch_dtype)

        mapping = _build_weight_mapping(cfg)
        sd = m.state_dict()
        loaded = set()
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(safetensors_path):
            from safetensors.torch import load_file
            hfs = load_file(safetensors_path, device="cpu")
        else:
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map="cpu")
            hfs = hf.state_dict()
            del hf

        for hk, ek in mapping.items():
            if hk in hfs and ek in sd and hfs[hk].shape == sd[ek].shape:
                sd[ek] = hfs[hk]
                loaded.add(ek)
        for k in sd:
            if k not in loaded and k in hfs and hfs[k].shape == sd[k].shape:
                sd[k] = hfs[k]
        m.load_state_dict(sd, strict=False)  # qr/kr 等 MSA 新增参数保持随机初始化
        del hfs

        if device_map == "auto" and torch.cuda.is_available():
            m = m.to("cuda")
        elif device_map != "cpu":
            m = m.to(device_map)
        return m

    def save_pretrained(self, save_path: str, safe_serialization: bool = True):
        import os
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        if safe_serialization:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in self.state_dict().items()},
                      os.path.join(save_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
