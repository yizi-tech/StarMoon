"""
StarMoon-z1 核心模型
===============
标准 Decoder-only Transformer 架构。
支持 RoPE、GQA、SwiGLU、RMSNorm、FlashAttention。
"""

from __future__ import annotations
import math
import logging
import warnings
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from StarMoonZ1.model.config import StarMoonZ1Config

logger = logging.getLogger("StarMoonZ1.Model")


class RMSNorm(nn.Module):
    """RMS Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs(
    dim: int, max_position: int, theta: float = 10000.0,
    rope_scaling: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    dim_half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, device=device).float() / dim_half))
    if rope_scaling is not None:
        stype = rope_scaling.get("type", "").lower()
        factor = rope_scaling.get("factor", 1.0)
        if stype == "linear":
            freqs = freqs / factor
        elif stype == "dynamic" and dim_half > 2:
            base = theta * (factor ** (dim_half / (dim_half - 2)))
            freqs = 1.0 / (base ** (torch.arange(0, dim_half, device=device).float() / dim_half))
    t = torch.arange(max_position, device=device).float()
    freqs = torch.outer(t, freqs)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    orig_dtype = x.dtype
    x_half = x.float().reshape(*x.shape[:-1], -1, 2)
    x_rot = torch.stack([-x_half[..., 1], x_half[..., 0]], dim=-1).flatten(-2)
    result = x.float() * cos + x_rot * sin
    return result.to(orig_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: StarMoonZ1Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.use_flash_attn = config.use_flash_attn
        self.sliding_window = config.sliding_window
        # QK-Norm: 对 Q/K 做 RMSNorm，稳定 attention logits，允许更大学习率
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._flash_available = None
        if self.use_flash_attn:
            try:
                import flash_attn  # noqa: F401
                self._flash_available = True
            except ImportError:
                self._flash_available = False

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # QK-Norm: 在 RoPE 之前归一化，稳定训练
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_kv = (k, v) if use_cache else None
        if self.num_groups > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_groups, -1, self.head_dim).reshape(B, self.num_heads, -1, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_groups, -1, self.head_dim).reshape(B, self.num_heads, -1, self.head_dim)
        if self._flash_available and self.use_flash_attn:
            from flash_attn import flash_attn_func  # type: ignore
            # 滑动窗口: flash_attn 通过 window_size 参数原生支持
            window_size = (self.sliding_window - 1, 0) if self.sliding_window else (-1, -1)
            attn_output = flash_attn_func(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True,
                window_size=window_size,
            ).transpose(1, 2)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[None, None, :, :]
                attn_weights = attn_weights + attention_mask[:, :, -T:, :]
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(self.attn_dropout(attn_weights), v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_output), new_kv


class SwiGLU(nn.Module):
    def __init__(self, config: StarMoonZ1Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, config: StarMoonZ1Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.gradient_checkpointing = False

    def _forward_impl(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        attn_out, kv_cache = self.self_attn(hidden_states, cos, sin, attention_mask, past_key_value, use_cache)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        return residual + self.mlp(hidden_states), kv_cache

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # 标志位模式梯度检查点：仅在训练时以计算换显存，避免 monkey-patch 双重实现
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._forward_impl,
                hidden_states, cos, sin, attention_mask, past_key_value, use_cache,
                use_reentrant=False,
            )
        return self._forward_impl(
            hidden_states, cos, sin, attention_mask, past_key_value, use_cache)


class StarMoonZ1Model(nn.Module):
    def __init__(self, config: StarMoonZ1Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        cos, sin = precompute_rope_freqs(config.head_dim, config.max_position_embeddings, config.rope_theta, config.rope_scaling)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.sliding_window = config.sliding_window

    def get_rope(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        cos = self.rope_cos[:seq_len].to(device=device, dtype=self.token_embedding.weight.dtype)
        sin = self.rope_sin[:seq_len].to(device=device, dtype=self.token_embedding.weight.dtype)
        return cos, sin

    def _build_causal_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """构建因果注意力掩码，支持滑动窗口 (向量化实现)"""
        if self.sliding_window is not None:
            # 向量化滑动窗口掩码: 避免 Python 循环，大序列下快 100x+
            row_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
            col_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
            # 因果: col <= row; 窗口: col >= row - window + 1
            causal = col_idx <= row_idx
            window = col_idx >= (row_idx - self.sliding_window + 1)
            mask = torch.where(causal & window, torch.tensor(0.0, device=device, dtype=dtype),
                               torch.tensor(float("-inf"), device=device, dtype=dtype))
            return mask[None, None, :, :]
        else:
            # 标准因果掩码
            return torch.triu(
                torch.full((T, T), float("-inf"), device=device, dtype=dtype), diagonal=1
            )[None, None, :, :]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
        B, T = input_ids.shape
        device = input_ids.device
        hidden_states = self.token_embedding(input_ids)
        if position_ids is not None:
            # 直接使用完整 RoPE 缓存按绝对/段内位置取用 (支持 Packing 与 KV-cache 增量解码)
            cos = self.rope_cos[position_ids].to(device=device, dtype=self.token_embedding.weight.dtype)
            sin = self.rope_sin[position_ids].to(device=device, dtype=self.token_embedding.weight.dtype)
        else:
            cos, sin = self.get_rope(T, device)
        if past_key_values is not None:
            # KV cache 增量解码: 滑动窗口下截断 KV 只保留窗口内的历史
            if self.sliding_window is not None:
                past_key_values = self._truncate_kv_cache(past_key_values)
            attention_mask = None
        elif attention_mask is None:
            attention_mask = self._build_causal_mask(T, device, hidden_states.dtype)
        elif attention_mask.dim() == 2:
            # 2D padding mask [B, T]: 1=attend, 0=ignore
            # 与 causal mask 组合，使用 torch.where 避免 0 * (-inf) = NaN
            causal = self._build_causal_mask(T, device, hidden_states.dtype)
            padding_mask = torch.where(
                attention_mask[:, None, None, :].bool(),
                torch.zeros(1, dtype=hidden_states.dtype, device=device),
                torch.full((1,), float("-inf"), dtype=hidden_states.dtype, device=device),
            )
            attention_mask = causal + padding_mask
        # 4D mask (e.g. block-diagonal from PackedSFTDataset) 直接使用
        new_pkv = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pk = past_key_values[i] if past_key_values is not None else None
            hidden_states, kv = layer(hidden_states, cos, sin, attention_mask, pk, use_cache)
            if use_cache:
                new_pkv.append(kv)
        return self.norm(hidden_states), new_pkv

    def _truncate_kv_cache(self, past_key_values: List[Tuple[Tensor, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        """截断 KV Cache，仅保留滑动窗口内的历史 token"""
        max_len = self.sliding_window - 1  # 窗口内最多保留的历史 token 数
        truncated = []
        for k, v in past_key_values:
            if k.shape[2] > max_len:
                truncated.append((k[:, :, -max_len:, :], v[:, :, -max_len:, :]))
            else:
                truncated.append((k, v))
        return truncated


class StarMoonZ1ForCausalLM(nn.Module):
    def __init__(self, config: StarMoonZ1Config):
        super().__init__()
        self.config = config
        self.model = StarMoonZ1Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.token_embedding.weight
        # 深度缩放初始化
        if config.depth_scale_init:
            self._apply_depth_scale_init()

    def _apply_depth_scale_init(self):
        """深度缩放初始化: 深层残差分支权重按 1/sqrt(2*N) 缩放，稳定深层训练"""
        n_layers = self.config.num_hidden_layers
        scale = 1.0 / math.sqrt(2.0 * n_layers)
        for layer in self.model.layers:
            # 缩放 attention output 和 FFN output 的投影层
            nn.init.normal_(layer.self_attn.o_proj.weight, std=self.config.initializer_range * scale)
            nn.init.normal_(layer.mlp.down_proj.weight, std=self.config.initializer_range * scale)

    def gradient_checkpointing_enable(self):
        """启用梯度检查点，以计算换显存（标志位模式，避免 monkey-patch 重复实现）"""
        for layer in self.model.layers:
            layer.gradient_checkpointing = True

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        hidden_states, new_pkv = self.model(input_ids, attention_mask, past_key_values, use_cache, position_ids)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            # Z-loss: 正则化 logits 幅度，防止 softmax 饱和，提升训练稳定性
            if self.config.z_loss_coeff > 0 and self.training:
                z_loss = self.config.z_loss_coeff * shift_logits.float().pow(2).mean()
                loss = loss + z_loss
        return {"logits": logits, "loss": loss, "past_key_values": new_pkv}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 2,
        do_sample: bool = True,
    ) -> Tensor:
        # 保存原始训练状态，生成结束后恢复，避免副作用
        was_training = self.training
        self.eval()
        pkv, gen = None, input_ids
        for _ in range(max_new_tokens):
            mi = gen[:, -1:] if pkv is not None else gen
            # KV-cache 增量解码时传入新 token 的绝对位置，确保 RoPE 正确 (修复潜伏的 RoPE 位置错误)
            pos_ids = None
            if pkv is not None:
                pos_ids = torch.full((gen.shape[0], 1), gen.shape[1] - 1,
                                     dtype=torch.long, device=gen.device)
            o = self.forward(mi, None, None, pkv, True, position_ids=pos_ids)
            pkv = o["past_key_values"]
            # 守卫温度除零：temperature<=0(贪心) 时退化为 argmax，避免 NaN
            nl = o["logits"][:, -1, :] / max(temperature, 1e-8)
            if repetition_penalty != 1.0:
                # 标准 repetition penalty: 正 logit 除以 penalty、负 logit 乘以 penalty，
                # 两者都降低重复 token 概率。注意负 logit 不能用除法(会反向增大概率)。
                penalty_mask = torch.zeros_like(nl)
                penalty_mask.scatter_(1, gen, 1.0)
                pen_mask = penalty_mask.bool()
                scale = torch.where(nl > 0, 1.0 / repetition_penalty, repetition_penalty)
                nl = torch.where(pen_mask, nl * scale, nl)
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
            nt = torch.multinomial(F.softmax(nl, dim=-1), 1) if do_sample else nl.argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nt], dim=-1)
            if (nt == eos_token_id).all():
                break
        # 恢复原始状态
        if was_training:
            self.train()
        return gen

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype=torch.bfloat16, device_map: str = "auto", use_flash_attn: bool = True):
        import os
        from transformers import AutoConfig, AutoModelForCausalLM
        hfcfg = AutoConfig.from_pretrained(model_path)
        cfg = StarMoonZ1Config.from_hf(hfcfg)
        cfg.use_flash_attn = use_flash_attn
        m = cls(cfg).to(torch_dtype)

        # 显存优化: 尝试直接加载 state_dict 避免实例化完整 HF 模型
        mapping = _build_weight_mapping(cfg)
        sd = m.state_dict()
        loaded = set()

        # 优先尝试 safetensors 直接加载 (峰值显存仅 1x)
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(safetensors_path):
            from safetensors.torch import load_file
            hfs = load_file(safetensors_path, device="cpu")
        else:
            # 回退: 加载 HF 模型 (2x 显存)
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
        m.load_state_dict(sd, strict=False)
        del hfs

        if device_map == "auto" and torch.cuda.is_available():
            m = m.to("cuda")
        elif device_map != "cpu":
            m = m.to(device_map)
        return m

    def save_pretrained(self, save_path: str, safe_serialization: bool = True):
        import os, json
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        if safe_serialization:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in self.state_dict().items()}, os.path.join(save_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        logger.info(f"Model saved to {save_path}")


def _build_weight_mapping(config: StarMoonZ1Config) -> Dict[str, str]:
    mapping = {"model.embed_tokens.weight": "model.token_embedding.weight", "lm_head.weight": "lm_head.weight"}
    for i in range(config.num_hidden_layers):
        for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mapping[f"model.layers.{i}.self_attn.{p}.weight"] = f"model.layers.{i}.self_attn.{p}.weight"
        for p in ["gate_proj", "up_proj", "down_proj"]:
            mapping[f"model.layers.{i}.mlp.{p}.weight"] = f"model.layers.{i}.mlp.{p}.weight"
        mapping[f"model.layers.{i}.input_layernorm.weight"] = f"model.layers.{i}.input_norm.weight"
        mapping[f"model.layers.{i}.post_attention_layernorm.weight"] = f"model.layers.{i}.post_attn_norm.weight"
    mapping["model.norm.weight"] = "model.norm.weight"
    return mapping
