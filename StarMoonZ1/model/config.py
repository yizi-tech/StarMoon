"""
StarMoon-z1 模型配置
==============
提供 1B/3B/7B/14B 标准预设及自定义配置能力。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class StarMoonZ1Config:
    """
    StarMoon-z1 模型超参数配置。
    
    支持标准预设 (1B, 3B, 7B, 14B) 及自定义参数。
    
    Args:
        vocab_size: 词表大小
        hidden_size: 隐藏层维度 (d_model)
        num_hidden_layers: Transformer 层数
        num_attention_heads: 注意力头数
        num_key_value_heads: GQA key/value 头数 (== num_attention_heads 时为 MHA)
        intermediate_size: FFN 中间层维度
        max_position_embeddings: 最大位置编码长度
        rms_norm_eps: RMSNorm epsilon
        rope_theta: RoPE base frequency
        rope_scaling: RoPE 缩放配置 (可选)
        hidden_act: 激活函数 (默认 silu)
        use_flash_attn: 是否使用 FlashAttention
        attention_dropout: Attention dropout 概率
        hidden_dropout: Hidden dropout 概率
        tie_word_embeddings: 是否绑定输入输出 embedding
        initializer_range: 参数初始化范围
        bos_token_id: BOS token id
        eos_token_id: EOS token id
        pad_token_id: PAD token id
        torch_dtype: 模型权重数据类型
        use_cache: 是否使用 KV cache
    """
    vocab_size: int = 151936
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None  # GQA key/value 头数，None 表示与 num_attention_heads 相同 (MHA)
    intermediate_size: int = 5632
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    hidden_act: str = "silu"
    use_flash_attn: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    sliding_window: Optional[int] = None  # 滑动窗口注意力窗口大小，None 表示全局注意力
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    
    # 以下字段为训练推理辅助配置
    head_dim: Optional[int] = None  # 若为 None 则自动计算 hidden_size // num_attention_heads
    num_labels: int = 1  # 用于分类头 (备用)
    # ─── 训练稳定性 & 性能增强 ───
    qk_norm: bool = True              # QK-Norm: 稳定注意力 logits，防止训练发散
    z_loss_coeff: float = 1e-4        # Z-loss 系数: 正则化 logits 防止过大
    depth_scale_init: bool = True     # 深度缩放初始化: 深层权重更小，稳定深层网络训练
    
    # ─── MSA (Memory Sparse Attention) 长时记忆扩展 ───
    # 说明: 当 memory_layers 为 None 时，模型退化为标准 Decoder（与现有行为完全一致）。
    #       仅当 memory_layers 非空时启用稀疏记忆路由，其余层仍为标准 GQA。
    memory_layers: Optional[List[int]] = None      # 应用 MSA 稀疏路由的层索引；None=关闭
    router_top_k: int = 5                          # Top-k 文档/块选择数
    chunk_size: int = 128                          # Chunk-Mean Pooling 块大小（压缩粒度）
    routing_key_dim: Optional[int] = None          # 路由键维度；None=hidden_size
    document_wise_rope: bool = True                # 文档级(Parallel) RoPE，提升位置外推
    rope_offset_base: int = 0                      # 全局 RoPE 偏移基数
    enable_memory_interleave: bool = True          # 是否启用 Memory Interleave 多轮推理
    max_interleave_rounds: int = 3                 # 最大交错轮数
    msa_routing_loss_coeff: float = 0.0            # 辅助路由损失权重（0=不额外加 loss）

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self._validate()

    def _validate(self):
        """参数合法性校验，尽早暴露配置错误"""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if self.num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {self.num_key_value_heads}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {self.intermediate_size}")
        if self.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {self.max_position_embeddings}")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive or None, got {self.sliding_window}")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
        if self.z_loss_coeff < 0:
            raise ValueError(f"z_loss_coeff must be non-negative, got {self.z_loss_coeff}")
        if self.router_top_k <= 0:
            raise ValueError(f"router_top_k must be positive, got {self.router_top_k}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.max_interleave_rounds <= 0:
            raise ValueError(f"max_interleave_rounds must be positive, got {self.max_interleave_rounds}")
        if self.memory_layers is not None:
            if not isinstance(self.memory_layers, (list, tuple)):
                raise ValueError("memory_layers must be a list of ints or None")
            for idx in self.memory_layers:
                if not isinstance(idx, int) or idx < 0 or idx >= self.num_hidden_layers:
                    raise ValueError(
                        f"memory_layers contains invalid index {idx!r}; "
                        f"must be in [0, {self.num_hidden_layers - 1}]")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StarMoonZ1Config":
        return cls(**config_dict)
    
    # ──────────────────────────────────────────
    # 标准预设
    # ──────────────────────────────────────────
    
    @classmethod
    def preset_1b(cls) -> "StarMoonZ1Config":
        """~1.5B 参数预设 (优化版: 更深更窄，强化代码/推理能力)"""
        return cls(
            vocab_size=151936,
            hidden_size=2048,
            num_hidden_layers=28,        # 加深: 24→28，提升抽象推理能力
            num_attention_heads=16,
            num_key_value_heads=4,       # 更强 GQA: 8→4，省参数给深度
            intermediate_size=6144,      # 略增 FFN 宽度
            max_position_embeddings=32768,
            rope_theta=10000.0,
            qk_norm=True,
            z_loss_coeff=1e-4,
            depth_scale_init=True,
            tie_word_embeddings=True,    # 绑定 embedding，省出的参数给层数
        )
    
    @classmethod
    def preset_3b(cls) -> "StarMoonZ1Config":
        """~2.8B 参数预设"""
        return cls(
            vocab_size=151936,
            hidden_size=3200,
            num_hidden_layers=26,
            num_attention_heads=32,
            num_key_value_heads=8,   # GQA
            intermediate_size=8640,
            max_position_embeddings=32768,
            rope_theta=10000.0,
        )
    
    @classmethod
    def preset_7b(cls) -> "StarMoonZ1Config":
        """~6.7B 参数预设 (接近 LLaMA-2 7B)"""
        return cls(
            vocab_size=151936,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,   # GQA
            intermediate_size=11008,
            max_position_embeddings=65536,
            rope_theta=10000.0,
        )
    
    @classmethod
    def preset_14b(cls) -> "StarMoonZ1Config":
        """~13B 参数预设"""
        return cls(
            vocab_size=151936,
            hidden_size=5120,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=10,  # GQA
            intermediate_size=13824,
            max_position_embeddings=65536,
            rope_theta=10000.0,
        )
    
    @classmethod
    def from_hf(cls, hf_config: Any) -> "StarMoonZ1Config":
        """从 HuggingFace PretrainedConfig 导入"""
        return cls(
            vocab_size=getattr(hf_config, "vocab_size", 151936),
            hidden_size=getattr(hf_config, "hidden_size", 2048),
            num_hidden_layers=getattr(hf_config, "num_hidden_layers", 24),
            num_attention_heads=getattr(hf_config, "num_attention_heads", 16),
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", 16),
            intermediate_size=getattr(hf_config, "intermediate_size", 5632),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 32768),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            rope_scaling=getattr(hf_config, "rope_scaling", None),
            hidden_act=getattr(hf_config, "hidden_act", "silu"),
            use_flash_attn=getattr(hf_config, "use_flash_attn", True),
            sliding_window=getattr(hf_config, "sliding_window", None),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
            bos_token_id=getattr(hf_config, "bos_token_id", 1),
            eos_token_id=getattr(hf_config, "eos_token_id", 2),
            pad_token_id=getattr(hf_config, "pad_token_id", 0),
            torch_dtype=getattr(hf_config, "torch_dtype", "bfloat16"),
            # 训练/推理辅助配置: 从 StarMoonZ1 导出的 config.json 回读时必须保留，
            # 否则回落到默认值会导致与训练时不一致 (如 qk_norm 被错误关闭、head_dim 重算等)
            head_dim=getattr(hf_config, "head_dim", None),
            attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
            initializer_range=getattr(hf_config, "initializer_range", 0.02),
            qk_norm=getattr(hf_config, "qk_norm", True),
            z_loss_coeff=getattr(hf_config, "z_loss_coeff", 1e-4),
            depth_scale_init=getattr(hf_config, "depth_scale_init", True),
            # ─── MSA 扩展字段（从 StarMoonZ1 导出的 config.json 回读时必须保留）───
            memory_layers=getattr(hf_config, "memory_layers", None),
            router_top_k=getattr(hf_config, "router_top_k", 5),
            chunk_size=getattr(hf_config, "chunk_size", 128),
            routing_key_dim=getattr(hf_config, "routing_key_dim", None),
            document_wise_rope=getattr(hf_config, "document_wise_rope", True),
            rope_offset_base=getattr(hf_config, "rope_offset_base", 0),
            enable_memory_interleave=getattr(hf_config, "enable_memory_interleave", True),
            max_interleave_rounds=getattr(hf_config, "max_interleave_rounds", 3),
            msa_routing_loss_coeff=getattr(hf_config, "msa_routing_loss_coeff", 0.0),
        )
    
    @property
    def num_params_estimate(self) -> str:
        """估算参数量级"""
        v = self.vocab_size
        d = self.hidden_size
        n = self.num_hidden_layers
        h = self.num_attention_heads
        k = self.num_key_value_heads
        i = self.intermediate_size
        d_head = self.head_dim
        
        # Embedding
        embed_params = v * d * (1 if not self.tie_word_embeddings else 0)
        lm_head = v * d
        
        # Per layer: attention + FFN + norms
        attn_q = d * (h * d_head)
        attn_k = d * (k * d_head)
        attn_v = d * (k * d_head)
        attn_o = (h * d_head) * d
        attn_total = attn_q + attn_k + attn_v + attn_o
        
        ffn_gate = d * i
        ffn_up = d * i
        ffn_down = i * d
        ffn_total = ffn_gate + ffn_up + ffn_down
        
        norm_total = d * 2  # pre_attn_norm + pre_ffn_norm
        
        layer_total = attn_total + ffn_total + norm_total
        total = layer_total * n + embed_params + lm_head + d  # final_norm
        
        if total < 1e9:
            return f"{total/1e6:.1f}M"
        return f"{total/1e9:.2f}B"
