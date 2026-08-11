"""
StarMoon-z1 LoRA 高效微调模块
========================
低秩适配 (Low-Rank Adaptation) 实现。
支持标准 LoRA 和 RSLoRA (Rank-Stabilized LoRA)。
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("StarMoonZ1.LoRA")


@dataclass
class LoraConfig:
    """LoRA 配置"""
    r: int = 8
    lora_alpha: int = 16
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False  # Rank-Stabilized LoRA

    @property
    def scaling(self) -> float:
        if self.use_rslora:
            return self.lora_alpha / math.sqrt(self.r)
        return self.lora_alpha / self.r


class LoraLinear(nn.Module):
    """
    LoRA 增强线性层：包装原始 nn.Linear，注入低秩旁路。
    
    forward 时: output = W @ x + (B @ A @ x) * scaling
    训练时仅 lora_A / lora_B 参与梯度更新。
    """
    def __init__(self, original: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.05, scaling: float = 1.0):
        super().__init__()
        self.original = original
        self.r = r
        self.scaling = scaling
        in_f = original.in_features
        out_f = original.out_features

        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA 低秩矩阵: A (r x in), B (out x r)
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        # Kaiming 初始化 A，零初始化 B => 初始时 LoRA 输出为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始路径
        result = self.original(x)
        # LoRA 旁路: x -> dropout -> A^T -> B^T -> scale
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return result + lora_out * self.scaling

    @property
    def weight(self):
        """兼容访问原始权重"""
        return self.original.weight

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features


def apply_lora(model: nn.Module, config: LoraConfig, verbose: bool = True) -> nn.Module:
    """
    将 LoRA 适配器注入模型的目标线性层。
    
    直接替换目标 nn.Linear 为 LoraLinear，确保前向传播时 LoRA 生效。
    
    Args:
        model: 目标模型
        config: LoRA 配置
        verbose: 是否打印注入信息
    
    Returns:
        注入 LoRA 后的模型 (原地修改)
    """
    targets = config.target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    total_lora_params = 0
    replaced_count = 0

    # 遍历所有命名模块，找到目标 Linear 并替换
    modules_to_replace = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        # 取模块名最后一段判断是否为目标
        base_name = name.split(".")[-1]
        if base_name in targets:
            modules_to_replace.append((name, mod))

    for name, mod in modules_to_replace:
        # 构建 LoRA 替换模块
        lora_layer = LoraLinear(
            original=mod,
            r=config.r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            scaling=config.scaling,
        )
        # 在父模块中替换
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)

        total_lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
        replaced_count += 1
        if verbose:
            logger.info(f"  LoRA -> {name}: r={config.r}, scaling={config.scaling:.4f}")

    # 统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.info(
            f"  LoRA injected: {replaced_count} layers | "
            f"LoRA params: {total_lora_params:,} | "
            f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)"
        )
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    将 LoRA 权重合并回原始 Linear 层，用于推理部署。
    合并后恢复为普通 nn.Linear，无额外开销。
    """
    modules_to_merge = []
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLinear):
            modules_to_merge.append((name, mod))

    for name, lora_mod in modules_to_merge:
        # 计算 delta_W = B @ A * scaling
        delta = (lora_mod.lora_B @ lora_mod.lora_A) * lora_mod.scaling
        # 合并到原始权重
        original = lora_mod.original
        original.weight.data += delta.to(original.weight.device, original.weight.dtype)

        # 在父模块中恢复原始 Linear
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], original)

    logger.info(f"  Merged {len(modules_to_merge)} LoRA layers back to base weights")
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """提取仅包含 LoRA 参数的 state_dict，用于轻量保存"""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param
    return lora_state
