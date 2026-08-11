"""
StarMoon-z1: 小模型训练推理一站式框架
===================================
专为 1B ~ 14B 参数规模的语言模型设计，
提供从数据处理、模型训练到推理部署的全链路支持。

核心技术特性：
- 标准 Decoder-only Transformer 架构 (LLaMA/Qwen 兼容)
- RoPE 旋转位置编码
- Grouped-Query Attention (GQA)
- SwiGLU 门控前馈网络
- RMSNorm 预归一化
- FlashAttention v2 加速
- LoRA/QLoRA 高效微调
- FSDP 分布式训练
- 多后端推理 (PyTorch / vLLM / llama.cpp)
"""

__version__ = "0.1.0"

from StarMoonZ1.model.config import StarMoonZ1Config
from StarMoonZ1.model.model import StarMoonZ1Model, StarMoonZ1ForCausalLM
from StarMoonZ1.model.lora import LoraConfig, apply_lora
from StarMoonZ1.training.trainer import TrainerBase, TrainingArguments
from StarMoonZ1.training.sft import SFTTrainer
from StarMoonZ1.training.dpo import DPOTrainer
from StarMoonZ1.training.pretrain import PreTrainer, PretrainArguments
from StarMoonZ1.inference.engine import InferenceEngine
from StarMoonZ1.data.dataset import load_dataset, format_chat_template
from StarMoonZ1.evaluation.evaluator import Evaluator, EvalConfig
from StarMoonZ1.evaluation.benchmarks import BENCHMARK_REGISTRY

__all__ = [
    "StarMoonZ1Config",
    "StarMoonZ1Model",
    "StarMoonZ1ForCausalLM",
    "LoraConfig",
    "apply_lora",
    "TrainerBase",
    "TrainingArguments",
    "SFTTrainer",
    "DPOTrainer",
    "PreTrainer",
    "PretrainArguments",
    "InferenceEngine",
    "load_dataset",
    "format_chat_template",
    "Evaluator",
    "EvalConfig",
    "BENCHMARK_REGISTRY",
]
