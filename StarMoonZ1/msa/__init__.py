"""
StarMoon-z1 MSA (Memory Sparse Attention) 扩展包
================================================

提供长时记忆（亿级 Token）能力的原生实现：
  - MemorySparseAttention / ChunkMeanPooler / MSABlock  核心层
  - StarMoonZ1MSAModel / StarMoonZ1ForCausalLMWithMemory 模型
  - MemoryBank / save / load                             记忆库
  - DocumentRoPEHelper                                    双模式 RoPE
  - MSAEngine                                             推理引擎

用法概要：
  from StarMoonZ1.msa import MSAEngine, StarMoonZ1ForCausalLMWithMemory
"""

from StarMoonZ1.msa.memory_bank import (
    MemoryBank, MemoryLayerBank, save_memory_bank, load_memory_bank,
)
from StarMoonZ1.msa.rope import DocumentRoPEHelper
from StarMoonZ1.msa.layers import (
    MemorySparseAttention, ChunkMeanPooler, MSABlock,
)
from StarMoonZ1.msa.model import (
    StarMoonZ1MSAModel, StarMoonZ1ForCausalLMWithMemory,
)
from StarMoonZ1.msa.engine import MSAEngine, MemoryConfig

__all__ = [
    "MemoryBank", "MemoryLayerBank", "save_memory_bank", "load_memory_bank",
    "DocumentRoPEHelper",
    "MemorySparseAttention", "ChunkMeanPooler", "MSABlock",
    "StarMoonZ1MSAModel", "StarMoonZ1ForCausalLMWithMemory",
    "MSAEngine", "MemoryConfig",
]
