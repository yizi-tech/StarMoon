"""
MSA 记忆库 (Memory Bank) 数据结构与持久化
========================================

记忆库存储文档经过下层编码 + Chunk-Mean Pooling 后的压缩表示：
  - memory_k / memory_v : 压缩后的内容键/值（拼接进注意力）
  - memory_kr           : 压缩后的路由键（用于 Top-k 文档/块选择）
  - chunk_mask          : 有效块掩码（padding 块置 False）

为支持多层路由（每个 memory layer 的 K/V/KR 投影不同），
``MemoryBank`` 以 ``per_layer[layer_idx]`` 的形式保存每层的压缩表示。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class MemoryLayerBank:
    """单层记忆压缩表示"""
    memory_k: Tensor   # [N_docs, N_chunks, n_kv_heads, head_dim]
    memory_v: Tensor   # [N_docs, N_chunks, n_kv_heads, head_dim]
    memory_kr: Tensor  # [N_docs, N_chunks, n_kv_heads, head_dim]
    chunk_mask: Tensor  # [N_docs, N_chunks] bool，有效块


@dataclass
class MemoryBank:
    """MSA 记忆库（多层）"""
    per_layer: Dict[int, MemoryLayerBank] = field(default_factory=dict)
    doc_ids: List[str] = field(default_factory=list)
    doc_metadata: List[dict] = field(default_factory=list)
    doc_lengths: Optional[Tensor] = None   # [N_docs]
    chunk_counts: Optional[Tensor] = None  # [N_docs]
    meta: dict = field(default_factory=dict)  # 存储 chunk_size / router_top_k 等配置

    def num_docs(self) -> int:
        return len(self.doc_ids)

    def num_layers(self) -> int:
        return len(self.per_layer)

    def to(self, device: str | torch.device) -> "MemoryBank":
        """将全部张量搬到目标设备（用于 CPU↔GPU 分层存储切换）"""
        for lb in self.per_layer.values():
            lb.memory_k = lb.memory_k.to(device)
            lb.memory_v = lb.memory_v.to(device)
            lb.memory_kr = lb.memory_kr.to(device)
            lb.chunk_mask = lb.chunk_mask.to(device)
        if self.doc_lengths is not None:
            self.doc_lengths = self.doc_lengths.to(device)
        if self.chunk_counts is not None:
            self.chunk_counts = self.chunk_counts.to(device)
        return self


def save_memory_bank(mb: MemoryBank, path: str) -> str:
    """
    保存记忆库到磁盘。

    采用 torch.save 序列化（含张量与元信息），单文件便于迁移。
    返回最终保存路径。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # 将 dataclass 转为普通 dict 以便 torch.save 稳健序列化
    payload = {
        "per_layer": {
            str(k): {
                "memory_k": v.memory_k,
                "memory_v": v.memory_v,
                "memory_kr": v.memory_kr,
                "chunk_mask": v.chunk_mask,
            } for k, v in mb.per_layer.items()
        },
        "doc_ids": mb.doc_ids,
        "doc_metadata": mb.doc_metadata,
        "doc_lengths": mb.doc_lengths,
        "chunk_counts": mb.chunk_counts,
        "meta": mb.meta,
    }
    torch.save(payload, path)
    return path


def load_memory_bank(path: str) -> MemoryBank:
    """从磁盘加载记忆库"""
    payload = torch.load(path, map_location="cpu")
    per_layer = {
        int(k): MemoryLayerBank(
            memory_k=v["memory_k"],
            memory_v=v["memory_v"],
            memory_kr=v["memory_kr"],
            chunk_mask=v["chunk_mask"],
        ) for k, v in payload["per_layer"].items()
    }
    return MemoryBank(
        per_layer=per_layer,
        doc_ids=payload.get("doc_ids", []),
        doc_metadata=payload.get("doc_metadata", []),
        doc_lengths=payload.get("doc_lengths"),
        chunk_counts=payload.get("chunk_counts"),
        meta=payload.get("meta", {}),
    )
