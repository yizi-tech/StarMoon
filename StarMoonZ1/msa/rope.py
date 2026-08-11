"""
MSA 双模式 RoPE 位置编码辅助器
==============================

MSA 使用两种 RoPE 模式解决「训练短上下文 → 推理超长上下文」的位置外推：

1. Parallel (文档级) RoPE —— 编码阶段
   每个文档的位置从 0 重新开始，文档之间互不干扰，
   避免长文档拼接时的位置漂移，使 64K 训练可外推到 100M 推理。

2. Global (全局) RoPE —— 生成阶段
   活跃上下文（检索到的记忆块 + 查询 + 生成）使用全局连续位置。
   检索到的 k 个压缩块占据位置 [0, k-1]，查询从位置 k 开始，
   保持「背景 → 查询 → 生成」的因果顺序。
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor


class DocumentRoPEHelper:
    """文档级 / 全局 RoPE 位置计算辅助类（纯函数，无状态）"""

    @staticmethod
    def compute_parallel_positions(
        doc_lengths: List[int],
        max_length: Optional[int] = None,
    ) -> Tensor:
        """
        计算 Parallel RoPE 的位置 ID：每个文档从 0 开始连续编号。

        Args:
            doc_lengths: 每个文档的有效长度列表 [len_1, len_2, ...]
            max_length:   每个文档的占位长度（pad 到等长时，含 padding 位置）

        Returns:
            position_ids: [total_len] 拼接后的位置 ID（padding 位置也连续编号，
                         实际不参与路由评分，由 chunk_mask 屏蔽）
        """
        if max_length is None:
            max_length = max(doc_lengths)
        positions = []
        for length in doc_lengths:
            # 文档内位置 [0, max_length-1]，padding 部分位置继续递增（便于等长堆叠）
            positions.append(torch.arange(max_length, dtype=torch.long))
        return torch.cat(positions)  # [N * max_length]

    @staticmethod
    def compute_global_positions_with_offset(
        query_length: int,
        num_retrieved_chunks: int,
        max_gen_length: int = 2048,
    ) -> Tensor:
        """
        计算 Global RoPE 的位置 ID（带检索偏移）。

        布局： [检索记忆块 (0..k-1)] + [查询 (k..k+q-1)] + [生成预留 (k+q..)]

        Args:
            query_length:         查询 token 数
            num_retrieved_chunks: 检索到的压缩块数（即记忆前缀长度）
            max_gen_length:       预留的生成长度（用于预先分配位置范围）

        Returns:
            position_ids: [context_len + max_gen_length]
        """
        total = num_retrieved_chunks + query_length + max_gen_length
        return torch.arange(total, dtype=torch.long)
