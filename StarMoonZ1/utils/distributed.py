"""
分布式训练工具
=============
提供 DDP 初始化、模型包装、梯度同步、指标聚合等生产级工具函数。
"""

from __future__ import annotations
import os
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger("StarMoonZ1.Distributed")


def setup_distributed(backend: str = "nccl") -> tuple:
    """
    初始化分布式训练环境。
    
    自动检测环境变量 RANK / WORLD_SIZE / LOCAL_RANK，
    支持 torchrun / accelerate launch 等启动方式。
    
    Returns:
        (rank, world_size, local_rank) 三元组
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """清理分布式进程组"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def wrap_model_ddp(
    model: nn.Module,
    device_id: Optional[int] = None,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> nn.Module:
    """
    将模型包装为 DistributedDataParallel。
    
    如果未处于分布式环境，直接返回原模型。
    
    Args:
        model: 待包装模型 (应已在正确设备上)
        device_id: 当前 GPU 设备 ID (默认使用 LOCAL_RANK)
        find_unused_parameters: 是否检测未使用参数 (LoRA 场景建议 True)
        broadcast_buffers: 是否同步 buffers
    
    Returns:
        DDP 包装后的模型，或原始模型 (单卡)
    """
    if not is_distributed():
        return model

    if device_id is None:
        device_id = get_local_rank()

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )
    logger.info(f"Model wrapped with DDP on device {device_id}")
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """从 DDP/DataParallel 包装中提取原始模型"""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


@torch.no_grad()
def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """跨进程求均值 (用于 loss/metric 聚合)"""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


@torch.no_grad()
def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """跨进程求和"""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def barrier():
    """进程同步屏障"""
    if is_distributed():
        dist.barrier()


def create_distributed_sampler(dataset, shuffle: bool = True):
    """
    创建分布式采样器，确保各进程数据不重叠。
    
    Args:
        dataset: 数据集
        shuffle: 是否打乱
    
    Returns:
        DistributedSampler 或 None (单卡)
    """
    if not is_distributed():
        return None
    from torch.utils.data.distributed import DistributedSampler
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
    )
