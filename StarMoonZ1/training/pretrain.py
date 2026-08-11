"""
PreTrainer - 预训练训练器
支持: 从零预训练、多阶段训练 (pretrain → annealing → SFT)、流式数据、课程学习。
"""
from __future__ import annotations
import os, math, time, logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from StarMoonZ1.training.trainer import TrainerBase, TrainingArguments
from StarMoonZ1.utils.distributed import (
    is_main_process, wrap_model_ddp, unwrap_model, barrier,
)

logger = logging.getLogger("StarMoonZ1.PreTrainer")


@dataclass
class PretrainArguments(TrainingArguments):
    """预训练专用参数"""
    # 多阶段训练
    total_tokens: int = 100_000_000_000     # 总训练 token 数 (100B)
    annealing_ratio: float = 0.05           # 最后 5% 步数做退火 (高质量数据)
    annealing_lr_ratio: float = 0.1         # 退火阶段学习率降至 10%
    # 课程学习
    curriculum_stages: Optional[List[Dict[str, Any]]] = None
    # 数据
    streaming: bool = False
    tokens_per_step_log: int = 1_000_000    # 每处理 N tokens 打印一次


class PreTrainer(TrainerBase):
    """
    预训练训练器。
    
    核心特性:
    - 支持从零训练 (随机初始化) 或继续预训练 (加载 checkpoint)
    - 多阶段: 大规模预训练 → 退火 (高质量数据) → 交给 SFT
    - 流式数据: 支持 IterableDataset，无需全量加载
    - Token-based 调度: 按总 token 数计算步数和学习率
    - 课程学习: 可选多阶段数据切换
    """
    def __init__(self, model, args: PretrainArguments,
                 train_dataset=None, eval_dataset=None,
                 annealing_dataset=None):
        super().__init__(model, args, train_dataset, eval_dataset)
        self.annealing_dataset = annealing_dataset
        self.tokens_processed = 0
        self._in_annealing = False

    def _calc_total_steps(self, loader) -> int:
        """基于总 token 数计算训练步数"""
        args = self.args
        tokens_per_step = (args.per_device_train_batch_size
                           * args.gradient_accumulation_steps
                           * args.max_seq_length)
        total_steps = args.total_tokens // tokens_per_step
        return total_steps

    def train(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is required")

        args = self.args
        if is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
        barrier()

        self._setup_cuda_performance()
        self._setup_amp()
        self._setup_gradient_checkpointing()

        # 先移到设备，再创建优化器 (保证参数在正确设备上)
        self.model.to(self.device)

        # DDP 包装: 多卡自动启用
        self.model = wrap_model_ddp(self.model, device_id=self.local_rank)

        self.create_optimizer()

        # 计算总步数
        if isinstance(self.train_dataset, IterableDataset):
            # 流式数据: 基于 token 预算
            total_steps = self._calc_total_steps(None)
        else:
            loader = self._create_dataloader(self.train_dataset, args.per_device_train_batch_size)
            steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
            total_steps = steps_per_epoch * args.num_train_epochs

        self.create_scheduler(total_steps)

        # EMA
        if args.use_ema:
            from StarMoonZ1.training.trainer import EMAModel
            self.ema = EMAModel(unwrap_model(self.model), args.ema_decay)

        self._setup_tensorboard()
        if args.resume_from_checkpoint:
            self._resume_checkpoint(args.resume_from_checkpoint)

        self._setup_torch_compile()
        self.model.zero_grad(set_to_none=True)

        annealing_start = int(total_steps * (1 - args.annealing_ratio))

        if is_main_process():
            logger.info("***** Pretraining Config *****")
            logger.info(f"  Total steps = {total_steps:,}")
            logger.info(f"  Total tokens = {args.total_tokens:,}")
            logger.info(f"  Annealing starts at step {annealing_start:,}")
            logger.info(f"  AMP dtype = {self.amp_dtype}")
            logger.info(f"  World size = {self.world_size}")

        start_time = time.time()
        loss_accum = 0.0
        log_count = 0

        # 主训练循环
        # 使用 while 循环以支持退火阶段切换时替换 data_loader，而不错误地推进 epoch
        # 计数器，也不丢弃触发切换的那个 batch。
        for epoch in range(args.num_train_epochs):
            self.epoch = epoch
            data_loader = self._create_dataloader(
                self.train_dataset, args.per_device_train_batch_size, shuffle=True)
            step_in_epoch = 0
            while True:  # 退火切换后重跑本 while (新 loader)，不增加 epoch
                switched = False
                for batch in data_loader:
                    loss = self.training_step(batch)
                    loss_accum += loss.item()
                    log_count += 1
                    self.tokens_processed += (args.per_device_train_batch_size * args.max_seq_length)

                    if (step_in_epoch + 1) % args.gradient_accumulation_steps == 0:
                        self._optimizer_step()
                        self.global_step += 1

                        if self.ema is not None:
                            self.ema.update(unwrap_model(self.model))

                        # 日志 (仅 rank 0)
                        if self.global_step % args.logging_steps == 0 and log_count > 0 and is_main_process():
                            avg_loss = loss_accum / log_count
                            lr = self.scheduler.get_last_lr()[0]
                            elapsed = time.time() - start_time
                            tokens_b = self.tokens_processed / 1e9
                            tps = self.tokens_processed / max(elapsed, 1)
                            logger.info(
                                f"Step {self.global_step:,}/{total_steps:,} | "
                                f"Loss {avg_loss:.4f} | LR {lr:.2e} | "
                                f"Tokens {tokens_b:.2f}B | {tps/1e3:.1f}K tok/s")
                            self._log_tensorboard({
                                "train/loss": avg_loss, "train/lr": lr,
                                "train/tokens_B": tokens_b,
                            }, self.global_step)
                            loss_accum = 0.0
                            log_count = 0

                        # 评估
                        if self.eval_dataset and self.global_step % args.eval_steps == 0:
                            eval_loss = self.evaluate()
                            self._log_tensorboard({"eval/loss": eval_loss}, self.global_step)

                        # 保存 (仅 rank 0)
                        if self.global_step % args.save_steps == 0:
                            barrier()
                            if is_main_process():
                                self.save_checkpoint()
                            barrier()

                        if self.global_step >= total_steps:
                            break

                    step_in_epoch += 1

                    # 退火切换: 先处理完当前 batch，再换 loader，避免丢弃数据；
                    # 不重置 step_in_epoch，让 (step_in_epoch+1) % grad_accum 的模运算
                    # 在切换前后连续，保持梯度累积边界对齐。
                    if (not self._in_annealing and self.annealing_dataset is not None
                            and self.global_step >= annealing_start):
                        self._switch_to_annealing(annealing_start, total_steps)
                        data_loader = self._create_dataloader(
                            self.annealing_dataset, args.per_device_train_batch_size, shuffle=True)
                        switched = True
                        break

                if not switched:
                    break  # 当前 loader 正常耗尽，结束 while
                # 否则继续 while，使用退火 loader 重新迭代 (不增加 epoch)

            if self.global_step >= total_steps:
                break

        self._finalize_training()

    def _optimizer_step(self):
        """执行优化器步进 (含梯度裁剪)"""
        if self.args.max_grad_norm > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _switch_to_annealing(self, anneal_start: int, total_steps: int):
        """切换到退火阶段: 使用高质量数据 + 降低学习率"""
        self._in_annealing = True
        logger.info(f"=== Switching to annealing phase at step {self.global_step} ===")
        # 重建调度器: 从当前 lr 线性衰减到 min_lr
        remaining = total_steps - self.global_step
        from torch.optim.lr_scheduler import LambdaLR
        current_lr = self.scheduler.get_last_lr()[0]
        target_lr = self.args.min_lr
        ratio = target_lr / max(current_lr, 1e-10)

        def anneal_lambda(step):
            return max(ratio, 1.0 - step / max(remaining, 1) * (1.0 - ratio))

        self.scheduler = LambdaLR(self.optimizer, anneal_lambda)
        logger.info(f"  Annealing LR: {current_lr:.2e} → {target_lr:.2e} over {remaining} steps")
