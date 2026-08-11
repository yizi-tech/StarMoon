"""
TrainerBase - 训练器基类 (生产级)
提供 AMP 混合精度、梯度检查点、DDP 分布式训练、评估循环、早停、断点续训、TensorBoard 日志等功能。
"""
from __future__ import annotations
import os, json, math, time, logging, shutil
from contextlib import nullcontext
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler

from StarMoonZ1.utils.distributed import (
    setup_distributed, is_distributed, is_main_process,
    get_rank, get_world_size, get_local_rank,
    wrap_model_ddp, unwrap_model, create_distributed_sampler,
    all_reduce_mean, barrier,
)

logger = logging.getLogger("StarMoonZ1.Trainer")


@dataclass
class TrainingArguments:
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    min_lr: float = 1e-7                  # cosine 调度最低学习率
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"     # cosine / linear / constant
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: Optional[int] = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    fsdp: bool = False
    max_seq_length: int = 2048
    log_level: str = "info"
    report_to: str = "tensorboard"
    # ─── 新增优化参数 ───
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2    # 预取批次倍数
    early_stopping_patience: int = 0       # 0 = 不启用早停
    early_stopping_threshold: float = 0.0  # 最小改善阈值
    resume_from_checkpoint: Optional[str] = None
    use_ema: bool = False                  # 指数移动平均
    ema_decay: float = 0.999
    seed: int = 42
    # ─── 速度优化参数 ───
    torch_compile: bool = False            # PyTorch 2.0 图编译加速
    compile_mode: str = "reduce-overhead"  # default / reduce-overhead / max-autotune
    fused_optimizer: bool = True           # 使用 fused AdamW (减少 kernel launch)
    tf32: bool = True                      # 启用 TF32 矩阵乘法加速 (Ampere+)
    cudnn_benchmark: bool = True           # cuDNN 自动选择最快算法


class EMAModel:
    """指数移动平均 (Exponential Moving Average)"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                       for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """将 EMA 权重应用到模型 (评估时使用)"""
        self.backup = {name: p.clone() for name, p in model.named_parameters()
                       if name in self.shadow}
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """恢复原始权重"""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    """带 warmup 的 cosine 退火调度器 (含 min_lr)"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


class TrainerBase:
    def __init__(self, model: nn.Module, args: TrainingArguments,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.patience_counter = 0
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ema = None
        self.tb_writer = None
        self._resume_epoch = 0          # 断点续训: 恢复的起始 epoch
        self._resume_step_in_epoch = 0  # 断点续训: 恢复 epoch 内已完成的步数
        self._current_step_in_epoch = 0  # 当前 epoch 内的步数 (用于 checkpoint 保存)
        # 分布式环境自动检测
        self.rank, self.world_size, self.local_rank = setup_distributed()
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        self._setup_seed()

    def _setup_seed(self):
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

    def _setup_cuda_performance(self):
        """CUDA 性能优化: TF32、cuDNN benchmark、matmul 精度"""
        if not torch.cuda.is_available():
            return
        if self.args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        if self.args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        logger.info(f"CUDA perf: TF32={self.args.tf32}, cuDNN benchmark={self.args.cudnn_benchmark}")

    def _setup_torch_compile(self):
        """使用 torch.compile 进行图级别优化 (kernel fusion, 减少 overhead)"""
        if not self.args.torch_compile:
            return
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile requires PyTorch >= 2.0, skipping")
            return
        logger.info(f"Compiling model with mode='{self.args.compile_mode}'...")
        self.model = torch.compile(self.model, mode=self.args.compile_mode)
        logger.info("Model compiled successfully")

    # ──────────────────────────────────────────
    # 优化器 & 调度器
    # ──────────────────────────────────────────

    def create_optimizer(self):
        no_decay = ["bias", "layernorm", "norm", "rmsnorm"]
        decay_params = []
        no_decay_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n.lower() for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        # fused AdamW: 将多个参数的更新合并为单个 CUDA kernel，减少 launch overhead
        use_fused = self.args.fused_optimizer and torch.cuda.is_available()
        self.optimizer = AdamW(param_groups, lr=self.args.learning_rate,
                               betas=(self.args.adam_beta1, self.args.adam_beta2),
                               eps=self.args.adam_epsilon,
                               fused=use_fused)

    def create_scheduler(self, num_training_steps: int):
        warmup_steps = self.args.warmup_steps or int(num_training_steps * self.args.warmup_ratio)
        min_lr_ratio = self.args.min_lr / self.args.learning_rate
        if self.args.lr_scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, warmup_steps, num_training_steps, min_lr_ratio)
        elif self.args.lr_scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, warmup_steps, num_training_steps)
        else:
            self.scheduler = LambdaLR(self.optimizer, lambda _: 1.0)

    # ──────────────────────────────────────────
    # AMP & 梯度检查点
    # ──────────────────────────────────────────

    def _setup_amp(self):
        """配置自动混合精度"""
        if self.args.bf16:
            self.amp_dtype = torch.bfloat16
            self.scaler = None  # bf16 不需要 GradScaler
        elif self.args.fp16:
            self.amp_dtype = torch.float16
            self.scaler = GradScaler("cuda")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

    def _setup_gradient_checkpointing(self):
        """启用梯度检查点以节省显存（统一调用模型的标志位实现，避免重复 monkey-patch）"""
        if not self.args.gradient_checkpointing:
            return
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            logger.warning("模型未实现 gradient_checkpointing_enable，已跳过梯度检查点配置")

    # ──────────────────────────────────────────
    # 数据加载
    # ──────────────────────────────────────────

    def _create_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        # 分布式采样器: 各进程数据不重叠
        sampler = create_distributed_sampler(dataset, shuffle=shuffle)
        kwargs = dict(
            batch_size=batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory and torch.cuda.is_available(),
            drop_last=shuffle,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )
        if sampler is not None:
            kwargs["sampler"] = sampler  # 分布式时不能用 shuffle
        else:
            kwargs["shuffle"] = shuffle
        # prefetch_factor: 提前加载下 N 个 batch，隐藏 I/O 延迟
        if self.args.dataloader_num_workers > 0:
            kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(dataset, **kwargs)

    # ──────────────────────────────────────────
    # 训练核心
    # ──────────────────────────────────────────

    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)
        return outputs["loss"]

    def training_step(self, batch) -> torch.Tensor:
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        if self.amp_dtype != torch.float32:
            with autocast(device_type="cuda", dtype=self.amp_dtype):
                loss = self.compute_loss(self.model, batch)
        else:
            loss = self.compute_loss(self.model, batch)
        # 梯度累积: 缩放 loss
        scaled_loss = loss / self.args.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss.detach()

    def train(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is required")

        if is_main_process():
            os.makedirs(self.args.output_dir, exist_ok=True)
        barrier()

        self._setup_cuda_performance()
        self._setup_amp()
        self._setup_gradient_checkpointing()

        # 模型移到设备后再创建优化器
        self.model.to(self.device)

        # DDP 包装: 多卡自动启用
        self.model = wrap_model_ddp(self.model, device_id=self.local_rank)

        self.create_optimizer()

        train_loader = self._create_dataloader(
            self.train_dataset, self.args.per_device_train_batch_size, shuffle=True)
        num_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        total_steps = num_steps_per_epoch * self.args.num_train_epochs
        self.create_scheduler(total_steps)

        # EMA
        if self.args.use_ema:
            self.ema = EMAModel(unwrap_model(self.model), self.args.ema_decay)

        # TensorBoard (仅 rank 0)
        self._setup_tensorboard()

        # 断点续训
        if self.args.resume_from_checkpoint:
            self._resume_checkpoint(self.args.resume_from_checkpoint)

        self._setup_torch_compile()
        self.model.zero_grad(set_to_none=True)

        if is_main_process():
            logger.info(f"***** Training Config *****")
            logger.info(f"  Total steps = {total_steps}")
            logger.info(f"  Batch size = {self.args.per_device_train_batch_size}")
            logger.info(f"  Gradient accumulation = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Effective batch = {self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.world_size}")
            logger.info(f"  World size = {self.world_size}")
            logger.info(f"  AMP dtype = {self.amp_dtype}")
            logger.info(f"  LR = {self.args.learning_rate}, Min LR = {self.args.min_lr}")

        train_loss_accum = 0.0
        train_loss_count = 0
        start_time = time.time()

        for epoch in range(self._resume_epoch, self.args.num_train_epochs):
            self.epoch = epoch
            # 分布式采样器每 epoch 重新打乱
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            for step, batch in enumerate(train_loader):
                # 断点续训: 跳过恢复 epoch 内已训练的步数
                if epoch == self._resume_epoch and step < self._resume_step_in_epoch:
                    continue
                self._current_step_in_epoch = step + 1

                # DDP no_sync: 梯度累积期间跳过 allreduce，最后一步再同步
                is_accumulation_step = (step + 1) % self.args.gradient_accumulation_steps != 0
                ctx = self.model.no_sync() if (is_accumulation_step and is_distributed()
                                               and hasattr(self.model, 'no_sync')) else nullcontext()
                with ctx:
                    loss = self.training_step(batch)

                # NaN 检测: 立即停止训练避免浪费时间
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss detected at step {self.global_step}, epoch {epoch}. "
                                 f"Saving emergency checkpoint before abort.")
                    if is_main_process():
                        self.save_checkpoint()
                    barrier()
                    raise RuntimeError(f"Training aborted: NaN/Inf loss at global_step={self.global_step}")

                train_loss_accum += loss.item()
                train_loss_count += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.args.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm)

                    # 优化器步进
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # EMA 更新
                    if self.ema is not None:
                        self.ema.update(unwrap_model(self.model))

                    # 日志 (仅 rank 0)
                    if self.global_step % self.args.logging_steps == 0 and is_main_process():
                        avg_loss = train_loss_accum / max(train_loss_count, 1)
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Epoch {epoch} | Step {self.global_step}/{total_steps} | "
                            f"Loss {avg_loss:.4f} | LR {lr:.2e} | "
                            f"Time {elapsed:.1f}s")
                        self._log_tensorboard({"train/loss": avg_loss, "train/lr": lr}, self.global_step)
                        train_loss_accum = 0.0
                        train_loss_count = 0

                    # 评估
                    if self.eval_dataset and self.global_step % self.args.eval_steps == 0:
                        eval_loss = self.evaluate()
                        if is_main_process():
                            self._log_tensorboard({"eval/loss": eval_loss}, self.global_step)
                            if self._check_early_stopping(eval_loss):
                                logger.info(f"Early stopping at step {self.global_step}")
                                self._finalize_training()
                                return

                    # 保存 (仅 rank 0)
                    if self.global_step % self.args.save_steps == 0:
                        barrier()
                        if is_main_process():
                            self.save_checkpoint()
                        barrier()

        self._finalize_training()

    def _finalize_training(self):
        """训练结束: 保存最终模型 (仅 rank 0)"""
        if not is_main_process():
            return
        final_path = os.path.join(self.args.output_dir, "final")
        raw_model = unwrap_model(self.model)
        if self.ema is not None:
            self.ema.apply(raw_model)
        self.save_model(final_path)
        if self.ema is not None:
            self.ema.restore(raw_model)
        if self.tb_writer:
            self.tb_writer.close()
        logger.info(f"Training complete. Model saved to {final_path}")

    # ──────────────────────────────────────────
    # 评估
    # ──────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        raw_model = unwrap_model(self.model)
        if self.ema is not None:
            self.ema.apply(raw_model)

        eval_loader = self._create_dataloader(
            self.eval_dataset, self.args.per_device_eval_batch_size, shuffle=False)
        total_loss, num_batches = 0.0, 0

        for batch in eval_loader:
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            if self.amp_dtype != torch.float32:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    loss = self.compute_loss(self.model, batch)
            else:
                loss = self.compute_loss(self.model, batch)
            total_loss += loss.item()
            num_batches += 1

        if self.ema is not None:
            self.ema.restore(raw_model)

        # 分布式聚合 eval loss
        avg_loss = total_loss / max(num_batches, 1)
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        loss_tensor = all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()

        if is_main_process():
            logger.info(f"  Eval Loss: {avg_loss:.4f}")
        self.model.train()
        return avg_loss

    # ──────────────────────────────────────────
    # 早停
    # ──────────────────────────────────────────

    def _check_early_stopping(self, eval_loss: float) -> bool:
        if self.args.early_stopping_patience <= 0:
            return False
        if eval_loss < self.best_eval_loss - self.args.early_stopping_threshold:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            # 保存最佳模型
            self.save_model(os.path.join(self.args.output_dir, "best"))
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.args.early_stopping_patience

    # ──────────────────────────────────────────
    # 检查点 & 断点续训
    # ──────────────────────────────────────────

    def save_checkpoint(self):
        out = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(out, exist_ok=True)
        # 保存模型 (unwrap DDP)
        raw_model = unwrap_model(self.model)
        if hasattr(raw_model, 'save_pretrained'):
            raw_model.save_pretrained(out)
        else:
            torch.save(raw_model.state_dict(), os.path.join(out, "model.pt"))
        # 保存训练状态
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "step_in_epoch": self._current_step_in_epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        torch.save(state, os.path.join(out, "training_state.pt"))
        logger.info(f"Saved checkpoint: {out}")
        self._cleanup_checkpoints()

    def _resume_checkpoint(self, checkpoint_path: str):
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if not os.path.exists(state_path):
            logger.warning(f"No training state found at {checkpoint_path}, starting fresh")
            return
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.global_step = state["global_step"]
        self._resume_epoch = state["epoch"]
        self.best_eval_loss = state.get("best_eval_loss", float("inf"))
        # 计算恢复 epoch 内已完成的原始步数 (gradient accumulation 前)
        self._resume_step_in_epoch = state.get("step_in_epoch", 0)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        logger.info(f"Resumed from checkpoint: {checkpoint_path} (step {self.global_step}, epoch {self._resume_epoch})")

    def _cleanup_checkpoints(self):
        """保留最近 N 个 checkpoint"""
        if self.args.save_total_limit is None:
            return
        ckpts = sorted(
            [d for d in os.listdir(self.args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1])
        )
        while len(ckpts) > self.args.save_total_limit:
            old = ckpts.pop(0)
            shutil.rmtree(os.path.join(self.args.output_dir, old), ignore_errors=True)

    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        raw_model = unwrap_model(self.model)
        if hasattr(raw_model, 'save_pretrained'):
            raw_model.save_pretrained(path)
        else:
            torch.save(raw_model.state_dict(), os.path.join(path, "model.pt"))

    # ──────────────────────────────────────────
    # TensorBoard
    # ──────────────────────────────────────────

    def _setup_tensorboard(self):
        if self.args.report_to != "tensorboard" or not is_main_process():
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.args.output_dir, "logs")
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        except ImportError:
            logger.warning("tensorboard not installed, skipping")

    def _log_tensorboard(self, metrics: Dict[str, float], step: int):
        if self.tb_writer is None:
            return
        for k, v in metrics.items():
            self.tb_writer.add_scalar(k, v, step)

    # ──────────────────────────────────────────
    # 工具
    # ──────────────────────────────────────────

    def _estimate_throughput(self, elapsed: float) -> float:
        total_tokens = (self.global_step * self.args.gradient_accumulation_steps
                        * self.args.per_device_train_batch_size * self.args.max_seq_length)
        return total_tokens / max(elapsed, 1.0)
