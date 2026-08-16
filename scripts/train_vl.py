#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StarMoon-y1 多模态两阶段训练脚本
================================

在 StarMoonY1ForCausalLMWithVision（基座 StarMoon-z1 Decoder + 视觉通路）上做
视觉-语言对齐 / SFT。采用自包含的原生 PyTorch 训练循环（不依赖框架的纯文本
SFTTrainer，因其不原生支持 pixel_values）。

多模态代码位于独立包 `StarMoonY1`（仓库根/多模态放这里/StarMoonY1），
与基座 `StarMoonZ1` 分离；本脚本通过 sys.path 同时挂载两者。

两阶段（设计稿 §7）：
  阶段 1 (对齐): 冻结 LLM + 冻结视觉塔，仅训练 Projector —— 让视觉 token 对齐到
                  LLM 隐藏空间，学习成本低、不易破坏文本能力。
  阶段 2 (SFT) : 冻结视觉塔，训练 LLM + Projector —— 在指令/对话数据上微调。

数据格式 (JSONL，每行一个样本):
  {"text": "描述这张图片 <image>", "images": ["/path/a.jpg"]}
  {"text": "纯文本样本也可以混训（无 images 字段）"}
  ※ 文本中的 <image> 占位符数量需与 images 列表长度一致（由 processor 展开）。

关键接线 (R1)：训练前 prepare_for_vision(tokenizer) 注册 <image> 特殊 token 并
扩展 embedding/lm_head，否则视觉 token 不会注入 inputs_embeds。

权重来源：VL_MODEL 指向「基础 LLM checkpoint」（其 config 无 vision 字段，视觉
部分随机初始化）。若需从基础 ckpt 起训，用环境变量 VISION_TOWER 注入视觉配置；
若 VL_MODEL 已是「带 vision 配置的 VL checkpoint」，则直接加载，无需注入。

用法:
  阶段 1:
    VISION_TOWER=google/siglip2-base-patch16-256 VISION_HIDDEN_SIZE=768 \
    STAGE=1 VL_MODEL=./checkpoints/starmoon-z1-base \
    BASE_MODEL=./models/Qwen2.5-1.5B-Base VL_DATA=./data/vl.jsonl \
    VL_OUTPUT=./output/vl_stage1 python scripts/train_vl.py
  阶段 2:
    STAGE=2 VL_MODEL=./output/vl_stage1/final \
    BASE_MODEL=./models/Qwen2.5-1.5B-Base VL_DATA=./data/vl_sft.jsonl \
    VL_OUTPUT=./output/vl_stage2 python scripts/train_vl.py
"""
from __future__ import annotations

import os
import sys
import json
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MULTIMODAL_DIR = os.path.join(_REPO_ROOT, "多模态放这里")
# 挂载：StarMoonY1 包（多模态代码，位于「多模态放这里/」）与 StarMoonZ1 包（基座）
sys.path.insert(0, _MULTIMODAL_DIR)
sys.path.insert(0, _REPO_ROOT)

from StarMoonY1.model import StarMoonY1ForCausalLMWithVision
from StarMoonY1.vision_tower import VisionTower
from StarMoonY1.projector import MultiModalProjector
from StarMoonY1.processor import StarMoonY1VLProcessor
from StarMoonY1.collator import VLCollator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_vl")

# ──────────────────────────────────────────
# 配置 (环境变量)
# ──────────────────────────────────────────
VL_MODEL = os.environ.get("VL_MODEL", "./checkpoints/starmoon-z1-base")
BASE_MODEL = os.environ.get("BASE_MODEL", "./models/Qwen2.5-1.5B-Base")
VL_DATA = os.environ.get("VL_DATA", "./data/vl.jsonl")
VL_OUTPUT = os.environ.get("VL_OUTPUT", "./output/vl_stage1")
STAGE = int(os.environ.get("STAGE", "1"))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-3" if STAGE == 1 else "2e-5"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "200"))       # 每 N 步保存
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))
GRAD_CHECKPOINT = os.environ.get("GRAD_CHECKPOINT", "1") == "1"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))        # Windows 下建议 0


class VLDataset(Dataset):
    """JSONL 多模态数据集：文本 + 可选图像 → processor 输出。"""

    def __init__(self, path: str, processor: StarMoonY1VLProcessor, max_length: int):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length
        logger.info(f"VLDataset 载入 {len(self.samples):,} 条样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        text = s["text"]
        imgs = None
        if s.get("images"):
            imgs = [Image.open(p).convert("RGB") for p in s["images"]]
        return self.processor.process(
            text, images=imgs, return_labels=True, max_length=self.max_length
        )


def _inject_vision_if_needed(model: StarMoonY1ForCausalLMWithVision):
    """若基础 ckpt 无 vision 配置，则从 VISION_TOWER 注入视觉塔 + Projector。"""
    if model.config.vision_tower is not None:
        return
    vt = os.environ.get("VISION_TOWER")
    if not vt:
        raise RuntimeError(
            "模型 config 无 vision_tower（推测从基础 LLM ckpt 起训）。\n"
            "请设置环境变量 VISION_TOWER=<视觉编码器名/路径> 注入视觉配置，\n"
            "例如 VISION_TOWER=google/siglip2-base-patch16-256"
        )
    model.config.vision_tower = vt
    vhs = os.environ.get("VISION_HIDDEN_SIZE")
    if vhs:
        model.config.vision_hidden_size = int(vhs)
    else:
        # 未显式给定维度时，在线加载视觉塔读取（需 transformers + 网络）
        logger.info("VISION_HIDDEN_SIZE 未设置，在线加载视觉塔读取输出维度...")
        model.vision_tower = VisionTower(model.config)
        model.vision_tower.load_model()
        model.config.vision_hidden_size = model.vision_tower.output_hidden_size
    model.vision_tower = VisionTower(model.config)
    model.projector = MultiModalProjector(model.config)
    logger.info(f"已注入视觉配置: tower={vt}, hidden={model.config.vision_hidden_size}")


def build_optimizer(model: StarMoonY1ForCausalLMWithVision, lr: float):
    """按阶段设置冻结策略并返回优化器（仅优化 requires_grad 参数）。"""
    if STAGE == 1:
        # 冻结 LLM 全部参数 + 冻结视觉塔，仅训练 Projector
        for p in model.language_model.parameters():
            p.requires_grad_(False)
        if model.vision_tower is not None:
            model.vision_tower.requires_grad_(False)
        params = list(model.projector.parameters())
        tag = "Stage1(仅 Projector)"
    else:
        # Stage2：解冻 LLM，视觉塔维持冻结（默认 freeze_vision_tower=True）
        for p in model.language_model.parameters():
            p.requires_grad_(True)
        if model.vision_tower is not None:
            model.vision_tower.requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
        tag = "Stage2(LLM+Projector)"
    trainable = sum(p.numel() for p in params)
    logger.info(f"[{tag}] 可训参数: {trainable / 1e6:.1f}M / 总参 {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)


def main():
    logger.info("=" * 64)
    logger.info("StarMoon-y1 多模态训练")
    logger.info(f"  阶段      : {STAGE}  ({'对齐' if STAGE == 1 else 'SFT 微调'})")
    logger.info(f"  模型来源  : {VL_MODEL}")
    logger.info(f"  Tokenizer : {BASE_MODEL}")
    logger.info(f"  训练数据  : {VL_DATA}")
    logger.info(f"  输出目录  : {VL_OUTPUT}")
    logger.info(f"  序列长度  : {MAX_SEQ_LENGTH}")
    logger.info(f"  BS×Accum  : {BATCH_SIZE} × {GRAD_ACCUM}")
    logger.info(f"  Epochs    : {NUM_EPOCHS}")
    logger.info(f"  LR        : {LEARNING_RATE}")
    logger.info("=" * 64)

    # 1. tokenizer（始终用基础 tokenizer，<image> 由 prepare_for_vision 注册）
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型（视觉部分若无对应权重则随机初始化）
    logger.info("加载模型...")
    model = StarMoonY1ForCausalLMWithVision.from_pretrained(
        VL_MODEL, torch_dtype=torch.bfloat16
    )
    _inject_vision_if_needed(model)
    model.prepare_for_vision(tokenizer)  # R1 接线：注册 <image> + 扩展 embedding/lm_head
    if GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable()

    # 3. 数据集 + collator
    processor = StarMoonY1VLProcessor(model.config, tokenizer, vision_tower=model.vision_tower)
    dataset = VLDataset(VL_DATA, processor, MAX_SEQ_LENGTH)
    collator = VLCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=NUM_WORKERS,
    )

    # 4. 优化器 + 设备
    optimizer = build_optimizer(model, LEARNING_RATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    logger.info(f"设备: {device}")

    # 5. 训练循环（梯度累积）
    total_steps = len(loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=LEARNING_RATE * 0.1
    )
    global_step = 0
    running_loss = 0.0
    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            image_grids = batch.get("image_grids")

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grids=image_grids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"] / GRAD_ACCUM
            loss.backward()

            running_loss += outputs["loss"].item()
            global_step += 1

            if global_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY
                lr_now = scheduler.get_last_lr()[0]
                logger.info(f"[ep{epoch + 1}] step {global_step}/{total_steps} loss={avg:.4f} lr={lr_now:.2e}")
                running_loss = 0.0

            if global_step % SAVE_EVERY == 0:
                save_dir = os.path.join(VL_OUTPUT, f"step_{global_step}")
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                logger.info(f"已保存检查点: {save_dir}")

    # 6. 最终保存
    final_dir = os.path.join(VL_OUTPUT, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"训练完成! 最终模型保存在: {final_dir}")


if __name__ == "__main__":
    main()
