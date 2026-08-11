"""
阶段 4: DPO 偏好对齐
在 Code 模型基础上，使用偏好数据提升回复质量和安全性。

用法:
    单卡: python scripts/train_dpo.py
    多卡: torchrun --nproc_per_node=8 scripts/train_dpo.py
"""
import os
import sys
import logging
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from StarMoonZ1.model.model import StarMoonZ1ForCausalLM
from StarMoonZ1.training.trainer import TrainingArguments
from StarMoonZ1.training.dpo import DPOTrainer, DPODataset
from StarMoonZ1.data.dataset import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_dpo")

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
MODEL_PATH = os.environ.get("DPO_MODEL", "./output/stage4_agent/final")
BASE_MODEL = os.environ.get("BASE_MODEL", "./models/Qwen2.5-1.5B-Base")
DATA_PATH = os.environ.get("DPO_DATA", "./data/dpo_pairs.jsonl")
OUTPUT_DIR = os.environ.get("DPO_OUTPUT", "./output/stage5_dpo")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-7"))
BETA = float(os.environ.get("DPO_BETA", "0.1"))
LOSS_TYPE = os.environ.get("DPO_LOSS", "dpo")  # "dpo" or "ipo"


def main():
    logger.info("=" * 60)
    logger.info("阶段 5: DPO 偏好对齐")
    logger.info(f"  模型路径: {MODEL_PATH}")
    logger.info(f"  偏好数据: {DATA_PATH}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  序列长度: {MAX_SEQ_LENGTH}")
    logger.info(f"  Batch Size: {BATCH_SIZE} × {GRAD_ACCUM} (grad accum)")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info(f"  Beta: {BETA}")
    logger.info(f"  Loss Type: {LOSS_TYPE}")
    logger.info("=" * 60)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型 (从阶段 3 输出)
    logger.info("加载模型...")
    model = StarMoonZ1ForCausalLM.from_pretrained(
        MODEL_PATH,
        use_flash_attn=True,
        torch_dtype=torch.bfloat16,
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count / 1e9:.2f}B")

    # 3. 加载偏好数据
    logger.info("加载 DPO 偏好数据...")
    raw_data = load_dataset(DATA_PATH)
    logger.info(f"偏好对数量: {len(raw_data):,}")

    train_dataset = DPODataset(raw_data, tokenizer, max_length=MAX_SEQ_LENGTH)

    # 4. 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_seq_length=MAX_SEQ_LENGTH,
        learning_rate=LEARNING_RATE,
        min_lr=LEARNING_RATE * 0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        torch_compile=False,             # DPO 有 ref_model，compile 可能冲突
        fused_optimizer=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        dataloader_num_workers=4,
    )

    # 5. 启动 DPO 训练
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        beta=BETA,
        loss_type=LOSS_TYPE,
        length_normalize=True,
        ref_model_offload=True,          # ref_model 放 CPU 省显存
    )

    logger.info("开始 DPO 训练...")
    trainer.train()
    logger.info(f"阶段 5 完成! 最终模型保存在: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
