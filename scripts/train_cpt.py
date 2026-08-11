"""
阶段 1: 继续预训练 (Continued Pre-Training)
基于 Qwen2.5-1.5B-Base，使用 8GB 领域数据注入领域知识。

用法:
    单卡: python scripts/train_cpt.py
    多卡: torchrun --nproc_per_node=8 scripts/train_cpt.py
"""
import os
import sys
import logging
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from StarMoonZ1.model.model import StarMoonZ1ForCausalLM
from StarMoonZ1.training.pretrain import PreTrainer, PretrainArguments
from StarMoonZ1.training.sft import PackedSFTDataset
from StarMoonZ1.data.dataset import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_cpt")

# ──────────────────────────────────────────
# 配置 (按需修改)
# ──────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "./models/Qwen2.5-1.5B-Base")
DATA_PATH = os.environ.get("CPT_DATA", "./data/pretrain_8gb.jsonl")
OUTPUT_DIR = os.environ.get("CPT_OUTPUT", "./output/stage1_cpt")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))


def main():
    logger.info("=" * 60)
    logger.info("阶段 1: 继续预训练 (CPT)")
    logger.info(f"  基座模型: {BASE_MODEL}")
    logger.info(f"  训练数据: {DATA_PATH}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  序列长度: {MAX_SEQ_LENGTH}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info("=" * 60)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载基座模型
    logger.info("加载基座模型...")
    model = StarMoonZ1ForCausalLM.from_pretrained(
        BASE_MODEL,
        use_flash_attn=True,
        torch_dtype=torch.bfloat16,
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count / 1e9:.2f}B")

    # 3. 准备数据 (使用 PackedDataset 最大化 GPU 利用率)
    logger.info("加载训练数据...")
    raw_data = load_dataset(DATA_PATH)
    logger.info(f"原始样本数: {len(raw_data):,}")

    train_dataset = PackedSFTDataset(raw_data, tokenizer, max_length=MAX_SEQ_LENGTH)
    logger.info(f"Packed 后样本数: {len(train_dataset):,}")

    # 4. 训练参数
    args = PretrainArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_seq_length=MAX_SEQ_LENGTH,
        learning_rate=LEARNING_RATE,
        min_lr=LEARNING_RATE * 0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=False,    # 80GB 显存不需要
        torch_compile=True,
        fused_optimizer=True,
        tf32=True,
        cudnn_benchmark=True,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        # 退火: 最后 5% 步用高质量数据 + 降低 lr
        annealing_ratio=0.05,
        annealing_lr_ratio=0.1,
    )

    # 5. 启动训练
    trainer = PreTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        annealing_dataset=None,  # 如有高质量退火数据，在此传入
    )

    logger.info("开始训练...")
    trainer.train()
    logger.info(f"阶段 1 完成! 模型保存在: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
