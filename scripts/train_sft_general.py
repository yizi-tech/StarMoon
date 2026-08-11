"""
阶段 2: 通用 SFT (Supervised Fine-Tuning)
在 CPT 模型基础上，使用 14GB 指令数据赋予对话能力。

用法:
    单卡: python scripts/train_sft_general.py
    多卡: torchrun --nproc_per_node=8 scripts/train_sft_general.py
"""
import os
import sys
import logging
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from StarMoonZ1.model.model import StarMoonZ1ForCausalLM
from StarMoonZ1.training.trainer import TrainingArguments
from StarMoonZ1.training.sft import SFTTrainer, SFTDataset
from StarMoonZ1.data.dataset import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_sft")

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
# 输入: 阶段 1 CPT 模型 (若跳过 CPT，直接用基座)
MODEL_PATH = os.environ.get("SFT_MODEL", "./output/stage1_cpt/final")
BASE_MODEL = os.environ.get("BASE_MODEL", "./models/Qwen2.5-1.5B-Base")  # tokenizer 来源
DATA_PATH = os.environ.get("SFT_DATA", "./data/sft_14gb.jsonl")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT", "./output/stage2_sft")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "2"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))


def main():
    logger.info("=" * 60)
    logger.info("阶段 2: 通用 SFT")
    logger.info(f"  模型路径: {MODEL_PATH}")
    logger.info(f"  训练数据: {DATA_PATH}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  序列长度: {MAX_SEQ_LENGTH}")
    logger.info(f"  Batch Size: {BATCH_SIZE} × {GRAD_ACCUM} (grad accum)")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info("=" * 60)

    # 1. 加载 tokenizer (始终用基座的 tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型 (从阶段 1 输出)
    logger.info("加载模型...")
    model = StarMoonZ1ForCausalLM.from_pretrained(
        MODEL_PATH,
        use_flash_attn=True,
        torch_dtype=torch.bfloat16,
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count / 1e9:.2f}B")

    # 3. 加载 SFT 数据
    logger.info("加载 SFT 数据...")
    raw_data = load_dataset(DATA_PATH)
    logger.info(f"样本数: {len(raw_data):,}")

    train_dataset = SFTDataset(
        raw_data, tokenizer,
        max_length=MAX_SEQ_LENGTH,
        mask_instruction=True,  # 仅对 assistant 回复计算 loss
    )

    # 4. 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_seq_length=MAX_SEQ_LENGTH,
        learning_rate=LEARNING_RATE,
        min_lr=LEARNING_RATE * 0.1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,     # seq=4096 + bs=8 建议开
        torch_compile=True,
        fused_optimizer=True,
        tf32=True,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        early_stopping_patience=3,       # 防过拟合
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    # 5. 启动训练
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        pad_token_id=tokenizer.pad_token_id,
    )

    logger.info("开始 SFT 训练...")
    trainer.train()
    logger.info(f"阶段 2 完成! 模型保存在: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
