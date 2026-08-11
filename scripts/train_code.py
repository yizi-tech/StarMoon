"""
阶段 3: Code 专项训练
在 SFT 模型基础上，用代码数据强化代码生成/补全/调试/推理能力。
混入 30% 通用数据防止灾难性遗忘。

用法:
    单卡: python scripts/train_code.py
    多卡: torchrun --nproc_per_node=8 scripts/train_code.py
"""
import os
import sys
import logging
import random
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from StarMoonZ1.model.model import StarMoonZ1ForCausalLM
from StarMoonZ1.training.trainer import TrainingArguments
from StarMoonZ1.training.sft import SFTTrainer, SFTDataset
from StarMoonZ1.data.dataset import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_code")

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
MODEL_PATH = os.environ.get("CODE_MODEL", "./output/stage2_sft/final")
BASE_MODEL = os.environ.get("BASE_MODEL", "./models/Qwen2.5-1.5B-Base")
CODE_DATA = os.environ.get("CODE_DATA", "./data/code_sft.jsonl")
GENERAL_DATA = os.environ.get("GENERAL_DATA", "./data/sft_general_subset.jsonl")
OUTPUT_DIR = os.environ.get("CODE_OUTPUT", "./output/stage3_code")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "8192"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))
CODE_RATIO = float(os.environ.get("CODE_RATIO", "0.7"))  # 代码数据占比


def mix_data(code_data, general_data, code_ratio=0.7):
    """按比例混合代码数据和通用数据，防止灾难性遗忘"""
    # 计算通用数据需要的数量
    n_code = len(code_data)
    n_general_target = int(n_code * (1 - code_ratio) / code_ratio)

    # 从通用数据中采样
    if len(general_data) >= n_general_target:
        general_subset = random.sample(general_data, n_general_target)
    else:
        # 通用数据不够，全部使用并重复采样
        repeats = n_general_target // len(general_data) + 1
        general_subset = (general_data * repeats)[:n_general_target]

    mixed = code_data + general_subset
    random.shuffle(mixed)

    logger.info(f"数据混合: 代码 {n_code:,} + 通用 {len(general_subset):,} = {len(mixed):,}")
    logger.info(f"代码占比: {n_code / len(mixed) * 100:.1f}%")
    return mixed


def main():
    logger.info("=" * 60)
    logger.info("阶段 3: Code 专项训练")
    logger.info(f"  模型路径: {MODEL_PATH}")
    logger.info(f"  代码数据: {CODE_DATA}")
    logger.info(f"  通用数据: {GENERAL_DATA}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  序列长度: {MAX_SEQ_LENGTH}")
    logger.info(f"  Batch Size: {BATCH_SIZE} × {GRAD_ACCUM} (grad accum)")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info(f"  代码占比: {CODE_RATIO}")
    logger.info("=" * 60)

    random.seed(42)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型 (从阶段 2 输出)
    logger.info("加载模型...")
    model = StarMoonZ1ForCausalLM.from_pretrained(
        MODEL_PATH,
        use_flash_attn=True,
        torch_dtype=torch.bfloat16,
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count / 1e9:.2f}B")

    # 3. 加载并混合数据
    logger.info("加载代码数据...")
    code_data = load_dataset(CODE_DATA)
    logger.info(f"代码样本数: {len(code_data):,}")

    logger.info("加载通用数据 (防遗忘)...")
    general_data = load_dataset(GENERAL_DATA)
    logger.info(f"通用样本数: {len(general_data):,}")

    mixed_data = mix_data(code_data, general_data, code_ratio=CODE_RATIO)

    train_dataset = SFTDataset(
        mixed_data, tokenizer,
        max_length=MAX_SEQ_LENGTH,
        mask_instruction=True,
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
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,     # 8192 长度必须开
        torch_compile=True,
        fused_optimizer=True,
        tf32=True,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        save_total_limit=3,
        early_stopping_patience=3,
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

    logger.info("开始 Code 训练...")
    trainer.train()
    logger.info(f"阶段 3 完成! 模型保存在: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
