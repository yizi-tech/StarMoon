"""
阶段 4: Agent 能力训练
在 Code 模型基础上，赋予工具调用、ReAct 多步推理、任务规划、CodeAct 能力。
混入代码+通用数据防止灾难性遗忘。

用法:
    单卡: python scripts/train_agent.py
    多卡: torchrun --nproc_per_node=8 scripts/train_agent.py
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
logger = logging.getLogger("train_agent")

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
MODEL_PATH = os.environ.get("AGENT_MODEL", "./output/stage3_code/final")
BASE_MODEL = os.environ.get("BASE_MODEL", "./models/Qwen2.5-1.5B-Base")
AGENT_DATA = os.environ.get("AGENT_DATA", "./data/agent_sft.jsonl")
CODE_DATA = os.environ.get("CODE_DATA_SUBSET", "./data/code_sft_subset.jsonl")
GENERAL_DATA = os.environ.get("GENERAL_DATA", "./data/sft_general_subset.jsonl")
OUTPUT_DIR = os.environ.get("AGENT_OUTPUT", "./output/stage4_agent")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "8192"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "8e-6"))

# 数据混合比例
AGENT_RATIO = 0.60   # Agent 数据占比
CODE_RATIO = 0.30    # 代码数据占比 (防遗忘)
GENERAL_RATIO = 0.10 # 通用数据占比 (防遗忘)


def mix_data(agent_data, code_data, general_data):
    """按比例混合 Agent + Code + 通用数据"""
    n_agent = len(agent_data)

    # 根据 Agent 数据量计算其他数据的采样数
    n_code_target = int(n_agent * CODE_RATIO / AGENT_RATIO)
    n_general_target = int(n_agent * GENERAL_RATIO / AGENT_RATIO)

    # 采样代码数据
    if len(code_data) >= n_code_target:
        code_subset = random.sample(code_data, n_code_target)
    else:
        repeats = n_code_target // max(len(code_data), 1) + 1
        code_subset = (code_data * repeats)[:n_code_target]

    # 采样通用数据
    if len(general_data) >= n_general_target:
        general_subset = random.sample(general_data, n_general_target)
    else:
        repeats = n_general_target // max(len(general_data), 1) + 1
        general_subset = (general_data * repeats)[:n_general_target]

    mixed = agent_data + code_subset + general_subset
    random.shuffle(mixed)

    total = len(mixed)
    logger.info(f"数据混合:")
    logger.info(f"  Agent:   {n_agent:,} ({n_agent/total*100:.1f}%)")
    logger.info(f"  Code:    {len(code_subset):,} ({len(code_subset)/total*100:.1f}%)")
    logger.info(f"  General: {len(general_subset):,} ({len(general_subset)/total*100:.1f}%)")
    logger.info(f"  总计:    {total:,}")
    return mixed


def main():
    logger.info("=" * 60)
    logger.info("阶段 4: Agent 能力训练")
    logger.info(f"  模型路径: {MODEL_PATH}")
    logger.info(f"  Agent 数据: {AGENT_DATA}")
    logger.info(f"  Code 数据: {CODE_DATA}")
    logger.info(f"  通用数据: {GENERAL_DATA}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  序列长度: {MAX_SEQ_LENGTH}")
    logger.info(f"  Batch Size: {BATCH_SIZE} × {GRAD_ACCUM} (grad accum)")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info(f"  混合比例: Agent {AGENT_RATIO:.0%} / Code {CODE_RATIO:.0%} / General {GENERAL_RATIO:.0%}")
    logger.info("=" * 60)

    random.seed(42)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型 (从阶段 3 Code 输出)
    logger.info("加载模型...")
    model = StarMoonZ1ForCausalLM.from_pretrained(
        MODEL_PATH,
        use_flash_attn=True,
        torch_dtype=torch.bfloat16,
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count / 1e9:.2f}B")

    # 3. 加载并混合数据
    logger.info("加载 Agent 数据...")
    agent_data = load_dataset(AGENT_DATA)
    logger.info(f"Agent 样本数: {len(agent_data):,}")

    logger.info("加载代码数据 (防遗忘)...")
    code_data = load_dataset(CODE_DATA)
    logger.info(f"代码样本数: {len(code_data):,}")

    logger.info("加载通用数据 (防遗忘)...")
    general_data = load_dataset(GENERAL_DATA)
    logger.info(f"通用样本数: {len(general_data):,}")

    mixed_data = mix_data(agent_data, code_data, general_data)

    train_dataset = SFTDataset(
        mixed_data, tokenizer,
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

    logger.info("开始 Agent 训练...")
    trainer.train()
    logger.info(f"阶段 4 (Agent) 完成! 模型保存在: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
