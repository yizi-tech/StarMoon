#!/usr/bin/env python3
"""
StarMoon-z1 SFT training example
=============================
演示如何用 SFTTrainer + LoRA 对 StarMoonZ1 1B 预设模型做指令微调。

运行前准备:
    pip install -e .
    # 准备一份 JSONL 训练数据，每行格式:
    # {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""
import logging

logging.basicConfig(level=logging.INFO)

from transformers import AutoTokenizer

from StarMoonZ1 import StarMoonZ1Config, StarMoonZ1ForCausalLM, LoraConfig, apply_lora
from StarMoonZ1.training import SFTTrainer, TrainingArguments
from StarMoonZ1.training.sft import SFTDataset
from StarMoonZ1.data.dataset import load_dataset


def main():
    # ── 1. 模型与 tokenizer ──────────────────────────────
    config = StarMoonZ1Config.preset_1b()
    model = StarMoonZ1ForCausalLM(config)
    print(f"Model params: {model.num_parameters():,}")

    # 用 Qwen 系 tokenizer（与 1B 预设词表大小 151936 对齐）
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. LoRA 注入（参数高效微调）─────────────────────────
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05)
    apply_lora(model, lora_cfg)

    # ── 3. 数据 ──────────────────────────────────────────
    # load_dataset 返回 List[dict]；SFTDataset 负责指令感知 masking
    raw = load_dataset("data/sft.jsonl")
    train_dataset = SFTDataset(raw, tokenizer, max_length=2048, mask_instruction=True)

    # ── 4. 训练 ──────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir="./output/sft_example",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=2048,
        logging_steps=10,
        save_steps=200,
    )

    trainer = SFTTrainer(
        model, train_args,
        train_dataset=train_dataset,
        pad_token_id=tokenizer.pad_token_id,
    )

    trainer.train()
    print("Training done. Model saved to ./output/sft_example")


if __name__ == "__main__":
    main()
