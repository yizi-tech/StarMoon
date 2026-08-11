#!/bin/bash
# StarMoon-z1 training script

MODEL_PATH=${1:-"Qwen/Qwen2.5-7B"}
DATA_PATH=${2:-"./data/train.jsonl"}
OUTPUT_DIR=${3:-"./output"}

python -m StarMoonZ1.cli train \
    --model "$MODEL_PATH" \
    --data "$DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-5 \
    --lora-r 8 \
    --max-length 2048
