#!/bin/bash
# StarMoon-z1 inference script

MODEL_PATH=${1:-"./output/checkpoint-final"}
PROMPT=${2:-"Hello, how are you?"}

python -m StarMoonZ1.cli infer \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --temperature 0.7 \
    --max-new-tokens 512
