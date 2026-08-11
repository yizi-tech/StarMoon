#!/bin/bash
# ============================================================
# StarMoon-z1 一键全流程训练脚本
# 基座: Qwen2.5-1.5B-Base
# 流程: CPT → SFT → Code → DPO → 评估
# 硬件: 8× A800 80GB (NVLink)
# ============================================================
set -e

# ──────────────────────────────────────────
# 全局配置 (按需修改)
# ──────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=8
MASTER_PORT=29500
BASE_MODEL="./models/Qwen2.5-1.5B-Base"

# 数据路径
CPT_DATA="./data/pretrain_8gb.jsonl"
SFT_DATA="./data/sft_14gb.jsonl"
CODE_DATA="./data/code_sft.jsonl"
AGENT_DATA="./data/agent_sft.jsonl"
GENERAL_DATA="./data/sft_general_subset.jsonl"
DPO_DATA="./data/dpo_pairs.jsonl"

# 输出路径
CPT_OUTPUT="./output/stage1_cpt"
SFT_OUTPUT="./output/stage2_sft"
CODE_OUTPUT="./output/stage3_code"
AGENT_OUTPUT="./output/stage4_agent"
DPO_OUTPUT="./output/stage5_dpo"

# 是否跳过某阶段 (设为 true 跳过)
SKIP_CPT=false
SKIP_SFT=false
SKIP_CODE=false
SKIP_AGENT=false
SKIP_DPO=true  # DPO 可选，默认跳过

echo "============================================================"
echo " StarMoon-z1 全流程训练"
echo " 基座模型: ${BASE_MODEL}"
echo " GPU 数量: ${NUM_GPUS}"
echo "============================================================"
echo ""

# ──────────────────────────────────────────
# 阶段 1: 继续预训练 (CPT)
# ──────────────────────────────────────────
if [ "$SKIP_CPT" = false ]; then
    echo "========== 阶段 1: 继续预训练 (CPT) =========="
    echo "数据: ${CPT_DATA}"
    echo "输出: ${CPT_OUTPUT}"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        scripts/train_cpt.py

    echo ""
    echo "✓ 阶段 1 完成!"
    echo ""
else
    echo "========== 跳过阶段 1 (CPT) =========="
fi

# ──────────────────────────────────────────
# 阶段 2: 通用 SFT
# ──────────────────────────────────────────
if [ "$SKIP_SFT" = false ]; then
    echo "========== 阶段 2: 通用 SFT =========="
    echo "数据: ${SFT_DATA}"
    echo "输出: ${SFT_OUTPUT}"
    echo ""

    # 如果跳过了 CPT，直接从基座开始
    if [ "$SKIP_CPT" = true ]; then
        export SFT_MODEL="${BASE_MODEL}"
    fi

    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        scripts/train_sft_general.py

    echo ""
    echo "✓ 阶段 2 完成!"
    echo ""
else
    echo "========== 跳过阶段 2 (SFT) =========="
fi

# ──────────────────────────────────────────
# 阶段 3: Code 专项训练
# ──────────────────────────────────────────
if [ "$SKIP_CODE" = false ]; then
    echo "========== 阶段 3: Code 专项训练 =========="
    echo "代码数据: ${CODE_DATA}"
    echo "通用数据: ${GENERAL_DATA}"
    echo "输出: ${CODE_OUTPUT}"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        scripts/train_code.py

    echo ""
    echo "✓ 阶段 3 完成!"
    echo ""
else
    echo "========== 跳过阶段 3 (Code) =========="
fi

# ──────────────────────────────────────────
# 阶段 4: Agent 能力训练
# ──────────────────────────────────────────
if [ "$SKIP_AGENT" = false ]; then
    echo "========== 阶段 4: Agent 能力训练 =========="
    echo "Agent 数据: ${AGENT_DATA}"
    echo "输出: ${AGENT_OUTPUT}"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        scripts/train_agent.py

    echo ""
    echo "✓ 阶段 4 完成!"
    echo ""
else
    echo "========== 跳过阶段 4 (Agent) =========="
fi

# ──────────────────────────────────────────
# 阶段 5: DPO 偏好对齐 (可选)
# ──────────────────────────────────────────
if [ "$SKIP_DPO" = false ]; then
    echo "========== 阶段 5: DPO 偏好对齐 =========="
    echo "数据: ${DPO_DATA}"
    echo "输出: ${DPO_OUTPUT}"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        scripts/train_dpo.py

    echo ""
    echo "✓ 阶段 5 完成!"
    echo ""
else
    echo "========== 跳过阶段 5 (DPO) =========="
fi

# ──────────────────────────────────────────
# 评估
# ──────────────────────────────────────────
echo "========== 模型评估 =========="

# 确定最终模型路径
if [ "$SKIP_DPO" = false ]; then
    FINAL_MODEL="${DPO_OUTPUT}/final"
elif [ "$SKIP_AGENT" = false ]; then
    FINAL_MODEL="${AGENT_OUTPUT}/final"
elif [ "$SKIP_CODE" = false ]; then
    FINAL_MODEL="${CODE_OUTPUT}/final"
elif [ "$SKIP_SFT" = false ]; then
    FINAL_MODEL="${SFT_OUTPUT}/final"
else
    FINAL_MODEL="${CPT_OUTPUT}/final"
fi

echo "最终模型: ${FINAL_MODEL}"
echo ""

python -m StarMoonZ1.cli eval \
    --model "${FINAL_MODEL}" \
    --benchmarks humaneval,gsm8k,mmlu,perplexity \
    --output ./eval_results \
    --batch-size 8 \
    --max-new-tokens 512

echo ""
echo "============================================================"
echo " 全部完成!"
echo " 最终模型: ${FINAL_MODEL}"
echo " 评估结果: ./eval_results/"
echo ""
echo " 快速测试:"
echo "   python -m StarMoonZ1.cli infer --model ${FINAL_MODEL} --interactive"
echo ""
echo " 启动服务:"
echo "   python -m StarMoonZ1.cli serve --model ${FINAL_MODEL} --port 8000 --webui"
echo "============================================================"
