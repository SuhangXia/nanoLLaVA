#!/bin/bash
# Train VLA Action Head on RLBench-Panda HDF5 Data
# 
# Phase 1: Action-only VLA training (Vision → Action)
# Tactile data (Force/Torque, Contact Depth) is dormant for Phase 2 VTLA.

set -e

# Configuration
MODEL_NAME="BAAI/Bunny-v1_0-2B-zh"
MODEL_TYPE="qwen1.5-1.8b"
VISION_TOWER="siglip-so400m-patch14-384"
DATA_ROOT="./data/rlbench_panda_vtla"
OUTPUT_DIR="./outputs/vla_panda_phase1_bf16"

# Training hyperparameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
NUM_EPOCHS=3
NUM_WORKERS=0

# Checkpointing
SAVE_STEPS=500
LOGGING_STEPS=10

# BF16 (recommended for RTX 5070 Ti / sm_120)
BF16_FLAG="--bf16"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# Alternative: 4-bit quantization (for older GPUs)
# BF16_FLAG=""
# GRADIENT_CHECKPOINTING=""

echo "=============================================="
echo "VLA Training: RLBench-Panda Phase 1"
echo "=============================================="
echo "Model: ${MODEL_NAME} (${MODEL_TYPE})"
echo "Data: ${DATA_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION_STEPS} (accum)"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=============================================="

# Check data availability
if [ ! -d "${DATA_ROOT}" ] || [ -z "$(ls -A ${DATA_ROOT}/*.hdf5 2>/dev/null)" ]; then
    echo "ERROR: No HDF5 data found in ${DATA_ROOT}"
    echo "  Run data collection first: bash scripts/run_collect_panda_data.sh"
    exit 1
fi

# Count episodes
NUM_FILES=$(ls ${DATA_ROOT}/*.hdf5 2>/dev/null | wc -l)
echo "Found ${NUM_FILES} HDF5 episodes in ${DATA_ROOT}"
echo ""

# Run training
python train_vla_panda.py \
    --model_name_or_path ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --vision_tower ${VISION_TOWER} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    ${BF16_FLAG} \
    ${GRADIENT_CHECKPOINTING}

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Plot loss: python scripts/plot_vla_loss.py --log ${OUTPUT_DIR}/train_loss.jsonl"
echo "  2. Evaluate: python eval_vla_panda.py --vla_ckpt ${OUTPUT_DIR}"
echo "=============================================="
