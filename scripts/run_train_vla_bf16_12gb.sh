#!/bin/bash
# VLA Phase1：BF16 全量 + 梯度检查点，适用于 12GB 显存、PyTorch 2.11+ / sm_120 (RTX 5070 Ti)
# 当 4-bit 出现 "weight is not an nn.Module" 时使用此脚本（完全避开 bitsandbytes 量化）

cd /workspace/nanoLLaVA

python train_vla.py \
  --model_name_or_path BAAI/Bunny-v1_0-2B-zh \
  --model_type qwen1.5-1.8b \
  --vision_tower siglip-so400m-patch14-384 \
  --data_root /datasets/bridge_numpy \
  --output_dir ./outputs/vla_phase1_bf16 \
  --no_4bit \
  --gradient_checkpointing \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --warmup_steps 500 \
  --num_train_epochs 5 \
  --save_steps 500 \
  --logging_steps 10
