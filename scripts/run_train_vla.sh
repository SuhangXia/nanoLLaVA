#!/bin/bash
# 使用 HuggingFace 模型 ID 启动 VLA Phase1 训练（4-bit，约 12GB）
# 若出现 "weight is not an nn.Module"（PyTorch 2.11+ / sm_120），请改用 run_train_vla_bf16_12gb.sh

cd /workspace/nanoLLaVA

python train_vla.py \
  --model_name_or_path BAAI/Bunny-v1_0-2B-zh \
  --model_type qwen1.5-1.8b \
  --vision_tower siglip-so400m-patch14-384 \
  --data_root /datasets/bridge_numpy \
  --output_dir ./outputs/vla_phase1 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --save_steps 500 \
  --logging_steps 10
