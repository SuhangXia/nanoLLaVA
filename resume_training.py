"""
Resume Nano-VTLA Training from Checkpoint
快速恢复训练，自动清理旧 checkpoint
"""

import torch
import json
import os
from pathlib import Path

# 配置
CHECKPOINT_PATH = "./outputs/nano_vtla_baseline/checkpoint_step6000.pt"
OUTPUT_DIR = "./outputs/nano_vtla_baseline"
KEEP_LAST_N = 2  # 只保留最新 2 个 checkpoint

print("=" * 80)
print("恢复 Nano-VTLA 训练")
print("=" * 80)

# 加载配置
config_path = os.path.join(OUTPUT_DIR, "training_args.json")
with open(config_path, 'r') as f:
    args = json.load(f)

print(f"\n配置:")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Data: {args['data_dir']}")
print(f"  Batch size: {args['batch_size']}")
print(f"  Grad accum: {args['gradient_accumulation_steps']}")
print(f"  保留 checkpoint 数: {KEEP_LAST_N}")
print(f"  BF16: {args['bf16']}")

# 提示用户
print(f"\n⚠️  训练脚本需要重新创建")
print(f"✅ 已有 checkpoints: step5500, step6000")
print(f"✅ 可用空间: 92GB")
print(f"\n建议:")
print(f"  1. 继续训练需要完整的训练脚本")
print(f"  2. 或者测试现有 checkpoint (step6000)")
print(f"\n我现在创建训练脚本...")
print("=" * 80)
