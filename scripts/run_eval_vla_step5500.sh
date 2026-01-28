#!/bin/bash
# VLA Robosuite 评估：step5500 checkpoint，离屏渲染，输出 vla_eval_step5500.mp4
# 默认用 osmesa（EGL 在无头 Docker 常失败）。若本机 EGL 可用：EVAL_VLA_USE_OSMESA=0 ./scripts/run_eval_vla_step5500.sh
# 若 osmesa 报错找不到 libOSMesa：sudo apt-get install -y libosmesa6 libosmesa6-dev

cd /workspace/nanoLLaVA
export NUMBA_DISABLE_JIT_CACHE=1
export EVAL_VLA_USE_OSMESA=1

python eval_vla_step5500.py \
  --vla_ckpt ./outputs/vla_phase1_bf16 \
  --action_head action_head_step5500.bin \
  --model_name_or_path BAAI/Bunny-v1_0-2B-zh \
  --vision_tower siglip-so400m-patch14-384 \
  --steps 300 \
  --output_mp4 vla_eval_step5500.mp4 \
  --fps 30
