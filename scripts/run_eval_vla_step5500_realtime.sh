#!/bin/bash
# VLA Robosuite 评估：realtime 实时窗口，Lift + Panda，输出 vla_eval_step5500.mp4
# 适用：本机有显示屏（laptop/工作站）。使用 --realtime 强制 EGL/glfw，不用 osmesa，避免 osmesa 首次创建卡顿。
# 需 DISPLAY。若未设置：export DISPLAY=:0 或 :1 后执行。

cd /workspace/nanoLLaVA
export NUMBA_DISABLE_JIT_CACHE=1
# 不设置 EVAL_VLA_USE_OSMESA，让脚本用 EGL（无 DISPLAY）或 glfw（有 DISPLAY）

python eval_vla_step5500.py --realtime \
  --vla_ckpt ./outputs/vla_phase1_bf16 \
  --action_head action_head_step5500.bin \
  --model_name_or_path BAAI/Bunny-v1_0-2B-zh \
  --vision_tower siglip-so400m-patch14-384 \
  --steps 300 \
  --output_mp4 vla_eval_step5500.mp4 \
  --fps 30
