#!/bin/bash
# VLA PyBullet 评估：WidowX 250 / Franka Panda，384 相机，IK，输出 vla_eval_pybullet.mp4
# 默认 离屏：--direct --tiny_renderer（不依赖 X11/OpenGL，最稳）。要开 GUI：去掉 --direct --tiny_renderer，加 --gui。

cd /workspace/nanoLLaVA
export NUMBA_DISABLE_JIT_CACHE=1

# WidowX 250 (Bridge V2)：Black arm，相机 Top-Right Over-the-shoulder，FOV 75，xyz_scale 0.02
python eval_vla_pybullet.py \
  --direct \
  --tiny_renderer \
  --robot widowx \
  --vla_ckpt ./outputs/vla_phase1_bf16 \
  --action_head action_head_step5500.bin \
  --model_name_or_path BAAI/Bunny-v1_0-2B-zh \
  --vision_tower siglip-so400m-patch14-384 \
  --scene cube \
  --steps 600 \
  --output_mp4 vla_eval_pybullet.mp4 \
  --fps 30 \
  --cam_size 384
# 若仍撞桌可 --xyz_scale 0.015。用 Franka Panda：--robot panda --xyz_scale 0.05
