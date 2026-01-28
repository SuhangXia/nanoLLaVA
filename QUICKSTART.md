# Nano-VTLA 快速开始指南

从零开始训练 Vision-Tactile-Language-Action 模型

## 🚀 完整流程

### 1️⃣ 下载 ViTaMIn-B 数据集

**在宿主机上执行：**

```bash
# 进入项目目录
cd /home/suhang/projects/nanoLLaVA

# 运行下载脚本
bash scripts/download_vitamin_b.sh
```

这会下载以下数据：
- `beaker_wiping.zip` (5.93 GB)
- `bean_scooping.zip` (7.78 GB)
- `cube_storage.zip` (4.7 GB)
- `weight_placement.zip` (8.57 GB)

**总大小**: ~27 GB

数据来源: https://huggingface.co/datasets/chuanyune/ViTaMIn-B_data_and_ckpt

### 2️⃣ 解压并组织数据

```bash
# 解压数据
bash scripts/extract_vitamin_b.sh

# 手动组织成 train/val 结构（根据实际数据格式调整）
cd /home/suhang/vitamin_b_data

# 示例：将前 80% 作为训练集
# 具体命令取决于解压后的文件结构
```

**期望的目录结构：**
```
/home/suhang/vitamin_b_data/
├── train/
│   ├── episode_0000.hdf5
│   ├── episode_0001.hdf5
│   └── ...
└── val/
    ├── episode_0800.hdf5
    └── ...
```

### 3️⃣ 启动 Docker 容器并挂载数据

```bash
# 停止旧容器（如果存在）
docker stop nanollava_vla_panda
docker rm nanollava_vla_panda

# 启动新容器，挂载数据集
docker run -it --gpus all \
  --privileged \
  --device=/dev/dri \
  --shm-size=4g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --net=host \
  -v /dev:/dev \
  -v /home/suhang/projects/nanoLLaVA:/workspace/nanoLLaVA \
  -v /home/suhang/vitamin_b_data:/datasets/vitamin_b \
  --name nanollava_vtla \
  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
  /bin/bash
```

**关键挂载点：**
- `/workspace/nanoLLaVA`: 代码目录
- `/datasets/vitamin_b`: 数据集目录

### 4️⃣ 在容器内安装依赖

```bash
# 进入工作目录
cd /workspace/nanoLLaVA

# 安装系统依赖
apt-get update && apt-get install -y git wget

# 安装 Python 依赖
pip install -U \
  torch torchvision \
  transformers accelerate \
  h5py pillow tqdm \
  tensorboard \
  huggingface_hub

# 安装项目
pip install -e .
```

### 5️⃣ 测试组件

```bash
# 测试所有 VTLA 组件
python test_vtla_components.py
```

**预期输出：**
```
================================================================================
Nano-VTLA Component Tests
================================================================================

Testing TactileEncoder (ResNet-18)
================================================================================
Input shape: torch.Size([4, 3, 128, 128])
Output shape: torch.Size([4, 512])
Expected: (4, 512)
✅ TactileEncoder works!

[... 更多测试 ...]

✅ All components tested successfully!
```

### 6️⃣ 测试数据加载

```bash
# 测试数据集加载（无需 checkpoint）
python test_nano_vtla_pipeline.py \
  --data_dir /datasets/vitamin_b \
  --split train \
  --sample_idx 0 \
  --visualize
```

**预期输出：**
```
[ViTaMInBDataset] Loaded 1000 episodes from /datasets/vitamin_b/train
[ViTaMInBDataset] Total 50000 timesteps
[ViTaMInBDataset] Computing action statistics...
...
================================================================================
Testing Sample 0
================================================================================

[Sample Info]
  Image shape: torch.Size([3, 384, 384])
  Tactile shape: torch.Size([3, 128, 128])
  Instruction: Pick up the red block
  Ground Truth Action: [0.05, -0.12, 0.23, 0.10, -0.05, 0.08, 1.00]
...
```

### 7️⃣ 开始训练（从头开始）

```bash
# Stage 1: 训练 Tactile Projector + Action Head
bash scripts/run_training_from_scratch.sh
```

**训练参数：**
- Batch size: 8
- Epochs: 20
- Learning rate: 1e-4
- 冻结: Vision Tower (SigLIP) + LLM (InternLM2)
- 训练: Tactile Projector + Action Head

**训练输出：**
```
============================================
Nano-VTLA 从头训练
============================================

配置:
  数据目录: /datasets/vitamin_b
  输出目录: ./outputs/nano_vtla_from_scratch/stage1
  批次大小: 8
  训练轮数: 20
  学习率: 0.0001

[Trainer] Initialized on device: cuda
[Trainer] Total parameters: 1,234,567,890
[Trainer] Trainable parameters: 45,678,900

Epoch 0: 100%|████████| 6250/6250 [1:23:45<00:00, 74.50it/s, loss=0.0234]
[Eval] Step 1000 | Val Loss: 0.0198 | Metrics: {'mae_translation': 0.015, ...}
...
```

### 8️⃣ 监控训练

在宿主机上启动 TensorBoard：

```bash
tensorboard --logdir /home/suhang/projects/nanoLLaVA/outputs/nano_vtla_from_scratch/stage1/tensorboard/ \
  --host 0.0.0.0 \
  --port 6006
```

在浏览器打开：`http://localhost:6006`

### 9️⃣ Stage 2 训练（可选）

```bash
# Stage 2: LoRA 微调 LLM
bash scripts/run_stage2_lora.sh
```

**训练参数：**
- Batch size: 4
- Epochs: 5
- Learning rate: 5e-5
- LoRA rank: 8
- 从 Stage 1 checkpoint 继续

### 🔟 测试训练好的模型

```bash
# 使用最新的 checkpoint 测试
python test_nano_vtla_pipeline.py \
  --data_dir /datasets/vitamin_b \
  --split val \
  --sample_idx 0 \
  --checkpoint ./outputs/nano_vtla_from_scratch/stage1/checkpoint_step5000.pt \
  --visualize
```

## 📊 预期性能指标

### Stage 1 (10 epochs)
- Translation MAE: < 3 cm
- Rotation MAE: < 15°
- Gripper Accuracy: > 85%
- Training time: ~2-3 hours (8x A100)

### Stage 2 (5 epochs)
- Translation MAE: < 2 cm
- Rotation MAE: < 10°
- Gripper Accuracy: > 90%
- Training time: ~1-2 hours (4x A100)

## 🐛 常见问题

### Q1: 数据集下载太慢
```bash
# 使用镜像加速
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ...
```

### Q2: CUDA Out of Memory
```bash
# 减小 batch size
python train_nano_vtla.py --batch_size 4 --gradient_accumulation_steps 2
```

### Q3: 数据格式不匹配
检查 HDF5 文件结构：
```python
import h5py
with h5py.File('episode_0000.hdf5', 'r') as f:
    print(list(f.keys()))
    print(f['observation/image'].shape)
    print(f['action'].shape)
```

### Q4: 容器内找不到数据
检查挂载：
```bash
# 在容器内
ls -la /datasets/vitamin_b/train/
```

## 📁 文件位置总结

| 位置 | 路径 |
|------|------|
| 代码（宿主机） | `/home/suhang/projects/nanoLLaVA` |
| 代码（容器内） | `/workspace/nanoLLaVA` |
| 数据（宿主机） | `/home/suhang/vitamin_b_data` |
| 数据（容器内） | `/datasets/vitamin_b` |
| 输出（容器内） | `/workspace/nanoLLaVA/outputs` |

## 🎯 下一步

1. ✅ 完成 Stage 1 训练
2. ✅ 评估验证集性能
3. ✅ (可选) Stage 2 LoRA 微调
4. ✅ 在真实机器人上测试
5. ✅ 集成 Octopi Reasoning Head

## 📚 参考文档

- 完整架构说明: `NANO_VTLA_README.md`
- 组件测试: `test_vtla_components.py`
- 数据格式: `bunny/data/vitamin_b_dataset.py`

---

**开始训练吧！** 🚀
