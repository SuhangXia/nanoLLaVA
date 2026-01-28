# 快速开始 - Docker 环境配置

## 🔥 立即执行（宿主机上）

### 步骤 1: 停止并删除旧容器

```bash
# 停止旧容器
docker stop nanollava_vla_final

# 删除旧容器
docker rm nanollava_vla_final
```

### 步骤 2: 重新启动 Docker 容器

```bash
docker run -it --gpus all \
    --privileged \
    --net=host \
    -v /dev:/dev \
    -v /home/suhang/projects/nanoLLaVA:/workspace/nanoLLaVA \
    -v /home/suhang/robot_datasets:/datasets \
    --name nanollava_vla_panda \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
    /bin/bash
```

### 步骤 3: 在 Docker 容器内执行

容器启动后，您会看到提示符变为 `root@Ubuntu22-suhang:/workspace#`

然后执行以下命令：

```bash
# 1. 安装 git 和基础工具
apt-get update && apt-get install -y git wget build-essential

# 2. 进入项目目录
cd /workspace/nanoLLaVA

# 3. 运行自动配置脚本
bash docker_setup.sh
```

## ⚡ 或者：一键启动（宿主机上）

```bash
# 清理旧容器并启动新容器，自动运行配置
docker stop nanollava_vla_final 2>/dev/null || true
docker rm nanollava_vla_final 2>/dev/null || true

docker run -it --gpus all \
    --privileged \
    --net=host \
    -v /dev:/dev \
    -v /home/suhang/projects/nanoLLaVA:/workspace/nanoLLaVA \
    -v /home/suhang/robot_datasets:/datasets \
    --name nanollava_vla_panda \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
    /bin/bash -c "apt-get update && apt-get install -y git && cd /workspace/nanoLLaVA && bash docker_setup.sh && /bin/bash"
```

## 🔍 验证安装

在 Docker 容器内运行：

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

应该看到：
```
PyTorch: 2.1.0+cu121
CUDA available: True
```

## ❌ 如果还是有问题

### 手动安装（在 Docker 容器内）：

```bash
# 1. 安装 git
apt-get update
apt-get install -y git wget build-essential

# 2. 进入项目目录
cd /workspace/nanoLLaVA
pwd  # 应该显示: /workspace/nanoLLaVA

# 3. 安装 Python 依赖
pip install numpy scipy pillow tqdm h5py opencv-python transformers accelerate

# 4. 验证
python3 -c "import torch, h5py, transformers; print('All OK')"
```

## 📝 注意事项

- ✅ **宿主机路径**: `/home/suhang/projects/nanoLLaVA`
- ✅ **容器内路径**: `/workspace/nanoLLaVA`
- ✅ **不要**在宿主机上执行 `cd /workspace/nanoLLaVA`（这个路径只在容器内存在）
- ✅ **容器名称已改**: `nanollava_vla_panda`（避免冲突）
