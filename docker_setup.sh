#!/bin/bash
# Docker 环境设置脚本 - VTLA Panda 项目
# 在 Docker 容器内运行此脚本以配置完整的环境

set -e  # 遇到错误立即退出

echo "=============================================="
echo "Docker 环境设置 - VTLA Panda"
echo "=============================================="

# 1. 安装基础系统工具（仅在缺失时）
echo ""
echo "[1/6] 检查基础系统工具 (git, wget, build-essential)..."
if command -v git >/dev/null 2>&1; then
    echo "✓ git 已安装，跳过系统依赖安装"
else
    echo "未检测到 git，开始安装系统工具..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y \
        git \
        wget \
        build-essential \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1
    echo "✓ 系统工具安装完成"
fi

# 2. 导航到项目目录
echo ""
echo "[2/6] 导航到项目目录..."
cd /workspace/nanoLLaVA || {
    echo "错误: 找不到 /workspace/nanoLLaVA 目录"
    echo "请确保 Docker 挂载正确: -v /home/suhang/projects/nanoLLaVA:/workspace/nanoLLaVA"
    exit 1
}
echo "✓ 当前目录: $(pwd)"

# 3. 安装 Python 基础依赖
echo ""
echo "[3/6] 安装 Python 基础依赖..."
if [ -f "requirements_vtla.txt" ]; then
    pip install --no-cache-dir -r requirements_vtla.txt
else
    pip install --no-cache-dir \
        numpy \
        scipy \
        pillow \
        tqdm \
        matplotlib \
        opencv-python \
        h5py \
        imageio \
        imageio-ffmpeg
fi

echo "✓ Python 基础依赖安装完成"

# 4. 安装 transformers 和相关库
echo ""
echo "[4/6] 安装 transformers 和 NanoLLaVA 依赖..."
pip install --no-cache-dir \
    transformers \
    accelerate \
    sentencepiece \
    protobuf

# 安装 bitsandbytes (可选，用于 4-bit 量化)
pip install --no-cache-dir -q bitsandbytes || echo "⚠ bitsandbytes 安装失败（可选）"

# 安装 peft (可选，用于 LoRA)
pip install --no-cache-dir -q peft || echo "⚠ peft 安装失败（可选）"

echo "✓ transformers 依赖安装完成"

# 5. 安装 NanoLLaVA（如果尚未安装）
echo ""
echo "[5/6] 安装 NanoLLaVA..."
if [ "${SKIP_NANOLLAVA_INSTALL:-0}" = "1" ]; then
    echo "⚠ SKIP_NANOLLAVA_INSTALL=1，跳过 NanoLLaVA 安装"
elif python3 - << 'EOF' >/dev/null 2>&1
from bunny.model import BunnyQwenForCausalLM
EOF
then
    echo "✓ 检测到 NanoLLaVA 已安装，跳过安装"
elif [ -f "pyproject.toml" ]; then
    pip install -e .
    echo "✓ NanoLLaVA 安装完成"
else
    echo "⚠ 找不到 pyproject.toml，跳过 NanoLLaVA 安装"
fi

# 6. 验证安装
echo ""
echo "[6/6] 验证安装..."
python3 << 'EOF'
import sys
try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except:
    print("✗ numpy 未安装")
    sys.exit(1)

try:
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except:
    print("✗ torch 未安装")
    sys.exit(1)

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except:
    print("✗ transformers 未安装")
    sys.exit(1)

try:
    import h5py
    print(f"✓ h5py {h5py.__version__}")
except:
    print("✗ h5py 未安装")
    sys.exit(1)

try:
    import bunny  # noqa: F401
    print("✓ NanoLLaVA (Bunny) 已安装")
except Exception as e:
    print(f"⚠ NanoLLaVA (Bunny) 未安装或未在当前环境可见: {e}")

print("\n✓ 核心依赖验证通过")
EOF

# 7. 显示 RLBench 安装说明
echo ""
echo "=============================================="
echo "环境设置完成！"
echo "=============================================="
echo ""
echo "核心依赖已安装。如需使用 RLBench 模拟器："
echo ""
echo "1. 下载并安装 CoppeliaSim:"
echo "   wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
echo "   tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
echo "   export COPPELIASIM_ROOT=/workspace/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
echo "   export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT"
echo ""
echo "2. 安装 PyRep 和 RLBench:"
echo "   pip install git+https://github.com/stepjam/PyRep.git"
echo "   pip install git+https://github.com/stepjam/RLBench.git"
echo ""
echo "3. 测试 VTLA 管道:"
echo "   cd /workspace/nanoLLaVA"
echo "   python3 test_vtla_pipeline.py"
echo ""
echo "=============================================="
