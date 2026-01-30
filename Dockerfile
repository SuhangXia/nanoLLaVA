# Nano-VTLA Docker 镜像
# 基于 PyTorch CUDA 镜像，包含完整的 Nano-VTLA 环境

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 安装 Python 基础依赖
RUN pip install --no-cache-dir \
    numpy>=1.20.0 \
    scipy>=1.7.0 \
    pillow>=9.0.0 \
    tqdm>=4.60.0 \
    matplotlib>=3.5.0 \
    opencv-python>=4.5.0 \
    h5py>=3.7.0 \
    imageio>=2.9.0 \
    imageio-ffmpeg>=0.4.0 \
    zarr>=2.10.0 \
    numcodecs>=0.11.0 \
    imagecodecs>=2023.1.0

# 安装 transformers 和相关库
RUN pip install --no-cache-dir \
    transformers>=4.30.0 \
    accelerate \
    sentencepiece \
    protobuf \
    tokenizers \
    tiktoken

# 安装 FastAPI 和相关 Web 服务依赖
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    gradio \
    requests

# 安装深度学习相关库
RUN pip install --no-cache-dir \
    einops \
    einops-exts \
    timm \
    peft \
    bitsandbytes

# 安装 apex (NVIDIA Apex)
RUN pip install ninja && \
    git clone https://github.com/NVIDIA/apex /tmp/apex && \
    cd /tmp/apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./ && \
    rm -rf /tmp/apex

# 安装 flash-attention (可选，如果失败不影响)
RUN pip install packaging && \
    pip install flash-attn --no-build-isolation || echo "⚠️  flash-attn 安装失败（可选）"

# 安装 xformers (可选)
RUN pip install xformers || echo "⚠️  xformers 安装失败（可选）"

# 安装 deepspeed (可选)
RUN pip install deepspeed || echo "⚠️  deepspeed 安装失败（可选）"

# 复制项目文件（注意：实际使用时，项目代码通过 volume 挂载）
# 这里只是创建目录结构
RUN mkdir -p /workspace/nanoLLaVA

# 设置工作目录为项目目录
WORKDIR /workspace/nanoLLaVA

# 创建启动脚本
RUN echo '#!/bin/bash\n\
cd /workspace/nanoLLaVA\n\
if [ -f "pyproject.toml" ]; then\n\
    pip install -e . || echo "⚠️  NanoLLaVA 安装失败"\n\
fi\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# 设置入口点
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# 默认命令
CMD ["/bin/bash"]

# 标签
LABEL maintainer="Nano-VTLA"
LABEL description="Nano-VTLA Docker image with PyTorch, CUDA, and all dependencies"
LABEL version="1.0"
