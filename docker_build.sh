#!/bin/bash
# 使用 Dockerfile 构建镜像
# 用法: ./docker_build.sh [镜像名称] [标签]

IMAGE_NAME="${1:-nanollava-vtla}"
TAG="${2:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "=============================================="
echo "使用 Dockerfile 构建镜像"
echo "=============================================="
echo "镜像名称: $FULL_IMAGE_NAME"
echo "Dockerfile: ./Dockerfile"
echo "=============================================="

# 检查 Dockerfile 是否存在
if [ ! -f "Dockerfile" ]; then
    echo "❌ 错误: Dockerfile 不存在"
    exit 1
fi

# 构建镜像
echo ""
echo "正在构建镜像（这可能需要 10-30 分钟）..."
docker build \
    --tag $FULL_IMAGE_NAME \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 镜像构建成功: $FULL_IMAGE_NAME"
    echo ""
    echo "镜像信息:"
    docker images $FULL_IMAGE_NAME
    echo ""
    echo "=============================================="
    echo "使用新镜像启动容器:"
    echo "=============================================="
    echo "docker run -it --gpus all \\"
    echo "    --privileged \\"
    echo "    --net=host \\"
    echo "    -v /dev:/dev \\"
    echo "    -v /home/suhang/projects/nanoLLaVA:/workspace/nanoLLaVA \\"
    echo "    -v /home/suhang/robot_datasets:/datasets \\"
    echo "    --name nanollava_vtla_new \\"
    echo "    $FULL_IMAGE_NAME \\"
    echo "    /bin/bash"
    echo ""
    echo "保存镜像到文件:"
    echo "  docker save $FULL_IMAGE_NAME | gzip > nanollava-vtla.tar.gz"
    echo "=============================================="
else
    echo ""
    echo "❌ 镜像构建失败"
    exit 1
fi
