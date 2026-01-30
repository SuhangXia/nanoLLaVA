#!/bin/bash
# 快速固化当前运行的容器为镜像
# 用法: ./docker_commit.sh [镜像名称] [标签]

CONTAINER_NAME="nanollava_vtla"
IMAGE_NAME="${1:-nanollava-vtla}"
TAG="${2:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "=============================================="
echo "固化 Docker 容器为镜像"
echo "=============================================="
echo "容器名称: $CONTAINER_NAME"
echo "镜像名称: $FULL_IMAGE_NAME"
echo "=============================================="

# 检查容器是否存在
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ 错误: 容器 '$CONTAINER_NAME' 不存在"
    echo ""
    echo "请先启动容器:"
    echo "  docker start $CONTAINER_NAME"
    echo ""
    echo "或检查容器名称:"
    echo "  docker ps -a"
    exit 1
fi

# 检查容器是否运行
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "⚠️  容器 '$CONTAINER_NAME' 未运行，正在启动..."
    docker start $CONTAINER_NAME
    sleep 2
fi

# 提交容器为镜像
echo ""
echo "正在提交容器为镜像..."
docker commit \
    --author "Nano-VTLA" \
    --message "Nano-VTLA environment with all dependencies installed" \
    $CONTAINER_NAME \
    $FULL_IMAGE_NAME

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 镜像创建成功: $FULL_IMAGE_NAME"
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
    echo ""
    echo "从文件加载镜像:"
    echo "  docker load < nanollava-vtla.tar.gz"
    echo "=============================================="
else
    echo ""
    echo "❌ 镜像创建失败"
    exit 1
fi
