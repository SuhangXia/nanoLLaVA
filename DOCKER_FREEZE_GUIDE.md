# Docker 容器固化指南

本指南提供两种方式将当前容器环境固化为可重用的 Docker 镜像。

---

## 方法 1: 快速固化（docker commit）⭐ 推荐

**适用场景**: 当前容器已经配置好所有环境，想快速保存当前状态

### 步骤

1. **确保容器正在运行**
   ```bash
   docker ps | grep nanollava_vtla
   # 如果未运行，启动它
   docker start nanollava_vtla
   ```

2. **运行固化脚本**
   ```bash
   cd /home/suhang/projects/nanoLLaVA
   chmod +x docker_commit.sh
   ./docker_commit.sh
   ```

   或者手动执行：
   ```bash
   docker commit \
       --author "Nano-VTLA" \
       --message "Nano-VTLA environment with all dependencies" \
       nanollava_vtla \
       nanollava-vtla:latest
   ```

3. **验证镜像**
   ```bash
   docker images nanollava-vtla
   ```

4. **使用新镜像启动容器**
   ```bash
   docker run -it --gpus all \
       --privileged \
       --net=host \
       -v /dev:/dev \
       -v /home/suhang/projects/nanoLLaVA:/workspace/nanoLLaVA \
       -v /home/suhang/robot_datasets:/datasets \
       --name nanollava_vtla_new \
       nanollava-vtla:latest \
       /bin/bash
   ```

---

## 方法 2: 使用 Dockerfile 构建（可重现）

**适用场景**: 想要一个可重现、可维护的构建过程

### 步骤

1. **构建镜像**
   ```bash
   cd /home/suhang/projects/nanoLLaVA
   chmod +x docker_build.sh
   ./docker_build.sh
   ```

   或者手动执行：
   ```bash
   docker build -t nanollava-vtla:latest -f Dockerfile .
   ```

2. **使用新镜像**（同方法1的步骤4）

---

## 导出和导入镜像

### 导出镜像到文件

```bash
# 导出为压缩文件
docker save nanollava-vtla:latest | gzip > nanollava-vtla.tar.gz

# 或导出为未压缩文件
docker save nanollava-vtla:latest -o nanollava-vtla.tar
```

### 从文件导入镜像

```bash
# 从压缩文件导入
gunzip -c nanollava-vtla.tar.gz | docker load

# 或从未压缩文件导入
docker load < nanollava-vtla.tar
```

### 传输到其他机器

```bash
# 在源机器上
docker save nanollava-vtla:latest | gzip > nanollava-vtla.tar.gz
scp nanollava-vtla.tar.gz user@other-machine:/path/to/

# 在目标机器上
gunzip -c nanollava-vtla.tar.gz | docker load
```

---

## 镜像大小优化

### 查看镜像大小

```bash
docker images nanollava-vtla
```

### 清理未使用的镜像和缓存

```bash
# 清理未使用的镜像
docker image prune -a

# 清理构建缓存
docker builder prune

# 查看磁盘使用
docker system df
```

---

## 完整示例

### 场景：保存当前容器并创建新容器

```bash
# 1. 固化当前容器
./docker_commit.sh nanollava-vtla v1.0

# 2. 导出镜像（可选，用于备份）
docker save nanollava-vtla:v1.0 | gzip > nanollava-vtla-v1.0.tar.gz

# 3. 使用新镜像启动容器
docker run -it --gpus all \
    --privileged \
    --net=host \
    -v /dev:/dev \
    -v /home/suhang/projects/nanoLLaVA:/workspace/nanoLLaVA \
    -v /home/suhang/robot_datasets:/datasets \
    --name nanollava_vtla_v1 \
    nanollava-vtla:v1.0 \
    /bin/bash
```

---

## 验证固化结果

在新容器中验证环境：

```bash
# 进入新容器
docker exec -it nanollava_vtla_v1 /bin/bash

# 验证 Python 环境
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 验证依赖
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import fastapi; print('FastAPI OK')"
python3 -c "import bunny; print('NanoLLaVA OK')"

# 验证项目代码（如果挂载了）
cd /workspace/nanoLLaVA
ls -la
```

---

## 常见问题

### Q1: 镜像太大怎么办？

**A**: 
- 使用多阶段构建（Dockerfile）
- 清理 apt 缓存：`RUN apt-get clean && rm -rf /var/lib/apt/lists/*`
- 使用 `.dockerignore` 排除不必要的文件

### Q2: 如何更新固化镜像？

**A**: 
- 方法1: 在容器中更新，然后重新 commit
- 方法2: 修改 Dockerfile，重新构建

### Q3: 容器中的文件会保存吗？

**A**: 
- ✅ 会保存：安装的软件包、环境变量、系统配置
- ❌ 不会保存：通过 `-v` 挂载的卷（这些是宿主机文件）

### Q4: 如何查看镜像包含什么？

**A**: 
```bash
# 查看镜像历史
docker history nanollava-vtla:latest

# 查看镜像详细信息
docker inspect nanollava-vtla:latest
```

---

## 推荐工作流

1. **开发阶段**: 使用 volume 挂载，代码在宿主机编辑
2. **测试阶段**: 在容器中测试，确保环境一致
3. **固化阶段**: 使用 `docker commit` 快速保存
4. **部署阶段**: 使用 Dockerfile 构建生产镜像
5. **备份阶段**: 导出镜像文件，保存到安全位置

---

## 相关文件

- `Dockerfile`: 完整的构建定义
- `docker_commit.sh`: 快速固化脚本
- `docker_build.sh`: Dockerfile 构建脚本
- `docker_setup.sh`: 容器内环境配置脚本

---

**提示**: 建议定期固化容器，特别是在重要配置更改后。
