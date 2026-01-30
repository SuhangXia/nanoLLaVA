# Nano-VTLA API 部署指南

FastAPI 服务部署，供远程仿真容器调用

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
# 在容器内
pip install fastapi uvicorn python-multipart
```

### 2️⃣ 启动服务

```bash
# 在容器内
cd /workspace/nanoLLaVA
python serve_vtla_api.py
```

**输出**：
```
================================================================================
Nano-VTLA FastAPI Service
================================================================================
Checkpoint: ./outputs/nano_vtla_baseline/checkpoint_step70000.pt
Model: BAAI/Bunny-v1_0-2B-zh
Device: cuda
Dtype: BF16
================================================================================

Starting server on http://0.0.0.0:8000
API 文档: http://0.0.0.0:8000/docs
================================================================================

Loading Nano-VTLA Model...
✅ Model Ready for Inference
INFO:     Started server process [1234]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3️⃣ 测试服务

**在另一个终端**（容器内或宿主机）：

```bash
# 测试健康检查
curl http://localhost:8000/health

# 运行完整测试
python test_vtla_api.py
```

## 📡 API 文档

### Endpoint 1: `/predict` (POST)

**预测 7-DoF 机器人动作**

**输入**：
- `image` (File): RGB 图像文件
- `instruction` (String, optional): 语言指令
- `use_dummy_tactile` (Boolean, default=True): 是否使用 dummy 触觉

**输出**：
```json
{
  "success": true,
  "action": [0.0015, 0.0024, -0.0012, 0.0018, 0.0032, -0.0015, 0.1295],
  "action_breakdown": {
    "translation": {
      "dx": 0.0015,
      "dy": 0.0024,
      "dz": -0.0012,
      "unit": "meters"
    },
    "rotation": {
      "droll": 0.0018,
      "dpitch": 0.0032,
      "dyaw": -0.0015,
      "unit": "radians"
    },
    "gripper": {
      "value": 0.1295,
      "range": "0 (open) to 1 (closed)"
    }
  },
  "metadata": {
    "instruction": "Pick up the red cube",
    "image_size": [640, 480],
    "used_dummy_tactile": true
  }
}
```

### Endpoint 2: `/health` (GET)

**检查服务健康状态**

**输出**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "gpu_memory": "8.23 GB"
}
```

### Endpoint 3: `/stats` (GET)

**获取动作归一化统计**

**输出**：
```json
{
  "mean": [-5.6e-06, 1.7e-05, 4.0e-05, -5.5e-05, -2.4e-04, -9.5e-05, 0.1132],
  "std": [0.005, 0.0066, 0.0068, 0.0136, 0.0172, 0.0182, 0.021],
  "description": {
    "0-2": "translation (dx, dy, dz) in meters",
    "3-5": "rotation (droll, dpitch, dyaw) in radians",
    "6": "gripper (0=open, 1=closed)"
  }
}
```

## 🐍 Python 客户端示例

```python
import requests
from PIL import Image

# 1. 加载图像
image = Image.open("robot_view.png")

# 2. 发送请求
files = {'image': open("robot_view.png", 'rb')}
data = {
    'instruction': "Pick up the red block",
    'use_dummy_tactile': True
}

response = requests.post("http://localhost:8000/predict", files=files, data=data)
result = response.json()

# 3. 获取动作
if result['success']:
    action = result['action']
    print(f"Action: {action}")
    
    # 提取分量
    dx, dy, dz = action[0:3]  # 平移 (米)
    droll, dpitch, dyaw = action[3:6]  # 旋转 (弧度)
    gripper = action[6]  # 夹爪 (0-1)
    
    # 发送给仿真器执行...
```

## 🔌 与仿真容器集成

### 方案 1: Docker 网络

```bash
# 创建 Docker 网络
docker network create vtla-network

# 启动 VTLA 服务容器（加入网络）
docker run ... --network vtla-network --name vtla-service ...

# 启动仿真容器（加入同一网络）
docker run ... --network vtla-network --name sim-container ...

# 在仿真容器内调用
curl http://vtla-service:8000/predict ...
```

### 方案 2: Host 网络

```bash
# VTLA 服务使用 host 网络（已配置）
docker run ... --net=host ...

# 仿真容器也使用 host 网络
docker run ... --net=host ...

# 两者都可以通过 localhost:8000 通信
```

## 🛡️ 性能 & 优化

**推理速度**：
- 单次预测: ~150ms (包含图像预处理)
- 吞吐量: ~6-7 requests/s

**显存占用**：
- 模型: ~8GB
- 推理: ~1GB (临时)
- 总计: ~9GB (11GB GPU 足够)

**优化建议**：
- 批量预测：一次处理多个图像
- 模型量化：使用 INT8 减少显存
- TensorRT：加速推理

## 📊 监控

```bash
# 查看服务状态
curl http://localhost:8000/health

# 查看请求日志
# 服务会实时打印每个请求的信息
```

## 🐛 故障排除

### Q: ModuleNotFoundError: No module named 'fastapi'
```bash
pip install fastapi uvicorn python-multipart
```

### Q: CUDA out of memory
减小模型显存或关闭其他 GPU 程序

### Q: 连接拒绝
检查服务是否启动：`curl http://localhost:8000/health`

---

**部署完成后，仿真容器就可以调用 API 获取动作了！** 🚀
