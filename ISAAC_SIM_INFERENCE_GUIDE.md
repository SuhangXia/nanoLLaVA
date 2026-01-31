# Isaac Sim + Nano-VTLA 推理指南

使用 Isaac Sim 仿真环境调用 Nano-VTLA API 进行视觉-语言-动作推理。

---

## 前置条件

### 1. Isaac Sim 环境

确保已安装 Isaac Sim 并激活 conda 环境：

```bash
conda activate /home/suhang/datasets/isaac_sim_work/envs/isaacsim
```

### 2. VTLA API 服务

在 Docker 容器中运行 VTLA API 服务：

```bash
# 在宿主机上
docker start nanollava_vtla_new
docker exec -it nanollava_vtla_new bash

# 在容器内
cd /workspace/nanoLLaVA
python serve_vtla_api.py --checkpoint ./outputs/nano_vtla_baseline/checkpoint_step70000.pt --port 8000
```

服务启动后，应看到：
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 快速开始

### 步骤 1: 测试 API 连接（推荐先做）

在 Isaac Sim conda 环境中：

```bash
cd /home/suhang/projects/nanoLLaVA
python test_isaac_api_simple.py --api-url http://localhost:8000
```

**预期输出**:
```
[1/3] 健康检查...
  ✅ API 健康
     状态: healthy
     模型已加载: True

[2/3] 测试 Isaac Sim 接口 (POST /predict_isaac)...
  HTTP 状态码: 200
  ✅ 请求成功
     delta_pose: [0.001234, -0.002345, 0.000567, ...]

[3/3] 多次测试（验证稳定性）...
  成功率: 5/5 (100.0%)

✅ 所有测试通过！Isaac Sim 接口工作正常
```

### 步骤 2: 运行完整推理（Isaac Sim 场景）

```bash
cd /home/suhang/projects/nanoLLaVA
python isaac_sim_vtla_inference.py \
    --api-url http://localhost:8000 \
    --prompt "pick up the red cube" \
    --max-steps 50 \
    --num-episodes 1
```

---

## 脚本说明

### `test_isaac_api_simple.py`

**功能**: 轻量级测试脚本，不需要完整 Isaac Sim 场景
- 测试 API 健康状态
- 测试 POST /predict_isaac 接口
- 验证响应格式和数据正确性

**用法**:
```bash
python test_isaac_api_simple.py --api-url http://localhost:8000
```

### `isaac_sim_vtla_inference.py`

**功能**: 完整的 Isaac Sim 推理脚本
- 初始化 Isaac Sim 场景（Franka Panda + 目标物体 + 相机）
- 闭环控制：图像 → API → delta_pose → 执行动作
- 支持多回合运行

**用法**:
```bash
python isaac_sim_vtla_inference.py \
    --api-url http://localhost:8000 \
    --prompt "pick up the red cube" \
    --max-steps 50 \
    --num-episodes 3
```

**参数**:
- `--api-url`: VTLA API 服务地址（默认 `http://localhost:8000`）
- `--prompt`: 语言指令（默认 `"pick up the red cube"`）
- `--max-steps`: 每回合最大步数（默认 50）
- `--num-episodes`: 运行回合数（默认 1）

---

## 工作流程

```
┌─────────────────────┐
│   Isaac Sim 场景     │
│  (Franka + Camera)  │
└──────────┬──────────┘
           │
           │ 1. 捕获 RGB 图像 (384x384)
           │ 2. 获取当前位姿 [x,y,z,qx,qy,qz,qw]
           ▼
┌─────────────────────┐
│  isaac_sim_vtla_    │
│  inference.py       │
└──────────┬──────────┘
           │
           │ 3. Base64 编码图像
           │ 4. POST /predict_isaac (JSON)
           ▼
┌─────────────────────┐
│  VTLA API 服务       │
│  (Docker 容器内)     │
└──────────┬──────────┘
           │
           │ 5. 模型推理
           │ 6. 返回 delta_pose [dx,dy,dz,drx,dry,drz]
           ▼
┌─────────────────────┐
│  isaac_sim_vtla_    │
│  inference.py       │
└──────────┬──────────┘
           │
           │ 7. 计算目标位姿
           │ 8. 执行动作（IK/OSC）
           ▼
┌─────────────────────┐
│   Isaac Sim 场景     │
│  (更新机器人状态)    │
└──────────┬──────────┘
           │
           └───► 循环回到步骤 1
```

---

## 常见问题

### Q1: `ImportError: No module named 'omni'`

**原因**: 未在 Isaac Sim conda 环境中运行

**解决**:
```bash
conda activate /home/suhang/datasets/isaac_sim_work/envs/isaacsim
python isaac_sim_vtla_inference.py
```

### Q2: API 连接失败 `Connection refused`

**原因**: VTLA API 服务未运行或地址不对

**解决**:
1. 检查 API 服务是否运行（在 Docker 容器内）
2. 如果容器和宿主机网络隔离，使用 `--net=host` 或端口映射
3. 检查防火墙设置

### Q3: `delta_pose` 全是零或很小

**原因**: 可能的原因：
1. 模型未正确训练
2. 图像质量问题
3. 语言指令与训练数据不匹配

**调试**:
```bash
# 查看 API 日志（在容器内）
# 应该看到 [Response] Action: [非零值...]

# 测试不同的图像和指令
python test_isaac_api_simple.py --api-url http://localhost:8000
```

### Q4: 机器人动作不稳定或抖动

**原因**: 控制频率、增益、或坐标系转换问题

**解决**:
1. 降低控制频率（增加 `time.sleep()`）
2. 检查 `apply_delta_pose()` 中的坐标系转换
3. 添加平滑滤波器

---

## 自定义修改

### 修改机器人类型

在 `isaac_sim_vtla_inference.py` 中修改：

```python
# 当前使用 Franka Panda
from omni.isaac.franka import Franka
self.robot = self.world.scene.add(Franka(...))

# 改为 UR5
from omni.isaac.universal_robots import UR5
self.robot = self.world.scene.add(UR5(...))
```

### 修改相机位置

```python
self.camera = Camera(
    prim_path="/World/Franka/panda_hand/camera",
    position=np.array([0.0, 0.0, 0.1]),  # 调整相对于手的位置
    frequency=20,
    resolution=(384, 384)
)
```

### 添加任务成功检测

在 `run_episode()` 中添加：

```python
# 检查物体是否被抓取
object_pos, _ = target_cube.get_world_pose()
ee_pos, _ = self.robot.end_effector.get_world_pose()
distance = np.linalg.norm(object_pos - ee_pos)

if distance < 0.05:  # 5cm 内认为抓取成功
    print("  ✅ 任务成功！")
    break
```

---

## 完整运行示例

### Terminal 1: 启动 VTLA API 服务（Docker 容器内）

```bash
docker start nanollava_vtla_new
docker exec -it nanollava_vtla_new bash
cd /workspace/nanoLLaVA
python serve_vtla_api.py --checkpoint ./outputs/nano_vtla_baseline/checkpoint_step70000.pt --port 8000
```

### Terminal 2: 运行 Isaac Sim 推理（宿主机）

```bash
conda activate /home/suhang/datasets/isaac_sim_work/envs/isaacsim
cd /home/suhang/projects/nanoLLaVA

# 先测试连接
python test_isaac_api_simple.py

# 再运行完整推理
python isaac_sim_vtla_inference.py --prompt "pick up the red cube" --max-steps 50
```

---

## 性能优化

### 减少延迟

1. **使用本地 API**（避免网络延迟）
2. **增加控制频率**（减少 `time.sleep()`）
3. **批量推理**（如果 API 支持）

### GPU 内存优化

如果 GPU 内存不足（API 和 Isaac Sim 共用）：

1. **使用 FP16 而非 BF16**:
   ```bash
   python serve_vtla_api.py --checkpoint ... --fp16
   ```

2. **降低 Isaac Sim 渲染质量**

3. **使用不同的 GPU**（如果有多卡）

---

## 调试技巧

### 1. 查看 API 日志

在 Docker 容器的终端查看实时日志，确认请求是否到达和响应内容。

### 2. 保存中间结果

在脚本中添加：

```python
# 保存图像
Image.fromarray(rgb_image).save(f"debug_step_{step}.jpg")

# 保存位姿
np.save(f"debug_pose_{step}.npy", current_pose)
```

### 3. 可视化 delta_pose

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, delta_pose[0], delta_pose[1], delta_pose[2], color='r')
plt.show()
```

---

## 参考

- **API 文档**: `http://localhost:8000/docs`
- **Isaac Sim 文档**: https://docs.omniverse.nvidia.com/isaacsim/latest/
- **VTLA API 格式**: 见 `API_ISAAC_SIM.md`
