# Robosuite + Nano-VTLA 推理指南

使用 Robosuite 仿真环境调用 Nano-VTLA API 进行视觉-语言-动作推理。

## 前置条件

### 1. Robosuite 环境

确保已安装 Robosuite 并激活 conda 环境：

```bash
conda activate robosuite_env
```

### 2. VTLA API 服务

在 Docker 容器中运行 VTLA API 服务：

```bash
docker start nanollava_vtla_new
docker exec -it nanollava_vtla_new bash
cd /workspace/nanoLLaVA
python serve_vtla_api.py --checkpoint ./outputs/nano_vtla_baseline/checkpoint_step70000.pt --port 8000
```

## 运行

```bash
# 激活 Robosuite 环境
conda activate robosuite_env
cd /home/suhang/projects/nanoLLaVA

# 运行推理（PickPlaceCan + Panda，默认显示仿真窗口）
python robosuite_vtla_inference.py \
    --api-url http://localhost:8000 \
    --env-name PickPlaceCan \
    --robot Panda \
    --prompt "pick up the can" \
    --max-steps 200
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-url` | http://localhost:8000 | VTLA API 地址 |
| `--env-name` | PickPlaceCan | 环境名称（PickPlace, Lift, Stack 等） |
| `--robot` | Panda | 机器人类型 |
| `--prompt` | "pick up the can" | 语言指令 |
| `--max-steps` | 200 | 每回合最大步数 |
| `--action-scale` | 50.0 | 动作缩放因子 |
| `--camera-dir` | ./robosuite_camera_views | 图像保存目录 |
| `--no-render` | - | 无界面运行（headless） |
| `--no-save-api-input` | - | 不保存 API 输入图像 |
| `--save-every-step` | 5 | 每 N 步保存一张手眼相机图像到文件 |

### 环境示例

- **PickPlaceCan**: 抓取罐子放入容器（单物体）
- **PickPlace**: 多物体抓放
- **Lift**: 抬起方块
- **Stack**: 堆叠方块

## 工作流程

```
Robosuite 环境
     ↓
获取 robot0_eye_in_hand 手眼相机图像 (384x384)
     ↓
获取当前末端位姿 [x,y,z,qx,qy,qz,qw]
     ↓
POST /predict_isaac (Base64 图像 + 位姿 + 指令)
     ↓
VTLA API 返回 delta_pose [dx,dy,dz,drx,dry,drz]
     ↓
缩放并转换为 Robosuite OSC_POSE 动作 [dx,dy,dz,droll,dpitch,dyaw,gripper]
     ↓
env.step(action)
     ↓
循环
```

## 相机配置（手眼 Eye-in-Hand）

推理脚本使用 **手眼相机**（`robot0_eye_in_hand`），即相机装在机械臂末端、随手腕移动，与 ViTaMIn 等 VLA 数据标准一致。相机画面会保存到本地，方便查看模型实际看到的视角。

## 调试

发送给 API 的图像会保存到 `./robosuite_camera_views/`：

- `api_input_latest.jpg` - 最近发送给 API 的图像（手眼相机视角）
- `eye_in_hand_ep0_step_00000.jpg`, ... - 每 `--save-every-step` 步保存（默认 5）

```bash
# 查看模型实际看到的图像（手眼相机）
eog ./robosuite_camera_views/api_input_latest.jpg
```

## 动作缩放

VTLA 模型输出的 delta 较小（约 0.001m）。若机械臂几乎不动，增大 `--action-scale`：

```bash
python robosuite_vtla_inference.py --action-scale 100
```

若动作过大或不稳定，减小：

```bash
python robosuite_vtla_inference.py --action-scale 20
```

## 无界面运行

```bash
python robosuite_vtla_inference.py --no-render --env-name PickPlaceCan
```

## 完整运行示例

### Terminal 1: VTLA API（Docker 容器内）

```bash
docker start nanollava_vtla_new
docker exec -it nanollava_vtla_new bash
cd /workspace/nanoLLaVA
python serve_vtla_api.py --checkpoint ./outputs/nano_vtla_baseline/checkpoint_step70000.pt --port 8000
```

### Terminal 2: Robosuite 推理（宿主机）

```bash
conda activate robosuite_env
cd /home/suhang/projects/nanoLLaVA
python robosuite_vtla_inference.py --api-url http://localhost:8000 --prompt "pick up the can"
```
