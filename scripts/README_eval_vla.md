# VLA 评估

## 方案一：PyBullet（推荐，替代 Robosuite/MuJoCo）

**`eval_vla_pybullet.py`**：无 Robosuite/MuJoCo 依赖，支持 **WidowX 250 (Bridge V2)** 或 Franka Panda。

- **WidowX 250（默认）**：`assets/widowx_250/widowx_250.urdf`。相机 Over-the-shoulder 对齐 Bridge V2 主视角：`eye=[0.15,-0.15,0.3]`，`target=[0.2,-0.08,0.06]`，FOV 75°。桌子、方块、机械臂相对位置与训练数据（Bridge）的桌台场景一致；`xyz_scale` 默认 0.05，撞桌时可试 `--xyz_scale 0.02`。
- **Panda**：`--robot panda --xyz_scale 0.05`，相机与工作区同前。
- 384×384 `p.getCameraImage` → VLA → 7D → IK → 输出 `vla_eval_pybullet.mp4`。

```bash
./scripts/run_eval_vla_pybullet.sh
# 脚本默认 --direct --tiny_renderer 离屏（不依赖 X11/OpenGL）。要开 GUI：改脚本去掉 --direct --tiny_renderer，加 --gui
# Panda：--robot panda --xyz_scale 0.05
python eval_vla_pybullet.py --direct --tiny_renderer --robot widowx --scene cube --steps 600 --output_mp4 vla_eval_pybullet.mp4
```

---

## 方案二：Robosuite (eval_vla_step5500.py)

### 功能

- 加载 Phase 1 训练的 `action_head_step5500.bin` 与 `action_mean_std.pkl`
- 在 Robosuite Lift + Panda 中闭环 300 步，每步：图像 → 模型 → 7D 动作 → `env.step` → 存帧
- 离屏渲染，输出 `vla_eval_step5500.mp4`

### 运行

```bash
cd /workspace/nanoLLaVA

# 本机有显示屏（laptop/工作站）：优先用 realtime，避免 osmesa 卡顿
./scripts/run_eval_vla_step5500_realtime.sh
# 需 DISPLAY；若未设置：export DISPLAY=:0 或 :1

# 默认（osmesa 离屏，无头或 EGL 不可用时）
./scripts/run_eval_vla_step5500.sh

# 或直接（有屏时在其它参数前加 --realtime）
python eval_vla_step5500.py --realtime \
  --vla_ckpt ./outputs/vla_phase1_bf16 \
  --action_head action_head_step5500.bin \
  --steps 300 \
  --output_mp4 vla_eval_step5500.mp4
```

## 常见问题

### 1. EGL：`Cannot initialize a EGL device display`

`run_eval_vla_step5500.sh` 已默认使用 osmesa，直接跑即可。若仍用 EGL，可手动：

```bash
EVAL_VLA_USE_OSMESA=1 python eval_vla_step5500.py ...
# 或
python eval_vla_step5500.py --osmesa ...
```

**osmesa 依赖**：若报错找不到 `libOSMesa` 或 `mujoco.osmesa`，先安装：

```bash
sudo apt-get update && sudo apt-get install -y libosmesa6 libosmesa6-dev
```

### 2. Numba：`cannot cache function 'mat2quat': no locator available`

在只读或容器内 `site-packages` 时，禁用 numba 缓存：

```bash
export NUMBA_DISABLE_JIT_CACHE=1
python eval_vla_step5500.py ...
```

`run_eval_vla_step5500.sh` 已设置 `NUMBA_DISABLE_JIT_CACHE=1`。

### 3. osmesa 卡在 “Creating Lift” 很久（10+ 分钟）

osmesa 首次创建 GL 会做大量 CPU 初始化，脚本已把离屏分辨率从 640×480 降到 256×256 以加速。若仍卡很久：

- **推荐（本机有屏）**：用 `--realtime` 强制 EGL/glfw，不用 osmesa：  
  `./scripts/run_eval_vla_step5500_realtime.sh` 或  
  `python eval_vla_step5500.py --realtime ...`  
  需 `DISPLAY`（如 `export DISPLAY=:0`）。
- 再等 5–10 分钟，首跑 15 分钟也偶有发生。
- 或在本机用 EGL 离屏：`EVAL_VLA_USE_OSMESA=0 ./scripts/run_eval_vla_step5500.sh`（需 nvidia 驱动）。

### 4. 缩短测试

```bash
python eval_vla_step5500.py --steps 10 --output_mp4 /tmp/vla_test.mp4
```

## 输出

- 每步打印：`[step=xxx] raw: dx=... dy=... dz=... dr=... dp=... dyaw=... g=... -> env_act: ...`
- 结束：`[eval] Saved vla_eval_step5500.mp4 (300 frames, 30 fps)`

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--vla_ckpt` | `./outputs/vla_phase1_bf16` | 含 `action_head_*.bin` 与 `action_mean_std.pkl` 的目录 |
| `--action_head` | `action_head_step5500.bin` | action head 文件名 |
| `--steps` | 300 | 仿真步数 |
| `--output_mp4` | `vla_eval_step5500.mp4` | 输出视频 |
| `--fps` | 30 | 视频帧率 |
| `--osmesa` | - | 使用 osmesa 代替 EGL（须在其他参数前） |
| `--realtime` | - | 有屏时 EGL/glfw 实时窗口，避免 osmesa 卡顿；须在其他参数前，需 DISPLAY |

---

## PyBullet 参数 (eval_vla_pybullet.py)

| 参数 | 默认 | 说明 |
|------|------|------|
| `--vla_ckpt` | `./outputs/vla_phase1_bf16` | 含 `action_head_*.bin` 与 `action_mean_std.pkl` |
| `--action_head` | `action_head_step5500.bin` | action head 文件名 |
| `--prompt` | `Lift the red cube` | 文本提示（当前占位） |
| `--steps` | 300 | 仿真步数 |
| `--output_mp4` | `vla_eval_pybullet.mp4` | 输出视频 |
| `--cam_size` | 384 | getCameraImage 宽高 |
| `--gui` | - | 强制 p.GUI |
| `--direct` | - | 强制 p.DIRECT 离屏 |
