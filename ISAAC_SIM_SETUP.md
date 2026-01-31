# Isaac Sim 环境配置指南

## 问题诊断

你遇到的问题：虽然 conda 环境显示 `(isaacsim)`，但 Python 找不到 `omni` 模块。

**原因**: Isaac Sim 通常有自己的 Python 解释器，不是通过标准 conda 安装的。

---

## 解决方案

### 方案 1: 使用 Isaac Sim 自带的 Python 启动器（推荐）

Isaac Sim 通常提供 `python.sh` 或类似的启动脚本：

```bash
# 查找 Isaac Sim Python 启动器
find /home/suhang/datasets/isaac_sim_work -name "python.sh" -o -name "isaac-sim.sh" 2>/dev/null

# 使用找到的启动器运行脚本
/path/to/isaac_sim/python.sh /home/suhang/projects/nanoLLaVA/isaac_sim_vtla_standalone.py \
    --api-url http://localhost:8000 \
    --prompt "pick up the red cube"
```

### 方案 2: 使用独立测试脚本（不需要 Isaac Sim）

我已经创建了 `isaac_sim_vtla_standalone.py`，它会尝试自动找到 Isaac Sim 路径。

如果只是想测试 API 通信（不需要完整的 Isaac Sim 场景），继续使用：

```bash
python test_isaac_api_simple.py --api-url http://localhost:8000
```

这个脚本已经验证通过 ✅，说明 API 端工作正常。

### 方案 3: 检查 Isaac Sim 安装

```bash
# 1. 检查 Isaac Sim 是否安装
ls -la /home/suhang/datasets/isaac_sim_work/

# 2. 查找 Python 可执行文件
find /home/suhang/datasets/isaac_sim_work -name "python*" -type f 2>/dev/null | grep -v ".pyc"

# 3. 查找 omni 模块
find /home/suhang/datasets/isaac_sim_work -name "omni" -type d 2>/dev/null | head -5
```

---

## 当前状态

✅ **API 通信已验证**: `test_isaac_api_simple.py` 测试通过
- API 健康检查: ✓
- POST /predict_isaac 接口: ✓
- 返回正确的 delta_pose: ✓

❌ **Isaac Sim 场景未运行**: 需要正确的 Isaac Sim Python 环境

---

## 三种运行模式

### 模式 1: 纯 API 测试（无需 Isaac Sim）✅ 已工作

```bash
cd /home/suhang/projects/nanoLLaVA
python test_isaac_api_simple.py --api-url http://localhost:8000
```

**用途**: 验证 VTLA 模型推理是否正常

### 模式 2: Isaac Sim 简化场景

```bash
# 使用 Isaac Sim 的 Python 启动器
/path/to/isaac_sim/python.sh isaac_sim_vtla_standalone.py \
    --isaac-sim-path /home/suhang/datasets/isaac_sim_work \
    --api-url http://localhost:8000
```

**用途**: 在 Isaac Sim 中可视化，但不需要完整机器人模型

### 模式 3: 完整 Isaac Sim 机器人场景

```bash
# 使用 Isaac Sim 的 Python 启动器
/path/to/isaac_sim/python.sh isaac_sim_vtla_inference.py \
    --api-url http://localhost:8000 \
    --prompt "pick up the red cube" \
    --max-steps 50
```

**用途**: 完整的机器人仿真（需要 Franka Panda 模型）

---

## 常见 Isaac Sim 安装位置

Isaac Sim 可能安装在以下位置之一：

```bash
# Omniverse Launcher 安装
~/.local/share/ov/pkg/isaac_sim-*/
~/.local/share/ov/pkg/isaac-sim-*/

# 手动安装
/opt/nvidia/isaac_sim/
~/isaac_sim/
/home/suhang/datasets/isaac_sim_work/

# 检查所有可能的位置
find ~ -maxdepth 3 -name "*isaac*sim*" -type d 2>/dev/null
```

找到后，Python 启动器通常在：
```
<isaac_sim_path>/python.sh
<isaac_sim_path>/kit/python/bin/python3
```

---

## 快速诊断命令

在你的终端运行：

```bash
# 1. 检查当前 Python 路径
which python
python -c "import sys; print('\n'.join(sys.path))"

# 2. 尝试导入 omni
python -c "import omni; print('✓ omni 可用')" 2>&1

# 3. 检查 Isaac Sim 安装
ls -la /home/suhang/datasets/isaac_sim_work/

# 4. 查找 Isaac Sim Python 启动器
find /home/suhang/datasets/isaac_sim_work -name "*.sh" -type f 2>/dev/null | grep -i python
```

把输出贴给我，我可以帮你进一步诊断。

---

## 推荐的工作流程

由于 API 通信已经验证通过，建议：

### 选项 A: 仅使用 API 测试（最简单）

```bash
# 已经工作 ✅
python test_isaac_api_simple.py --api-url http://localhost:8000
```

这已经足够验证你的 VTLA 模型能正确处理图像并返回动作预测。

### 选项 B: 在你现有的仿真环境中集成

如果你已经有其他仿真环境（如 RLBench、PyBullet 等），可以：

1. 使用 `test_isaac_api_simple.py` 中的 API 调用逻辑
2. 集成到你现有的仿真循环中
3. 不需要专门配置 Isaac Sim

### 选项 C: 配置完整 Isaac Sim（如果需要）

1. 找到 Isaac Sim Python 启动器（见上面的命令）
2. 使用启动器运行脚本
3. 或者在你的 Python 脚本开头添加 Isaac Sim 路径

---

## 总结

**当前状态**: ✅ VTLA API 工作正常，可以接收图像并返回动作预测

**下一步**:
1. 如果只需要测试模型推理，当前的 `test_isaac_api_simple.py` 已经足够
2. 如果需要 Isaac Sim 可视化，需要找到正确的 Isaac Sim Python 启动器
3. 或者，可以将 API 集成到其他仿真环境中

需要我帮你进一步配置 Isaac Sim 吗？请运行上面的"快速诊断命令"并把输出发给我。
