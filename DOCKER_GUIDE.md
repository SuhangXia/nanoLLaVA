# Docker 环境配置指南 - VTLA Panda

本指南帮助您在 Docker 容器内正确配置 VTLA Panda 项目环境。

---

## 当前问题分析

从您的终端输出看到的问题：

1. ❌ **路径错误**: 在 `/workspace` 而非 `/workspace/nanoLLaVA`
2. ❌ **缺少 git**: Docker 镜像未预装 git
3. ❌ **文件未找到**: `requirements_vtla.txt` 路径不正确

---

## 快速修复（在 Docker 容器内执行）

### 方法 1: 使用自动配置脚本（推荐）

```bash
# 1. 进入正确的目录
cd /workspace/nanoLLaVA

# 2. 给脚本添加执行权限
chmod +x docker_setup.sh

# 3. 运行配置脚本（自动安装所有依赖）
bash docker_setup.sh
```

脚本会自动：
- ✅ 安装 git 和其他系统工具
- ✅ 安装所有 Python 依赖
- ✅ 验证环境配置
- ✅ 显示 RLBench 安装说明

---

### 方法 2: 手动安装（逐步执行）

如果您想手动控制安装过程：

#### 步骤 1: 安装基础工具

```bash
# 更新 apt 并安装基础工具
apt-get update
apt-get install -y git wget build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

#### 步骤 2: 进入项目目录

```bash
cd /workspace/nanoLLaVA
pwd  # 应该显示: /workspace/nanoLLaVA
```

#### 步骤 3: 安装 Python 核心依赖

```bash
pip install numpy scipy torch transformers pillow tqdm h5py opencv-python
```

#### 步骤 4: 安装 NanoLLaVA

```bash
# 如果有 requirements.txt
pip install -r requirements.txt

# 安装 NanoLLaVA 包
pip install -e .
```

#### 步骤 5: 验证安装

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import h5py; print(f'h5py: {h5py.__version__}')"
```

---

## RLBench 模拟器安装（可选）

如果您需要使用 RLBench 进行数据收集：

### 1. 下载并解压 CoppeliaSim

```bash
cd /workspace

# 下载 CoppeliaSim 4.1.0 (约 200MB)
wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# 解压
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# 设置环境变量
export COPPELIASIM_ROOT=/workspace/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT

# 添加到 bashrc（永久生效）
echo "export COPPELIASIM_ROOT=/workspace/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> ~/.bashrc
```

### 2. 安装 PyRep 和 RLBench

```bash
# 安装 PyRep (CoppeliaSim Python API)
pip install git+https://github.com/stepjam/PyRep.git

# 安装 RLBench (机器人任务库)
pip install git+https://github.com/stepjam/RLBench.git

# 验证安装
python3 -c "from pyrep import PyRep; print('PyRep OK')"
python3 -c "from rlbench import Environment; print('RLBench OK')"
```

---

## 测试 VTLA 管道

安装完成后，测试核心组件：

```bash
cd /workspace/nanoLLaVA

# 运行管道测试
python3 test_vtla_pipeline.py
```

**预期输出**:
```
============================================================
VTLA Pipeline Validation
============================================================

[Test 1/4] Testing VTLA HDF5 DataLoader...
  ✓ Dataset loaded: 100 samples
  ✓ Sample loaded: image (3, 384, 384), action (7,)
✓ Test 1 PASSED: HDF5 DataLoader works correctly

[Test 2/4] Testing Oracle Policy...
  ✓ Oracle policy generated action: [...]
✓ Test 2 PASSED: Oracle Policy works correctly

...
```

---

## 常见问题解决

### Q1: `pip install -r requirements_vtla.txt` 失败

**原因**: 文件路径不正确或不在项目根目录

**解决**:
```bash
cd /workspace/nanoLLaVA
ls requirements_vtla.txt  # 确认文件存在
pip install -r requirements_vtla.txt
```

### Q2: `git` 命令找不到

**原因**: Docker 基础镜像未包含 git

**解决**:
```bash
apt-get update && apt-get install -y git
```

### Q3: `No module named 'bunny'`

**原因**: NanoLLaVA 未正确安装

**解决**:
```bash
cd /workspace/nanoLLaVA
pip install -e .
```

### Q4: CUDA 不可用

**原因**: Docker 未正确配置 GPU

**检查**:
```bash
nvidia-smi  # 应该显示 GPU 信息
python3 -c "import torch; print(torch.cuda.is_available())"  # 应该返回 True
```

**解决**: 确保 Docker 启动时使用了 `--gpus all`

### Q5: CoppeliaSim 找不到

**原因**: 环境变量未设置

**解决**:
```bash
export COPPELIASIM_ROOT=/workspace/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
```

---

## 完整的 Docker 启动命令

如果需要重新启动容器，使用以下命令：

```bash
docker run -it --gpus all \
    --privileged \
    --net=host \
    -v /dev:/dev \
    -v /home/suhang/projects/nanoLLaVA:/workspace/nanoLLaVA \
    -v /home/suhang/robot_datasets:/datasets \
    --name nanollava_vla_panda \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
    /bin/bash -c "cd /workspace/nanoLLaVA && bash docker_setup.sh && /bin/bash"
```

这会自动运行配置脚本然后进入 shell。

---

## 检查清单

安装完成后，确认以下项目：

- [ ] Git 已安装: `git --version`
- [ ] Python 依赖已安装: `pip list | grep -E "(numpy|torch|transformers|h5py)"`
- [ ] CUDA 可用: `python3 -c "import torch; print(torch.cuda.is_available())"`
- [ ] NanoLLaVA 已安装: `python3 -c "from bunny.model import BunnyQwenForCausalLM"`
- [ ] 在正确目录: `pwd` 显示 `/workspace/nanoLLaVA`
- [ ] 测试脚本可运行: `python3 test_vtla_pipeline.py`

---

## 下一步

环境配置完成后：

1. **如果有 BridgeV2 数据**: 可直接使用旧的训练脚本
2. **如果要收集 Panda 数据**: 
   ```bash
   # 安装 RLBench (见上文)
   bash scripts/run_collect_panda_data.sh
   ```
3. **如果已有 HDF5 数据**: 
   ```bash
   bash scripts/run_train_vla_panda.sh
   ```

---

## 参考资源

- Docker 镜像: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
- CoppeliaSim 下载: https://www.coppeliarobotics.com/downloads
- PyRep 文档: https://github.com/stepjam/PyRep
- RLBench 文档: https://github.com/stepjam/RLBench

---

**有任何问题，请运行 `bash docker_setup.sh` 查看详细的安装日志。**
