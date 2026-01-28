# train_vla 训练环境问题排查

## 1. 安装 transformers

```bash
pip install transformers
```

若已用 `pip install -e .` 安装项目，理论上会带上 `transformers`；若仍报 `ModuleNotFoundError: No module named 'transformers'`，可单独安装上述命令。

### 1.1 `register_pytree_node` 与旧版 PyTorch 不兼容

**现象**：`AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'. Did you mean: '_register_pytree_node'?`

**原因**：transformers 4.49+ 使用 `register_pytree_node(..., serialized_type_name=...)`，而旧版 PyTorch 只有 `_register_pytree_node(type, flatten, unflatten)`，且签名不同。

**处理**：`train_vla.py` 已在 `import transformers` 前加入兼容补丁（包装 `_register_pytree_node`，吸收 `serialized_type_name` 等新参数），一般无需再改。若你希望不依赖补丁，可改用与旧 PyTorch 兼容的 transformers：

```bash
pip install "transformers>=4.36,<4.46"
```

---

## 2. NumPy 与 PyTorch 版本

**现象**：`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6`，或 `_ARRAY_API not found`。

**处理**：当前环境中的 PyTorch 按 NumPy 1.x 编译，需使用 NumPy 1.x：

```bash
pip install "numpy<2"
```

你已安装 `numpy==1.26.4`，可保持不变。

**opencv-python 冲突**：`opencv-python 4.13` 要求 `numpy>=2`，与 `numpy<2` 冲突。

- **仅跑 `train_vla`**：`train_vla` 与 `vla_dataloader` 使用 PIL + numpy，**不依赖 opencv**，可先忽略该冲突，训练可正常进行。
- **若需要 opencv**（如 `infer_vla --realsense` 或其他脚本）：
  - 降级：`pip install "opencv-python<4.10"`（例如 `opencv-python==4.8.1.78` 支持 numpy 1.x），或
  - 在单独虚拟环境中用 `numpy>=2` + 新版 opencv，仅在该环境中跑依赖 opencv 的脚本。

---

## 3. model_name_or_path / checkpoints 不存在

**现象**：`/workspace/nanoLLaVA/checkpoints` 不存在，`ls` 报 `No such file or directory`。

**处理**：`--model_name_or_path` 可以是：

1. **本地目录**：已下载的 Bunny/NanoLLaVA 权重目录，且该路径必须存在。
2. **HuggingFace 模型 ID**：会自动从 HF 下载，无需事先有 `checkpoints` 目录。

推荐直接用 HuggingFace ID（需网络），例如与 `--model_type qwen1.5-1.8b`、`--vision_tower siglip-so400m-patch14-384` 匹配的 2B 级 Bunny 模型：

```bash
--model_name_or_path BAAI/Bunny-v1_0-2B-zh
```

若已有本地权重，例如在 `/path/to/bunny-2b`，则：

```bash
--model_name_or_path /path/to/bunny-2b
```

---

## 4. 推荐训练命令（修正后）

在 **numpy<2**、**transformers 已安装**、**能访问 HuggingFace** 的前提下：

```bash
cd /workspace/nanoLLaVA

python train_vla.py \
  --model_name_or_path BAAI/Bunny-v1_0-2B-zh \
  --model_type qwen1.5-1.8b \
  --vision_tower siglip-so400m-patch14-384 \
  --data_root /datasets/bridge_numpy \
  --output_dir ./outputs/vla_phase1 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --save_steps 500 \
  --logging_steps 10
```

若 HF 模型内已正确配置 `mm_vision_tower`，可省略 `--vision_tower`，由配置自动读取。

**首次运行**：会从 HuggingFace 下载 `BAAI/Bunny-v1_0-2B-zh` 与 `google/siglip-so400m-patch14-384`（若 vision 单独指定），请保证网络与 `~/.cache/huggingface` 可用。

---

## 5. 启动前快速自检

```bash
# 1) NumPy 版本（期望 1.x，如 1.26.4）
python -c "import numpy; print(numpy.__version__)"

# 2) torch + transformers
python -c "import torch; import transformers; print('torch, transformers OK')"

# 3) 数据目录存在且有 npz
ls /datasets/bridge_numpy | head -5
```

若 1、2 通过且 3 有 `.npz` 或子目录，可按上面命令启动训练。

---

## 6. `AttributeError: 'weight' is not an nn.Module`（4-bit 与 PyTorch 2.11+ / sm_120）

**现象**：在 PyTorch 2.11+、CUDA 12.8、bitsandbytes 0.49、RTX 5070 Ti (sm_120) 上，用 4-bit 加载时在 `_initialize_missing_keys` / `get_module_from_name` 中报错：`AttributeError: 'weight' is not an nn.Module`。

**原因**：transformers 的量化初始化逻辑与当前 PyTorch / bitsandbytes 组合不兼容，在解析 `weight` 等参数名时误当作子模块路径。

**处理**：改用 **BF16 全量 + 梯度检查点**，完全避开 bitsandbytes，在 12GB 下可跑：

```bash
python train_vla.py \
  --model_name_or_path BAAI/Bunny-v1_0-2B-zh \
  --model_type qwen1.5-1.8b \
  --vision_tower siglip-so400m-patch14-384 \
  --data_root /datasets/bridge_numpy \
  --output_dir ./outputs/vla_phase1_bf16 \
  --no_4bit \
  --gradient_checkpointing \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --save_steps 500 \
  --logging_steps 10
```

或直接运行：

```bash
./scripts/run_train_vla_bf16_12gb.sh
```

- `--no_4bit` 或 `--bf16`：不量化，BF16 加载。
- `--gradient_checkpointing`：降低激活显存，配合 `--batch_size 2` 更稳。

---

## 7. 其他依赖

若出现 `ModuleNotFoundError: No module named 'einops'` 等，请安装项目依赖，例如：

```bash
pip install einops peft
# 或从项目安装： pip install -e .
```

### 参数说明（`--bf16` / `--gradient_checkpointing`）

- `--bf16 True` 会报 `unrecognized arguments`：应使用 **`--bf16`** 或 **`--no_4bit`**，无需写 `True`（二者均为 `action="store_true"`）。
- `--gradient_checkpointing True` 同理，应写 **`--gradient_checkpointing`**。
