# BridgeData V2 → VLA NumPy 转换

## 用法

```bash
# 完整流程（需对 /datasets 或指定目录有写权限）
python scripts/run_bridgedata_to_vla_numpy.py

# 若 /datasets 无写权限，可用 workspace 或 /tmp：
python scripts/run_bridgedata_to_vla_numpy.py \
  --bridge-output /workspace/nanoLLaVA/bridge_numpy_npy \
  --output /workspace/nanoLLaVA/bridge_numpy

# 仅做 npy→npz 与完整性检查（已有 out.npy 时）
python scripts/run_bridgedata_to_vla_numpy.py --skip-bridge \
  --bridge-output /path/to/out_npy_root --output /datasets/bridge_numpy

# 多进程加速 bridge（需宿主支持 multiprocessing）
python scripts/run_bridgedata_to_vla_numpy.py --num-workers 8
```

## 路径与 depth

- **原始数据**: `/datasets/scripted_raw/{任务名如 pnp_utensils_11-18}/{日期如 2022-11-18_16-32-03}/raw/traj_group*/traj*/`
- **depth=2**: 把 `input_path` 下一层当作“任务”目录，其子目录中**仅**匹配 `YYYY-MM-DD_HH-MM-SS` 的当作日期目录，避免把 `traj*`、`traj_group*` 当日期解析导致 `ValueError: time data 'traj51' does not match format`。
- **Bridge 输出**: `--bridge-output` 下的 `{任务}/train/out.npy`、`{任务}/val/out.npy`。
- **VLA .npz**: `--output` 下保持 `{任务}/train/traj_*.npz`、`{任务}/val/traj_*.npz`，供 `train_vla.py --data_root=...` 使用。

## 完整性检查

脚本会统计 `--output` 下的 `.npz`，并尝试加载一个样本检查 `image`、`actions` 的 shape，**不**依赖 `torch`，避免 NumPy 2 / torch 冲突。

## 与 train_vla 的衔接

转换完成后：

```bash
python train_vla.py --data_root /datasets/bridge_numpy ...
# 若 npz 在别处：
python train_vla.py --data_root /workspace/nanoLLaVA/bridge_numpy ...
```
