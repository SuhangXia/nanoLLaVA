"""
VLA (Vision-Language-Action) DataLoader for bridge-style numpy data.

Reads trajectories from /datasets/bridge_numpy. Each trajectory has images and 7D actions:
  (dx, dy, dz, dr, dp, dyaw, gripper).
Computes dataset action mean/std and applies normalization.
"""

from __future__ import annotations

import os
import glob
import pickle
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Action dim: dx, dy, dz, droll, dpitch, dyaw, gripper
ACTION_DIM = 7
# Cache filename for precomputed action statistics
ACTION_STATS_CACHE = "vla_action_mean_std.pkl"


def _find_episode_files(data_root: str) -> List[str]:
    """Collect .npz episode files from data_root, including nested task/train|val/..."""
    data_root = os.path.expanduser(data_root)
    # 递归查找所有 *.npz，兼容 data_root/task/train/traj_*.npz 等结构
    out = glob.glob(os.path.join(data_root, "**", "*.npz"), recursive=True)
    return sorted(out)


def _load_episode_obs_actions(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and actions from one .npz.
    Returns: images (T, 3, H, W), actions (T, 7).
    """
    data = np.load(path, allow_pickle=True)

    # Resolve observations -> image
    obs = data.get("observations", None)
    if obs is None:
        obs = data
    if hasattr(obs, "item") and isinstance(obs, np.ndarray) and obs.dtype == object:
        obs = obs.item()
    if isinstance(obs, dict):
        im = obs.get("image", obs.get("images", obs.get("pixels", None)))
    else:
        im = data.get("image", data.get("images", None))

    # Resolve actions
    acts = data.get("actions", data.get("action", None))
    if acts is None and "action" in data.files:
        acts = data["action"]
    if acts is None:
        raise KeyError(f"No 'actions' or 'action' in {path}")

    acts = np.asarray(acts, dtype=np.float32)
    if acts.ndim == 1:
        acts = acts.reshape(1, -1)
    if acts.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected action dim {ACTION_DIM}, got {acts.shape[-1]} in {path}")

    # Align T: use min(len(im), len(acts))
    T = min(len(im), len(acts))
    im = im[:T]
    acts = acts[:T]

    # Image: (T, H, W, 3) -> (T, 3, H, W)
    if im.shape[-1] == 3:
        im = np.transpose(im, (0, 3, 1, 2))
    im = np.asarray(im, dtype=np.float32)
    if im.max() > 1.5:
        im = im / 255.0

    return im, acts


def compute_action_stats(
    data_root: str,
    cache_path: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute action mean and std over the dataset.
    Returns: mean (7,), std (7,). std has a minimum of 1e-6 to avoid div-by-zero.
    """
    cache_path = cache_path or os.path.join(os.path.dirname(data_root) or ".", ACTION_STATS_CACHE)
    try:
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                d = pickle.load(f)
            return d["mean"], d["std"]
    except Exception:
        pass

    files = _find_episode_files(data_root)
    all_acts: List[np.ndarray] = []
    n = 0
    for p in files:
        try:
            _, a = _load_episode_obs_actions(p)
            all_acts.append(a)
            n += a.shape[0]
            if num_samples is not None and n >= num_samples:
                break
        except Exception:
            continue

    if not all_acts:
        # Fallback: zero mean, ones std
        mean = np.zeros(ACTION_DIM, dtype=np.float32)
        std = np.ones(ACTION_DIM, dtype=np.float32)
    else:
        concat = np.concatenate(all_acts, axis=0)
        if num_samples is not None and concat.shape[0] > num_samples:
            concat = concat[:num_samples]
        mean = np.mean(concat, axis=0).astype(np.float32)
        std = np.std(concat, axis=0).astype(np.float32)
        std = np.maximum(std, 1e-6)

    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"mean": mean, "std": std}, f)
    except Exception:
        pass

    return mean, std


class BridgeNumpyVLA(Dataset):
    """
    Dataset over /datasets/bridge_numpy: (image_t, action_t) pairs.
    Actions are normalized with (a - mean) / std.
    """

    def __init__(
        self,
        data_root: str = "/datasets/bridge_numpy",
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        compute_stats: bool = True,
        stats_cache: Optional[str] = None,
    ):
        self.data_root = os.path.expanduser(data_root)
        self.episode_files = _find_episode_files(self.data_root)

        if action_mean is not None and action_std is not None:
            self.mean = np.asarray(action_mean, dtype=np.float32)
            self.std = np.asarray(action_std, dtype=np.float32)
        elif compute_stats:
            self.mean, self.std = compute_action_stats(self.data_root, cache_path=stats_cache)
        else:
            self.mean = np.zeros(ACTION_DIM, dtype=np.float32)
            self.std = np.ones(ACTION_DIM, dtype=np.float32)

        # Build index: (file_idx, frame_idx) -> global_idx
        self.index: List[Tuple[int, int]] = []
        for i, p in enumerate(self.episode_files):
            try:
                _, a = _load_episode_obs_actions(p)
                for t in range(len(a)):
                    self.index.append((i, t))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.index)

    def _load_at(self, file_idx: int, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self.episode_files[file_idx]
        im, a = _load_episode_obs_actions(path)
        return im[frame_idx], a[frame_idx]

    def get_action_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean.copy(), self.std.copy()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, frame_idx = self.index[idx]
        im, act = self._load_at(file_idx, frame_idx)
        # Normalize action: (a - mean) / std
        act_norm = (act - self.mean) / self.std
        return {
            "image": torch.from_numpy(im).float(),
            "action": torch.from_numpy(act_norm).float(),
            "action_raw": torch.from_numpy(act).float(),
        }


def build_vla_dataloader(
    data_root: str = "/datasets/bridge_numpy",
    batch_size: int = 4,
    num_workers: int = 0,
    action_mean: Optional[np.ndarray] = None,
    action_std: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Build DataLoader and return (dataloader, action_mean, action_std).
    action_mean/std are the raw (un-normalized) statistics for potential denormalization.
    """
    ds = BridgeNumpyVLA(
        data_root=data_root,
        action_mean=action_mean,
        action_std=action_std,
        compute_stats=(action_mean is None or action_std is None),
    )
    mean, std = ds.get_action_mean_std()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **kwargs,
    )
    return loader, mean, std
