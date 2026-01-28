"""
VTLA HDF5 DataLoader for ViTaMIn-B multimodal data.

This loader reads RLBench-Panda trajectories in HDF5 format and
provides (image, action) pairs for VLA Phase 1 training.

Data Format:
  observations/
    images/eye_in_hand: (T, 384, 384, 3) uint8
    instruction: string
    proprioception/joint_positions: (T, 7)
    proprioception/gripper_width: (T,)
    tactile/force_torque: (T, 6)
    tactile/contact_depth: (T, 384, 384)
  actions: (T, 7) [dx, dy, dz, dr, dp, dy, gripper]
"""

from __future__ import annotations

import os
import glob
import pickle
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Action dimension: dx, dy, dz, droll, dpitch, dyaw, gripper
ACTION_DIM = 7


def find_hdf5_files(data_root: str) -> List[str]:
    """
    Recursively find all HDF5 episode files.
    
    Args:
        data_root: Root directory containing HDF5 files
        
    Returns:
        List of HDF5 file paths
    """
    data_root = os.path.expanduser(data_root)
    pattern = os.path.join(data_root, "**", "*.hdf5")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def load_episode_from_hdf5(filepath: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load episode data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        images: (T, 384, 384, 3) uint8 RGB
        actions: (T, 7) float32
        instruction: string
    """
    with h5py.File(filepath, 'r') as f:
        # Load images
        images = f['observations/images/eye_in_hand'][:]
        
        # Load actions
        actions = f['actions'][:]
        
        # Load instruction
        instruction = f['observations/instruction'][()].decode('utf-8') if isinstance(f['observations/instruction'][()], bytes) else str(f['observations/instruction'][()])
        
        # Optional: Load proprioception for future use
        # joint_positions = f['observations/proprioception/joint_positions'][:]
        # gripper_width = f['observations/proprioception/gripper_width'][:]
    
    return images, actions, instruction


def compute_action_stats_hdf5(
    data_root: str,
    cache_path: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute action mean and std over HDF5 dataset.
    
    Args:
        data_root: Root directory containing HDF5 files
        cache_path: Path to save/load cached statistics
        num_samples: Maximum number of samples (None = all)
        
    Returns:
        mean: (7,) action mean
        std: (7,) action std
    """
    # Try to load from cache
    if cache_path is not None and os.path.isfile(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                stats = pickle.load(f)
            return stats['mean'], stats['std']
        except Exception:
            pass
    
    # Compute from scratch
    files = find_hdf5_files(data_root)
    if not files:
        print(f"[WARNING] No HDF5 files found in {data_root}, using default stats")
        return np.zeros(ACTION_DIM, dtype=np.float32), np.ones(ACTION_DIM, dtype=np.float32)
    
    all_actions = []
    total_samples = 0
    
    for filepath in files:
        try:
            _, actions, _ = load_episode_from_hdf5(filepath)
            all_actions.append(actions)
            total_samples += len(actions)
            
            if num_samples is not None and total_samples >= num_samples:
                break
        except Exception as e:
            print(f"[WARNING] Failed to load {filepath}: {e}")
            continue
    
    if not all_actions:
        print(f"[WARNING] No valid episodes loaded, using default stats")
        return np.zeros(ACTION_DIM, dtype=np.float32), np.ones(ACTION_DIM, dtype=np.float32)
    
    # Concatenate and compute statistics
    actions_concat = np.concatenate(all_actions, axis=0)
    if num_samples is not None and len(actions_concat) > num_samples:
        actions_concat = actions_concat[:num_samples]
    
    mean = np.mean(actions_concat, axis=0).astype(np.float32)
    std = np.std(actions_concat, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)  # Avoid division by zero
    
    # Save to cache
    if cache_path is not None:
        try:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({'mean': mean, 'std': std}, f)
        except Exception as e:
            print(f"[WARNING] Failed to save stats cache: {e}")
    
    return mean, std


class VTLAHDF5Dataset(Dataset):
    """
    VTLA HDF5 Dataset for VLA training.
    
    Loads (image, action) pairs from RLBench-Panda HDF5 trajectories.
    Actions are normalized with (a - mean) / std.
    """
    
    def __init__(
        self,
        data_root: str = "./data/rlbench_panda_vtla",
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        compute_stats: bool = True,
        stats_cache: Optional[str] = None,
    ):
        """
        Initialize VTLA HDF5 Dataset.
        
        Args:
            data_root: Root directory containing HDF5 files
            action_mean: Precomputed action mean (7,)
            action_std: Precomputed action std (7,)
            compute_stats: Compute statistics if not provided
            stats_cache: Path to save/load cached statistics
        """
        self.data_root = os.path.expanduser(data_root)
        self.hdf5_files = find_hdf5_files(self.data_root)
        
        if not self.hdf5_files:
            raise ValueError(f"No HDF5 files found in {self.data_root}")
        
        print(f"[VTLAHDF5Dataset] Found {len(self.hdf5_files)} HDF5 files in {self.data_root}")
        
        # Load or compute action statistics
        if action_mean is not None and action_std is not None:
            self.mean = np.asarray(action_mean, dtype=np.float32)
            self.std = np.asarray(action_std, dtype=np.float32)
        elif compute_stats:
            cache_path = stats_cache or os.path.join(self.data_root, "action_mean_std.pkl")
            self.mean, self.std = compute_action_stats_hdf5(self.data_root, cache_path=cache_path)
        else:
            self.mean = np.zeros(ACTION_DIM, dtype=np.float32)
            self.std = np.ones(ACTION_DIM, dtype=np.float32)
        
        print(f"[VTLAHDF5Dataset] Action mean: {self.mean}")
        print(f"[VTLAHDF5Dataset] Action std: {self.std}")
        
        # Build index: (file_idx, frame_idx) -> global_idx
        self.index: List[Tuple[int, int]] = []
        
        for file_idx, filepath in enumerate(self.hdf5_files):
            try:
                with h5py.File(filepath, 'r') as f:
                    episode_length = f['actions'].shape[0]
                    for frame_idx in range(episode_length):
                        self.index.append((file_idx, frame_idx))
            except Exception as e:
                print(f"[WARNING] Failed to index {filepath}: {e}")
                continue
        
        print(f"[VTLAHDF5Dataset] Total samples: {len(self.index)}")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def get_action_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get action statistics."""
        return self.mean.copy(), self.std.copy()
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get (image, action) pair.
        
        Returns:
            Dict with keys:
                'image': (3, 384, 384) float32 [0, 1]
                'action': (7,) float32 normalized
                'action_raw': (7,) float32 raw (unnormalized)
        """
        file_idx, frame_idx = self.index[idx]
        filepath = self.hdf5_files[file_idx]
        
        with h5py.File(filepath, 'r') as f:
            # Load image: (384, 384, 3) uint8 -> (3, 384, 384) float32
            image = f['observations/images/eye_in_hand'][frame_idx]  # (384, 384, 3)
            image = np.transpose(image, (2, 0, 1))  # (3, 384, 384)
            image = image.astype(np.float32) / 255.0  # [0, 1]
            
            # Load action
            action = f['actions'][frame_idx].astype(np.float32)
        
        # Normalize action
        action_norm = (action - self.mean) / self.std
        
        return {
            'image': torch.from_numpy(image).float(),
            'action': torch.from_numpy(action_norm).float(),
            'action_raw': torch.from_numpy(action).float(),
        }


def build_vtla_dataloader(
    data_root: str = "./data/rlbench_panda_vtla",
    batch_size: int = 4,
    num_workers: int = 0,
    action_mean: Optional[np.ndarray] = None,
    action_std: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Build VTLA DataLoader.
    
    Args:
        data_root: Root directory containing HDF5 files
        batch_size: Batch size
        num_workers: Number of worker processes
        action_mean: Precomputed action mean
        action_std: Precomputed action std
        
    Returns:
        dataloader: PyTorch DataLoader
        action_mean: Action mean (7,)
        action_std: Action std (7,)
    """
    dataset = VTLAHDF5Dataset(
        data_root=data_root,
        action_mean=action_mean,
        action_std=action_std,
        compute_stats=(action_mean is None or action_std is None),
    )
    
    mean, std = dataset.get_action_mean_std()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **kwargs,
    )
    
    return dataloader, mean, std


if __name__ == "__main__":
    # Quick test
    print("Testing VTLA HDF5 DataLoader...")
    
    # Test with dummy data
    data_root = "./data/rlbench_panda_vtla"
    
    if not os.path.exists(data_root):
        print(f"Data directory {data_root} does not exist. Creating dummy episode...")
        os.makedirs(data_root, exist_ok=True)
        
        # Create a dummy HDF5 file for testing
        dummy_file = os.path.join(data_root, "episode_000000_test.hdf5")
        with h5py.File(dummy_file, 'w') as f:
            obs_group = f.create_group('observations')
            images_group = obs_group.create_group('images')
            prop_group = obs_group.create_group('proprioception')
            tactile_group = obs_group.create_group('tactile')
            
            # Dummy data
            T = 50
            images = np.random.randint(0, 255, (T, 384, 384, 3), dtype=np.uint8)
            actions = np.random.randn(T, 7).astype(np.float32) * 0.01
            instruction = "Pick and place object"
            
            images_group.create_dataset('eye_in_hand', data=images)
            obs_group.create_dataset('instruction', data=instruction)
            prop_group.create_dataset('joint_positions', data=np.zeros((T, 7), dtype=np.float32))
            prop_group.create_dataset('gripper_width', data=np.zeros(T, dtype=np.float32))
            tactile_group.create_dataset('force_torque', data=np.zeros((T, 6), dtype=np.float32))
            tactile_group.create_dataset('contact_depth', data=np.zeros((T, 384, 384), dtype=np.float32))
            f.create_dataset('actions', data=actions)
        
        print(f"Created dummy episode: {dummy_file}")
    
    try:
        # Test dataset
        dataset = VTLAHDF5Dataset(data_root=data_root)
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Action shape: {sample['action'].shape}")
        print(f"Action (normalized): {sample['action']}")
        print(f"Action (raw): {sample['action_raw']}")
        
        # Test dataloader
        loader, mean, std = build_vtla_dataloader(data_root=data_root, batch_size=2, num_workers=0)
        print(f"\nAction mean: {mean}")
        print(f"Action std: {std}")
        
        batch = next(iter(loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch action shape: {batch['action'].shape}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
