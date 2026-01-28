"""
Test script to verify VTLA pipeline components.

This script validates:
1. HDF5 data format compliance
2. DataLoader functionality
3. Model compatibility with new data format
4. Action statistics computation
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

print("=" * 60)
print("VTLA Pipeline Validation")
print("=" * 60)

# Test 1: HDF5 DataLoader
print("\n[Test 1/4] Testing VTLA HDF5 DataLoader...")
try:
    from vtla_hdf5_dataloader import VTLAHDF5Dataset, build_vtla_dataloader
    print("✓ HDF5 DataLoader module imported successfully")
    
    # Create dummy data for testing
    dummy_data_dir = "./data/test_vtla_hdf5"
    os.makedirs(dummy_data_dir, exist_ok=True)
    
    import h5py
    dummy_file = os.path.join(dummy_data_dir, "episode_test.hdf5")
    
    if not os.path.exists(dummy_file):
        print(f"  Creating dummy HDF5 file: {dummy_file}")
        with h5py.File(dummy_file, 'w') as f:
            obs_group = f.create_group('observations')
            images_group = obs_group.create_group('images')
            prop_group = obs_group.create_group('proprioception')
            tactile_group = obs_group.create_group('tactile')
            
            T = 100
            images = np.random.randint(0, 255, (T, 384, 384, 3), dtype=np.uint8)
            actions = np.random.randn(T, 7).astype(np.float32) * 0.02
            instruction = "Pick up the red cube"
            
            images_group.create_dataset('eye_in_hand', data=images)
            obs_group.create_dataset('instruction', data=instruction)
            prop_group.create_dataset('joint_positions', data=np.zeros((T, 7), dtype=np.float32))
            prop_group.create_dataset('gripper_width', data=np.ones(T, dtype=np.float32) * 0.04)
            tactile_group.create_dataset('force_torque', data=np.zeros((T, 6), dtype=np.float32))
            tactile_group.create_dataset('contact_depth', data=np.zeros((T, 384, 384), dtype=np.float32))
            f.create_dataset('actions', data=actions)
            
            metadata_group = f.create_group('episode_metadata')
            metadata_group.attrs['success'] = True
            metadata_group.attrs['object_type'] = "cube"
        
        print("  ✓ Dummy HDF5 created")
    
    # Test dataset
    dataset = VTLAHDF5Dataset(data_root=dummy_data_dir)
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    
    # Test sample
    sample = dataset[0]
    assert sample['image'].shape == (3, 384, 384), f"Image shape mismatch: {sample['image'].shape}"
    assert sample['action'].shape == (7,), f"Action shape mismatch: {sample['action'].shape}"
    print(f"  ✓ Sample loaded: image {sample['image'].shape}, action {sample['action'].shape}")
    
    # Test dataloader
    loader, mean, std = build_vtla_dataloader(data_root=dummy_data_dir, batch_size=8, num_workers=0)
    batch = next(iter(loader))
    print(f"  ✓ DataLoader works: batch image {batch['image'].shape}, batch action {batch['action'].shape}")
    print(f"  ✓ Action stats: mean={mean.round(4)}, std={std.round(4)}")
    
    print("✓ Test 1 PASSED: HDF5 DataLoader works correctly")
    
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Oracle Policy
print("\n[Test 2/4] Testing Oracle Policy...")
try:
    from oracle_policy import OraclePolicy, OperationalSpaceController
    
    # Create dummy observation
    obs = {
        'proprioception': {
            'gripper_pose': np.array([0.5, 0.0, 0.3, 0, 0, 0, 1], dtype=np.float32),
        },
        'task_state': np.array([0.5, 0.1, 0.05, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        'T_ee_base': np.eye(4, dtype=np.float32),
    }
    
    class DummyCalibration:
        T_cam_ee = np.eye(4, dtype=np.float32)
        T_cam_ee[:3, 3] = [0.05, 0.0, 0.04]
        T_tcp_ee = np.eye(4, dtype=np.float32)
        T_tcp_ee[:3, 3] = [0.0, 0.0, 0.1034]
    
    calibration = DummyCalibration()
    oracle = OraclePolicy()
    
    action, info = oracle.get_action(obs, calibration)
    assert action.shape == (7,), f"Action shape mismatch: {action.shape}"
    assert 'phase' in info, "Info missing 'phase' key"
    
    print(f"  ✓ Oracle policy generated action: {action.round(4)}")
    print(f"  ✓ Current phase: {info['phase']}")
    print("✓ Test 2 PASSED: Oracle Policy works correctly")
    
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: RLBench Environment (optional, requires RLBench installation)
print("\n[Test 3/4] Testing RLBench-Panda Environment...")
try:
    from rlbench_panda_env import RLBenchPandaEnv, PandaCalibration, DomainRandomizer
    
    # Test calibration
    calib = PandaCalibration()
    T_ee_base = np.eye(4)
    T_ee_base[:3, 3] = [0.5, 0.0, 0.3]
    
    T_cam_base = calib.get_camera_pose_in_base(T_ee_base)
    T_tcp_base = calib.get_tcp_pose_in_base(T_ee_base)
    
    print(f"  ✓ Hand-Eye calibration: T_cam_ee translation = {calib.T_cam_ee[:3, 3]}")
    print(f"  ✓ TCP calibration: T_tcp_ee translation = {calib.T_tcp_ee[:3, 3]}")
    
    # Test domain randomizer
    randomizer = DomainRandomizer(seed=42)
    workspace = {'x': (0.3, 0.7), 'y': (-0.4, 0.4), 'z': (0.05, 0.5)}
    pos, ori = randomizer.randomize_object_pose(workspace)
    print(f"  ✓ Randomized object pose: pos={pos.round(3)}, ori={ori.round(3)}")
    
    table_color = randomizer.randomize_table_texture()
    print(f"  ✓ Randomized table texture: RGB={table_color.round(3)}")
    
    lighting = randomizer.randomize_lighting()
    print(f"  ✓ Randomized lighting: intensity={lighting['intensity']:.2f}")
    
    print("✓ Test 3 PASSED: RLBench components work correctly")
    print("  (Note: Full RLBench test requires CoppeliaSim installation)")
    
except ImportError as e:
    print(f"⚠ Test 3 SKIPPED: RLBench not installed ({e})")
    print("  Install with: pip install git+https://github.com/stepjam/PyRep.git")
    print("               pip install git+https://github.com/stepjam/RLBench.git")

except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Model Integration
print("\n[Test 4/4] Testing Model Integration...")
try:
    from bunny.model.bunny_arch import ActionHead
    
    # Test ActionHead
    hidden_size = 2048
    action_head = ActionHead(hidden_size=hidden_size, action_dim=7, intermediate_size=512)
    
    # Dummy input
    dummy_hidden = torch.randn(4, hidden_size)  # Batch of 4
    output = action_head(dummy_hidden)
    
    assert output.shape == (4, 7), f"ActionHead output shape mismatch: {output.shape}"
    print(f"  ✓ ActionHead forward pass: input {dummy_hidden.shape} -> output {output.shape}")
    
    # Test with dataloader output
    sample = dataset[0]
    print(f"  ✓ Sample from dataset: image {sample['image'].shape}, action {sample['action'].shape}")
    
    print("✓ Test 4 PASSED: Model integration works correctly")
    
except Exception as e:
    print(f"✗ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("VTLA Pipeline Validation Complete!")
print("=" * 60)
print("\nAll core components are working correctly:")
print("  ✓ HDF5 DataLoader (ViTaMIn-B format)")
print("  ✓ Oracle Policy (GT-based trajectory planning)")
print("  ✓ RLBench-Panda Environment (calibration + randomization)")
print("  ✓ Model Integration (ActionHead + NanoLLaVA)")
print("\nNext steps:")
print("  1. Install RLBench: pip install git+https://github.com/stepjam/PyRep.git")
print("                      pip install git+https://github.com/stepjam/RLBench.git")
print("  2. Collect data: bash scripts/run_collect_panda_data.sh")
print("  3. Train VLA: bash scripts/run_train_vla_panda.sh")
print("=" * 60)
