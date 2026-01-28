"""
VTLA Data Collector: RLBench-Panda → ViTaMIn-B HDF5 Format.

This script orchestrates:
1. RLBench-Panda environment with domain randomization
2. Oracle policy for expert demonstrations
3. HDF5 saving in ViTaMIn-B multimodal standard

Data Format (per episode HDF5):
  observations/
    images/
      eye_in_hand: (T, 384, 384, 3) uint8 RGB
    instruction: string (language command)
    proprioception/
      joint_positions: (T, 7) float32
      gripper_width: (T,) float32
    tactile/
      force_torque: (T, 6) float32 (6D F/T at wrist)
      contact_depth: (T, 384, 384) float32 (depth proxy for GelSight)
  actions: (T, 7) float32 [dx, dy, dz, dr, dp, dy, gripper]
  episode_metadata:
    success: bool
    object_type: string
    domain_randomization: dict
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm

from rlbench_panda_env import RLBenchPandaEnv, PandaCalibration
from oracle_policy import OraclePolicy


class ViTaMInBDataWriter:
    """
    HDF5 writer for ViTaMIn-B multimodal data standard.
    
    Schema:
      observations/
        images/eye_in_hand: (T, H, W, 3)
        instruction: string
        proprioception/joint_positions: (T, 7)
        proprioception/gripper_width: (T,)
        tactile/force_torque: (T, 6)
        tactile/contact_depth: (T, H, W)
      actions: (T, 7)
      episode_metadata: {...}
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize data writer.
        
        Args:
            output_dir: Directory to save HDF5 files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[ViTaMInBDataWriter] Output directory: {self.output_dir}")
    
    def save_episode(
        self,
        episode_data: Dict,
        episode_id: int,
        success: bool,
        object_type: str = "unknown",
        domain_randomization: Optional[Dict] = None,
    ):
        """
        Save episode to HDF5 file.
        
        Args:
            episode_data: Dict with 'observations', 'actions', 'instruction'
            episode_id: Episode ID
            success: Task success flag
            object_type: YCB object type
            domain_randomization: Domain randomization parameters
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"episode_{episode_id:06d}_{timestamp}.hdf5"
        
        with h5py.File(filename, 'w') as f:
            # Create groups
            obs_group = f.create_group('observations')
            images_group = obs_group.create_group('images')
            proprioception_group = obs_group.create_group('proprioception')
            tactile_group = obs_group.create_group('tactile')
            
            # Extract trajectory data
            observations = episode_data['observations']
            actions = episode_data['actions']
            instruction = episode_data.get('instruction', 'Pick and place object')
            
            T = len(observations)
            
            # Vision: eye_in_hand (T, 384, 384, 3)
            eye_in_hand_list = []
            for obs in observations:
                img = obs['images']['eye_in_hand']
                if img is not None:
                    eye_in_hand_list.append(img)
                else:
                    # Fallback: black image
                    eye_in_hand_list.append(np.zeros((384, 384, 3), dtype=np.uint8))
            
            eye_in_hand = np.stack(eye_in_hand_list, axis=0).astype(np.uint8)
            images_group.create_dataset('eye_in_hand', data=eye_in_hand, compression='gzip')
            
            # Language: instruction
            obs_group.create_dataset('instruction', data=instruction, dtype=h5py.string_dtype())
            
            # Proprioception: joint_positions (T, 7) and gripper_width (T,)
            joint_positions_list = []
            gripper_width_list = []
            
            for obs in observations:
                prop = obs['proprioception']
                joint_pos = prop.get('joint_positions', np.zeros(7, dtype=np.float32))
                gripper_open = prop.get('gripper_open', 0.5)
                
                joint_positions_list.append(joint_pos)
                # Convert gripper_open [0, 1] to width in meters (Panda: 0-0.08m)
                gripper_width = gripper_open * 0.08
                gripper_width_list.append(gripper_width)
            
            joint_positions = np.stack(joint_positions_list, axis=0).astype(np.float32)
            gripper_width = np.array(gripper_width_list, dtype=np.float32)
            
            proprioception_group.create_dataset('joint_positions', data=joint_positions, compression='gzip')
            proprioception_group.create_dataset('gripper_width', data=gripper_width, compression='gzip')
            
            # Tactile (Proxy for future VTLA Phase 2)
            # For now, generate dummy data as placeholders
            
            # Force-Torque: 6D sensor at wrist (F_x, F_y, F_z, T_x, T_y, T_z)
            # TODO: Extract from RLBench if available, otherwise zeros
            force_torque = np.zeros((T, 6), dtype=np.float32)
            tactile_group.create_dataset('force_torque', data=force_torque, compression='gzip')
            
            # Contact Depth: Depth map from Eye-in-Hand camera as GelSight proxy
            # TODO: Extract depth from wrist camera if available
            contact_depth = np.zeros((T, 384, 384), dtype=np.float32)
            tactile_group.create_dataset('contact_depth', data=contact_depth, compression='gzip')
            
            # Actions: (T, 7) [dx, dy, dz, dr, dp, dy, gripper]
            actions_array = np.stack(actions, axis=0).astype(np.float32)
            f.create_dataset('actions', data=actions_array, compression='gzip')
            
            # Episode Metadata
            metadata_group = f.create_group('episode_metadata')
            metadata_group.attrs['success'] = success
            metadata_group.attrs['object_type'] = object_type
            metadata_group.attrs['episode_length'] = T
            metadata_group.attrs['timestamp'] = timestamp
            
            if domain_randomization is not None:
                for key, value in domain_randomization.items():
                    if isinstance(value, (int, float, str, bool)):
                        metadata_group.attrs[f'domain_rand_{key}'] = value
        
        print(f"[ViTaMInBDataWriter] Saved episode {episode_id} → {filename} (T={T}, success={success})")


def collect_episodes(
    num_episodes: int,
    output_dir: str,
    headless: bool = True,
    randomize_domain: bool = True,
    seed: Optional[int] = None,
    min_episode_length: int = 20,
    max_episode_length: int = 200,
    task_name: str = "PickAndPlace",
):
    """
    Collect expert demonstration episodes using Oracle policy.
    
    Args:
        num_episodes: Number of episodes to collect
        output_dir: Directory to save HDF5 files
        headless: Run RLBench without GUI
        randomize_domain: Enable domain randomization
        seed: Random seed
        min_episode_length: Minimum episode length to save
        max_episode_length: Maximum episode steps
    """
    print(f"[VTLA Data Collector] Starting collection of {num_episodes} episodes")
    print(f"[VTLA Data Collector] Output directory: {output_dir}")
    print(f"[VTLA Data Collector] Domain randomization: {randomize_domain}")
    
    # Initialize environment
    env = RLBenchPandaEnv(
        headless=headless,
        randomize_domain=randomize_domain,
        seed=seed,
        task_name=task_name,
    )
    
    calibration = env.calibration
    
    # Initialize Oracle policy
    oracle = OraclePolicy()
    
    # Initialize data writer
    writer = ViTaMInBDataWriter(output_dir)
    
    # Collection statistics
    success_count = 0
    episode_lengths = []
    
    for episode_id in tqdm(range(num_episodes), desc="Collecting episodes"):
        try:
            # region agent log
            import json, time, os, subprocess
            _log_data = {
                "id": f"log_{int(time.time()*1000)}_pre_reset",
                "timestamp": int(time.time() * 1000),
                "location": "vtla_data_collector.py:224",
                "message": "Before env.reset()",
                "data": {
                    "episode_id": episode_id,
                    "DISPLAY": os.environ.get("DISPLAY"),
                    "COPPELIASIM_ROOT": os.environ.get("COPPELIASIM_ROOT"),
                    "LD_LIBRARY_PATH_prefix": os.environ.get("LD_LIBRARY_PATH", "")[:300]
                },
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H1,H4,H6"
            }
            try:
                with open("/workspace/nanoLLaVA/debug.log", "a") as f:
                    f.write(json.dumps(_log_data) + "\n")
            except: pass
            # 诊断 GLX/EGL 可用性
            try:
                glxinfo = subprocess.check_output("glxinfo -B 2>&1 || echo 'glxinfo_failed'", shell=True, text=True)
                eglinfo = subprocess.check_output("eglinfo 2>&1 || echo 'eglinfo_failed'", shell=True, text=True)
                _diag = {
                    "id": f"log_{int(time.time()*1000)}_opengl_diag",
                    "timestamp": int(time.time() * 1000),
                    "location": "vtla_data_collector.py:224",
                    "message": "OpenGL diagnostics",
                    "data": {
                        "glxinfo": glxinfo[:500],
                        "eglinfo": eglinfo[:500]
                    },
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H6,H7"
                }
                with open("/workspace/nanoLLaVA/debug.log", "a") as f:
                    f.write(json.dumps(_diag) + "\n")
            except: pass
            # endregion
            
            # Reset environment and oracle
            obs = env.reset()
            
            # region agent log
            _log_data = {
                "id": f"log_{int(time.time()*1000)}_post_reset",
                "timestamp": int(time.time() * 1000),
                "location": "vtla_data_collector.py:248",
                "message": "After env.reset() success",
                "data": {
                    "episode_id": episode_id,
                    "obs_keys": list(obs.keys()) if isinstance(obs, dict) else "not_dict"
                },
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H3,H8"
            }
            try:
                with open("/workspace/nanoLLaVA/debug.log", "a") as f:
                    f.write(json.dumps(_log_data) + "\n")
            except: pass
            # endregion
            
            oracle.reset()
            
            episode_observations = []
            episode_actions = []
            instruction = "Pick and place object"  # TODO: Get from env
            
            # Episode loop
            for step in range(max_episode_length):
                # Get action from oracle
                action, info = oracle.get_action(obs, calibration)
                
                # Execute action
                next_obs, reward, done, env_info = env.step(action)
                
                # Store data
                episode_observations.append(obs)
                episode_actions.append(action)
                
                # Update observation
                obs = next_obs
                
                # Check termination
                if oracle.is_done() or done:
                    break
            
            # Check success
            success = env_info.get('success', False)
            episode_length = len(episode_observations)
            
            # Only save successful episodes with sufficient length
            if success and episode_length >= min_episode_length:
                episode_data = {
                    'observations': episode_observations,
                    'actions': episode_actions,
                    'instruction': instruction,
                }
                
                writer.save_episode(
                    episode_data=episode_data,
                    episode_id=episode_id,
                    success=success,
                    object_type="ycb_object",  # TODO: Get actual object type
                    domain_randomization={'enabled': randomize_domain},
                )
                
                success_count += 1
                episode_lengths.append(episode_length)
            else:
                reason = "too short" if episode_length < min_episode_length else "failed"
                print(f"[VTLA Data Collector] Episode {episode_id} skipped ({reason}): "
                      f"length={episode_length}, success={success}")
        
        except Exception as e:
            print(f"[VTLA Data Collector] Episode {episode_id} error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Close environment
    env.close()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("[VTLA Data Collector] Collection Complete!")
    print("=" * 60)
    print(f"Total episodes attempted: {num_episodes}")
    print(f"Successful episodes saved: {success_count}")
    print(f"Success rate: {success_count / num_episodes * 100:.1f}%")
    if episode_lengths:
        print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
        print(f"Episode length range: [{np.min(episode_lengths)}, {np.max(episode_lengths)}]")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="VTLA Data Collector: RLBench-Panda → ViTaMIn-B HDF5"
    )
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to collect')
    parser.add_argument('--output_dir', type=str, default='./data/rlbench_panda_vtla',
                        help='Output directory for HDF5 files')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run without GUI (default: True)')
    parser.add_argument('--gui', action='store_true',
                        help='Run with GUI (overrides --headless)')
    parser.add_argument('--no_randomize', action='store_true',
                        help='Disable domain randomization')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--min_episode_length', type=int, default=20,
                        help='Minimum episode length to save')
    parser.add_argument('--max_episode_length', type=int, default=200,
                        help='Maximum episode steps')
    parser.add_argument('--task_name', type=str, default="PickAndPlace",
                        help='RLBench task class name (e.g., PickAndPlace, PickAndLift)')
    
    args = parser.parse_args()
    
    headless = not args.gui if args.gui else args.headless
    randomize_domain = not args.no_randomize
    
    collect_episodes(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        headless=headless,
        randomize_domain=randomize_domain,
        seed=args.seed,
        min_episode_length=args.min_episode_length,
        max_episode_length=args.max_episode_length,
        task_name=args.task_name,
    )


if __name__ == "__main__":
    main()
