"""
Oracle Scripted Policy for Expert Demonstration Generation.

This module implements a ground-truth-based policy that uses:
1. GT object poses from RLBench
2. Trajectory planning (Hover → Approach → Grasp → Lift)
3. Operational Space Controller (OSC) execution
4. Action frame transformation (Camera → Base)

The Oracle bypasses the VLA model to generate high-quality training data.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
import scipy.spatial.transform as tf


class PickAndPlacePhase(Enum):
    """Pick and place task phases."""
    HOVER = 0      # Move above object (10cm clearance)
    APPROACH = 1   # Move down to grasp height
    GRASP = 2      # Close gripper
    LIFT = 3       # Lift object
    MOVE_TO_PLACE = 4  # Move to place location
    PLACE = 5      # Open gripper and place
    RETREAT = 6    # Move back to neutral
    DONE = 7


class OraclePolicy:
    """
    Oracle policy using ground truth for expert demonstrations.
    
    This policy:
    1. Perceives: Retrieves GT object pose from RLBench
    2. Plans: Generates trajectory based on current phase
    3. Executes: Uses OSC to follow trajectory, outputs actions in camera frame
    """
    
    def __init__(
        self,
        hover_height: float = 0.10,  # 10cm above object
        approach_speed: float = 0.02,  # 2cm per step
        lift_height: float = 0.15,  # 15cm lift
        place_offset: np.ndarray = None,  # Offset from pick to place
        dt: float = 0.05,  # Control frequency (20Hz)
    ):
        """
        Initialize Oracle policy.
        
        Args:
            hover_height: Clearance height above object (meters)
            approach_speed: Approach velocity (m/s)
            lift_height: Height to lift object (meters)
            place_offset: [dx, dy, dz] offset for place location
            dt: Control timestep (seconds)
        """
        self.hover_height = hover_height
        self.approach_speed = approach_speed
        self.lift_height = lift_height
        self.dt = dt
        
        # Default place offset: 20cm to the right
        self.place_offset = place_offset if place_offset is not None else np.array([0.0, 0.2, 0.0])
        
        # Internal state
        self.phase = PickAndPlacePhase.HOVER
        self.target_pose = None  # Target TCP pose in base frame
        self.object_pose = None  # Object pose in base frame
        self.pick_pose = None  # Grasp pose in base frame
        self.place_pose = None  # Place pose in base frame
        
        # Phase transition conditions
        self.position_tolerance = 0.01  # 1cm
        self.gripper_close_steps = 10  # Steps to close gripper
        self.gripper_step_counter = 0
    
    def reset(self):
        """Reset policy for new episode."""
        self.phase = PickAndPlacePhase.HOVER
        self.target_pose = None
        self.object_pose = None
        self.pick_pose = None
        self.place_pose = None
        self.gripper_step_counter = 0
    
    def perceive(self, observation: Dict) -> bool:
        """
        Extract ground truth object pose from observation.
        
        Args:
            observation: Observation dict with 'task_state'
            
        Returns:
            True if object pose successfully extracted
        """
        task_state = observation.get('task_state', None)
        if task_state is None or len(task_state) < 7:
            return False
        
        # Assume task_state contains object pose: [x, y, z, qx, qy, qz, qw, ...]
        self.object_pose = np.array(task_state[:7], dtype=np.float32)
        
        # Compute grasp pose (above object, gripper pointing down)
        self.pick_pose = self._compute_grasp_pose(self.object_pose)
        
        # Compute place pose
        place_position = self.object_pose[:3] + self.place_offset
        self.place_pose = np.concatenate([place_position, self.object_pose[3:7]])
        
        return True
    
    def plan_trajectory(self, current_tcp_pose: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Plan next waypoint based on current phase.
        
        Args:
            current_tcp_pose: Current TCP pose [x, y, z, qx, qy, qz, qw]
            
        Returns:
            target_pose: Target TCP pose [x, y, z, qx, qy, qz, qw]
            gripper_command: Gripper command [0=close, 1=open]
        """
        if self.phase == PickAndPlacePhase.HOVER:
            # Hover above object
            hover_position = self.pick_pose[:3].copy()
            hover_position[2] += self.hover_height
            target_pose = np.concatenate([hover_position, self.pick_pose[3:7]])
            gripper_command = 1.0  # Open
            
        elif self.phase == PickAndPlacePhase.APPROACH:
            # Approach grasp pose
            target_pose = self.pick_pose.copy()
            gripper_command = 1.0  # Keep open
            
        elif self.phase == PickAndPlacePhase.GRASP:
            # Hold position, close gripper
            target_pose = current_tcp_pose.copy()
            gripper_command = 0.0  # Close
            
        elif self.phase == PickAndPlacePhase.LIFT:
            # Lift object
            lift_position = self.pick_pose[:3].copy()
            lift_position[2] += self.lift_height
            target_pose = np.concatenate([lift_position, self.pick_pose[3:7]])
            gripper_command = 0.0  # Keep closed
            
        elif self.phase == PickAndPlacePhase.MOVE_TO_PLACE:
            # Move to place location (with clearance)
            place_position = self.place_pose[:3].copy()
            place_position[2] += self.hover_height
            target_pose = np.concatenate([place_position, self.place_pose[3:7]])
            gripper_command = 0.0  # Keep closed
            
        elif self.phase == PickAndPlacePhase.PLACE:
            # Lower to place height
            target_pose = self.place_pose.copy()
            gripper_command = 1.0  # Open
            
        elif self.phase == PickAndPlacePhase.RETREAT:
            # Retreat to neutral
            retreat_position = self.place_pose[:3].copy()
            retreat_position[2] += self.hover_height
            target_pose = np.concatenate([retreat_position, self.place_pose[3:7]])
            gripper_command = 1.0  # Open
            
        else:  # DONE
            target_pose = current_tcp_pose.copy()
            gripper_command = 1.0
        
        self.target_pose = target_pose
        return target_pose, gripper_command
    
    def update_phase(self, current_tcp_pose: np.ndarray):
        """
        Update phase based on current TCP pose.
        
        Args:
            current_tcp_pose: Current TCP pose [x, y, z, qx, qy, qz, qw]
        """
        if self.target_pose is None:
            return
        
        # Check if reached target
        position_error = np.linalg.norm(current_tcp_pose[:3] - self.target_pose[:3])
        reached_target = position_error < self.position_tolerance
        
        # Phase transitions
        if self.phase == PickAndPlacePhase.HOVER and reached_target:
            self.phase = PickAndPlacePhase.APPROACH
            
        elif self.phase == PickAndPlacePhase.APPROACH and reached_target:
            self.phase = PickAndPlacePhase.GRASP
            self.gripper_step_counter = 0
            
        elif self.phase == PickAndPlacePhase.GRASP:
            self.gripper_step_counter += 1
            if self.gripper_step_counter >= self.gripper_close_steps:
                self.phase = PickAndPlacePhase.LIFT
                
        elif self.phase == PickAndPlacePhase.LIFT and reached_target:
            self.phase = PickAndPlacePhase.MOVE_TO_PLACE
            
        elif self.phase == PickAndPlacePhase.MOVE_TO_PLACE and reached_target:
            self.phase = PickAndPlacePhase.PLACE
            self.gripper_step_counter = 0
            
        elif self.phase == PickAndPlacePhase.PLACE:
            self.gripper_step_counter += 1
            if self.gripper_step_counter >= self.gripper_close_steps:
                self.phase = PickAndPlacePhase.RETREAT
                
        elif self.phase == PickAndPlacePhase.RETREAT and reached_target:
            self.phase = PickAndPlacePhase.DONE
    
    def compute_action_camera_frame(
        self,
        current_tcp_pose: np.ndarray,
        target_tcp_pose: np.ndarray,
        gripper_command: float,
        T_ee_base: np.ndarray,
        T_cam_ee: np.ndarray,
    ) -> np.ndarray:
        """
        Compute 7D action in camera frame.
        
        The VLA Action Head predicts ΔP = [dx, dy, dz, dr, dp, dy, g] in Camera Frame.
        This function computes the delta in camera frame from base frame trajectory.
        
        Args:
            current_tcp_pose: Current TCP pose [x, y, z, qx, qy, qz, qw]
            target_tcp_pose: Target TCP pose [x, y, z, qx, qy, qz, qw]
            gripper_command: Gripper [0=close, 1=open]
            T_ee_base: End-effector pose in base frame (4x4)
            T_cam_ee: Camera pose relative to EE (4x4)
            
        Returns:
            action: 7D action [dx, dy, dz, droll, dpitch, dyaw, gripper] in camera frame
        """
        # Compute position delta in base frame
        delta_pos_base = target_tcp_pose[:3] - current_tcp_pose[:3]
        
        # Compute orientation delta in base frame
        current_rot = tf.Rotation.from_quat(current_tcp_pose[3:7])
        target_rot = tf.Rotation.from_quat(target_tcp_pose[3:7])
        delta_rot = target_rot * current_rot.inv()
        delta_euler_base = delta_rot.as_euler('xyz')
        
        # Transform delta to camera frame
        # Camera frame: T_cam_base = T_ee_base @ T_cam_ee
        T_cam_base = T_ee_base @ T_cam_ee
        R_cam_base = T_cam_base[:3, :3]
        
        # Delta position in camera frame: R_cam_base^T @ delta_pos_base
        delta_pos_cam = R_cam_base.T @ delta_pos_base
        
        # Delta orientation in camera frame
        # (Simplified: assume small angles, use same rotation)
        delta_euler_cam = R_cam_base.T @ delta_euler_base
        
        # Gripper command: [0, 1] → [-1, 1] for VLA format
        gripper_vla = 2.0 * gripper_command - 1.0
        
        # Assemble 7D action
        action = np.concatenate([
            delta_pos_cam,
            delta_euler_cam,
            [gripper_vla]
        ]).astype(np.float32)
        
        return action
    
    def get_action(
        self,
        observation: Dict,
        calibration,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate action for current observation.
        
        Args:
            observation: Observation dict with proprioception and task state
            calibration: PandaCalibration instance
            
        Returns:
            action: 7D action [dx, dy, dz, droll, dpitch, dyaw, gripper] in camera frame
            info: Dict with phase, target_pose, etc.
        """
        # Extract current TCP pose
        gripper_pose = observation['proprioception']['gripper_pose']
        current_tcp_pose = gripper_pose  # Assuming gripper_pose is TCP
        
        # Extract transformation matrices
        T_ee_base = observation['T_ee_base']
        T_cam_ee = calibration.T_cam_ee
        
        # Perceive (first time only)
        if self.object_pose is None:
            success = self.perceive(observation)
            if not success:
                # Return null action if perception fails
                return np.zeros(7, dtype=np.float32), {'phase': 'FAILED'}
        
        # Plan trajectory
        target_tcp_pose, gripper_command = self.plan_trajectory(current_tcp_pose)
        
        # Compute action in camera frame
        action = self.compute_action_camera_frame(
            current_tcp_pose=current_tcp_pose,
            target_tcp_pose=target_tcp_pose,
            gripper_command=gripper_command,
            T_ee_base=T_ee_base,
            T_cam_ee=T_cam_ee,
        )
        
        # Update phase
        self.update_phase(current_tcp_pose)
        
        info = {
            'phase': self.phase.name,
            'target_pose': target_tcp_pose,
            'position_error': np.linalg.norm(current_tcp_pose[:3] - target_tcp_pose[:3]),
        }
        
        return action, info
    
    def is_done(self) -> bool:
        """Check if task is complete."""
        return self.phase == PickAndPlacePhase.DONE
    
    @staticmethod
    def _compute_grasp_pose(object_pose: np.ndarray) -> np.ndarray:
        """
        Compute grasp pose for object.
        
        Args:
            object_pose: Object pose [x, y, z, qx, qy, qz, qw]
            
        Returns:
            Grasp pose [x, y, z, qx, qy, qz, qw] (gripper pointing down)
        """
        # Position: slightly above object center
        grasp_position = object_pose[:3].copy()
        grasp_position[2] += 0.02  # 2cm above object center
        
        # Orientation: gripper pointing down (Z-axis down)
        # Rotation: 180° around X-axis (tool frame convention)
        grasp_orientation = tf.Rotation.from_euler('xyz', [np.pi, 0, 0]).as_quat()
        
        grasp_pose = np.concatenate([grasp_position, grasp_orientation])
        return grasp_pose.astype(np.float32)


class OperationalSpaceController:
    """
    Operational Space Controller (OSC) for Cartesian control.
    
    This controller computes joint velocities to achieve desired
    end-effector velocity in Cartesian space.
    """
    
    def __init__(
        self,
        gains_position: float = 1.0,
        gains_orientation: float = 0.5,
        max_velocity: float = 0.5,  # m/s
    ):
        """
        Initialize OSC.
        
        Args:
            gains_position: Position control gain
            gains_orientation: Orientation control gain
            max_velocity: Maximum end-effector velocity (m/s)
        """
        self.Kp_pos = gains_position
        self.Kp_ori = gains_orientation
        self.max_velocity = max_velocity
    
    def compute_joint_velocities(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
        jacobian: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute joint velocities using OSC.
        
        Args:
            current_pose: Current TCP pose [x, y, z, qx, qy, qz, qw]
            target_pose: Target TCP pose [x, y, z, qx, qy, qz, qw]
            jacobian: End-effector Jacobian (6 x n_joints)
            dt: Control timestep
            
        Returns:
            Joint velocities (n_joints,)
        """
        # Position error
        pos_error = target_pose[:3] - current_pose[:3]
        
        # Orientation error (axis-angle)
        current_rot = tf.Rotation.from_quat(current_pose[3:7])
        target_rot = tf.Rotation.from_quat(target_pose[3:7])
        ori_error = (target_rot * current_rot.inv()).as_rotvec()
        
        # Desired Cartesian velocity
        vel_pos = self.Kp_pos * pos_error
        vel_ori = self.Kp_ori * ori_error
        vel_desired = np.concatenate([vel_pos, vel_ori])
        
        # Clip velocity
        vel_norm = np.linalg.norm(vel_desired[:3])
        if vel_norm > self.max_velocity:
            vel_desired[:3] *= self.max_velocity / vel_norm
        
        # Compute joint velocities: q_dot = J^+ @ vel_desired
        J_pinv = np.linalg.pinv(jacobian)
        q_dot = J_pinv @ vel_desired
        
        return q_dot


if __name__ == "__main__":
    # Quick test
    print("Testing Oracle Policy...")
    
    # Dummy observation
    obs = {
        'proprioception': {
            'gripper_pose': np.array([0.5, 0.0, 0.3, 0, 0, 0, 1], dtype=np.float32),
        },
        'task_state': np.array([0.5, 0.1, 0.05, 0, 0, 0, 1], dtype=np.float32),
        'T_ee_base': np.eye(4, dtype=np.float32),
    }
    
    # Dummy calibration
    class DummyCalibration:
        T_cam_ee = np.eye(4, dtype=np.float32)
        T_cam_ee[:3, 3] = [0.05, 0.0, 0.04]
    
    calibration = DummyCalibration()
    
    # Test oracle
    oracle = OraclePolicy()
    action, info = oracle.get_action(obs, calibration)
    
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    print(f"Phase: {info['phase']}")
    print(f"Target pose: {info['target_pose']}")
    
    print("Test completed successfully!")
