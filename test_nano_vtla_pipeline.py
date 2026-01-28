"""
Test Pipeline for Nano-VTLA Model

This script:
1. Loads one HDF5 sample from ViTaMIn-B dataset
2. Runs a forward pass through the VTLA model
3. Prints predicted action vector and ground truth
4. Provides a placeholder for future Octopi reasoning logic (material/hardness prediction)

Usage:
    python test_nano_vtla_pipeline.py --data_dir ./data/vitamin_b --checkpoint ./outputs/checkpoint_step5000.pt
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path

from bunny.data.vitamin_b_dataset import ViTaMInBDataset

# TODO: Import actual VTLA model when integrated
# from bunny.model.language_model.bunny_vtla import BunnyVTLAForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Test Nano-VTLA Pipeline")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing train/ and val/ subdirectories")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to use (train/val)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index to test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--action_stats_path", type=str, default=None,
                        help="Path to action_mean_std.pkl")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize images")
    
    args = parser.parse_args()
    return args


class NanoVTLAPipeline:
    """
    Test pipeline for Nano-VTLA model.
    
    Demonstrates:
    1. Loading and preprocessing multimodal data (Vision, Tactile, Language)
    2. Running forward pass through VTLA model
    3. Predicting 7-DoF relative actions
    4. Placeholder for future physical property prediction (Octopi reasoning)
    """
    
    def __init__(self, model, dataset, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[Pipeline] Initialized on device: {self.device}")
    
    @torch.no_grad()
    def predict_action(self, sample):
        """
        Predict action from a single sample.
        
        Args:
            sample: Dictionary with 'image', 'tactile', 'instruction', 'action'
        
        Returns:
            predicted_action: (7,) tensor - Predicted action [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Move to device
        image = sample['image'].unsqueeze(0).to(self.device)  # (1, 3, H, W)
        tactile = sample['tactile'].unsqueeze(0).to(self.device)  # (1, 3, 128, 128)
        instruction = sample['instruction']  # str
        
        # TODO: Replace with actual model forward pass
        # This should:
        # 1. Encode vision with SigLIP
        # 2. Encode tactile with ResNet-18
        # 3. Tokenize instruction
        # 4. Fuse tokens: [Instruction, Vision, Tactile]
        # 5. Pass through InternLM2-1.8B
        # 6. Extract last hidden state
        # 7. Pass through ActionHead
        
        # Placeholder prediction
        predicted_action = torch.randn(7, device=self.device)
        
        return predicted_action
    
    def predict_physical_properties(self, sample):
        """
        Placeholder for future Octopi-style reasoning head.
        
        This hook allows future extensions to predict:
        - Material type (soft, rigid, deformable)
        - Hardness (Shore A scale)
        - Compliance
        - etc.
        
        Based on Octopi.pdf: "The LLM output hidden states are accessible 
        for a future Reasoning Head (Material/Hardness prediction)"
        
        Args:
            sample: Dictionary with multimodal inputs
        
        Returns:
            properties: Dictionary with predicted physical properties
                        Currently returns None (not implemented)
        """
        # TODO: Implement when Octopi reasoning logic is added
        # This will:
        # 1. Extract hidden states from LLM
        # 2. Pass through ReasoningHead
        # 3. Predict material/hardness/compliance
        
        if hasattr(self.model, 'reasoning_head') and self.model.reasoning_head.enabled:
            # Future implementation
            properties = {
                'material_type': None,
                'hardness': None,
                'compliance': None
            }
            return properties
        else:
            return None
    
    def test_single_sample(self, sample_idx=0, visualize=False):
        """
        Test pipeline on a single sample.
        
        Args:
            sample_idx: Index of sample to test
            visualize: Whether to visualize images
        """
        print(f"\n{'='*80}")
        print(f"Testing Sample {sample_idx}")
        print(f"{'='*80}")
        
        # Load sample
        sample = self.dataset[sample_idx]
        
        # Print sample info
        print(f"\n[Sample Info]")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Tactile shape: {sample['tactile'].shape}")
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Ground Truth Action (normalized): {sample['action'].numpy()}")
        
        # Denormalize ground truth action
        action_gt_denorm = self.dataset.denormalize_action(sample['action'])
        print(f"  Ground Truth Action (denormalized): {action_gt_denorm.numpy()}")
        print(f"    Translation (dx, dy, dz): {action_gt_denorm[:3].numpy()}")
        print(f"    Rotation (droll, dpitch, dyaw): {action_gt_denorm[3:6].numpy()}")
        print(f"    Gripper: {action_gt_denorm[6].item():.4f}")
        
        # Predict action
        print(f"\n[Action Prediction]")
        predicted_action = self.predict_action(sample)
        print(f"  Predicted Action (normalized): {predicted_action.cpu().numpy()}")
        
        # Denormalize predicted action
        predicted_action_denorm = self.dataset.denormalize_action(predicted_action.cpu())
        print(f"  Predicted Action (denormalized): {predicted_action_denorm.numpy()}")
        print(f"    Translation (dx, dy, dz): {predicted_action_denorm[:3].numpy()}")
        print(f"    Rotation (droll, dpitch, dyaw): {predicted_action_denorm[3:6].numpy()}")
        print(f"    Gripper: {predicted_action_denorm[6].item():.4f}")
        
        # Compute error
        error = torch.abs(predicted_action.cpu() - sample['action'])
        error_denorm = torch.abs(predicted_action_denorm - action_gt_denorm)
        print(f"\n[Error Analysis]")
        print(f"  MAE (normalized): {error.mean().item():.4f}")
        print(f"  MAE (denormalized): {error_denorm.mean().item():.4f}")
        print(f"    Translation MAE: {error_denorm[:3].mean().item():.4f}")
        print(f"    Rotation MAE: {error_denorm[3:6].mean().item():.4f}")
        print(f"    Gripper MAE: {error_denorm[6].item():.4f}")
        
        # Predict physical properties (placeholder)
        print(f"\n[Physical Property Prediction - Octopi Hook]")
        properties = self.predict_physical_properties(sample)
        if properties is not None:
            print(f"  Material Type: {properties['material_type']}")
            print(f"  Hardness: {properties['hardness']}")
            print(f"  Compliance: {properties['compliance']}")
        else:
            print(f"  [Not Implemented] This is a placeholder for future Octopi reasoning logic.")
            print(f"  Future capabilities:")
            print(f"    - Material type prediction (soft/rigid/deformable)")
            print(f"    - Hardness estimation (Shore A scale)")
            print(f"    - Compliance/stiffness prediction")
        
        # Visualize if requested
        if visualize:
            self._visualize_sample(sample)
        
        print(f"\n{'='*80}")
        print(f"Test Completed Successfully!")
        print(f"{'='*80}\n")
    
    def _visualize_sample(self, sample):
        """Visualize vision and tactile images"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Vision image
            image = sample['image'].permute(1, 2, 0).numpy()  # (H, W, 3)
            axes[0].imshow(image)
            axes[0].set_title("Vision (RGB)")
            axes[0].axis('off')
            
            # Tactile image
            tactile = sample['tactile'].permute(1, 2, 0).numpy()  # (128, 128, 3)
            axes[1].imshow(tactile)
            axes[1].set_title("Tactile (GelSight)")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig("test_sample_visualization.png", dpi=150, bbox_inches='tight')
            print(f"\n[Visualization] Saved to test_sample_visualization.png")
            plt.close()
            
        except ImportError:
            print("[Visualization] matplotlib not available, skipping visualization")


def main():
    args = parse_args()
    
    print("="*80)
    print("Nano-VTLA Test Pipeline")
    print("="*80)
    print(f"Data Dir: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Sample Index: {args.sample_idx}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*80)
    
    # Setup action stats path
    if args.action_stats_path is None:
        args.action_stats_path = os.path.join(
            os.path.dirname(args.checkpoint) if args.checkpoint else "./outputs",
            "action_mean_std.pkl"
        )
    
    # Create dataset
    print("\n[Data] Loading dataset...")
    dataset = ViTaMInBDataset(
        data_dir=args.data_dir,
        split=args.split,
        action_stats_path=args.action_stats_path,
        compute_action_stats=False,
        max_episodes=None  # Load all episodes
    )
    
    print(f"[Data] Dataset size: {len(dataset)} samples")
    
    # Load model
    print("\n[Model] Loading Nano-VTLA model...")
    if args.checkpoint:
        # TODO: Load actual model checkpoint
        # checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # model = BunnyVTLAForCausalLM.from_pretrained(...)
        # model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Model] TODO: Load checkpoint from {args.checkpoint}")
    
    # Placeholder model
    import torch.nn as nn
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)
    
    model = DummyModel()
    print("[Model] WARNING: Using dummy model. Replace with actual VTLA model.")
    
    # Create pipeline
    pipeline = NanoVTLAPipeline(
        model=model,
        dataset=dataset,
        device=args.device
    )
    
    # Test on single sample
    pipeline.test_single_sample(
        sample_idx=args.sample_idx,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
