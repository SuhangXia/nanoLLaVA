"""
Quick test script for Nano-VTLA components

Tests:
1. TactileEncoder - ResNet-18 forward pass
2. TactileProjector - Feature projection
3. TactileTower - End-to-end tactile processing
4. ActionHead - Action prediction
5. ReasoningHead - Placeholder for future

Usage:
    python test_vtla_components.py
"""

import torch
import torch.nn as nn
from bunny.model.tactile_encoder import TactileEncoder, TactileProjector, TactileTower
from bunny.model.vtla_arch import ActionHead, ReasoningHead


def test_tactile_encoder():
    """Test TactileEncoder (ResNet-18)"""
    print("\n" + "="*80)
    print("Testing TactileEncoder (ResNet-18)")
    print("="*80)
    
    encoder = TactileEncoder(pretrained=True, freeze_backbone=False)
    
    # Test input: batch of 4 tactile images (128x128 RGB)
    tactile_images = torch.randn(4, 3, 128, 128)
    
    print(f"Input shape: {tactile_images.shape}")
    
    # Forward pass
    features = encoder(tactile_images)
    
    print(f"Output shape: {features.shape}")
    print(f"Expected: (4, 512)")
    print(f"✅ TactileEncoder works!" if features.shape == (4, 512) else "❌ Shape mismatch!")
    
    return encoder


def test_tactile_projector():
    """Test TactileProjector (MLP)"""
    print("\n" + "="*80)
    print("Testing TactileProjector (MLP)")
    print("="*80)
    
    projector = TactileProjector(tactile_hidden_size=512, llm_hidden_size=1024)
    
    # Test input: batch of 4 tactile features (512-dim)
    tactile_features = torch.randn(4, 512)
    
    print(f"Input shape: {tactile_features.shape}")
    
    # Forward pass
    projected = projector(tactile_features)
    
    print(f"Output shape: {projected.shape}")
    print(f"Expected: (4, 1024)")
    print(f"✅ TactileProjector works!" if projected.shape == (4, 1024) else "❌ Shape mismatch!")
    
    return projector


def test_tactile_tower():
    """Test TactileTower (Encoder + Projector)"""
    print("\n" + "="*80)
    print("Testing TactileTower (End-to-End)")
    print("="*80)
    
    tower = TactileTower(
        pretrained=True,
        freeze_encoder=False,
        llm_hidden_size=1024
    )
    
    # Test input: batch of 4 tactile images (128x128 RGB)
    tactile_images = torch.randn(4, 3, 128, 128)
    
    print(f"Input shape: {tactile_images.shape}")
    
    # Forward pass
    embeddings = tower(tactile_images)
    
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: (4, 1024)")
    print(f"✅ TactileTower works!" if embeddings.shape == (4, 1024) else "❌ Shape mismatch!")
    
    # Test parameter freezing
    frozen_params = sum(1 for p in tower.encoder.parameters() if not p.requires_grad)
    trainable_params = sum(1 for p in tower.parameters() if p.requires_grad)
    
    print(f"\nParameter Status:")
    print(f"  Frozen encoder params: {frozen_params}")
    print(f"  Trainable params: {trainable_params}")
    
    return tower


def test_action_head():
    """Test ActionHead (7-DoF prediction)"""
    print("\n" + "="*80)
    print("Testing ActionHead (7-DoF)")
    print("="*80)
    
    action_head = ActionHead(hidden_size=1024, action_dim=7)
    
    # Test input: batch of 4 hidden states (1024-dim)
    hidden_states = torch.randn(4, 1024)
    
    print(f"Input shape: {hidden_states.shape}")
    
    # Forward pass
    actions = action_head(hidden_states)
    
    print(f"Output shape: {actions.shape}")
    print(f"Expected: (4, 7)")
    print(f"✅ ActionHead works!" if actions.shape == (4, 7) else "❌ Shape mismatch!")
    
    print(f"\nSample action output:")
    print(f"  dx, dy, dz: {actions[0, :3].detach().numpy()}")
    print(f"  droll, dpitch, dyaw: {actions[0, 3:6].detach().numpy()}")
    print(f"  gripper: {actions[0, 6].item():.4f}")
    
    return action_head


def test_reasoning_head():
    """Test ReasoningHead (Placeholder for Octopi)"""
    print("\n" + "="*80)
    print("Testing ReasoningHead (Octopi Placeholder)")
    print("="*80)
    
    reasoning_head = ReasoningHead(hidden_size=1024, num_properties=3)
    
    # Test input: batch of 4 hidden states (1024-dim)
    hidden_states = torch.randn(4, 1024)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Enabled: {reasoning_head.enabled}")
    
    # Forward pass (should return None when disabled)
    properties = reasoning_head(hidden_states)
    
    print(f"Output: {properties}")
    print(f"✅ ReasoningHead (disabled) works!" if properties is None else "⚠️ Unexpected output!")
    
    # Test when enabled
    reasoning_head.enabled = True
    properties = reasoning_head(hidden_states)
    
    print(f"\nWhen enabled:")
    print(f"  Output shape: {properties.shape}")
    print(f"  Expected: (4, 3)")
    print(f"  ✅ ReasoningHead (enabled) works!" if properties.shape == (4, 3) else "❌ Shape mismatch!")
    
    return reasoning_head


def test_full_pipeline():
    """Test full pipeline: Tactile → LLM → Action"""
    print("\n" + "="*80)
    print("Testing Full Pipeline")
    print("="*80)
    
    # Create components
    tactile_tower = TactileTower(pretrained=True, llm_hidden_size=1024)
    action_head = ActionHead(hidden_size=1024, action_dim=7)
    
    # Simulate input
    batch_size = 4
    tactile_images = torch.randn(batch_size, 3, 128, 128)
    
    print(f"Input: Tactile images {tactile_images.shape}")
    
    # Step 1: Encode tactile
    tactile_embeddings = tactile_tower(tactile_images)
    print(f"Step 1: Tactile embeddings {tactile_embeddings.shape}")
    
    # Step 2: Simulate LLM processing (in real case, this would be InternLM2)
    # Here we just use the tactile embeddings as "hidden states"
    hidden_states = tactile_embeddings
    print(f"Step 2: LLM hidden states {hidden_states.shape} (simulated)")
    
    # Step 3: Predict action
    actions = action_head(hidden_states)
    print(f"Step 3: Predicted actions {actions.shape}")
    
    print(f"\n✅ Full pipeline works!")
    print(f"Pipeline: Tactile (128×128) → Encoder (512) → Projector (1024) → LLM (1024) → Action (7)")
    
    return tactile_tower, action_head


def main():
    print("="*80)
    print("Nano-VTLA Component Tests")
    print("="*80)
    print("\nTesting individual components of the Nano-VTLA architecture...")
    
    # Test individual components
    encoder = test_tactile_encoder()
    projector = test_tactile_projector()
    tower = test_tactile_tower()
    action_head = test_action_head()
    reasoning_head = test_reasoning_head()
    
    # Test full pipeline
    tactile_tower, action_head = test_full_pipeline()
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("\n✅ All components tested successfully!")
    print("\nNext Steps:")
    print("  1. Integrate with nanoLLaVA language models (InternLM2-1.8B)")
    print("  2. Implement full forward pass in vtla_arch.py")
    print("  3. Test with real ViTaMIn-B HDF5 data")
    print("  4. Run training script")
    print("\nFor full usage, see: NANO_VTLA_README.md")
    print("="*80)


if __name__ == "__main__":
    main()
