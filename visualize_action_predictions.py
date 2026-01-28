"""
Nano-VTLA 动作预测可视化
从验证集随机采样，对比预测动作 vs Ground Truth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import random

from bunny.data.vitamin_b_zarr_dataset import ViTaMInBZarrDataset
from bunny.model.builder import load_pretrained_model
from bunny.model.tactile_encoder import TactileTower
import torch.nn as nn
from bunny.constants import IMAGE_TOKEN_INDEX
from PIL import Image as PILImage


def load_vtla_model(checkpoint_path, device='cuda', bf16=True):
    """加载 VTLA 模型"""
    print(f"[Model] Loading from {checkpoint_path}")
    
    # Load base model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path="BAAI/Bunny-v1_0-2B-zh",
        model_base=None,
        model_name="BAAI/Bunny-v1_0-2B-zh",
        model_type="qwen1.5-1.8b",
        device=device
    )
    
    if bf16:
        model = model.to(torch.bfloat16)
    
    # Add tactile
    model.model.config.use_tactile = True
    dtype = torch.bfloat16 if bf16 else torch.float16
    model.model.tactile_tower = TactileTower(
        pretrained=True, freeze_encoder=False, llm_hidden_size=model.config.hidden_size
    ).to(device=device, dtype=dtype)
    model.model.tactile_pos_embedding = nn.Parameter(
        torch.zeros(1, 1, model.config.hidden_size, device=device, dtype=dtype)
    )
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    print(f"✅ Model loaded from step {ckpt.get('global_step', 0)}")
    
    return model, tokenizer, image_processor


@torch.no_grad()
def predict_action(model, tokenizer, image_processor, image, tactile, device='cuda', bf16=True):
    """预测动作"""
    model.eval()
    
    # Preprocess image
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    pil_image = PILImage.fromarray(img_np)
    processed_image = image_processor([pil_image], return_tensors='pt')['pixel_values']
    processed_image = processed_image.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float16)
    
    # Tokenize
    text = "What action should the robot take?"
    ids = [IMAGE_TOKEN_INDEX] + tokenizer(text).input_ids
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Add batch dimension to tactile
    tactile = tactile.unsqueeze(0).to(device)
    
    # Forward
    num_image_tokens = 729  # SigLIP
    with torch.cuda.amp.autocast(enabled=bf16, dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=processed_image,
            tactile_images=tactile,
            num_image_tokens=num_image_tokens,
            use_cache=False,
            return_dict=True
        )
    
    action_pred = outputs.action_prediction
    return action_pred.squeeze(0).cpu()  # (7,)


def visualize_predictions(samples, predictions, gts, dataset, save_path='action_predictions.png'):
    """
    可视化预测结果
    
    samples: list of 5 samples
    predictions: (5, 7) tensor - 预测的动作
    gts: (5, 7) tensor - Ground Truth 动作
    """
    num_samples = len(samples)
    
    # Denormalize actions
    predictions_denorm = torch.stack([dataset.denormalize_action(p) for p in predictions])
    gts_denorm = torch.stack([dataset.denormalize_action(g) for g in gts])
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    for idx in range(num_samples):
        sample = samples[idx]
        pred = predictions_denorm[idx]
        gt = gts_denorm[idx]
        
        # Row 1: RGB image
        ax_rgb = plt.subplot(3, num_samples, idx + 1)
        rgb = sample['image'].permute(1, 2, 0).numpy()
        ax_rgb.imshow(rgb)
        ax_rgb.set_title(f"Sample {idx+1}: RGB", fontsize=10)
        ax_rgb.axis('off')
        
        # Row 2: Tactile image
        ax_tac = plt.subplot(3, num_samples, num_samples + idx + 1)
        tactile = sample['tactile'].permute(1, 2, 0).numpy()
        ax_tac.imshow(tactile)
        ax_tac.set_title(f"Tactile", fontsize=10)
        ax_tac.axis('off')
        
        # Row 3: 3D action vectors
        ax_3d = plt.subplot(3, num_samples, 2 * num_samples + idx + 1, projection='3d')
        
        # Extract translation (dx, dy, dz)
        pred_trans = pred[:3].numpy()
        gt_trans = gt[:3].numpy()
        
        # Plot vectors from origin
        origin = [0, 0, 0]
        
        # Predicted (red arrow)
        ax_3d.quiver(origin[0], origin[1], origin[2], 
                     pred_trans[0], pred_trans[1], pred_trans[2],
                     color='red', arrow_length_ratio=0.3, linewidth=2, label='Predicted')
        
        # Ground Truth (green arrow)
        ax_3d.quiver(origin[0], origin[1], origin[2],
                     gt_trans[0], gt_trans[1], gt_trans[2],
                     color='green', arrow_length_ratio=0.3, linewidth=2, label='GT')
        
        # Compute L2 distance
        l2_dist = torch.norm(pred - gt).item()
        trans_l2 = torch.norm(pred[:3] - gt[:3]).item()
        rot_l2 = torch.norm(pred[3:6] - gt[3:6]).item()
        
        ax_3d.set_xlabel('dx (m)', fontsize=8)
        ax_3d.set_ylabel('dy (m)', fontsize=8)
        ax_3d.set_zlabel('dz (m)', fontsize=8)
        ax_3d.set_title(f'Action Vectors\nL2={l2_dist:.4f}\nTrans={trans_l2:.4f}, Rot={rot_l2:.4f}', fontsize=9)
        ax_3d.legend(fontsize=7)
        
        # Set limits
        max_range = max(np.abs(pred_trans).max(), np.abs(gt_trans).max()) * 1.5
        ax_3d.set_xlim([-max_range, max_range])
        ax_3d.set_ylim([-max_range, max_range])
        ax_3d.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化保存到: {save_path}")
    plt.close()


def print_action_comparison(idx, pred, gt, dataset):
    """打印动作对比"""
    pred_denorm = dataset.denormalize_action(pred)
    gt_denorm = dataset.denormalize_action(gt)
    
    print(f"\n{'='*60}")
    print(f"Sample {idx+1}")
    print(f"{'='*60}")
    
    print(f"\n预测动作 (Predicted):")
    print(f"  Translation (dx, dy, dz): {pred_denorm[:3].numpy()}")
    print(f"  Rotation (droll, dpitch, dyaw): {pred_denorm[3:6].numpy()}")
    print(f"  Gripper: {pred_denorm[6].item():.4f}")
    
    print(f"\nGround Truth 动作:")
    print(f"  Translation (dx, dy, dz): {gt_denorm[:3].numpy()}")
    print(f"  Rotation (droll, dpitch, dyaw): {gt_denorm[3:6].numpy()}")
    print(f"  Gripper: {gt_denorm[6].item():.4f}")
    
    # Error analysis
    error = torch.abs(pred_denorm - gt_denorm)
    l2_total = torch.norm(pred_denorm - gt_denorm).item()
    l2_trans = torch.norm(pred_denorm[:3] - gt_denorm[:3]).item()
    l2_rot = torch.norm(pred_denorm[3:6] - gt_denorm[3:6]).item()
    
    print(f"\n误差分析:")
    print(f"  总 L2 距离: {l2_total:.4f}")
    print(f"  平移 L2: {l2_trans:.4f} (m)")
    print(f"  旋转 L2: {l2_rot:.4f} (rad)")
    print(f"  夹爪误差: {error[6].item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_dir", type=str, default="/datasets/vitamin_b")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--output", type=str, default="action_predictions.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()
    
    print("=" * 80)
    print("Nano-VTLA 动作预测可视化")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print(f"Samples: {args.num_samples}")
    print("=" * 80)
    
    # Load dataset
    print("\n[Data] Loading validation dataset...")
    action_stats_path = "./outputs/nano_vtla_baseline/action_mean_std.pkl"
    dataset = ViTaMInBZarrDataset(
        data_dir=args.data_dir,
        action_stats_path=action_stats_path,
        compute_action_stats=False
    )
    
    # Random sample from validation split (last 20%)
    val_start = int(0.8 * len(dataset))
    val_indices = list(range(val_start, len(dataset)))
    sampled_indices = random.sample(val_indices, args.num_samples)
    
    samples = [dataset[i] for i in sampled_indices]
    print(f"✅ Sampled {args.num_samples} validation samples")
    
    # Load model
    model, tokenizer, image_processor = load_vtla_model(args.checkpoint, args.device, args.bf16)
    
    # Predict actions
    print(f"\n[Prediction] Running inference...")
    predictions = []
    gts = []
    
    for idx, sample in enumerate(samples):
        image = sample['image']
        tactile = sample['tactile']
        action_gt = sample['action']
        
        # Predict
        action_pred = predict_action(model, tokenizer, image_processor, image, tactile, args.device, args.bf16)
        
        predictions.append(action_pred)
        gts.append(action_gt)
        
        # Print comparison
        print_action_comparison(idx, action_pred, action_gt, dataset)
    
    # Visualize
    print(f"\n[Visualization] Creating comparison plot...")
    predictions = torch.stack(predictions)
    gts = torch.stack(gts)
    
    visualize_predictions(samples, predictions, gts, dataset, args.output)
    
    # Overall statistics
    print(f"\n{'='*60}")
    print(f"总体统计 (共 {args.num_samples} 个样本)")
    print(f"{'='*60}")
    
    predictions_denorm = torch.stack([dataset.denormalize_action(p) for p in predictions])
    gts_denorm = torch.stack([dataset.denormalize_action(g) for g in gts])
    
    mae = torch.abs(predictions_denorm - gts_denorm).mean(dim=0)
    l2_mean = torch.norm(predictions_denorm - gts_denorm, dim=1).mean()
    
    print(f"平均绝对误差 (MAE):")
    print(f"  Translation: {mae[:3].mean().item():.4f} m")
    print(f"  Rotation: {mae[3:6].mean().item():.4f} rad ({np.rad2deg(mae[3:6].mean().item()):.2f}°)")
    print(f"  Gripper: {mae[6].item():.4f}")
    print(f"\n平均 L2 距离: {l2_mean.item():.4f}")
    print(f"\n✅ 完成！可视化保存到 {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
