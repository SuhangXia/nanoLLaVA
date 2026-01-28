"""
Nano-VTLA Training Script
自动清理旧 checkpoint，只保留最新 2 个
"""

import argparse
import os
import json
import glob
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from bunny.data.vitamin_b_zarr_dataset import ViTaMInBZarrDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_llm", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--keep_checkpoints", type=int, default=2, help="Keep last N checkpoints")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def cleanup_old_checkpoints(output_dir, keep_last_n=2):
    """只保留最新 N 个 checkpoints"""
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint_step*.pt")),
        key=os.path.getmtime
    )
    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            try:
                os.remove(old_ckpt)
                size_gb = os.path.getsize(old_ckpt) / (1024**3) if os.path.exists(old_ckpt) else 0
                print(f"[Cleanup] 删除旧 checkpoint: {os.path.basename(old_ckpt)}")
            except Exception as e:
                pass

def train_step(model, batch, tokenizer, image_processor, device, bf16=False):
    """单步训练"""
    images = batch['image'].to(device)
    tactile = batch['tactile'].to(device)
    actions_gt = batch['action'].to(device)
    batch_size = images.shape[0]
    
    # Preprocess images for SigLIP
    from PIL import Image as PILImage
    pil_images = []
    for img in images:
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        pil_images.append(PILImage.fromarray(img_np))
    
    processed_images = image_processor(pil_images, return_tensors='pt')['pixel_values']
    processed_images = processed_images.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float16)
    
    # Tokenize with IMAGE_TOKEN_INDEX
    from bunny.constants import IMAGE_TOKEN_INDEX
    text = "What action should the robot take?"
    input_ids_list = []
    for _ in range(batch_size):
        ids = [IMAGE_TOKEN_INDEX] + tokenizer(text).input_ids
        input_ids_list.append(ids)
    
    max_len = max(len(ids) for ids in input_ids_list)
    padded_ids = []
    for ids in input_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
        padded_ids.append(padded)
    
    input_ids = torch.tensor(padded_ids, dtype=torch.long).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    
    # Compute num_image_tokens
    num_image_tokens = 729  # SigLIP 384x384
    
    # Forward
    with torch.cuda.amp.autocast(enabled=bf16, dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=processed_images,
            tactile_images=tactile,
            action_labels=actions_gt,
            num_image_tokens=num_image_tokens,
            use_cache=False,
            return_dict=True
        )
    
    loss = outputs.loss
    actions_pred = outputs.action_prediction if hasattr(outputs, 'action_prediction') else torch.zeros_like(actions_gt)
    
    return loss, actions_pred

def main():
    args = parse_args()
    
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'None (从头开始)'}")
    print(f"保留 checkpoints: {args.keep_checkpoints}")
    print("=" * 80)
    
    # Create dataset
    print("\n[Data] Loading dataset...")
    action_stats_path = os.path.join(args.output_dir, "action_mean_std.pkl")
    
    dataset = ViTaMInBZarrDataset(
        data_dir=args.data_dir,
        action_stats_path=action_stats_path,
        compute_action_stats=False
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Load model
    print("\n[Model] Loading...")
    from bunny.model.builder import load_pretrained_model
    
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path="BAAI/Bunny-v1_0-2B-zh",
        model_base=None,
        model_name="BAAI/Bunny-v1_0-2B-zh",
        model_type="qwen1.5-1.8b",
        device=args.device
    )
    
    if args.bf16:
        model = model.to(torch.bfloat16)
    
    # Add tactile
    model.model.config.use_tactile = True
    from bunny.model.tactile_encoder import TactileTower
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model.model.tactile_tower = TactileTower(
        pretrained=True, freeze_encoder=False, llm_hidden_size=model.config.hidden_size
    ).to(device=args.device, dtype=dtype)
    model.model.tactile_pos_embedding = nn.Parameter(
        torch.zeros(1, 1, model.config.hidden_size, device=args.device, dtype=dtype)
    )
    
    print(f"✅ Model loaded (BF16={args.bf16})")
    
    # Freeze
    if args.freeze_vision:
        for p in model.get_vision_tower().parameters():
            p.requires_grad = False
    if args.freeze_llm:
        for name, p in model.named_parameters():
            if 'tactile' not in name and 'action_head' not in name:
                p.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n[Resume] Loading {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        # Skip optimizer state (may have incompatible param groups)
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        print(f"✅ Resumed model weights from epoch {start_epoch}, step {global_step}")
        print(f"⚠️  Optimizer state reset (learning from scratch)")
    
    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    
    # Training loop
    print(f"\n[Training] Starting from epoch {start_epoch}, step {global_step}...")
    model.train()
    
    # Calculate how many batches to skip (if resuming)
    total_batches_per_epoch = len(train_loader)
    batches_to_skip = (global_step * args.gradient_accumulation_steps) % total_batches_per_epoch
    
    if batches_to_skip > 0:
        print(f"[Resume] Skipping first {batches_to_skip} batches to reach step {global_step}")
    
    for epoch in range(start_epoch, args.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (Step {global_step}+)")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Skip already trained batches when resuming
            if epoch == start_epoch and batch_idx < batches_to_skip:
                continue
            # Forward & backward
            loss, _ = train_step(model, batch, tokenizer, image_processor, args.device, args.bf16)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log
                if global_step % args.log_interval == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # Save checkpoint
                if global_step % args.save_interval == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint_step{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"\n[Checkpoint] Saved: step{global_step}.pt")
                    
                    # Auto cleanup
                    cleanup_old_checkpoints(args.output_dir, args.keep_checkpoints)
        
        print(f"\n[Epoch {epoch}] Completed")
    
    print("\n✅ Training completed!")
    writer.close()

if __name__ == "__main__":
    main()
