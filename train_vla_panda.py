"""
Phase 1: VLA Training with RLBench-Panda HDF5 Data.

This script trains the VLA Action Head using:
- RLBench-Panda trajectories in HDF5 format (ViTaMIn-B standard)
- Franka Panda Eye-in-Hand camera (384x384)
- 7D actions in camera frame: [dx, dy, dz, dr, dp, dy, gripper]

Usage:
  python train_vla_panda.py \
    --model_name_or_path BAAI/Bunny-v1_0-2B-zh \
    --model_type qwen1.5-1.8b \
    --data_root ./data/rlbench_panda_vtla \
    --output_dir ./outputs/vla_panda_phase1 \
    --batch_size 4 \
    --num_epochs 3
"""

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
if not hasattr(torch.utils._pytree, "register_pytree_node") and hasattr(torch.utils._pytree, "_register_pytree_node"):
    def _compat_register_pytree_node(ty, flatten, unflatten, *args, serialized_type_name=None, **kwargs):
        return torch.utils._pytree._register_pytree_node(ty, flatten, unflatten)
    torch.utils._pytree.register_pytree_node = _compat_register_pytree_node

import transformers
from torch.utils.data import Dataset
from transformers import BitsAndBytesConfig, get_scheduler

from bunny.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from bunny.model import BunnyPhiForCausalLM, BunnyQwenForCausalLM, BunnyStableLMForCausalLM
from vtla_hdf5_dataloader import VTLAHDF5Dataset
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Train VLA on RLBench-Panda HDF5 data.")
    p.add_argument("--model_name_or_path", type=str, required=True, help="Path to NanoLLaVA model.")
    p.add_argument("--model_type", type=str, required=True,
                   choices=["phi-1.5", "phi-2", "qwen1.5-0.5b", "qwen1.5-1.8b", "stablelm-2"],
                   help="Backbone type.")
    p.add_argument("--vision_tower", type=str, default=None,
                   help="Vision tower name/path. If None, read from model config.")
    p.add_argument("--data_root", type=str, default="./data/rlbench_panda_vtla",
                   help="Path to RLBench-Panda HDF5 data.")
    p.add_argument("--output_dir", type=str, default="./outputs/vla_panda_phase1",
                   help="Output directory for checkpoints.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--no_4bit", action="store_true",
                   help="Use BF16 without 4-bit quantization (for PyTorch 2.11+/sm_120).")
    p.add_argument("--bf16", action="store_true",
                   help="Same as --no_4bit: BF16 full precision.")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM.")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine",
                   choices=["cosine", "linear", "none"],
                   help="LR scheduler type.")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear warmup steps.")
    return p.parse_args()


def get_model_and_tokenizer(args):
    """Load model and tokenizer with BF16 or 4-bit quantization."""
    use_bf16 = getattr(args, "no_4bit", False) or getattr(args, "bf16", False)

    if use_bf16:
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "low_cpu_mem_usage": True}
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs = {"quantization_config": bnb_config, "device_map": "auto", "low_cpu_mem_usage": True}

    if args.model_type in ("phi-1.5", "phi-2"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = BunnyPhiForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    elif args.model_type in ("qwen1.5-0.5b", "qwen1.5-1.8b"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = BunnyQwenForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    elif args.model_type == "stablelm-2":
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
        model = BunnyStableLMForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    vision_tower = args.vision_tower
    if vision_tower is None:
        vision_tower = getattr(model.config, "mm_vision_tower", None)
    if vision_tower is None:
        raise ValueError("Provide --vision_tower or use a model config with mm_vision_tower.")

    model_args = SimpleNamespace(
        vision_tower=vision_tower,
        pretrain_mm_mlp_adapter=getattr(args, "pretrain_mm_mlp_adapter", None),
        mm_projector_type="mlp2x_gelu",
    )
    model.get_model().initialize_vision_modules(model_args=model_args)

    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt_dtype = torch.bfloat16 if use_bf16 else torch.float16
    vt.to(device=torch.device("cuda"), dtype=vt_dtype)

    model.config.use_cache = False
    return model, tokenizer, model_args


class VLAPandaDataset(Dataset):
    """
    Wraps VTLAHDF5Dataset and adds input_ids/labels for VLA training.
    """

    def __init__(self, hdf5_dataset: VTLAHDF5Dataset, eos_token_id: int = 0):
        self.hdf5_dataset = hdf5_dataset
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.hdf5_dataset)

    def __getitem__(self, i):
        item = self.hdf5_dataset[i]
        # [IMAGE_TOKEN, EOS] for prepare_inputs_labels
        input_ids = torch.tensor([IMAGE_TOKEN_INDEX, self.eos_token_id], dtype=torch.long)
        labels = torch.tensor([IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long)
        return {
            "image": item["image"],
            "action": item["action"],
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_vla(instances, image_processor, num_image_tokens, pad_id=0):
    """Collate batch for VLA training."""
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence([x["input_ids"] for x in instances], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([x["labels"] for x in instances], batch_first=True, padding_value=IGNORE_INDEX)
    actions = torch.stack([x["action"] for x in instances], dim=0)

    # (3,H,W) float [0,1] -> PIL -> processor
    images = []
    for x in instances:
        arr = (x["image"].numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]

    attention_mask = input_ids.ne(pad_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": pixel_values,
        "action_labels": actions,
        "num_image_tokens": num_image_tokens,
    }


def main():
    args = parse_args()

    if not os.path.exists(args.data_root):
        print(f"ERROR: Data root not found: {args.data_root}")
        print("  Run vtla_data_collector.py to generate RLBench-Panda HDF5 data first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    use_bf16 = getattr(args, "no_4bit", False) or getattr(args, "bf16", False)
    if use_bf16:
        print("[train_vla_panda] Using BF16 (no 4-bit).")

    # Model and tokenizer
    model, tokenizer, _ = get_model_and_tokenizer(args)
    image_processor = model.get_vision_tower().image_processor
    num_image_tokens = model.get_vision_tower().num_patches

    # Freeze Vision Tower and LLM; train only ActionHead
    model.requires_grad_(False)
    model.action_head.requires_grad_(True)

    # Gradient checkpointing
    if use_bf16 and getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    elif not use_bf16:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=getattr(args, "gradient_checkpointing", False))
        except Exception:
            pass

    # Data: VTLAHDF5Dataset
    hdf5_ds = VTLAHDF5Dataset(data_root=args.data_root)
    action_mean, action_std = hdf5_ds.get_action_mean_std()
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    vla_ds = VLAPandaDataset(hdf5_ds, eos_token_id=eos_id)

    if len(vla_ds) == 0:
        raise RuntimeError(f"No data found in {args.data_root}. Run vtla_data_collector.py first.")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = torch.utils.data.DataLoader(
        vla_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: collate_vla(b, image_processor, num_image_tokens, pad_id=pad_id),
    )

    # Save action stats
    import pickle
    with open(os.path.join(args.output_dir, "action_mean_std.pkl"), "wb") as f:
        pickle.dump({"mean": action_mean, "std": action_std}, f)

    log_path = os.path.join(args.output_dir, "train_loss.jsonl")

    # Optimizer
    opt = torch.optim.AdamW(model.action_head.parameters(), lr=args.lr)

    num_batches = len(loader)
    n_acc = args.gradient_accumulation_steps
    steps_per_epoch = (num_batches + n_acc - 1) // n_acc
    total_steps = steps_per_epoch * args.num_epochs

    # LR Scheduler
    if getattr(args, "lr_scheduler_type", "cosine") == "none":
        scheduler = None
    else:
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            opt,
            num_warmup_steps=getattr(args, "warmup_steps", 500),
            num_training_steps=total_steps,
        )

    # Training loop
    model.train()
    device = next(model.parameters()).device
    global_step = 0
    t_start = time.perf_counter()

    for epoch in range(args.num_epochs):
        opt.zero_grad()
        for step, batch in enumerate(loader):
            for k in list(batch.keys()):
                v = batch[k]
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
            with amp_ctx:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    images=batch["images"],
                    action_labels=batch["action_labels"],
                    num_image_tokens=batch["num_image_tokens"],
                )
                loss = out.loss / n_acc
                loss.backward()

            if (step + 1) % n_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), 1.0)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                opt.zero_grad()
                global_step += 1
                loss_val = out.loss.item()

                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": global_step, "epoch": epoch + 1, "loss": loss_val}) + "\n")

                if global_step % args.logging_steps == 0:
                    elapsed = time.perf_counter() - t_start
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
                    eta_sec = (total_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0.0
                    eta_str = str(timedelta(seconds=int(eta_sec)))
                    print(
                        f"[Epoch {epoch+1}/{args.num_epochs}] step={global_step}/{total_steps} "
                        f"loss={loss_val:.4f} | ETA {eta_str}"
                    )

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f"action_head_step{global_step}.bin")
                    torch.save(model.action_head.state_dict(), ckpt)
                    print(f"Saved {ckpt}")

        print(f"Epoch {epoch+1} finished.")

    # Final save
    torch.save(model.action_head.state_dict(), os.path.join(args.output_dir, "action_head.bin"))
    print(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
