"""
Phase 1: VLA action baseline training.

- 4-bit (default): bitsandbytes for ~12GB VRAM. If "weight is not an nn.Module" on PyTorch 2.11+ / sm_120, use --no_4bit.
- BF16 (--no_4bit / --bf16): full precision + --gradient_checkpointing, no bitsandbytes; for RTX 5070 Ti (sm_120) etc.
- Freezes Vision Tower and LLM; trains only ActionHead with MSE loss.
- Uses /datasets/bridge_numpy and vla_dataloader for (image, 7D action) pairs.
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
# 兼容：transformers 4.49+ 调用 register_pytree_node(..., serialized_type_name=)；
# 旧版 PyTorch 仅有 _register_pytree_node(type, flatten, unflatten)，无 serialized_type_name。
if not hasattr(torch.utils._pytree, "register_pytree_node") and hasattr(torch.utils._pytree, "_register_pytree_node"):
    def _compat_register_pytree_node(ty, flatten, unflatten, *args, serialized_type_name=None, **kwargs):
        return torch.utils._pytree._register_pytree_node(ty, flatten, unflatten)
    torch.utils._pytree.register_pytree_node = _compat_register_pytree_node
import transformers
from torch.utils.data import Dataset
from transformers import BitsAndBytesConfig, get_scheduler

from bunny.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from bunny.model import BunnyPhiForCausalLM, BunnyQwenForCausalLM, BunnyStableLMForCausalLM
from vla_dataloader import BridgeNumpyVLA
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Train VLA ActionHead on bridge_numpy.")
    p.add_argument("--model_name_or_path", type=str, required=True, help="Path to Bunny/NanoLLaVA model.")
    p.add_argument("--model_type", type=str, required=True,
                   choices=["phi-1.5", "phi-2", "qwen1.5-0.5b", "qwen1.5-1.8b", "stablelm-2"],
                   help="Backbone type.")
    p.add_argument("--vision_tower", type=str, default=None,
                   help="Vision tower name/path. If None, read from model config.")
    p.add_argument("--data_root", type=str, default="/datasets/bridge_numpy", help="Path to bridge_numpy data.")
    p.add_argument("--output_dir", type=str, default="./outputs/vla_phase1", help="Checkpoints and action_head.bin.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--learning_rate", type=float, default=None, help="Same as --lr (HuggingFace style).")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--num_train_epochs", type=int, default=None, help="Same as --num_epochs (HuggingFace style).")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)
    # 12GB / sm_120: 若 4-bit 在 PyTorch 2.11+bitsandbytes 下报 "weight is not an nn.Module"，改用 BF16
    p.add_argument("--no_4bit", action="store_true",
                   help="Load in BF16 without 4-bit (avoids bitsandbytes; for PyTorch 2.11+ / sm_120 when 4-bit fails).")
    p.add_argument("--bf16", action="store_true",
                   help="Same as --no_4bit: BF16 full precision, no quantization.")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM (use with --no_4bit for 12GB).")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine",
                   choices=["cosine", "linear", "none"],
                   help="LR scheduler: cosine (default), linear, or none.")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear warmup steps (for cosine/linear). Default 500.")
    return p.parse_args()


def get_model_and_tokenizer(args):
    """Load model and tokenizer. Use BF16 (no 4-bit) when --no_4bit/--bf16 to avoid bitsandbytes+PyTorch2.11/sm_120 issues."""
    use_bf16 = getattr(args, "no_4bit", False) or getattr(args, "bf16", False)

    if use_bf16:
        # BF16 全量：避开 bitsandbytes，避免 "weight is not an nn.Module"（transformers 量化与 PyTorch 2.11 / sm_120 不兼容）
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

    # Resolve vision tower from config if not given
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

    # Load vision and move to device/dtype
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt_dtype = torch.bfloat16 if use_bf16 else torch.float16
    vt.to(device=torch.device("cuda"), dtype=vt_dtype)

    model.config.use_cache = False
    return model, tokenizer, model_args


class VLADataset(Dataset):
    """Wraps BridgeNumpyVLA and adds input_ids/labels. Uses [IMAGE_TOKEN, EOS] to avoid empty segments in prepare_inputs_labels."""

    def __init__(self, bridge_ds: BridgeNumpyVLA, eos_token_id: int = 0):
        self.bridge = bridge_ds
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.bridge)

    def __getitem__(self, i):
        item = self.bridge[i]
        # [IMAGE_TOKEN, EOS] so prepare_inputs_labels has non-empty text segment; labels ignored (action loss only).
        input_ids = torch.tensor([IMAGE_TOKEN_INDEX, self.eos_token_id], dtype=torch.long)
        labels = torch.tensor([IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long)
        return {
            "image": item["image"],
            "action": item["action"],
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_vla(instances, image_processor, num_image_tokens, pad_id=0):
    """Collate batch: stack tensors, run image_processor, add num_image_tokens and action_labels."""
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
    if args.learning_rate is not None:
        args.lr = args.learning_rate
    if args.num_train_epochs is not None:
        args.num_epochs = args.num_train_epochs

    # Check local model path exists (HuggingFace IDs like BAAI/Bunny-2-8B are not paths)
    if (args.model_name_or_path.startswith("/") or args.model_name_or_path.startswith("./")) and not os.path.exists(args.model_name_or_path):
        print(f"Error: model_name_or_path not found: {args.model_name_or_path}")
        print("  Put the Bunny/NanoLLaVA model there, or use a HuggingFace ID, e.g.:")
        print("  --model_name_or_path BAAI/Bunny-v1_0-2B-zh  (qwen1.5-1.8b + siglip)")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    use_bf16 = getattr(args, "no_4bit", False) or getattr(args, "bf16", False)
    if use_bf16:
        print("[train_vla] Using BF16 (no 4-bit) to avoid bitsandbytes/quantization issues on PyTorch 2.11+ / sm_120.")

    # Model and tokenizer
    model, tokenizer, _ = get_model_and_tokenizer(args)
    image_processor = model.get_vision_tower().image_processor
    num_image_tokens = model.get_vision_tower().num_patches

    # Freeze Vision Tower and LLM; train only ActionHead
    model.requires_grad_(False)
    model.action_head.requires_grad_(True)

    # Gradient checkpointing & 4-bit 准备
    if use_bf16 and getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    elif not use_bf16:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=getattr(args, "gradient_checkpointing", False))
        except Exception:
            pass

    # Data: BridgeNumpyVLA for (image, action) + VLADataset adds input_ids/labels
    bridge_ds = BridgeNumpyVLA(data_root=args.data_root)
    action_mean, action_std = bridge_ds.get_action_mean_std()
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    vla_ds = VLADataset(bridge_ds, eos_token_id=eos_id)
    if len(vla_ds) == 0:
        raise RuntimeError(f"No (image, action) pairs found in data_root={args.data_root}. "
                           "Ensure .npz files with 'observations'/'image' and 'actions' (T, 7) exist.")

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

    # Save action stats for inference/denormalization
    import pickle
    with open(os.path.join(args.output_dir, "action_mean_std.pkl"), "wb") as f:
        pickle.dump({"mean": action_mean, "std": action_std}, f)

    # Training loss log (for plotting): one JSON per line
    log_path = os.path.join(args.output_dir, "train_loss.jsonl")

    # Optimizer
    opt = torch.optim.AdamW(model.action_head.parameters(), lr=args.lr)

    num_batches = len(loader)
    n_acc = args.gradient_accumulation_steps
    steps_per_epoch = (num_batches + n_acc - 1) // n_acc
    total_steps = steps_per_epoch * args.num_epochs

    # LR Scheduler: cosine/linear with warmup (none = 固定 lr)
    if getattr(args, "lr_scheduler_type", "cosine") == "none":
        scheduler = None
    else:
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            opt,
            num_warmup_steps=getattr(args, "warmup_steps", 500),
            num_training_steps=total_steps,
        )

    # Training loop with gradient accumulation and progress logging
    model.train()
    device = next(model.parameters()).device
    global_step = 0
    t_start = time.perf_counter()

    for epoch in range(args.num_epochs):
        opt.zero_grad()
        for step, batch in enumerate(loader):
            # Move tensors to device; leave num_image_tokens (int) as is
            for k in list(batch.keys()):
                v = batch[k]
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            # Forward: model expects images, action_labels, num_image_tokens (int)
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

                # Append to loss log (for scripts/plot_vla_loss.py)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": global_step, "epoch": epoch + 1, "loss": loss_val}) + "\n")

                # Progress logging (含 ETA)
                if global_step % args.logging_steps == 0:
                    elapsed = time.perf_counter() - t_start
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
                    eta_sec = (total_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0.0
                    eta_str = str(timedelta(seconds=int(eta_sec)))
                    print(
                        f"[Epoch {epoch+1}/{args.num_epochs}] step={global_step}/{total_steps} "
                        f"loss={loss_val:.4f} | ETA {eta_str}"
                    )

                # Save ActionHead periodically
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f"action_head_step{global_step}.bin")
                    torch.save(model.action_head.state_dict(), ckpt)
                    print(f"Saved {ckpt}")

        # End-of-epoch message
        print(f"Epoch {epoch+1} finished.")

    # Final save
    torch.save(model.action_head.state_dict(), os.path.join(args.output_dir, "action_head.bin"))
    print(f"Training done. action_head.bin and action_mean_std.pkl saved under {args.output_dir}")


if __name__ == "__main__":
    main()
