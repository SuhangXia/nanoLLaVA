"""
VLA inference: image -> 7D action (dx, dy, dz, dr, dp, dyaw, gripper).

Supports:
  --image /path/to.jpg     single image
  --image_dir /path        all .jpg/.png in dir
  --realsense              D435 live (needs: pip install pyrealsense2)

Without a robot: use this to view predicted actions from D435 or images.
Denormalized action is in the same space as the training data (e.g. Bridge).
"""

import argparse
import os
import pickle
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from bunny.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from bunny.model import BunnyPhiForCausalLM, BunnyQwenForCausalLM, BunnyStableLMForCausalLM
from transformers import BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser(description="VLA inference: image -> 7D action")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--model_type", type=str, required=True,
                   choices=["phi-1.5", "phi-2", "qwen1.5-0.5b", "qwen1.5-1.8b", "stablelm-2"])
    p.add_argument("--vision_tower", type=str, default=None)
    p.add_argument("--vla_ckpt", type=str, required=True,
                   help="Dir with action_head.bin and action_mean_std.pkl (e.g. ./outputs/vla_phase1)")
    p.add_argument("--image", type=str, default=None, help="Single image path")
    p.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    p.add_argument("--realsense", action="store_true", help="Live D435 (pip install pyrealsense2)")
    p.add_argument("--save_csv", type=str, default=None, help="If set, append step,dx,dy,dz,dr,dp,dyaw,gripper to CSV")
    return p.parse_args()


def load_model_and_stats(args):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    kwargs = {"quantization_config": bnb, "device_map": "auto", "low_cpu_mem_usage": True}

    if args.model_type in ("phi-1.5", "phi-2"):
        model = BunnyPhiForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    elif args.model_type in ("qwen1.5-0.5b", "qwen1.5-1.8b"):
        model = BunnyQwenForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    elif args.model_type == "stablelm-2":
        model = BunnyStableLMForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    else:
        raise ValueError(args.model_type)

    vt = args.vision_tower or getattr(model.config, "mm_vision_tower", None)
    if vt is None:
        raise ValueError("--vision_tower or model mm_vision_tower")
    model.get_model().initialize_vision_modules(SimpleNamespace(vision_tower=vt, pretrain_mm_mlp_adapter=None, mm_projector_type="mlp2x_gelu"))
    v = model.get_vision_tower()
    if not v.is_loaded:
        v.load_model()
    v.to(device=torch.device("cuda"), dtype=torch.float16)

    # Load trained ActionHead
    ah_path = os.path.join(args.vla_ckpt, "action_head.bin")
    if not os.path.isfile(ah_path):
        raise FileNotFoundError(ah_path)
    model.action_head.load_state_dict(torch.load(ah_path, map_location="cpu"))
    model.eval()

    pkl_path = os.path.join(args.vla_ckpt, "action_mean_std.pkl")
    with open(pkl_path, "rb") as f:
        stats = pickle.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    return model, mean, std


def preprocess_image(im: Image.Image, processor) -> torch.Tensor:
    return processor(im.convert("RGB"), return_tensors="pt")["pixel_values"]


def predict(model, pixel_values, num_image_tokens, tokenizer, device):
    B = pixel_values.shape[0]
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, tokenizer.eos_token_id or 0]] * B, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    labels = torch.full_like(input_ids, IGNORE_INDEX, device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=pixel_values,
            action_labels=None,
            num_image_tokens=num_image_tokens,
        )
    return out.action_prediction.cpu().float().numpy()


def denormalize(act_norm, mean, std):
    return act_norm * std + mean


def run_image(model, processor, num_image_tokens, mean, std, tokenizer, device, path: str, save_csv=None, step=0):
    im = Image.open(path).convert("RGB")
    pv = preprocess_image(im, processor).to(device)
    act_norm = predict(model, pv, num_image_tokens, tokenizer, device)
    act = denormalize(act_norm[0], mean, std)
    print(f"[{path}] normalized: {act_norm[0].tolist()}")
    print(f"         denorm:    dx={act[0]:.4f} dy={act[1]:.4f} dz={act[2]:.4f} dr={act[3]:.4f} dp={act[4]:.4f} dyaw={act[5]:.4f} gripper={act[6]:.4f}")
    if save_csv:
        with open(save_csv, "a") as f:
            f.write(f"{step},{act[0]},{act[1]},{act[2]},{act[3]},{act[4]},{act[5]},{act[6]}\n")
    return act


def run_realsense(model, processor, num_image_tokens, mean, std, tokenizer, device, save_csv=None):
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("D435 needs: pip install pyrealsense2")
        return
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    step = 0
    try:
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            arr = np.asarray(color.get_data())
            im = Image.fromarray(arr[:, :, ::-1])
            pv = preprocess_image(im, processor).to(device)
            with torch.no_grad():
                act_norm = predict(model, pv, num_image_tokens, tokenizer, device)
            act = denormalize(act_norm[0], mean, std)
            print(f"\r[realsense] dx={act[0]:.3f} dy={act[1]:.3f} dz={act[2]:.3f} dr={act[3]:.3f} dp={act[4]:.3f} dyaw={act[5]:.3f} grip={act[6]:.3f}  ", end="", flush=True)
            if save_csv:
                with open(save_csv, "a") as f:
                    f.write(f"{step},{act[0]},{act[1]},{act[2]},{act[3]},{act[4]},{act[5]},{act[6]}\n")
            step += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pipe.stop()


def main():
    args = parse_args()
    import transformers
    if args.model_type in ("phi-1.5", "phi-2", "qwen1.5-0.5b", "qwen1.5-1.8b"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    model, mean, std = load_model_and_stats(args)
    processor = model.get_vision_tower().image_processor
    num_image_tokens = model.get_vision_tower().num_patches
    device = next(model.parameters()).device

    if args.save_csv:
        with open(args.save_csv, "w") as f:
            f.write("step,dx,dy,dz,dr,dp,dyaw,gripper\n")

    if args.realsense:
        run_realsense(model, processor, num_image_tokens, mean, std, tokenizer, device, args.save_csv)
        return

    if args.image:
        run_image(model, processor, num_image_tokens, mean, std, tokenizer, device, args.image, args.save_csv, 0)
        return

    if args.image_dir:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = [os.path.join(args.image_dir, f) for f in sorted(os.listdir(args.image_dir)) if f.lower().endswith(exts)]
        for i, p in enumerate(paths):
            run_image(model, processor, num_image_tokens, mean, std, tokenizer, device, p, args.save_csv, i)
        return

    print("Use --image, --image_dir, or --realsense.")


if __name__ == "__main__":
    main()
