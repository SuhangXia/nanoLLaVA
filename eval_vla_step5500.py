#!/usr/bin/env python3
"""
VLA Robosuite 评估：离屏渲染，Lift+Panda，将帧保存为 vla_eval_step5500.mp4。

- 必须在 import 任何 OpenGL/mujoco/robosuite 之前设置 MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl。
- 模型：Bunny-Qwen BF16 + action_head_step5500.bin；动作 denorm 后按 0.05 缩放 xyz，clip 到 [-1,1] 传入 OSC_POSE。
- Phase 1 为 image-only，指令 "Lift the red cube" 未参与训练，此处仅作占位；如需语言条件可后续扩展。
"""

# 1) 必须在 import robosuite/mujoco/OpenGL 之前设置。EGL 失败时用 EVAL_VLA_USE_OSMESA=1 或 --osmesa 切到 osmesa。
#    --realtime：有显示屏时用 EGL/glfw 实时窗口，强制不用 osmesa，避免 osmesa 首次创建卡顿。
import os
import sys
# 避免 numba 在 site-packages 只读或容器环境下 cache 报错
os.environ.setdefault("NUMBA_DISABLE_JIT_CACHE", "1")

# --realtime 最先解析，以便后续强制不用 osmesa、选用 EGL/glfw
_realtime = "--realtime" in sys.argv
if _realtime:
    sys.argv = [a for a in sys.argv if a != "--realtime"]

_use_osmesa = os.environ.get("EVAL_VLA_USE_OSMESA", "").strip().lower() in ("1", "true", "yes")
if "--osmesa" in sys.argv:
    _use_osmesa = True
    sys.argv = [a for a in sys.argv if a != "--osmesa"]
if _realtime:
    _use_osmesa = False  # 实时模式强制不用 osmesa，用 EGL 或 glfw

if _use_osmesa:
    os.environ["MUJOCO_GL"] = "osmesa"
    if "PYOPENGL_PLATFORM" in os.environ:
        del os.environ["PYOPENGL_PLATFORM"]
elif _realtime and os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "glfw"  # 有屏时用 glfw 出实时窗口
    if "PYOPENGL_PLATFORM" in os.environ:
        del os.environ["PYOPENGL_PLATFORM"]
else:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import pickle
import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

# PyTorch / transformers 兼容（旧版 PyTorch 的 register_pytree_node）
if not hasattr(torch.utils._pytree, "register_pytree_node") and hasattr(torch.utils._pytree, "_register_pytree_node"):
    def _compat_rpn(ty, fn, uf, *a, serialized_type_name=None, **k):
        return torch.utils._pytree._register_pytree_node(ty, fn, uf)
    torch.utils._pytree.register_pytree_node = _compat_rpn

import transformers
from bunny.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from bunny.model import BunnyQwenForCausalLM

# robosuite 与 imageio 在 main() 内按需 import，便于 --help 在无 robosuite 环境下也能执行


# --- 默认路径 ---
DEFAULT_VLA_CKPT = "./outputs/vla_phase1_bf16"
DEFAULT_ACTION_HEAD = "action_head_step5500.bin"
DEFAULT_MODEL = "BAAI/Bunny-v1_0-2B-zh"
DEFAULT_VISION = "siglip-so400m-patch14-384"


def _get_img_from_obs(obs, keys=("agentview_image", "frontview_image")):
    for k in keys:
        if k in obs and hasattr(obs[k], "shape") and len(obs[k].shape) >= 3:
            return np.asarray(obs[k])
    for k, v in obs.items():
        if "image" in k.lower() and hasattr(v, "shape") and len(v.shape) >= 3:
            return np.asarray(v)
    raise KeyError(f"No image in obs. Keys: {list(obs.keys())}")


def load_model_and_stats(vla_ckpt: str, action_head_file: str, model_name: str, vision_tower: str):
    """BF16 加载 Bunny-Qwen，加载 action_head 的 state_dict 与 action_mean_std.pkl。"""
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "low_cpu_mem_usage": True}
    model = BunnyQwenForCausalLM.from_pretrained(model_name, **kwargs)

    model.get_model().initialize_vision_modules(SimpleNamespace(
        vision_tower=vision_tower, pretrain_mm_mlp_adapter=None, mm_projector_type="mlp2x_gelu",
    ))
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    vt.to(device=torch.device("cuda"), dtype=torch.bfloat16)

    ah_path = os.path.join(vla_ckpt, action_head_file)
    if not os.path.isfile(ah_path):
        ah_alt = os.path.join(vla_ckpt, "action_head.bin")
        if os.path.isfile(ah_alt):
            ah_path = ah_alt
        else:
            raise FileNotFoundError(f"Not found: {ah_path} or {ah_alt}")
    model.action_head.load_state_dict(torch.load(ah_path, map_location="cpu"))
    model.eval()

    pkl_path = os.path.join(vla_ckpt, "action_mean_std.pkl")
    with open(pkl_path, "rb") as f:
        stats = pickle.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return model, mean, std


def preprocess_image(arr: np.ndarray, processor, size=(384, 384)) -> torch.Tensor:
    """ (H,W,3) [0,1] -> resize 384x384 -> PIL -> processor -> pixel_values """
    if arr.max() <= 1.0:
        arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB").resize(size, Image.BILINEAR)
    return processor(im, return_tensors="pt")["pixel_values"]


def predict(model, pixel_values, num_image_tokens, tokenizer, device):
    B = pixel_values.shape[0]
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    input_ids = torch.tensor([[IMAGE_TOKEN_INDEX, eos]] * B, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    labels = torch.full_like(input_ids, IGNORE_INDEX, device=device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=pixel_values.to(device),
            action_labels=None,
            num_image_tokens=num_image_tokens,
        )
    return out.action_prediction.cpu().float().numpy()


def denormalize(act_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return act_norm * std + mean


def model_action_to_env_action(act_raw: np.ndarray, xyz_scale: float = 0.05) -> np.ndarray:
    """
    模型 denorm 后的 7D -> env.step 的 7D（OSC_POSE 期望 [-1,1]）。xyz clip 后 Panda 乘 0.05→5cm；xyz_scale 可微调。
    """
    act = np.asarray(act_raw, dtype=np.float32).reshape(-1)[:7]
    xyz = np.clip(act[0:3], -1.0, 1.0) * (xyz_scale / 0.05)
    ori = np.clip(act[3:6], -1.0, 1.0)
    gripper = np.clip(act[6:7], -1.0, 1.0)
    return np.concatenate([xyz, ori, gripper], axis=0).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description="VLA Robosuite 评估：Lift + Panda，离屏渲染，保存 mp4。",
        epilog="若 EGL 不可用，请设置 EVAL_VLA_USE_OSMESA=1 或加 --osmesa 改用 osmesa。"
        " 有显示屏时可在其它参数前加 --realtime：强制用 EGL/glfw 实时窗口，避免 osmesa 首次创建卡顿；需 DISPLAY。",
    )
    p.add_argument("--vla_ckpt", type=str, default=DEFAULT_VLA_CKPT, help="含 action_head_step5500.bin 与 action_mean_std.pkl 的目录")
    p.add_argument("--action_head", type=str, default=DEFAULT_ACTION_HEAD, help="action_head 文件名，如 action_head_step5500.bin")
    p.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--vision_tower", type=str, default=DEFAULT_VISION)
    p.add_argument("--steps", type=int, default=300, help="仿真步数")
    p.add_argument("--output_mp4", type=str, default="vla_eval_step5500.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--realtime", action="store_true", help="(由脚本最前部解析) 有屏时 EGL/glfw 实时窗口，避免 osmesa 卡顿；需 DISPLAY")
    return p.parse_args()


def main():
    args = parse_args()
    import robosuite as suite
    import imageio

    # 模型与统计量
    print("[eval] Loading model (BF16) and action_head ...")
    model, mean, std = load_model_and_stats(
        args.vla_ckpt, args.action_head, args.model_name_or_path, args.vision_tower,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    processor = model.get_vision_tower().image_processor
    num_image_tokens = model.get_vision_tower().num_patches
    device = next(model.parameters()).device

    # osmesa 时：降低离屏与相机分辨率以加速首次 GL 创建，并打补丁
    if _use_osmesa:
        import robosuite.utils.binding_utils as _bu
        import robosuite.environments.base as _base

        _OrigOff = _bu.MjRenderContextOffscreen

        class _SmallOffscreen(_OrigOff):
            def __init__(self, sim, device_id=-1, max_width=256, max_height=256):
                _OrigOff.__init__(self, sim, device_id, 256, 256)

        _base.MjRenderContextOffscreen = _SmallOffscreen
        print("[eval] Creating Lift (Panda), osmesa+256x256 离屏. 首次创建约 2–8 分钟，请勿中断…")
    elif _realtime:
        print("[eval] Creating Lift (Panda), realtime (glfw/EGL) has_renderer=True ...")
    else:
        print("[eval] Creating Lift (Panda) with has_offscreen_renderer=True ...")

    mk = dict(
        robots="Panda",
        has_renderer=_realtime,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
    )
    if _realtime:
        mk["render_camera"] = "agentview"
    try:
        env = suite.make("Lift", **mk)
    except Exception as e:
        if "EGL" in str(e) or "PLATFORM_DEVICE" in str(e) or "Cannot initialize" in str(e) or "glfw" in str(e).lower():
            print("[eval] EGL/glfw 不可用。若 --realtime，请确保 DISPLAY 已设置，例: export DISPLAY=:0")
            print("  否则请用 osmesa: EVAL_VLA_USE_OSMESA=1 python eval_vla_step5500.py ...")
        raise

    obs = env.reset()
    if _realtime:
        env.render()
    frames = []
    # 用于录制的图像：优先 frontview（若存在），否则 agentview
    cam_key = "frontview_image" if "frontview_image" in obs else "agentview_image"

    for step in range(args.steps):
        # 1) 从 obs 取图，resize 384x384，送入模型
        img_arr = _get_img_from_obs(obs)
        pv = preprocess_image(img_arr, processor, size=(384, 384))
        act_norm = predict(model, pv, num_image_tokens, tokenizer, device)
        act_norm = act_norm[0]
        act_raw = denormalize(act_norm, mean, std)
        env_act = model_action_to_env_action(act_raw, xyz_scale=0.05)

        # 2) 调试：打印 step 与原始 7D
        print(
            f"[step={step:4d}] raw: dx={act_raw[0]:.3f} dy={act_raw[1]:.3f} dz={act_raw[2]:.3f} "
            f"dr={act_raw[3]:.3f} dp={act_raw[4]:.3f} dyaw={act_raw[5]:.3f} g={act_raw[6]:.3f}  "
            f"-> env_act: {env_act[0]:.2f} {env_act[1]:.2f} {env_act[2]:.2f} {env_act[3]:.2f} {env_act[4]:.2f} {env_act[5]:.2f} {env_act[6]:.2f}"
        )

        # 3) 执行动作，收集帧（用于视频）
        obs, reward, done, info = env.step(env_act)
        view = obs.get(cam_key, obs.get("agentview_image", _get_img_from_obs(obs)))
        if view.max() <= 1.0:
            view = (np.clip(view, 0, 1) * 255.0).astype(np.uint8)
        else:
            view = np.clip(view, 0, 255).astype(np.uint8)
        frames.append(view)
        if _realtime:
            env.render()

        if done:
            obs = env.reset()

    # 保存 mp4（优先 FFMPEG+libx264；若报错则退化为默认）
    out_path = args.output_mp4
    try:
        imageio.mimsave(out_path, frames, fps=args.fps, format="FFMPEG", codec="libx264")
    except TypeError:
        imageio.mimsave(out_path, frames, fps=args.fps)
    print(f"[eval] Saved {out_path} ({len(frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
