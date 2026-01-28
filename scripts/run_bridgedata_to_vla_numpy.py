#!/usr/bin/env python3
"""
Wrapper: BridgeData V2 raw -> NumPy (out.npy) -> VLA-ready .npz.

1) Runs bridgedata_raw_to_numpy.py with depth=2 so that:
   - input_path/ = /datasets/scripted_raw (task folders: pnp_utensils_11-18, ...)
   - Each task's children are treated as date folders (only YYYY-MM-DD_HH-MM-SS)
   - Output: /datasets/bridge_numpy/{task}/train/out.npy and val/out.npy

2) Converts each out.npy to traj_*.npz with top-level 'image' (T,H,W,3) and
   'actions' (T,7), padding/trimming actions to 7D for vla_dataloader.

3) Runs an integrity check so /datasets/bridge_numpy works with train_vla.py.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

import numpy as np

# Bridge script and paths
# Bridge writes out.npy to BRIDGE_NPY_ROOT; we then write .npz to OUTPUT_NUMPY for train_vla
BRIDGE_SCRIPT = "/workspace/bridge_data_v2/data_processing/bridgedata_raw_to_numpy.py"
INPUT_RAW = "/datasets/scripted_raw"
BRIDGE_NPY_ROOT = "/workspace/nanoLLaVA/bridge_numpy_npy"  # intermediate (workspace writable)
OUTPUT_NUMPY = "/datasets/bridge_numpy"  # final .npz for vla_dataloader (override with --output if needed)
ACTION_DIM = 7


def run_bridge(depth: int = 2, overwrite: bool = False, num_workers: int = 8, output_path: str | None = None) -> int:
    """Run bridgedata_raw_to_numpy.py. Returns 0 on success."""
    out = output_path or BRIDGE_NPY_ROOT
    try:
        os.makedirs(out, exist_ok=True)
    except PermissionError:
        pass  # bridge’s tf.io.gfile.makedirs will create leaf dirs
    cmd = [
        sys.executable,
        BRIDGE_SCRIPT,
        f"--input_path={INPUT_RAW}",
        f"--output_path={out}",
        f"--depth={depth}",
        f"--num_workers={num_workers}",
    ]
    if overwrite:
        cmd.append("--overwrite")
    print("[run_bridgedata] Running:", " ".join(cmd))
    ret = subprocess.run(cmd)
    return ret.returncode


def _first_image_key(obs: list) -> str:
    first = obs[0]
    if "images0" in first:
        return "images0"
    for k in first:
        if "image" in k.lower() and isinstance(first.get(k), np.ndarray):
            arr = first[k]
            if arr.ndim >= 2 and (arr.shape[-1] == 3 or len(arr.shape) >= 3):
                return k
    return "images0"


def _to_7d(acts: np.ndarray) -> np.ndarray:
    acts = np.asarray(acts, dtype=np.float32)
    if acts.ndim == 1 and (acts.dtype == object or acts.size == 0):
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    if acts.ndim == 1:
        acts = acts.reshape(-1, 1)
    n = acts.shape[1]
    if n < ACTION_DIM:
        acts = np.pad(acts, ((0, 0), (0, ACTION_DIM - n)), constant_values=0)
    elif n > ACTION_DIM:
        acts = acts[:, :ACTION_DIM]
    return acts


def convert_out_npy_to_npz(npy_root: str = BRIDGE_NPY_ROOT, npz_root: str = OUTPUT_NUMPY) -> tuple[int, int]:
    """
    Find all out.npy under npy_root, split each into traj_*.npz (image, actions 7D)
    under npz_root preserving rel path (e.g. task/train, task/val).
    Returns (num_files_converted, num_trajs_written).
    """
    pattern = os.path.join(npy_root, "**", "out.npy")
    files = glob.glob(pattern, recursive=True)
    total_trajs = 0
    for path in files:
        try:
            arr = np.load(path, allow_pickle=True)
            lst = arr.item() if (hasattr(arr, "ndim") and arr.ndim == 0) else arr.tolist()
        except Exception as e:
            print(f"[convert] Skip {path}: {e}")
            continue
        if not isinstance(lst, list) or len(lst) == 0:
            continue
        # Preserve rel path: npy_root/task/train|val/out.npy -> npz_root/task/train|val/traj_*.npz
        rel = os.path.relpath(os.path.dirname(path), npy_root)
        base = os.path.join(npz_root, rel)
        os.makedirs(base, exist_ok=True)
        for i, traj in enumerate(lst):
            try:
                obs = traj.get("observations", [])
                acts = traj.get("actions", [])
                if obs is None or acts is None or len(obs) == 0:
                    continue
                if isinstance(acts, np.ndarray) and acts.size == 0:
                    continue
                if isinstance(acts, (list, tuple)) and len(acts) == 0:
                    continue
                img_key = _first_image_key(obs)
                images = np.stack([o[img_key] for o in obs if img_key in o])
                if len(images) == 0:
                    continue
                acts = _to_7d(acts)
                T = min(len(images), len(acts))
                images = images[:T]
                acts = acts[:T]
                if T == 0:
                    continue
                out_path = os.path.join(base, f"traj_{i:06d}.npz")
                np.savez(out_path, image=images, actions=acts)
                total_trajs += 1
            except Exception as e:
                print(f"[convert] traj {i} in {path}: {e}")
                continue
    return len(files), total_trajs


def _find_npz(root: str) -> list:
    out = []
    for d, _, fils in os.walk(root):
        for f in fils:
            if f.endswith(".npz"):
                out.append(os.path.join(d, f))
    return sorted(out)


def integrity_check(root: str = OUTPUT_NUMPY) -> dict:
    """
    Check .npz under root: count, load one sample, check image/actions shapes.
    Does not import torch/vla_dataloader to avoid NumPy 2 / torch conflicts.
    """
    files = _find_npz(root)
    report = {"npz_count": len(files), "sample_loaded": False, "image_shape": None, "action_shape": None, "error": None}
    if len(files) == 0:
        report["error"] = "No .npz found"
        return report
    try:
        with np.load(files[0], allow_pickle=True) as d:
            im = d.get("image", d.get("images", None))
            if im is None and "observations" in d.files:
                obs = d["observations"]
                if hasattr(obs, "item"):
                    obs = obs.item()
                im = obs.get("image", obs.get("images", None)) if isinstance(obs, dict) else None
            ac = d.get("actions", d.get("action", None))
        if im is not None and ac is not None:
            report["sample_loaded"] = True
            report["image_shape"] = np.asarray(im).shape
            report["action_shape"] = np.asarray(ac).shape
        else:
            report["error"] = "Missing 'image' or 'actions' in " + files[0]
    except Exception as e:
        report["error"] = str(e)
    return report


def main():
    import argparse

    p = argparse.ArgumentParser(description="BridgeData raw -> bridge_numpy (VLA .npz)")
    p.add_argument("--skip-bridge", action="store_true", help="Only run npy->npz + integrity check")
    p.add_argument("--skip-convert", action="store_true", help="Only run bridge + integrity (no npy->npz)")
    p.add_argument("--overwrite", action="store_true", help="Pass --overwrite to the bridge script")
    p.add_argument("--depth", type=int, default=2, help="Depth for bridge (2: input_path/task/date/raw/...)")
    p.add_argument("--num-workers", type=int, default=1, help="Workers for bridge (1 to avoid Pool/SemLock)")
    p.add_argument("--bridge-output", type=str, default=BRIDGE_NPY_ROOT, help="Where bridge writes out.npy")
    p.add_argument("--output", type=str, default=OUTPUT_NUMPY, help="Where to write .npz for train_vla")
    args = p.parse_args()

    if not args.skip_bridge:
        ret = run_bridge(depth=args.depth, overwrite=args.overwrite, num_workers=args.num_workers, output_path=args.bridge_output)
        if ret != 0:
            print("[run_bridgedata] Bridge script exited with", ret)
            sys.exit(ret)

    if not args.skip_convert:
        n_files, n_trajs = convert_out_npy_to_npz(npy_root=args.bridge_output, npz_root=args.output)
        print(f"[run_bridgedata] Converted {n_files} out.npy -> {n_trajs} traj_*.npz under {args.output}")

    report = integrity_check(args.output)
    print("[run_bridgedata] Integrity:", report)
    if report.get("error"):
        sys.exit(1)
    if report.get("npz_count", 0) == 0 and not args.skip_convert:
        sys.exit(1)
    print(f"[run_bridgedata] Done. {args.output} is ready for train_vla.py (--data_root={args.output}).")


if __name__ == "__main__":
    main()
