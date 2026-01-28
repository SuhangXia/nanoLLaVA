#!/usr/bin/env python3
"""
Plot VLA training loss from train_loss.jsonl (written by train_vla.py) or --csv.

Usage:
  python scripts/plot_vla_loss.py --output_dir ./outputs/vla_phase1
  python scripts/plot_vla_loss.py --csv steps_loss.csv --out loss.png
"""

import argparse
import json
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./outputs/vla_phase1",
                   help="Directory containing train_loss.jsonl (ignored if --csv)")
    p.add_argument("--csv", type=str, default=None,
                   help="CSV with step,loss (or header step,loss). Overrides output_dir log.")
    p.add_argument("--out", type=str, default=None,
                   help="Output image path (default: <output_dir>/train_loss.png)")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="EMA smoothing: 0=no smoothing, 0.9=heavy. Uses y = smooth*y_prev + (1-smooth)*x.")
    args = p.parse_args()

    steps, losses = [], []

    if args.csv and os.path.isfile(args.csv):
        import csv
        with open(args.csv) as f:
            r = csv.reader(f)
            row0 = next(r, None)
            if row0 and row0[0].lower() == "step":
                pass
            elif row0 and len(row0) >= 2:
                try:
                    steps.append(int(row0[0]))
                    losses.append(float(row0[1]))
                except ValueError:
                    pass
            for row in r:
                if len(row) >= 2:
                    try:
                        steps.append(int(row[0]))
                        losses.append(float(row[1]))
                    except ValueError:
                        pass
        if not steps:
            print("[plot_vla_loss] No rows in --csv.")
            return 1
    else:
        log_path = os.path.join(args.output_dir, "train_loss.jsonl")
        if not os.path.isfile(log_path):
            print(f"[plot_vla_loss] No {log_path} and no --csv. Re-run train_vla.py to get train_loss.jsonl.")
            return 1
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                steps.append(d["step"])
                losses.append(float(d["loss"]))

    if not steps:
        print("[plot_vla_loss] Empty log.")
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_vla_loss] matplotlib not installed: pip install matplotlib")
        return 1

    if args.smooth > 0:
        s = args.smooth
        y = [losses[0]]
        for i in range(1, len(losses)):
            y.append(s * y[-1] + (1 - s) * losses[i])
        losses = y

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(steps, losses, color="#2563eb", alpha=0.9, linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("VLA ActionHead Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.out:
        out = args.out
    elif args.csv:
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "train_loss.png")
    else:
        out = os.path.join(args.output_dir, "train_loss.png")
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"[plot_vla_loss] Saved {out} (steps 1–{steps[-1]}, loss {losses[0]:.4f} -> {losses[-1]:.4f})")
    return 0

if __name__ == "__main__":
    exit(main())
