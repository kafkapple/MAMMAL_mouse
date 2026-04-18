#!/usr/bin/env python3
"""G4 Savitzky-Golay vertex smoothing baseline (pop null hypothesis).

Applies SG filter (window=5, poly=3) along time axis to production_3600_slerp
vertex sequence. Measures max per-frame vertex acceleration reduction.

Purpose: establish whether temporal smoothing alone can remove pop, as
a baseline to compare against T2 canon re-interp (upstream fix).

Usage:
    python scripts/g4_sg_baseline.py \
        --obj-dir results/fitting/production_3600_slerp/obj/ \
        --output results/reports/g4_sg_baseline.csv
"""
import argparse
import csv
import glob
import os
import numpy as np
from scipy.signal import savgol_filter


def load_obj_verts(path):
    verts = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--polyorder", type=int, default=3)
    args = ap.parse_args()

    obj_files = sorted(glob.glob(os.path.join(args.obj_dir, "step_2_frame_*.obj")))
    if len(obj_files) < args.window:
        raise RuntimeError(f"Only {len(obj_files)} obj files found, need >= {args.window}")

    print(f"Loading {len(obj_files)} OBJ files...", flush=True)
    verts_list = []
    for i, f in enumerate(obj_files):
        verts_list.append(load_obj_verts(f))
        if (i + 1) % 500 == 0:
            print(f"  loaded {i+1}/{len(obj_files)}", flush=True)
    V = np.stack(verts_list)
    print(f"V shape: {V.shape}  (T frames, N verts, 3 dims)", flush=True)

    print(f"Applying Savitzky-Golay (window={args.window}, polyorder={args.polyorder})...", flush=True)
    V_smooth = savgol_filter(V, window_length=args.window, polyorder=args.polyorder, axis=0)

    def per_frame_max_accel(arr):
        a = arr[:-2] - 2 * arr[1:-1] + arr[2:]
        return np.linalg.norm(a, axis=-1).max(axis=-1)

    acc_orig = per_frame_max_accel(V)
    acc_smooth = per_frame_max_accel(V_smooth)

    print(f"\nOriginal peak accel:  mean={acc_orig.mean():.4f}  median={np.median(acc_orig):.4f}  max={acc_orig.max():.4f}")
    print(f"Smoothed peak accel:  mean={acc_smooth.mean():.4f}  median={np.median(acc_smooth):.4f}  max={acc_smooth.max():.4f}")
    reduction_mean = 1.0 - acc_smooth.mean() / max(acc_orig.mean(), 1e-9)
    reduction_max = 1.0 - acc_smooth.max() / max(acc_orig.max(), 1e-9)
    print(f"Reduction:            mean={reduction_mean:.1%}  max={reduction_max:.1%}")

    top_n = 10
    top_idx_orig = np.argsort(-acc_orig)[:top_n]
    print(f"\nTop {top_n} peak-accel frames (original):")
    for idx in top_idx_orig:
        frame_id = int(os.path.basename(obj_files[idx + 1]).split("_")[-1].replace(".obj", ""))
        print(f"  frame {frame_id:>5}: orig={acc_orig[idx]:.3f}  smooth={acc_smooth[idx]:.3f}  red={1-acc_smooth[idx]/max(acc_orig[idx],1e-9):.1%}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_idx", "frame_id", "accel_orig", "accel_smooth", "reduction_pct"])
        for i, (o, s) in enumerate(zip(acc_orig, acc_smooth)):
            fid = int(os.path.basename(obj_files[i + 1]).split("_")[-1].replace(".obj", ""))
            w.writerow([i + 1, fid, f"{o:.4f}", f"{s:.4f}", f"{100*(1-s/max(o,1e-9)):.2f}"])
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
