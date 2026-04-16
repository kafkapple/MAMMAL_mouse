#!/usr/bin/env python3
"""
Keyframe outlier detector.

Load all keyframe OBJs (or step_2 params) from production_900_merged/, compute
per-keyframe vertex velocity and acceleration against neighboring keyframes,
and flag keyframes whose motion-to-neighbors is anomalous.

Rationale: observed mesh pops (frame 9970 etc, accel_z=1935) are mid-interpolation
between keyframes, NOT at slerp-flagged intervals. This suggests one of the
bounding keyframes itself is a bad fit. Direct keyframe comparison isolates
which keyframes to re-fit.

Signals per keyframe k (stride=20):
  v_forward  = ||V_k - V_{k-1}|| mean  (motion from prev keyframe)
  v_backward = ||V_{k+1} - V_k|| mean  (motion to next keyframe)
  delta_v    = |v_forward - v_backward|  (asymmetry = "V-shape" signature of bad KF)

A bad KF typically has high delta_v (jumps in/out).

Also: robust-z of delta_v against local neighborhood.

Usage:
    python scripts/keyframe_outlier_detect.py \
        --obj-dir results/fitting/production_900_merged/obj/ \
        --output results/reports/keyframe_outliers.csv
"""

import argparse
import csv
import glob
import os
import re
import sys
from typing import List, Tuple

import numpy as np


def _load_obj_vertices(path: str) -> np.ndarray:
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(verts, dtype=np.float32)


def _frame_id(path: str) -> int:
    m = re.search(r"frame_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _load_sequence(obj_dir: str, pattern: str) -> Tuple[np.ndarray, List[int]]:
    files = sorted(glob.glob(os.path.join(obj_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No {pattern} in {obj_dir}")
    fids = [_frame_id(f) for f in files]
    first = _load_obj_vertices(files[0])
    N, V = len(files), first.shape[0]
    all_v = np.empty((N, V, 3), dtype=np.float32)
    all_v[0] = first
    for i, f in enumerate(files[1:], 1):
        v = _load_obj_vertices(f)
        if v.shape[0] != V:
            raise ValueError(f"vertex count mismatch {f}")
        all_v[i] = v
        if (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{N}")
    return all_v, fids


def _robust_z(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    half = window // 2
    z = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = x[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) + 1e-8
        z[i] = (x[i] - med) / (1.4826 * mad)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pattern", default="step_2_frame_*.obj")
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=5.0)
    args = ap.parse_args()

    print(f"Loading keyframe OBJs from {args.obj_dir}")
    all_v, fids = _load_sequence(args.obj_dir, args.pattern)
    N = all_v.shape[0]
    print(f"  {N} keyframes, {all_v.shape[1]} vertices each, frame range [{fids[0]},{fids[-1]}]")

    # Per-keyframe forward velocity = mean vertex motion to next keyframe
    # v_fwd[k] = ||V[k+1] - V[k]||.mean() for k in 0..N-2
    v_fwd = np.linalg.norm(all_v[1:] - all_v[:-1], axis=2).mean(axis=1)  # (N-1,)

    # Delta-v at each keyframe k (for k in 1..N-2):
    #   delta_v[k] = |v_fwd[k-1] - v_fwd[k]|   (motion asymmetry around keyframe)
    delta_v = np.abs(v_fwd[1:] - v_fwd[:-1])  # (N-2,)

    # Acceleration proxy: ||V[k+1] - 2*V[k] + V[k-1]||.mean()
    accel = np.linalg.norm(all_v[2:] - 2 * all_v[1:-1] + all_v[:-2], axis=2).mean(axis=1)  # (N-2,)

    # Robust z-scores
    delta_z = _robust_z(delta_v, args.window)
    accel_z = _robust_z(accel, args.window)

    rows = []
    for i, (dv, acc) in enumerate(zip(delta_v, accel)):
        fid = fids[i + 1]  # centered keyframe
        rows.append({
            "keyframe": fid,
            "v_fwd_prev": float(v_fwd[i]),
            "v_fwd_next": float(v_fwd[i + 1]),
            "delta_v": float(dv),
            "delta_v_z": float(delta_z[i]),
            "accel": float(acc),
            "accel_z": float(accel_z[i]),
            "flag": "BAD_KF" if accel_z[i] > args.threshold else "",
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")

    # Top-N outliers
    top = sorted(rows, key=lambda r: r["accel_z"], reverse=True)[: args.top_n]
    print(f"\nTop {args.top_n} keyframe outliers (ranked by accel_z):")
    print(f"  {'keyframe':>8s}  {'accel':>8s}  {'accel_z':>8s}  {'delta_v':>8s}  {'dv_z':>7s}  {'flag':>6s}")
    print("-" * 70)
    for r in top:
        print(f"  {r['keyframe']:>8d}  {r['accel']:>8.3f}  {r['accel_z']:>8.2f}  "
              f"{r['delta_v']:>8.4f}  {r['delta_v_z']:>7.2f}  {r['flag']:>6s}")

    # Summary
    bad = [r for r in rows if r["flag"] == "BAD_KF"]
    print(f"\nTotal BAD_KF (accel_z > {args.threshold}): {len(bad)} / {len(rows)} ({100*len(bad)/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
