#!/usr/bin/env python3
"""
Slerp pop diagnostic.

Iterate adjacent keyframe pairs, compute per-joint quaternion dot product,
flag intervals likely to produce visual pops under naive slerp:
  - wrong hemisphere (dot < 0)           → matrix-slerp picks long path
  - near-antipodal   (|dot| < 0.15)      → slerp numerically unstable

Report CSV with interval bounds, min/max/argmin joint dot, flag reason.
If a list of visually observed pop frames is supplied, also emit
a correlation summary (hit / miss / false-positive).

Usage:
    CUDA_VISIBLE_DEVICES= python scripts/diagnostic_slerp.py \
        --params-dir results/fitting/production_900_merged/ \
        --output results/reports/slerp_diagnostic.csv

    # Optional: cross-reference visual pops
    python scripts/diagnostic_slerp.py \
        --params-dir results/fitting/production_900_merged/ \
        --output results/reports/slerp_diagnostic.csv \
        --visual-pops 1320,5520,9480
"""

import argparse
import csv
import glob
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np

# Reuse canonical quaternion conversion from interpolation module so that
# diagnostic dot-products match the interpolator's internal math exactly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mammal_ext.fitting.interpolation import _axis_angle_to_quat  # noqa: E402


def _load_keyframe_thetas(params_dir: str) -> Dict[int, np.ndarray]:
    files = sorted(glob.glob(os.path.join(params_dir, "step_2_frame_*.pkl")))
    if not files:
        files = sorted(glob.glob(os.path.join(params_dir, "step_1_frame_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No param files in {params_dir}")

    out = {}
    for f in files:
        fid = int(f.split("frame_")[1].split(".")[0])
        p = pickle.load(open(f, "rb"))
        thetas = p["thetas"]
        if hasattr(thetas, "detach"):
            thetas = thetas.detach().cpu().numpy()
        elif hasattr(thetas, "numpy"):
            thetas = thetas.numpy()
        thetas = np.asarray(thetas)
        # Match interpolation.py convention: thetas shape (1, N_joints, 3).
        # Squeeze batch dim before reshape so joint axis is preserved.
        if thetas.ndim == 3:
            thetas = thetas[0]
        out[fid] = thetas.reshape(-1, 3)
    return out


def _pair_flags(theta_a: np.ndarray, theta_b: np.ndarray) -> Tuple[float, float, int, List[int], List[int]]:
    """Return (min_dot, max_dot, argmin_joint, wrong_hemisphere_joints, near_antipodal_joints)."""
    n_joints = theta_a.shape[0]
    dots = np.empty(n_joints, dtype=np.float64)
    for j in range(n_joints):
        q1 = _axis_angle_to_quat(theta_a[j])
        q2 = _axis_angle_to_quat(theta_b[j])
        dots[j] = float(np.dot(q1, q2))
    wrong = [int(j) for j in range(n_joints) if dots[j] < 0.0]
    near_anti = [int(j) for j in range(n_joints) if abs(dots[j]) < 0.15]
    return float(dots.min()), float(dots.max()), int(np.argmin(dots)), wrong, near_anti


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-dir", required=True)
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--visual-pops", default="",
                        help="Comma-separated video frame IDs observed as pops")
    args = parser.parse_args()

    thetas = _load_keyframe_thetas(args.params_dir)
    sorted_ids = sorted(thetas.keys())
    if len(sorted_ids) < 2:
        print("ERROR: need ≥2 keyframes", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    flagged = []
    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kf_prev", "kf_next", "min_dot", "max_dot", "argmin_joint",
                    "n_wrong_hemisphere", "n_near_antipodal", "flag"])
        for i in range(len(sorted_ids) - 1):
            a, b = sorted_ids[i], sorted_ids[i + 1]
            min_d, max_d, argmin, wrong, near_anti = _pair_flags(thetas[a], thetas[b])
            flag = ""
            if wrong:
                flag = "WRONG_HEMISPHERE"
            elif near_anti:
                flag = "NEAR_ANTIPODAL"
            w.writerow([a, b, f"{min_d:.6f}", f"{max_d:.6f}", argmin,
                        len(wrong), len(near_anti), flag])
            if flag:
                flagged.append((a, b, min_d, flag, wrong, near_anti))

    total = len(sorted_ids) - 1
    print(f"Keyframes: {len(sorted_ids)} ({sorted_ids[0]}..{sorted_ids[-1]})")
    print(f"Intervals: {total}")
    print(f"Flagged:   {len(flagged)} ({100.0 * len(flagged) / max(total, 1):.1f}%)")
    for a, b, min_d, flag, wrong, near_anti in flagged[:20]:
        print(f"  [{a:>6} → {b:>6}]  min_dot={min_d:+.4f}  {flag}  wrong={wrong}  near_anti={near_anti}")
    if len(flagged) > 20:
        print(f"  ... +{len(flagged) - 20} more")

    if args.visual_pops:
        visual = sorted({int(x) for x in args.visual_pops.split(",") if x.strip()})
        hit, miss, false_pos = 0, 0, 0
        flagged_intervals = [(a, b) for a, b, *_ in flagged]
        flagged_frames = set()
        for a, b in flagged_intervals:
            flagged_frames.update(range(a, b + 1))
        for v in visual:
            if v in flagged_frames:
                hit += 1
            else:
                miss += 1
        # False positives require knowing "clean" intervals — only upper bound here
        print(f"\nVisual pop correlation:")
        print(f"  HITS (visual frame inside flagged interval):   {hit}/{len(visual)}")
        print(f"  MISSES (visual frame outside flagged):          {miss}/{len(visual)}")
        if hit / max(len(visual), 1) >= 0.9:
            print("  → ≥90% correlation. Slerp hemisphere fix is indicated.")
        else:
            print("  → <90% correlation. Investigate keyframe fit failures as well.")


if __name__ == "__main__":
    main()
