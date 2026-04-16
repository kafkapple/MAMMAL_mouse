#!/usr/bin/env python3
"""
Quantitative mesh-pop detector.

Scan a directory of per-frame OBJ meshes in temporal order, compute
frame-to-frame vertex motion, and flag frames whose motion magnitude
is anomalously large versus its local neighborhood.

Two signals:
  1. velocity_mag:     ||V_t - V_{t-1}|| mean over vertices
  2. acceleration_mag: ||V_{t+1} - 2*V_t + V_{t-1}|| mean over vertices

A "pop" is a localized acceleration spike (a single frame deviating
much more than its neighbors). We use a robust outlier score:

  score_t = (accel_t - median(accel[t-W..t+W])) / MAD(accel[t-W..t+W])

Frames with score > THRESHOLD are reported. Also cross-references
with slerp_diagnostic.csv if provided to compute precision/recall of
the algorithmic slerp flag vs. observed motion outliers.

Usage:
    python scripts/quantitative_pop_detect.py \
        --obj-dir results/fitting/production_3600_slerp/obj/ \
        --output results/reports/pop_detect.csv \
        [--slerp-csv results/reports/slerp_diagnostic.csv] \
        [--top-n 30] [--window 11] [--threshold 5.0]
"""

import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_obj_vertices(path: str) -> np.ndarray:
    """Read only 'v x y z' lines. Fast path for large OBJ dirs."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                # "v x y z [optional]"
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(verts, dtype=np.float32)


def _frame_id(path: str) -> int:
    m = re.search(r"frame_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _load_sequence(obj_dir: str) -> Tuple[np.ndarray, List[int]]:
    files = sorted(glob.glob(os.path.join(obj_dir, "step_2_frame_*.obj")))
    if not files:
        raise FileNotFoundError(f"No step_2_frame_*.obj in {obj_dir}")
    fids = [_frame_id(f) for f in files]
    # Load first to get vertex count
    first = _load_obj_vertices(files[0])
    n_verts = first.shape[0]
    all_v = np.empty((len(files), n_verts, 3), dtype=np.float32)
    all_v[0] = first
    for i, f in enumerate(files[1:], start=1):
        v = _load_obj_vertices(f)
        if v.shape[0] != n_verts:
            raise ValueError(f"Vertex count mismatch {f}: {v.shape[0]} vs {n_verts}")
        all_v[i] = v
        if (i + 1) % 500 == 0:
            print(f"  loaded {i+1}/{len(files)}")
    return all_v, fids


def _robust_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Per-sample z-score against local (median, MAD) in a sliding window."""
    n = len(x)
    half = window // 2
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window_vals = x[lo:hi]
        med = np.median(window_vals)
        mad = np.median(np.abs(window_vals - med)) + 1e-8
        scores[i] = (x[i] - med) / (1.4826 * mad)
    return scores


def _load_slerp_intervals(csv_path: str) -> List[Tuple[int, int, str]]:
    """Return list of (frame_a, frame_b, flag) from slerp_diagnostic.csv."""
    intervals = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flag = row.get("flag", "") or ""
            if flag and flag != "OK":
                fa, fb = int(row["kf_prev"]), int(row["kf_next"])
                intervals.append((fa, fb, flag))
    return intervals


def _in_any_interval(fid: int, intervals: List[Tuple[int, int, str]]) -> Optional[str]:
    for fa, fb, flag in intervals:
        if fa <= fid <= fb:
            return flag
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", required=True)
    ap.add_argument("--output", required=True, help="CSV output path")
    ap.add_argument("--slerp-csv", default=None, help="optional slerp_diagnostic.csv for correlation")
    ap.add_argument("--top-n", type=int, default=30, help="print top N spike frames")
    ap.add_argument("--window", type=int, default=11, help="window size for robust z-score")
    ap.add_argument("--threshold", type=float, default=5.0, help="z-score threshold for pop flag")
    args = ap.parse_args()

    print(f"Loading OBJ sequence from {args.obj_dir}")
    all_v, fids = _load_sequence(args.obj_dir)
    N = all_v.shape[0]
    print(f"  {N} frames, {all_v.shape[1]} vertices each")

    # Per-frame mean vertex velocity
    dv = np.linalg.norm(all_v[1:] - all_v[:-1], axis=2).mean(axis=1)  # (N-1,)
    # Per-frame mean vertex acceleration (using centered difference)
    accel = np.linalg.norm(all_v[2:] - 2 * all_v[1:-1] + all_v[:-2], axis=2).mean(axis=1)  # (N-2,)

    # Align: velocity[i] = motion from frame i to i+1 → assign to frame i+1
    #        accel[i]    = curvature at frame i+1 → assign to frame i+1 (center)
    # Robust z-scores
    vel_z = _robust_zscore(dv, args.window)
    acc_z = _robust_zscore(accel, args.window)

    # Load slerp intervals if provided
    intervals = _load_slerp_intervals(args.slerp_csv) if args.slerp_csv and os.path.exists(args.slerp_csv) else []
    if intervals:
        print(f"  loaded {len(intervals)} flagged slerp intervals for correlation")

    # Build rows: one per frame (frame_id corresponds to fids[i+1] for accel)
    rows = []
    for i, acc in enumerate(accel):
        fid = fids[i + 1]  # centered frame
        slerp_flag = _in_any_interval(fid, intervals)
        rows.append({
            "frame_id": fid,
            "velocity_mean": float(dv[i + 1]) if i + 1 < len(dv) else float(dv[-1]),
            "accel_mean": float(acc),
            "vel_zscore": float(vel_z[i + 1]) if i + 1 < len(vel_z) else float(vel_z[-1]),
            "accel_zscore": float(acc_z[i]),
            "pop_flag": "POP" if acc_z[i] > args.threshold else "",
            "slerp_flag": slerp_flag or "",
        })

    # Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    # Top-N spike summary
    sorted_rows = sorted(rows, key=lambda r: r["accel_zscore"], reverse=True)
    top = sorted_rows[: args.top_n]
    print(f"\nTop {args.top_n} acceleration spikes:")
    print(f"  {'frame':>6s}  {'accel':>9s}  {'acc_z':>7s}  {'vel_z':>7s}  {'pop':>4s}  {'slerp':>17s}")
    for r in top:
        print(f"  {r['frame_id']:>6d}  {r['accel_mean']:>9.4f}  {r['accel_zscore']:>7.2f}  {r['vel_zscore']:>7.2f}  {r['pop_flag']:>4s}  {r['slerp_flag']:>17s}")

    # Aggregate stats
    pops = [r for r in rows if r["pop_flag"] == "POP"]
    print(f"\nTotal frames flagged POP (z>{args.threshold}): {len(pops)} / {len(rows)} ({100*len(pops)/len(rows):.1f}%)")
    if intervals:
        pops_with_slerp = [r for r in pops if r["slerp_flag"]]
        slerp_only = [r for r in rows if r["slerp_flag"] and r["pop_flag"] != "POP"]
        print(f"  POPs in slerp-flagged interval: {len(pops_with_slerp)} / {len(pops)} (precision)")
        # Recall = slerp-flagged frames that also show POP
        slerp_frames_total = sum(1 for r in rows if r["slerp_flag"])
        print(f"  Slerp-flagged frames shown as POP: {len(pops_with_slerp)} / {slerp_frames_total} (interval→frame recall)")
        if slerp_only:
            print(f"  Slerp-flagged but NOT POP: {len(slerp_only)} (flag may be false-positive or pop below threshold)")


if __name__ == "__main__":
    main()
