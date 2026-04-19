#!/usr/bin/env python3
"""P0: Scan all 3600 canon frames, identify belly-dent outliers.

Uses v3 belly metric (belly_metric_v2 logic) on every 5th frame in the
production_3600_canon/obj/ directory. Output: ranked candidate list.
"""
import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def load_obj_verts(p):
    vs = []
    with open(p) as fh:
        for ln in fh:
            if ln.startswith("v "):
                parts = ln.split()
                vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vs, dtype=np.float32)


def project(v, K, R, T):
    Pc = (R @ v.T).T + T.reshape(-1)
    uv = (K @ Pc.T).T
    return uv[:, :2] / (uv[:, 2:] + 1e-8), Pc[:, 2]


def compute_frame_delta(verts, cams, gt_masks_per_view, H, W, views=range(6)):
    # v3 belly
    y, z = verts[:, 1], verts[:, 2]
    z_med = float(np.percentile(z, 50))
    belly_idx = np.where((y >= 40.0) & (y <= 90.0) & (z >= 5.0) & (z <= z_med))[0]
    if len(belly_idx) < 3:
        return None
    deltas = []
    for vid in views:
        if gt_masks_per_view[vid] is None:
            continue
        K, R, T = cams[vid]["K"], cams[vid]["R"], cams[vid]["T"]
        uv_b, d_b = project(verts[belly_idx], K, R, T)
        valid = (d_b > 0) & (uv_b[:, 0] >= 0) & (uv_b[:, 0] < W) & \
                (uv_b[:, 1] >= 0) & (uv_b[:, 1] < H)
        if valid.sum() < 3:
            continue
        hull = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(uv_b[valid].astype(np.int32)), 1)
        hull = hull.astype(bool)
        uv_a, d_a = project(verts, K, R, T)
        va = (d_a > 0) & (uv_a[:, 0] >= 0) & (uv_a[:, 0] < W) & \
             (uv_a[:, 1] >= 0) & (uv_a[:, 1] < H)
        if va.sum() < 3:
            continue
        pred = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(pred, cv2.convexHull(uv_a[va].astype(np.int32)), 1)
        pred = pred.astype(bool)
        gt = gt_masks_per_view[vid]
        ib = (gt & pred & hull).sum() / max(((gt | pred) & hull).sum(), 1)
        ig = (gt & pred).sum() / max((gt | pred).sum(), 1)
        deltas.append(float(ib - ig))
    if not deltas:
        return None
    return float(np.mean(deltas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--output", default="docs/reports/260419_belly_outlier_scan.csv")
    ap.add_argument("--max-frame", type=int, default=17895)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--severe-thr", type=float, default=-0.2)
    ap.add_argument("--moderate-thr", type=float, default=-0.05)
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)

    # Open all 6 mask videos once
    mask_caps = [cv2.VideoCapture(os.path.join(args.data_dir, "simpleclick_undist", f"{v}.mp4"))
                 for v in range(6)]

    # Iterate every `step` frames
    frame_ids = list(range(0, args.max_frame + 1, args.step))
    print(f"Scanning {len(frame_ids)} frames (step={args.step})")

    # First frame: determine H, W
    cap_rgb = cv2.VideoCapture(os.path.join(args.data_dir, "videos_undist", "0.mp4"))
    ok, first = cap_rgb.read()
    cap_rgb.release()
    H, W = first.shape[:2]

    results = []
    t0 = time.time()
    n_ok = 0
    for i, fid in enumerate(frame_ids):
        obj_p = os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj")
        if not os.path.exists(obj_p):
            continue
        verts = load_obj_verts(obj_p)
        # Load 6 GT masks for this frame
        gt_masks = [None] * 6
        for vid in range(6):
            mask_caps[vid].set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, mframe = mask_caps[vid].read()
            if ok:
                gt_masks[vid] = mframe[:, :, 0] > 127
        delta = compute_frame_delta(verts, cams, gt_masks, H, W)
        if delta is None:
            continue
        results.append({"frame": fid, "delta": round(delta, 4)})
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(frame_ids)}]  ok={n_ok}  elapsed={time.time()-t0:.1f}s")

    for cap in mask_caps:
        cap.release()

    # Save CSV
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["delta"])
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "delta"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {out_path}  ({len(results)} frames)")

    # Summary
    severe = [r for r in results if r["delta"] < args.severe_thr]
    moderate = [r for r in results if args.severe_thr <= r["delta"] < args.moderate_thr]
    print(f"\nOutlier summary:")
    print(f"  Severe (Δ<{args.severe_thr}): {len(severe)} frames ({100*len(severe)/len(results):.1f}%)")
    for r in severe[:20]:
        print(f"    frame {r['frame']:5d}  Δ={r['delta']:+.3f}")
    print(f"  Moderate ({args.severe_thr}≤Δ<{args.moderate_thr}): {len(moderate)} frames")


if __name__ == "__main__":
    sys.exit(main())
