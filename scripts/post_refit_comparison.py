#!/usr/bin/env python3
"""Post-refit comparison: measure IoU + belly-dent improvement per frame.

Compares paper_fast canon fit vs accurate refit for outlier frames.
Produces: before/after metrics + visual grid.
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
    return np.array(vs, dtype=np.float32) if vs else None


def project(v, K, R, T):
    Pc = (R @ v.T).T + T.reshape(-1)
    uv = (K @ Pc.T).T
    return uv[:, :2] / (uv[:, 2:] + 1e-8), Pc[:, 2]


def frame_iou_and_belly(verts, cams, gt_masks, H, W):
    """Return (iou_global_mean_over_views, belly_iou_v2_mean, belly_delta_mean)."""
    y, z = verts[:, 1], verts[:, 2]
    z_med = float(np.percentile(z, 50))
    belly_idx = np.where((y >= 40.0) & (y <= 90.0) & (z >= 5.0) & (z <= z_med))[0]

    g_ious, b_ious, deltas = [], [], []
    for vid in range(6):
        if gt_masks[vid] is None:
            continue
        K, R, T = cams[vid]["K"], cams[vid]["R"], cams[vid]["T"]
        # Global
        uv_a, d_a = project(verts, K, R, T)
        va = (d_a > 0) & (uv_a[:, 0] >= 0) & (uv_a[:, 0] < W) & (uv_a[:, 1] >= 0) & (uv_a[:, 1] < H)
        if va.sum() < 3: continue
        pred = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(pred, cv2.convexHull(uv_a[va].astype(np.int32)), 1)
        pred = pred.astype(bool)
        gt = gt_masks[vid]
        iou_g = float((gt & pred).sum() / max((gt | pred).sum(), 1))
        g_ious.append(iou_g)
        # Belly hull
        if len(belly_idx) < 3:
            continue
        uv_b, d_b = project(verts[belly_idx], K, R, T)
        vb = (d_b > 0) & (uv_b[:, 0] >= 0) & (uv_b[:, 0] < W) & (uv_b[:, 1] >= 0) & (uv_b[:, 1] < H)
        if vb.sum() < 3: continue
        hull = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(uv_b[vb].astype(np.int32)), 1)
        hull = hull.astype(bool)
        iou_b = float((gt & pred & hull).sum() / max(((gt | pred) & hull).sum(), 1))
        b_ious.append(iou_b)
        deltas.append(iou_b - iou_g)
    if not g_ious:
        return None
    return {
        "iou_global": float(np.mean(g_ious)),
        "iou_belly_v2": float(np.mean(b_ious)) if b_ious else None,
        "belly_delta": float(np.mean(deltas)) if deltas else None,
        "n_views": len(g_ious),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--post-dir", default="results/fitting/refit_outliers_152/obj/")
    ap.add_argument("--frame-list", default="conf/frames/outlier_severe_152.txt")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--output", default="docs/reports/260419_refit_comparison.csv")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)
    mask_caps = [cv2.VideoCapture(os.path.join(args.data_dir, "simpleclick_undist", f"{v}.mp4"))
                 for v in range(6)]
    cap_rgb = cv2.VideoCapture(os.path.join(args.data_dir, "videos_undist", "0.mp4"))
    ok, first = cap_rgb.read(); cap_rgb.release()
    H, W = first.shape[:2]

    with open(args.frame_list) as f:
        frames = [int(ln.strip()) for ln in f if ln.strip()]

    rows = []
    for fid in frames:
        pre_p = os.path.join(args.pre_dir, f"step_2_frame_{fid:06d}.obj")
        post_p = os.path.join(args.post_dir, f"step_2_frame_{fid:06d}.obj")
        if not (os.path.exists(pre_p) and os.path.exists(post_p)):
            continue
        verts_pre = load_obj_verts(pre_p)
        verts_post = load_obj_verts(post_p)
        if verts_pre is None or verts_post is None: continue

        gt_masks = [None] * 6
        for vid in range(6):
            mask_caps[vid].set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, mf = mask_caps[vid].read()
            if ok: gt_masks[vid] = mf[:, :, 0] > 127

        pre_m = frame_iou_and_belly(verts_pre, cams, gt_masks, H, W)
        post_m = frame_iou_and_belly(verts_post, cams, gt_masks, H, W)
        if pre_m is None or post_m is None: continue
        row = {
            "frame": fid,
            "pre_iou_global": round(pre_m["iou_global"], 4),
            "post_iou_global": round(post_m["iou_global"], 4),
            "d_iou_global": round(post_m["iou_global"] - pre_m["iou_global"], 4),
        }
        if pre_m["iou_belly_v2"] is not None and post_m["iou_belly_v2"] is not None:
            row["pre_iou_belly"] = round(pre_m["iou_belly_v2"], 4)
            row["post_iou_belly"] = round(post_m["iou_belly_v2"], 4)
            row["d_iou_belly"] = round(post_m["iou_belly_v2"] - pre_m["iou_belly_v2"], 4)
        rows.append(row)

    for cap in mask_caps: cap.release()

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = set()
        for r in rows: keys.update(r.keys())
        keys = sorted(keys, key=lambda k: (k != "frame", k))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keys))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved: {out}  ({len(rows)} frame comparisons)")

    # Summary
    if rows:
        d_g = [r["d_iou_global"] for r in rows]
        print(f"\nGlobal IoU: pre mean {np.mean([r['pre_iou_global'] for r in rows]):.3f} → "
              f"post mean {np.mean([r['post_iou_global'] for r in rows]):.3f}  "
              f"(Δ {np.mean(d_g):+.3f})")
        improved = sum(1 for d in d_g if d > 0)
        print(f"Improved: {improved}/{len(rows)} ({100*improved/len(rows):.1f}%)")
        big_improve = sum(1 for d in d_g if d > 0.1)
        print(f"Δ > 0.1 improvement: {big_improve} frames")

        if "d_iou_belly" in rows[0]:
            d_b = [r["d_iou_belly"] for r in rows if "d_iou_belly" in r]
            print(f"\nBelly IoU: pre mean {np.mean([r['pre_iou_belly'] for r in rows if 'pre_iou_belly' in r]):.3f} → "
                  f"post mean {np.mean([r['post_iou_belly'] for r in rows if 'post_iou_belly' in r]):.3f}  "
                  f"(Δ {np.mean(d_b):+.3f})")


if __name__ == "__main__":
    sys.exit(main())
