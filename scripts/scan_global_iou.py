#!/usr/bin/env python3
"""Scan global silhouette IoU across all 3600 canon frames for paper-standard metric.

Output: docs/reports/260419_global_iou_scan.csv + summary.
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


def global_iou(verts, cams, gt_masks, H, W, views=range(6)):
    ious = []
    for vid in views:
        if gt_masks[vid] is None:
            continue
        K, R, T = cams[vid]["K"], cams[vid]["R"], cams[vid]["T"]
        uv, d = project(verts, K, R, T)
        valid = (d > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if valid.sum() < 3:
            continue
        pred = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(pred, cv2.convexHull(uv[valid].astype(np.int32)), 1)
        pred = pred.astype(bool)
        gt = gt_masks[vid]
        inter = (gt & pred).sum(); union = (gt | pred).sum()
        ious.append(float(inter / max(union, 1)))
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--output", default="docs/reports/260419_global_iou_scan.csv")
    ap.add_argument("--max-frame", type=int, default=17895)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)
    mask_caps = [cv2.VideoCapture(os.path.join(args.data_dir, "simpleclick_undist", f"{v}.mp4"))
                 for v in range(6)]
    cap_rgb = cv2.VideoCapture(os.path.join(args.data_dir, "videos_undist", "0.mp4"))
    ok, first = cap_rgb.read(); cap_rgb.release()
    H, W = first.shape[:2]

    frame_ids = list(range(0, args.max_frame + 1, args.step))
    print(f"Scanning {len(frame_ids)} frames for global IoU")

    rows = []
    t0 = time.time()
    for i, fid in enumerate(frame_ids):
        obj_p = os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj")
        if not os.path.exists(obj_p):
            continue
        verts = load_obj_verts(obj_p)
        gt_masks = [None] * 6
        for vid in range(6):
            mask_caps[vid].set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, mframe = mask_caps[vid].read()
            if ok:
                gt_masks[vid] = mframe[:, :, 0] > 127
        ious = global_iou(verts, cams, gt_masks, H, W)
        if not ious:
            continue
        row = {"frame": fid, "mean_iou": round(float(np.mean(ious)), 4)}
        for vid, iou in enumerate(ious):
            row[f"iou_v{vid}"] = round(iou, 4)
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(frame_ids)}] ok={len(rows)} elapsed={time.time()-t0:.1f}s")

    for cap in mask_caps:
        cap.release()

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out_path}  ({len(rows)} frames)")

    # Summary
    means = [r["mean_iou"] for r in rows]
    print(f"\nGlobal IoU summary (N={len(rows)} frames × 6 views):")
    print(f"  mean:   {np.mean(means):.3f}")
    print(f"  median: {np.median(means):.3f}")
    print(f"  std:    {np.std(means):.3f}")
    print(f"  min:    {np.min(means):.3f}")
    print(f"  max:    {np.max(means):.3f}")
    print(f"  ≥0.80:  {sum(1 for m in means if m >= 0.80)}/{len(means)} ({100*sum(1 for m in means if m >= 0.80)/len(means):.1f}%)")
    print(f"  ≥0.70:  {sum(1 for m in means if m >= 0.70)}/{len(means)} ({100*sum(1 for m in means if m >= 0.70)/len(means):.1f}%)")
    print(f"  <0.50:  {sum(1 for m in means if m < 0.50)}/{len(means)} ({100*sum(1 for m in means if m < 0.50)/len(means):.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
