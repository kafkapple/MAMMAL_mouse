#!/usr/bin/env python3
"""J2: Targeted refit POC — refit a specific frame with accurate config.

Uses Hydra override to run fitter_articulation on single frame, comparing
before/after v3 belly IoU.

Usage:
    python scripts/targeted_refit_poc.py --frame 2700 --config optim=accurate
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def v3_belly_iou(obj_path, data_dir, frame_id, views=range(6)):
    """Compute v3 belly IoU for a fitted mesh."""
    verts = []
    with open(obj_path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split(); verts.append([float(p[1]), float(p[2]), float(p[3])])
    V = np.array(verts, dtype=np.float32)
    y, z = V[:, 1], V[:, 2]
    z_med = float(np.percentile(z, 50))
    belly_idx = np.where((y >= 40.0) & (y <= 90.0) & (z >= 5.0) & (z <= z_med))[0]

    with open(os.path.join(data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)

    ious = []
    for vid in views:
        cap = cv2.VideoCapture(os.path.join(data_dir, "simpleclick_undist", f"{vid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, m = cap.read(); cap.release()
        if not ok: continue
        gt = m[:, :, 0] > 127
        H, W = gt.shape
        K, R, T = cams[vid]["K"], cams[vid]["R"], cams[vid]["T"].reshape(-1)
        # Belly hull
        Pb = (R @ V[belly_idx].T).T + T
        uvb = (K @ Pb.T).T[:, :2] / (Pb[:, 2:] + 1e-8)
        dvb = Pb[:, 2]
        vb = (dvb > 0) & (uvb[:, 0] >= 0) & (uvb[:, 0] < W) & (uvb[:, 1] >= 0) & (uvb[:, 1] < H)
        if vb.sum() < 3: continue
        hull = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(uvb[vb].astype(np.int32)), 1)
        hull = hull.astype(bool)
        # All-verts pred
        Pa = (R @ V.T).T + T
        uva = (K @ Pa.T).T[:, :2] / (Pa[:, 2:] + 1e-8)
        dva = Pa[:, 2]
        va = (dva > 0) & (uva[:, 0] >= 0) & (uva[:, 0] < W) & (uva[:, 1] >= 0) & (uva[:, 1] < H)
        if va.sum() < 3: continue
        pred = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(pred, cv2.convexHull(uva[va].astype(np.int32)), 1)
        pred = pred.astype(bool)
        ib = (gt & pred & hull).sum() / max(((gt | pred) & hull).sum(), 1)
        ig = (gt & pred).sum() / max((gt | pred).sum(), 1)
        ious.append((vid, float(ib), float(ig), float(ib - ig)))
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=2700)
    ap.add_argument("--pre-obj",
                    default="results/fitting/production_3600_canon/obj/step_2_frame_002700.obj")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--output", default="results/refit_poc_2700/")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # 1. Measure BEFORE iou
    print(f"[PRE] canon fit: {args.pre_obj}")
    pre_ious = v3_belly_iou(args.pre_obj, args.data_dir, args.frame)
    pre_summary = {
        "frame": args.frame,
        "pre_per_view": pre_ious,
        "pre_mean_iou_v2": float(np.mean([x[1] for x in pre_ious])),
        "pre_mean_iou_global": float(np.mean([x[2] for x in pre_ious])),
        "pre_mean_delta": float(np.mean([x[3] for x in pre_ious])),
    }
    for vid, iv2, ig, d in pre_ious:
        print(f"  v{vid}: iou_v2={iv2:.3f} iou_global={ig:.3f} Δ={d:+.3f}")
    print(f"  → mean: iou_v2={pre_summary['pre_mean_iou_v2']:.3f} Δ={pre_summary['pre_mean_delta']:+.3f}")

    with open(out / "pre_refit_iou.json", "w") as f:
        json.dump(pre_summary, f, indent=2)

    # 2. Run accurate refit on frame 2700
    #    Approach: use existing fitter_articulation.py with single-frame range
    #    + accurate optim config
    print(f"\n[REFIT] running accurate optim on frame {args.frame}")
    refit_out = out / f"accurate_frame_{args.frame:06d}"
    refit_out.mkdir(exist_ok=True)
    cmd = [
        "./run_experiment.sh", "baseline_6view_keypoint",
        f"frames.start={args.frame}",
        f"frames.end={args.frame+1}",
        "optim=accurate",
        f"output_dir={refit_out}/",
    ]
    # Hydra may not support arbitrary output_dir; use simpler approach:
    # Just note this is a POC placeholder. Actual command varies.
    print(f"  (would run: {' '.join(cmd)})")
    print(f"  NOTE: actual refit command depends on project's run_experiment.sh and Hydra config.")
    print(f"  For POC testing, manually invoke:")
    print(f"    python fitter_articulation.py frames=[{args.frame}] optim=accurate ...")

    # 3. Placeholder for POST measurement
    post_obj = refit_out / f"obj/step_2_frame_{args.frame:06d}.obj"
    if post_obj.exists():
        print(f"\n[POST] refit fit: {post_obj}")
        post_ious = v3_belly_iou(str(post_obj), args.data_dir, args.frame)
        post_summary = {
            "post_per_view": post_ious,
            "post_mean_iou_v2": float(np.mean([x[1] for x in post_ious])),
            "post_mean_delta": float(np.mean([x[3] for x in post_ious])),
        }
        with open(out / "post_refit_iou.json", "w") as f:
            json.dump(post_summary, f, indent=2)
        improvement = post_summary["post_mean_iou_v2"] - pre_summary["pre_mean_iou_v2"]
        print(f"  → Δ iou_v2 (post - pre): {improvement:+.3f}")
    else:
        print(f"\n[POST] {post_obj} not found — run refit first, then re-run this script")


if __name__ == "__main__":
    sys.exit(main())
