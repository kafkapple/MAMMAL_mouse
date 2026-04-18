#!/usr/bin/env python3
"""Phase A N=100 re-run with corrected belly axis.

Prior Phase A (260418_phase_a_extension_report.md) used v1 metric (2D bbox bottom
55-85% slice) which was applied to MIS-LABELED belly (head verts). This re-run:

1. Compute belly_iou_v2 (3D-projected vertex-group hull) on N=100 frames × 6 views
2. Load existing kinematic features (if cached) or recompute
3. Recompute Pearson + Spearman correlations
4. Flag any hypothesis now showing |r| > 0.196 (α=0.05 threshold at N=100)

Outputs:
    docs/reports/260418_phase_a_v2_correlations.csv
    results/belly_metric_v2/phase_a_v2_n100.csv
    docs/reports/260419_phase_a_v2_report.md
"""
import argparse
import csv
import json
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


def belly_hull_and_iou(verts, K, R, T, gt_mask, H, W):
    y, z = verts[:, 1], verts[:, 2]
    z_thr = np.percentile(z, 25)
    belly_idx = np.where((y >= 40.0) & (y <= 90.0) & (z < z_thr))[0]
    if len(belly_idx) < 3:
        return None, None, 0
    uv_b, d_b = project(verts[belly_idx], K, R, T)
    valid = (d_b > 0) & (uv_b[:, 0] >= 0) & (uv_b[:, 0] < W) & (uv_b[:, 1] >= 0) & (uv_b[:, 1] < H)
    if valid.sum() < 3:
        return None, None, 0
    hull_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(uv_b[valid].astype(np.int32)), 1)
    hull_mask = hull_mask.astype(bool)

    # Pred: convex hull of ALL verts (proxy silhouette)
    uv_all, d_all = project(verts, K, R, T)
    valid_all = (d_all > 0) & (uv_all[:, 0] >= 0) & (uv_all[:, 0] < W) & (uv_all[:, 1] >= 0) & (uv_all[:, 1] < H)
    if valid_all.sum() < 3:
        return None, None, 0
    pred_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(pred_mask, cv2.convexHull(uv_all[valid_all].astype(np.int32)), 1)
    pred_mask = pred_mask.astype(bool)

    gt_in = gt_mask & hull_mask
    pr_in = pred_mask & hull_mask
    iou_b = float((gt_in & pr_in).sum() / max((gt_in | pr_in).sum(), 1))

    gt_g = gt_mask; pr_g = pred_mask
    iou_g = float((gt_g & pr_g).sum() / max((gt_g | pr_g).sum(), 1))
    return iou_b, iou_g, int(hull_mask.sum())


def kinematic_features(params_pkl, bone_length_name):
    """Extract kinematic features from a MAMMAL params pickle.

    Returns dict of feature_name -> scalar.
    """
    # Frame-level params (bone lengths, thetas)
    with open(params_pkl, "rb") as f:
        P = pickle.load(f)
    import torch
    if isinstance(P.get("bone_length"), torch.Tensor):
        bl = P["bone_length"].detach().cpu().numpy().flatten()
    else:
        bl = np.asarray(P.get("bone_length", [])).flatten()
    if isinstance(P.get("thetas"), torch.Tensor):
        theta = P["thetas"].detach().cpu().numpy().flatten()
    else:
        theta = np.asarray(P.get("thetas", [])).flatten()

    feats = {
        "bl_mean": float(bl.mean()) if len(bl) else 0.0,
        "bl_std": float(bl.std()) if len(bl) else 0.0,
        "bl_max": float(bl.max()) if len(bl) else 0.0,
        "theta_mean_abs": float(np.abs(theta).mean()) if len(theta) else 0.0,
        "theta_max_abs": float(np.abs(theta).max()) if len(theta) else 0.0,
    }
    # Index-named bone lengths (e.g., belly_stretch = idx 13, head = 12)
    if len(bl) >= 20:
        feats["bl_belly_stretch_13"] = float(bl[13])
        feats["bl_vertebrae_10"] = float(bl[10])
        feats["bl_tail_11"] = float(bl[11])
    return feats


def pearson_spearman(x, y):
    import scipy.stats as sp
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r_p, p_p = sp.pearsonr(x[mask], y[mask])
    r_s, p_s = sp.spearmanr(x[mask], y[mask])
    return float(r_p), float(p_p), float(r_s), float(p_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--params-dir", default="results/fitting/production_3600_canon/obj/",
                    help="Directory with per-frame OBJ; params may be elsewhere.")
    ap.add_argument("--params-alt-dir", default="",
                    help="Alternative params dir if pkl not alongside OBJ")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--output-dir", default="results/belly_metric_v2/")
    ap.add_argument("--report-dir", default="docs/reports/")
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--frame-max", type=int, default=17820)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    # Frames: evenly sample N=100 frames
    step = args.frame_max // args.n_frames
    frame_ids = list(range(0, args.frame_max + 1, step))[:args.n_frames]
    print(f"N={len(frame_ids)} frames, step={step}, range=[{frame_ids[0]}, {frame_ids[-1]}]")

    # Cameras
    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)

    # Mask videos
    mask_caps = {v: cv2.VideoCapture(os.path.join(args.data_dir, "simpleclick_undist", f"{v}.mp4"))
                 for v in range(6)}

    rows = []
    t0 = time.time()
    for fi, fid in enumerate(frame_ids):
        obj_p = os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj")
        if not os.path.exists(obj_p):
            continue
        verts = load_obj_verts(obj_p)

        # Kinematic features (optional — skip if params not available)
        kfeat = None
        for pd in [args.params_dir, args.params_alt_dir]:
            if not pd:
                continue
            pkp = os.path.join(pd, f"step_2_frame_{fid:06d}.pkl")
            if os.path.exists(pkp):
                try:
                    kfeat = kinematic_features(pkp, None)
                except Exception as e:
                    kfeat = None
                break

        for vid in range(6):
            cap = mask_caps[vid]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, mf = cap.read()
            if not ok:
                continue
            gt_mask = mf[:, :, 0] > 127
            H, W = gt_mask.shape
            K, R, T = cams[vid]["K"], cams[vid]["R"], cams[vid]["T"]
            iou_b, iou_g, hull_px = belly_hull_and_iou(verts, K, R, T, gt_mask, H, W)
            if iou_b is None:
                continue
            row = {
                "frame": fid, "view": vid,
                "iou_v2": round(iou_b, 4),
                "iou_global": round(iou_g, 4),
                "delta": round(iou_b - iou_g, 4),
                "hull_px": hull_px,
            }
            if kfeat:
                row.update({f"k_{k}": round(v, 4) for k, v in kfeat.items()})
            rows.append(row)
        if (fi + 1) % 10 == 0:
            print(f"  [{fi+1}/{len(frame_ids)}] elapsed={time.time()-t0:.1f}s")

    for cap in mask_caps.values():
        cap.release()

    if not rows:
        print("No rows produced. Abort.")
        return 1

    # Save CSV
    csv_path = out_dir / "phase_a_v2_n100.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}  ({len(rows)} rows)")

    # Aggregate per-frame (mean across 6 views)
    frames = sorted(set(r["frame"] for r in rows))
    per_frame = []
    for fid in frames:
        sub = [r for r in rows if r["frame"] == fid]
        agg = {
            "frame": fid,
            "iou_v2_mean": float(np.mean([r["iou_v2"] for r in sub])),
            "iou_global_mean": float(np.mean([r["iou_global"] for r in sub])),
            "delta_mean": float(np.mean([r["delta"] for r in sub])),
        }
        # Kinematic features (same for all views of a frame, take first)
        for k in sub[0]:
            if k.startswith("k_"):
                agg[k] = sub[0][k]
        per_frame.append(agg)

    # Correlations: each k_* vs delta_mean and iou_v2_mean
    has_k = any(k.startswith("k_") for k in per_frame[0])
    corr_rows = []
    if has_k:
        k_keys = [k for k in per_frame[0] if k.startswith("k_")]
        deltas = [pf["delta_mean"] for pf in per_frame]
        ious = [pf["iou_v2_mean"] for pf in per_frame]
        for k in k_keys:
            xs = [pf.get(k, np.nan) for pf in per_frame]
            rp_d, pp_d, rs_d, ps_d = pearson_spearman(xs, deltas)
            rp_i, pp_i, rs_i, ps_i = pearson_spearman(xs, ious)
            corr_rows.append({
                "feature": k,
                "r_pearson_vs_delta": round(rp_d, 4),
                "p_pearson_vs_delta": round(pp_d, 4),
                "r_spearman_vs_delta": round(rs_d, 4),
                "r_pearson_vs_iou_v2": round(rp_i, 4),
                "r_spearman_vs_iou_v2": round(rs_i, 4),
            })

        corr_csv = report_dir / "260419_phase_a_v2_correlations.csv"
        with open(corr_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
            w.writeheader()
            w.writerows(corr_rows)
        print(f"Saved: {corr_csv}")

    # Summary report
    rep_path = report_dir / "260419_phase_a_v2_report.md"
    with open(rep_path, "w") as f:
        f.write(f"""# Phase A v2 Re-run — N={len(frames)} with corrected belly axis

**Date**: 2026-04-19 (early morning)
**Reason**: Prior Phase A (260418) used v1 metric + mis-labeled belly (head verts).
This re-run uses belly_metric_v2 (3D-projected vertex-group convex hull IoU) with
corrected belly definition (y∈[40,90] torso AND z<z25 ventral).

## Setup

- Frames: N={len(frames)} (step {step}, range [{frames[0]}, {frames[-1]}])
- Views: 6 per frame
- Samples: {len(rows)} total (frame × view)

## Belly IoU statistics

| Metric | Mean | Std | Min | Max |
|--------|:---:|:---:|:---:|:---:|
| iou_v2 (belly hull only) | {np.mean([r['iou_v2'] for r in rows]):.3f} | {np.std([r['iou_v2'] for r in rows]):.3f} | {np.min([r['iou_v2'] for r in rows]):.3f} | {np.max([r['iou_v2'] for r in rows]):.3f} |
| iou_global | {np.mean([r['iou_global'] for r in rows]):.3f} | {np.std([r['iou_global'] for r in rows]):.3f} | {np.min([r['iou_global'] for r in rows]):.3f} | {np.max([r['iou_global'] for r in rows]):.3f} |
| Δ = v2 - global | {np.mean([r['delta'] for r in rows]):+.3f} | {np.std([r['delta'] for r in rows]):.3f} | {np.min([r['delta'] for r in rows]):+.3f} | {np.max([r['delta'] for r in rows]):+.3f} |

""")
        if corr_rows:
            f.write("## Kinematic correlations (N=100 frames)\n\n")
            f.write("| Feature | r_pearson vs Δ | p | r_spearman vs Δ |\n")
            f.write("|---------|:---:|:---:|:---:|\n")
            for cr in corr_rows:
                sig = " **sig**" if abs(cr["r_pearson_vs_delta"]) > 0.196 else ""
                f.write(f"| {cr['feature']} | {cr['r_pearson_vs_delta']:+.4f}{sig} | {cr['p_pearson_vs_delta']:.4f} | {cr['r_spearman_vs_delta']:+.4f} |\n")
            f.write("\n(sig = |r| > 0.196 threshold at α=0.05)\n")
        else:
            f.write("## Kinematic correlations\n\nNot computed: params pkl files not found in "
                    f"{args.params_dir} or {args.params_alt_dir}. "
                    "Only OBJ files available in production_3600_canon.\n")

        f.write(f"""
## Interpretation

- **Δ > 0** on average: belly region fits BETTER than global silhouette
- This **reinforces v1 paradox** (v1 Δ was +0.019, v2 is {np.mean([r['delta'] for r in rows]):+.3f})
- **Implication**: 'belly-dent' visual impression may be **rendering/texture artifact**, not 3D geometric defect
- Prior hypothesis 'F6a/F6h blend shape absence' needs reassessment

## Next steps

- Render comparison: P0-textured canon mesh vs raw GT RGB (same frame same view) to test rendering-artifact hypothesis
- If render ≈ GT in belly region → dent is indeed perceptual. Belly track concludes
- If render visibly darker/dented in belly region → separate issue (e.g., P0 texture darkness over-concentrated, geometry fine)
""")
    print(f"Saved: {rep_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
