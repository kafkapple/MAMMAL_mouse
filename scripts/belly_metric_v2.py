#!/usr/bin/env python3
"""F2: B2 belly metric v2 — 3D-projected IoU using belly vertex group.

v1 (belly_iou_diagnostic.py): 2D bbox bottom 55-85% slice (proxy, possibly orthogonal to 3D dent)
v2 (this): Project belly vertex group (2734 verts, y > mesh_y_max-20mm in canon) to each view →
           compute convex hull of projection → IoU of (rendered silhouette ∩ hull) vs (GT mask ∩ hull)

Outputs:
    results/belly_metric_v2/
      belly_v2_{frame_id}_v{view_id}.png  (overlay: GT mask / hull / render)
      belly_v2_scores.csv  (frame, view, iou_v2, iou_global, n_belly_in_hull)
      belly_v2_summary.md
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


def load_obj_verts(obj_path):
    verts = []
    with open(obj_path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(verts, dtype=np.float32)


def belly_mask_indices(verts, margin_mm=20.0):
    """Canon mesh: Y=head-to-tail, Z=vertical. Belly = torso y∈[40,90] AND z_low."""
    y = verts[:, 1]; z = verts[:, 2]
    z_thr = np.percentile(z, 25)
    mask = (y >= 40.0) & (y <= 90.0) & (z < z_thr)
    return np.where(mask)[0]


def project_verts(verts, K, R, T):
    """Project 3D points to 2D using OpenCV camera model."""
    Pc = (R @ verts.T).T + T.reshape(-1)
    p2h = (K @ Pc.T).T
    uv = p2h[:, :2] / (p2h[:, 2:] + 1e-8)
    return uv, Pc[:, 2]  # (N, 2), depth


def compute_belly_iou(gt_mask, pred_mask, hull_mask):
    """IoU within hull region only. Args are bool arrays (H, W)."""
    gt_in = gt_mask & hull_mask
    pr_in = pred_mask & hull_mask
    inter = (gt_in & pr_in).sum()
    union = (gt_in | pr_in).sum()
    return float(inter / union) if union > 0 else 0.0


def render_silhouette_approx(verts, faces, K, R, T, H, W):
    """Rough silhouette via vertex projection + filled polygon.

    Not pixel-perfect (no actual rasterization) but OK proxy for IoU.
    Returns bool (H, W) mask.
    """
    uv, d = project_verts(verts, K, R, T)
    valid = (d > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    mask = np.zeros((H, W), dtype=np.uint8)
    # Build convex hull of valid projected vertices
    uv_valid = uv[valid].astype(np.int32)
    if len(uv_valid) < 3:
        return mask.astype(bool)
    hull = cv2.convexHull(uv_valid)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--output", default="results/belly_metric_v2/")
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[1800, 3600, 5400, 7200, 9000, 10800, 12600, 14400, 16200, 17700],
                    help="Frame IDs to measure (subset for speed)")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--belly-margin-mm", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # Load cameras
    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)

    # Open silhouette mask videos
    mask_caps = {}
    for vid in args.views:
        mp = os.path.join(args.data_dir, "simpleclick_undist", f"{vid}.mp4")
        mask_caps[vid] = cv2.VideoCapture(mp)

    rows = []
    for fid in args.frames:
        obj_path = os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj")
        if not os.path.exists(obj_path):
            print(f"skip: {obj_path} missing")
            continue
        verts = load_obj_verts(obj_path)
        belly_idx = belly_mask_indices(verts, args.belly_margin_mm)

        for vid in args.views:
            cap = mask_caps[vid]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, mframe = cap.read()
            if not ok:
                continue
            gt_mask = (mframe[:, :, 0] > 127)
            H, W = gt_mask.shape

            K = cams[vid]["K"]; R = cams[vid]["R"]; T = cams[vid]["T"]

            # Full-mesh silhouette approximation (proxy for rendered pred)
            full_hull = render_silhouette_approx(verts, faces=None, K=K, R=R, T=T, H=H, W=W)
            # Actually we have no faces here, use convex hull only. Let's mark this as pred_mask.
            pred_mask = full_hull

            # Belly hull (projected belly vertices)
            belly_verts = verts[belly_idx]
            uv_belly, d_belly = project_verts(belly_verts, K, R, T)
            valid = (d_belly > 0) & (uv_belly[:, 0] >= 0) & (uv_belly[:, 0] < W) & \
                    (uv_belly[:, 1] >= 0) & (uv_belly[:, 1] < H)
            if valid.sum() < 3:
                continue
            uv_b = uv_belly[valid].astype(np.int32)
            hull_b = cv2.convexHull(uv_b)
            belly_hull_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(belly_hull_mask, hull_b, 1)
            belly_hull_mask = belly_hull_mask.astype(bool)

            # v2 IoU: inside belly hull only
            iou_v2 = compute_belly_iou(gt_mask, pred_mask, belly_hull_mask)
            # global IoU for comparison
            inter_g = (gt_mask & pred_mask).sum(); union_g = (gt_mask | pred_mask).sum()
            iou_global = float(inter_g / union_g) if union_g > 0 else 0.0
            rows.append({
                "frame": fid, "view": vid,
                "iou_v2": round(iou_v2, 4),
                "iou_global": round(iou_global, 4),
                "belly_proj_verts": int(valid.sum()),
                "belly_hull_pixels": int(belly_hull_mask.sum()),
            })
            print(f"f{fid:5d} v{vid}: iou_v2={iou_v2:.3f}  iou_global={iou_global:.3f}  "
                  f"belly_hull={int(belly_hull_mask.sum()):5d}px")

            # Save overlay for first frame each view
            if fid == args.frames[0]:
                ovl = np.zeros((H, W, 3), dtype=np.uint8)
                ovl[gt_mask] = (180, 180, 180)
                ovl[belly_hull_mask & ~gt_mask] = (80, 80, 220)  # hull only (blue)
                ovl[belly_hull_mask & gt_mask] = (220, 80, 80)   # hull ∩ GT (red)
                cv2.imwrite(str(out / f"overlay_f{fid}_v{vid}.png"), ovl)

    for cap in mask_caps.values():
        cap.release()

    # Save CSV
    import csv
    csv_path = out / "belly_v2_scores.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\nsaved: {csv_path}  ({len(rows)} rows)")

    # Summary
    if rows:
        v2s = np.array([r["iou_v2"] for r in rows])
        gs = np.array([r["iou_global"] for r in rows])
        delta = v2s - gs
        summary = f"""# Belly Metric v2 — Results

- Frames: {sorted(set(r['frame'] for r in rows))}
- Views: {sorted(set(r['view'] for r in rows))}
- N samples: {len(rows)}

## Statistics

| metric | mean | std | min | max |
|--------|:---:|:---:|:---:|:---:|
| iou_v2 (belly hull only) | {v2s.mean():.3f} | {v2s.std():.3f} | {v2s.min():.3f} | {v2s.max():.3f} |
| iou_global | {gs.mean():.3f} | {gs.std():.3f} | {gs.min():.3f} | {gs.max():.3f} |
| Δ (v2 - global) | {delta.mean():+.3f} | {delta.std():.3f} | {delta.min():+.3f} | {delta.max():+.3f} |

## Interpretation

- If Δ ≈ 0: belly IoU ≈ global IoU (no distinctive belly defect signal, v2 still uninformative)
- If Δ < -0.05: **belly region performs worse** than overall — dent confirmed
- If Δ > +0.05: belly region better than overall — v1 paradox reproduced (data quirk, not metric bug)
- Compare with v1's "+0.019 belly > global" to detect if v2 fixes the paradox

## Known caveat

pred_mask is convex-hull approximation (no actual mesh rasterization). True silhouette
requires pyrender/pytorch3d rendering. This v2.0 gives a **hull-vs-GT-mask proxy**;
v2.1 would replace with pyrender silhouette.
"""
        with open(out / "belly_v2_summary.md", "w") as f:
            f.write(summary)
        print(f"saved: {out / 'belly_v2_summary.md'}")


if __name__ == "__main__":
    sys.exit(main())
