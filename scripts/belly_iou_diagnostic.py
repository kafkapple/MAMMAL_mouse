#!/usr/bin/env python3
"""
Regional IoU diagnostic for belly/abdomen.

Method (2D proxy, v1):
  1. Render mesh silhouette + load GT mask
  2. Compute GT bbox (min/max nonzero row/col)
  3. Define belly sub-bbox as row range [y_min + 0.55*h, y_min + 0.85*h]
     and col range [x_min + 0.20*w, x_min + 0.80*w] within GT bbox.
     Mouse is a quadruped oriented head-up in this dataset's views —
     the mid-lower slab is the belly region. Tune via --belly-* flags.
  4. IoU over the belly sub-bbox only.

Outputs CSV: frame, view, belly_iou, global_iou, delta.
v2 TODO: replace 2D proxy with true vertex-group belly IoU once
manual annotation of belly vertex indices is available.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/belly_iou_diagnostic.py \
        --obj-dir results/fitting/production_3600_slerp/obj/ \
        --output results/reports/belly_iou.csv \
        --frames 0,120,240,720,1320,5520,9480 \
        --views 3

    # All 100 baseline-grid frames
    python scripts/belly_iou_diagnostic.py \
        --obj-dir results/fitting/production_3600_slerp/obj/ \
        --output results/reports/belly_iou_grid100.csv \
        --all-100
"""

import argparse
import csv
import os
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(y1), int(x0), int(x1)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    parser.add_argument("--obj-dir", required=True)
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--frames", default="", help="Comma-separated frame IDs")
    parser.add_argument("--all-100", action="store_true",
                        help="Use baseline grid: 0, 120, 240, ..., 11880")
    parser.add_argument("--views", nargs="+", type=int, default=[3])
    parser.add_argument("--belly-y-lo", type=float, default=0.55,
                        help="Lower row fraction of GT bbox (0=top)")
    parser.add_argument("--belly-y-hi", type=float, default=0.85)
    parser.add_argument("--belly-x-lo", type=float, default=0.20)
    parser.add_argument("--belly-x-hi", type=float, default=0.80)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.all_100:
        frames = list(range(0, 11881, 120))
    elif args.frames:
        frames = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        print("ERROR: give --frames or --all-100", file=sys.stderr)
        sys.exit(1)

    from mammal_ext.visualization.mesh_comparison import MeshComparison, ComparisonConfig
    comp = MeshComparison(data_dir=args.data_dir, config=ComparisonConfig(), device=args.device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows: List[list] = []
    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "view", "global_iou", "belly_iou", "delta_belly_minus_global"])

        for fid in frames:
            obj_path = os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj")
            if not os.path.exists(obj_path):
                print(f"skip frame {fid}: OBJ missing")
                continue
            verts = comp._load_obj_vertices(obj_path)

            for vid in args.views:
                gt = comp._load_gt_mask(fid, vid).astype(bool)
                pred = comp._render_silhouette(verts, vid).astype(bool)
                if pred.shape != gt.shape:
                    print(f"skip frame {fid} view {vid}: shape mismatch pred={pred.shape} gt={gt.shape}")
                    continue
                bb = _bbox(gt)
                if bb is None:
                    print(f"skip frame {fid} view {vid}: empty GT mask")
                    continue
                y0, y1, x0, x1 = bb
                h, wd = (y1 - y0 + 1), (x1 - x0 + 1)
                by0 = y0 + int(args.belly_y_lo * h)
                by1 = y0 + int(args.belly_y_hi * h)
                bx0 = x0 + int(args.belly_x_lo * wd)
                bx1 = x0 + int(args.belly_x_hi * wd)
                belly_gt = gt[by0:by1 + 1, bx0:bx1 + 1]
                belly_pred = pred[by0:by1 + 1, bx0:bx1 + 1]
                global_iou = _iou(pred, gt)
                belly_iou = _iou(belly_pred, belly_gt)
                delta = belly_iou - global_iou
                w.writerow([fid, vid, f"{global_iou:.4f}", f"{belly_iou:.4f}", f"{delta:+.4f}"])
                rows.append([fid, vid, global_iou, belly_iou, delta])

    if not rows:
        print("No rows written.")
        return
    belly = np.array([r[3] for r in rows])
    glob = np.array([r[2] for r in rows])
    print(f"\nSamples: {len(rows)}")
    print(f"  Global IoU:  mean={glob.mean():.4f}  median={np.median(glob):.4f}  min={glob.min():.4f}")
    print(f"  Belly IoU:   mean={belly.mean():.4f}  median={np.median(belly):.4f}  min={belly.min():.4f}")
    print(f"  Belly - Global delta:  mean={(belly - glob).mean():+.4f}")
    worst = sorted(rows, key=lambda r: r[3])[:10]
    print("\n  Worst 10 belly IoU:")
    for fid, vid, g, b, d in worst:
        print(f"    frame {fid:>6} view {vid}: belly={b:.4f} global={g:.4f} delta={d:+.4f}")


if __name__ == "__main__":
    main()
