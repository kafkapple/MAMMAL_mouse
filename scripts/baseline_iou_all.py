#!/usr/bin/env python3
"""
Compute IoU baseline for all 100 OBJ frames against GT masks.
Ranks frames by IoU to identify the worst fitting results.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/baseline_iou_all.py
    CUDA_VISIBLE_DEVICES=5 python scripts/baseline_iou_all.py --views 0 1 2 3 4 5
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="IoU baseline for all 100 frames")
    parser.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    parser.add_argument("--obj-dir", default="/home/joon/data/synthetic/textured_obj/")
    parser.add_argument("--output", default="results/comparison/baseline_iou/")
    parser.add_argument("--views", nargs="+", type=int, default=[3])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # All 100 MAMMAL frame IDs (step=120, frames 0-11880)
    all_frames = list(range(0, 11881, 120))
    print(f"Computing IoU for {len(all_frames)} frames, views {args.views}")

    from mammal_ext.visualization.mesh_comparison import MeshComparison, ComparisonConfig

    config = ComparisonConfig()
    comp = MeshComparison(data_dir=args.data_dir, config=config, device=args.device)

    # Compute IoU for each frame (no comparison, just single set)
    results = {}
    for i, fid in enumerate(all_frames):
        obj_path = os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj")
        if not os.path.exists(obj_path):
            print(f"  [{i+1}/100] Frame {fid}: OBJ not found, skip")
            continue

        verts = comp._load_obj_vertices(obj_path)
        frame_ious = {}
        for vid in args.views:
            gt_mask = comp._load_gt_mask(fid, vid)
            sil = comp._render_silhouette(verts, vid)
            iou = comp.compute_iou(sil, gt_mask)
            frame_ious[vid] = iou

        mean_iou = np.mean(list(frame_ious.values()))
        results[fid] = {"per_view": frame_ious, "mean": float(mean_iou)}

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/100] Frame {fid}: mean IoU={mean_iou:.4f}")

    # Sort by mean IoU
    sorted_frames = sorted(results.items(), key=lambda x: x[1]["mean"])

    # Save results
    os.makedirs(args.output, exist_ok=True)

    # JSON with full results
    json_path = os.path.join(args.output, "baseline_iou.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Text report
    report_path = os.path.join(args.output, "baseline_iou_report.txt")
    with open(report_path, "w") as f:
        f.write(f"MAMMAL Mesh IoU Baseline (views: {args.views})\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Rank':>4} {'Frame':>8} {'Mean IoU':>10} {'Status':>8}\n")
        f.write(f"{'-'*60}\n")

        for rank, (fid, data) in enumerate(sorted_frames, 1):
            status = "GOOD" if data["mean"] >= config.iou_threshold else "BAD"
            f.write(f"{rank:>4} {fid:>8} {data['mean']:>10.4f} {status:>8}\n")

        f.write(f"{'-'*60}\n")
        all_ious = [d["mean"] for d in results.values()]
        n_bad = sum(1 for iou in all_ious if iou < config.iou_threshold)
        f.write(f"\nTotal: {len(all_ious)}, Bad (IoU < {config.iou_threshold}): {n_bad}\n")
        f.write(f"Mean IoU: {np.mean(all_ious):.4f}, Min: {np.min(all_ious):.4f}, Max: {np.max(all_ious):.4f}\n")

        # Worst 10
        f.write(f"\nWorst 10 frames:\n")
        for rank, (fid, data) in enumerate(sorted_frames[:10], 1):
            f.write(f"  {rank}. Frame {fid}: IoU={data['mean']:.4f}\n")

    print(f"\nReport: {report_path}")
    print(f"JSON: {json_path}")

    # Print summary
    all_ious = [d["mean"] for d in results.values()]
    print(f"\n{'='*50}")
    print(f"Mean IoU: {np.mean(all_ious):.4f}")
    print(f"Min: {np.min(all_ious):.4f} (frame {sorted_frames[0][0]})")
    print(f"Max: {np.max(all_ious):.4f} (frame {sorted_frames[-1][0]})")
    n_bad = sum(1 for iou in all_ious if iou < config.iou_threshold)
    print(f"Bad (IoU < {config.iou_threshold}): {n_bad}/100")
    print(f"\nWorst 5:")
    for rank, (fid, data) in enumerate(sorted_frames[:5], 1):
        print(f"  {rank}. Frame {fid}: IoU={data['mean']:.4f}")


if __name__ == "__main__":
    main()
