#!/usr/bin/env python3
"""
Compare fast vs accurate mesh fitting results.

Renders silhouettes, computes IoU, and creates comparison grids.

Usage:
    # Compare all 23 refit frames
    python scripts/compare_refit.py

    # Compare specific frames (partial results OK)
    python scripts/compare_refit.py --frames 720 1320

    # Compare from specific views
    python scripts/compare_refit.py --views 0 3 5

    # Custom OBJ directories
    python scripts/compare_refit.py \
        --obj-a /path/to/original/obj/ \
        --obj-b /path/to/refit/obj/
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BAD_FRAMES = [
    720, 1320, 1920, 2040, 2160, 2760, 3600,
    5160, 5520, 5880, 6000, 6120, 6960, 7200,
    8280, 8400, 9360, 9480, 9840, 10080,
    10680, 10800, 11880,
]


def main():
    parser = argparse.ArgumentParser(description="Compare fast vs accurate mesh fitting")
    parser.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/",
                        help="Dataset directory")
    parser.add_argument("--obj-a", default=None,
                        help="Original OBJ directory (default: /home/joon/data/synthetic/textured_obj/)")
    parser.add_argument("--obj-b", default="results/fitting/refit_accurate_23/obj/",
                        help="Refit OBJ directory")
    parser.add_argument("--output", default="results/comparison/fast_vs_accurate/",
                        help="Output directory")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="MAMMAL frame IDs to compare (default: all 23 bad frames)")
    parser.add_argument("--views", nargs="+", type=int, default=[3],
                        help="Camera view IDs (default: 3)")
    parser.add_argument("--label-a", default="fast", help="Label for set A")
    parser.add_argument("--label-b", default="accurate", help="Label for set B")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Default OBJ paths
    if args.obj_a is None:
        args.obj_a = "/home/joon/data/synthetic/textured_obj/"

    # Resolve frame list
    frame_ids = args.frames if args.frames else BAD_FRAMES

    # Filter to frames that have refit OBJ available
    available = []
    for fid in frame_ids:
        obj_b = os.path.join(args.obj_b, f"step_2_frame_{fid:06d}.obj")
        if os.path.exists(obj_b):
            available.append(fid)
        else:
            print(f"  Skip frame {fid}: refit OBJ not yet available")

    if not available:
        print("No refit OBJ files available yet. Wait for fitting to complete.")
        sys.exit(1)

    print(f"Comparing {len(available)}/{len(frame_ids)} frames (views: {args.views})")

    from mammal_ext.visualization.mesh_comparison import MeshComparison, ComparisonConfig

    config = ComparisonConfig()
    comp = MeshComparison(data_dir=args.data_dir, config=config, device=args.device)

    results = comp.compare(
        obj_dir_a=args.obj_a,
        obj_dir_b=args.obj_b,
        frame_ids=available,
        view_ids=args.views,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    for vid in args.views:
        output_dir = os.path.join(args.output, f"view_{vid}")
        comp.save_grid(results, output_dir, view_id=vid)

    # Print summary
    print(f"\n{'='*50}")
    print("Summary:")
    for r in results:
        for vid in args.views:
            iou_a = r.iou_a.get(vid, -1)
            iou_b = r.iou_b.get(vid, -1)
            delta = iou_b - iou_a if iou_a >= 0 and iou_b >= 0 else 0
            marker = "✓" if iou_b >= config.iou_threshold else "✗"
            print(f"  Frame {r.frame_id:>6}: fast={iou_a:.3f} → accurate={iou_b:.3f} ({delta:+.3f}) {marker}")


if __name__ == "__main__":
    main()
