#!/usr/bin/env python3
"""
Generate dense mesh sequence by interpolating between fitted keyframes.

Usage:
    # Interpolate between dense_accurate keyframes (M5 0-99)
    CUDA_VISIBLE_DEVICES=5 python scripts/interpolate_keyframes.py \
        --params-dir results/fitting/dense_accurate_0_100/params/ \
        --start 0 --end 500 --step 5 \
        --output results/fitting/interpolated_0_100/obj/

    # Use every 4th frame as keyframe, interpolate the rest
    CUDA_VISIBLE_DEVICES=5 python scripts/interpolate_keyframes.py \
        --params-dir results/fitting/dense_accurate_0_100/params/ \
        --keyframe-interval 4 \
        --start 0 --end 500 --step 5 \
        --output results/fitting/interpolated_interval4/obj/
"""

import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Interpolate keyframes to dense mesh sequence")
    parser.add_argument("--params-dir", required=True, help="Keyframe params directory")
    parser.add_argument("--start", type=int, default=0, help="Start video frame")
    parser.add_argument("--end", type=int, default=500, help="End video frame (exclusive)")
    parser.add_argument("--step", type=int, default=5, help="Frame step")
    parser.add_argument("--output", required=True, help="Output OBJ directory")
    parser.add_argument("--method", choices=["slerp", "lerp"], default="slerp")
    parser.add_argument("--keyframe-interval", type=int, default=None,
                        help="Use every Nth keyframe only (simulates sparse fitting)")
    parser.add_argument("--no-uv", action="store_true", help="Skip UV in output OBJs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from mammal_ext.fitting.interpolation import KeyframeInterpolator

    interp = KeyframeInterpolator(args.params_dir, device=args.device)

    # Optionally thin keyframes to simulate sparse fitting
    if args.keyframe_interval and args.keyframe_interval > 1:
        original_count = len(interp.sorted_ids)
        kept = interp.sorted_ids[::args.keyframe_interval]
        removed = set(interp.sorted_ids) - set(kept)
        for fid in removed:
            del interp.keyframes[fid]
        interp.sorted_ids = sorted(interp.keyframes.keys())
        print(f"Thinned keyframes: {original_count} → {len(interp.sorted_ids)} (interval={args.keyframe_interval})")

    vertices = interp.interpolate_range(args.start, args.end, args.step, args.method)
    interp.export_objs(vertices, args.output, with_uv=not args.no_uv)


if __name__ == "__main__":
    main()
