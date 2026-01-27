"""
Run All: Batch OBJ export + 6-view grid video in one command.

Usage:
    python -m mammal_ext.blender_export.run_all \
        --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
        --texture results/sweep/run_wild-sweep-9/texture_final.png \
        --output_dir exports/v012345_kp22_20251206/

    # Skip OBJ export (grid video only)
    python -m mammal_ext.blender_export.run_all \
        --result_dir ... --texture ... --output_dir ... --skip_obj

    # Skip video (OBJ only)
    python -m mammal_ext.blender_export.run_all \
        --result_dir ... --texture ... --output_dir ... --skip_video
"""

import os
import argparse
import time

from .batch_export import batch_export
from .sequence_renderer import render_sequence


def run_all(
    result_dir: str,
    texture_path: str,
    output_dir: str,
    data_dir: str = "data/examples/markerless_mouse_1_nerf",
    model_dir: str = "mouse_model/mouse_txt",
    transform: str = "mammal_to_blender",
    center: bool = True,
    scale_to_meters: bool = True,
    image_size: int = 512,
    fps: int = 15,
    max_frames: int = None,
    skip_obj: bool = False,
    skip_video: bool = False,
):
    """
    Run full pipeline: batch OBJ export + 6-view grid video.

    Args:
        result_dir: Fitting result directory
        texture_path: UV texture PNG
        output_dir: Root output directory
        data_dir: Data directory with camera calibration
        model_dir: Body model UV data directory
        transform: Coordinate transform
        center: Center at origin
        scale_to_meters: Convert mm to meters
        image_size: Render resolution per view
        fps: Output video FPS
        max_frames: Limit number of frames
        skip_obj: Skip OBJ export
        skip_video: Skip video rendering
    """
    exp_name = os.path.basename(result_dir)
    obj_dir = os.path.join(output_dir, "obj")
    render_dir = os.path.join(output_dir, "renders")

    print("=" * 60)
    print(f"Blender Export Pipeline")
    print(f"  Experiment: {exp_name}")
    print(f"  Texture:    {texture_path}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)
    print()

    t0 = time.time()

    # Step 1: Batch OBJ export
    if not skip_obj:
        print("[Step 1/2] Batch OBJ export")
        print("-" * 40)
        batch_export(
            result_dir=result_dir,
            texture_path=texture_path,
            output_dir=obj_dir,
            model_dir=model_dir,
            transform=transform,
            center=center,
            scale_to_meters=scale_to_meters,
        )
        print()
    else:
        print("[Step 1/2] Skipped (--skip_obj)")
        print()

    # Step 2: 6-view grid video
    if not skip_video:
        print("[Step 2/2] 6-view grid video")
        print("-" * 40)
        render_sequence(
            result_dir=result_dir,
            texture_path=texture_path,
            output_dir=render_dir,
            data_dir=data_dir,
            model_dir=model_dir,
            image_size=image_size,
            fps=fps,
            max_frames=max_frames,
        )
        print()
    else:
        print("[Step 2/2] Skipped (--skip_video)")
        print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Pipeline complete! ({elapsed:.1f}s)")
    print("=" * 60)
    print(f"\nOutput:")
    if not skip_obj:
        print(f"  OBJ files: {obj_dir}/")
    if not skip_video:
        print(f"  Renders:   {render_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Run full Blender export pipeline (OBJ + 6-view grid video)",
    )
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Fitting result directory')
    parser.add_argument('--texture', type=str, required=True,
                       help='UV texture PNG')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Root output directory')
    parser.add_argument('--data_dir', type=str,
                       default='data/examples/markerless_mouse_1_nerf',
                       help='Data directory with camera calibration')
    parser.add_argument('--model_dir', type=str, default='mouse_model/mouse_txt',
                       help='Model directory with UV definitions')
    parser.add_argument('--no_transform', action='store_true',
                       help='Skip coordinate transform')
    parser.add_argument('--no_center', action='store_true',
                       help='Skip centering')
    parser.add_argument('--no_scale', action='store_true',
                       help='Skip mm-to-meters conversion')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Render resolution per view')
    parser.add_argument('--fps', type=int, default=15,
                       help='Output video FPS')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Limit number of frames')
    parser.add_argument('--skip_obj', action='store_true',
                       help='Skip OBJ export')
    parser.add_argument('--skip_video', action='store_true',
                       help='Skip video rendering')

    args = parser.parse_args()

    run_all(
        result_dir=args.result_dir,
        texture_path=args.texture,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        transform="none" if args.no_transform else "mammal_to_blender",
        center=not args.no_center,
        scale_to_meters=not args.no_scale,
        image_size=args.image_size,
        fps=args.fps,
        max_frames=args.max_frames,
        skip_obj=args.skip_obj,
        skip_video=args.skip_video,
    )


if __name__ == '__main__':
    main()
