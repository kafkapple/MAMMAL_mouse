"""
Batch Export: Export all frames from a fitting result as textured OBJ for Blender.

Usage:
    python -m mammal_ext.blender_export.batch_export \
        --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356 \
        --texture results/sweep/run_wild-sweep-9/texture_final.png \
        --output_dir exports/sequence/

    # Keep MAMMAL coordinates (no transform)
    python -m mammal_ext.blender_export.batch_export \
        --result_dir ... --texture ... --output_dir ... --no_transform
"""

import os
import glob
import argparse
from typing import List

from .obj_exporter import export_single_frame


def find_obj_frames(result_dir: str) -> List[str]:
    """Find all step_2 OBJ files sorted by frame number."""
    obj_dir = os.path.join(result_dir, "obj")
    pattern = os.path.join(obj_dir, "step_2_frame_*.obj")
    files = sorted(glob.glob(pattern))
    return files


def batch_export(
    result_dir: str,
    texture_path: str,
    output_dir: str,
    model_dir: str = "mouse_model/mouse_txt",
    transform: str = "mammal_to_blender",
    center: bool = True,
    scale_to_meters: bool = True,
) -> List[str]:
    """
    Export all frames from a fitting result as textured OBJ.

    Args:
        result_dir: Fitting result directory (contains obj/ subdirectory)
        texture_path: UV texture PNG
        output_dir: Output directory for exported OBJ files
        model_dir: Body model UV data directory
        transform: Coordinate transform ("mammal_to_blender" or "none")
        center: Center mesh at origin
        scale_to_meters: Convert mm to meters

    Returns:
        List of exported OBJ file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    obj_files = find_obj_frames(result_dir)
    if not obj_files:
        raise FileNotFoundError(f"No step_2_frame_*.obj found in {result_dir}/obj/")

    print(f"Found {len(obj_files)} frames in {result_dir}/obj/")
    print(f"Texture: {texture_path}")
    print(f"Output: {output_dir}")
    print(f"Transform: {transform}, center={center}, scale_to_m={scale_to_meters}")
    print()

    exported = []
    for i, obj_path in enumerate(obj_files):
        basename = os.path.basename(obj_path)
        output_path = os.path.join(output_dir, basename)

        export_single_frame(
            mesh_path=obj_path,
            texture_path=texture_path,
            output_path=output_path,
            model_dir=model_dir,
            transform=transform,
            center=center,
            scale_to_meters=scale_to_meters,
        )

        exported.append(output_path)
        if (i + 1) % 50 == 0 or (i + 1) == len(obj_files):
            print(f"  [{i+1}/{len(obj_files)}] {basename}")

    print(f"\nExported {len(exported)} frames to {output_dir}")
    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Batch export all frames as textured OBJ for Blender",
    )
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Fitting result directory')
    parser.add_argument('--texture', type=str, required=True,
                       help='UV texture PNG')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--model_dir', type=str, default='mouse_model/mouse_txt',
                       help='Model directory with UV definitions')
    parser.add_argument('--no_transform', action='store_true',
                       help='Skip coordinate transform')
    parser.add_argument('--no_center', action='store_true',
                       help='Skip centering')
    parser.add_argument('--no_scale', action='store_true',
                       help='Skip mm-to-meters conversion')

    args = parser.parse_args()

    batch_export(
        result_dir=args.result_dir,
        texture_path=args.texture,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        transform="none" if args.no_transform else "mammal_to_blender",
        center=not args.no_center,
        scale_to_meters=not args.no_scale,
    )


if __name__ == '__main__':
    main()
