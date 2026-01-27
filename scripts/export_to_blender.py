#!/usr/bin/env python3
"""
Export mesh + UV texture for Blender visualization.

Thin wrapper around mammal_ext.blender_export module.
For batch export, use: python -m mammal_ext.blender_export.batch_export
For 6-view grid video, use: python -m mammal_ext.blender_export.sequence_renderer

Usage:
    python scripts/export_to_blender.py \
        --mesh results/fitting/.../obj/step_2_frame_000000.obj \
        --texture wandb_sweep_results/run_xxx/texture_final.png \
        --output exports/mouse_textured.obj

    # Keep MAMMAL coordinates (no transform)
    python scripts/export_to_blender.py \
        --mesh ... --texture ... --output ... --no_transform
"""

import argparse

from mammal_ext.blender_export.obj_exporter import export_single_frame


def main():
    parser = argparse.ArgumentParser(
        description="Export mesh with UV texture for Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export (auto: MAMMAL->Blender coords, center, mm->meters)
  python scripts/export_to_blender.py \\
      --mesh results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254/obj/step_2_frame_000000.obj \\
      --texture wandb_sweep_results/run_ancient-sweep-34/texture_final.png \\
      --output exports/mouse_frame0.obj

  # Keep MAMMAL coordinates (no transform)
  python scripts/export_to_blender.py \\
      --mesh ... --texture ... --output ... \\
      --no_transform --no_center --no_scale

  # Batch export all frames (use dedicated module)
  python -m mammal_ext.blender_export.batch_export \\
      --result_dir results/fitting/<experiment> \\
      --texture texture_final.png \\
      --output_dir exports/

  # 6-view grid video (use dedicated module)
  python -m mammal_ext.blender_export.sequence_renderer \\
      --result_dir results/fitting/<experiment> \\
      --texture texture_final.png \\
      --output_dir exports/renders/

Blender Import:
  1. File > Import > Wavefront (.obj)
  2. Select the exported .obj file
  3. Texture should auto-load from .mtl reference
  4. If not: Material Properties > Base Color > Image Texture > Select PNG
        """)

    parser.add_argument('--mesh', type=str, required=True,
                       help='Input OBJ mesh file')
    parser.add_argument('--texture', type=str, required=True,
                       help='UV texture PNG file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output OBJ file path')
    parser.add_argument('--model_dir', type=str, default='mouse_model/mouse_txt',
                       help='Model directory with UV definitions')
    parser.add_argument('--no_transform', action='store_true',
                       help='Skip coordinate transform (keep MAMMAL coords)')
    parser.add_argument('--no_center', action='store_true',
                       help='Skip centering at origin')
    parser.add_argument('--no_scale', action='store_true',
                       help='Skip mm-to-meters conversion')

    args = parser.parse_args()

    transform = "none" if args.no_transform else "mammal_to_blender"

    output_path = export_single_frame(
        mesh_path=args.mesh,
        texture_path=args.texture,
        output_path=args.output,
        model_dir=args.model_dir,
        transform=transform,
        center=not args.no_center,
        scale_to_meters=not args.no_scale,
    )

    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print(f"\nBlender import:")
    print(f"  File > Import > Wavefront (.obj) > Select {args.output}")


if __name__ == '__main__':
    main()
