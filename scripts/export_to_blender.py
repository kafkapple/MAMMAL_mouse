#!/usr/bin/env python3
"""
Export mesh + UV texture for Blender visualization.

Generates OBJ file with UV coordinates and MTL material file.

Usage:
    python scripts/export_to_blender.py \
        --mesh results/fitting/.../obj/step_2_frame_000000.obj \
        --texture wandb_sweep_results/run_xxx/texture_final.png \
        --output exports/mouse_textured.obj
"""

import os
import argparse
import numpy as np
import shutil


def load_uv_coordinates(model_dir: str = "mouse_model/mouse_txt") -> np.ndarray:
    """Load UV coordinates from textures.txt"""
    uv_path = os.path.join(model_dir, "textures.txt")
    return np.loadtxt(uv_path)


def load_faces_tex(model_dir: str = "mouse_model/mouse_txt") -> np.ndarray:
    """Load texture face indices from faces_tex.txt"""
    faces_path = os.path.join(model_dir, "faces_tex.txt")
    return np.loadtxt(faces_path, dtype=np.int32)


def load_faces_vert(model_dir: str = "mouse_model/mouse_txt") -> np.ndarray:
    """Load vertex face indices from faces_vert.txt"""
    faces_path = os.path.join(model_dir, "faces_vert.txt")
    return np.loadtxt(faces_path, dtype=np.int32)


def parse_obj_vertices(obj_path: str) -> np.ndarray:
    """Extract vertices from OBJ file"""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)


def create_mtl_file(mtl_path: str, texture_filename: str):
    """Create MTL material file"""
    mtl_content = f"""# Blender MTL File
# Material for mouse mesh with UV texture

newmtl mouse_material
Ns 225.000000
Ka 1.000000 1.000000 1.000000
Kd 0.800000 0.800000 0.800000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
map_Kd {texture_filename}
"""
    with open(mtl_path, 'w') as f:
        f.write(mtl_content)
    print(f"Created MTL: {mtl_path}")


def export_obj_with_uv(
    vertices: np.ndarray,
    uv_coords: np.ndarray,
    faces_vert: np.ndarray,
    faces_tex: np.ndarray,
    output_path: str,
    mtl_filename: str = None,
):
    """
    Export OBJ file with UV coordinates.

    Args:
        vertices: (N, 3) vertex positions
        uv_coords: (M, 2) UV coordinates
        faces_vert: (F, 3) vertex indices for faces (0-indexed)
        faces_tex: (F, 3) texture indices for faces (0-indexed)
        output_path: Output OBJ file path
        mtl_filename: Optional MTL filename to reference
    """
    with open(output_path, 'w') as f:
        # Header
        f.write("# Exported for Blender visualization\n")
        f.write(f"# Vertices: {len(vertices)}, UV coords: {len(uv_coords)}, Faces: {len(faces_vert)}\n")
        f.write("\n")

        # MTL reference
        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n\n")

        # Vertices
        f.write("# Vertices\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")

        # UV coordinates
        f.write("# Texture coordinates\n")
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("\n")

        # Use material
        if mtl_filename:
            f.write("usemtl mouse_material\n\n")

        # Faces (OBJ uses 1-indexed)
        f.write("# Faces (v/vt format)\n")
        for fv, ft in zip(faces_vert, faces_tex):
            # OBJ is 1-indexed
            v1, v2, v3 = fv[0] + 1, fv[1] + 1, fv[2] + 1
            t1, t2, t3 = ft[0] + 1, ft[1] + 1, ft[2] + 1
            f.write(f"f {v1}/{t1} {v2}/{t2} {v3}/{t3}\n")

    print(f"Created OBJ: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export mesh with UV texture for Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python scripts/export_to_blender.py \\
      --mesh results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254/obj/step_2_frame_000000.obj \\
      --texture wandb_sweep_results/run_ancient-sweep-34/texture_final.png \\
      --output exports/mouse_frame0.obj

  # Export multiple frames
  for i in $(seq 0 9); do
      python scripts/export_to_blender.py \\
          --mesh results/.../obj/step_2_frame_00000${i}.obj \\
          --texture texture_final.png \\
          --output exports/mouse_frame${i}.obj
  done

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

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading mesh vertices...")
    vertices = parse_obj_vertices(args.mesh)
    print(f"  Loaded {len(vertices)} vertices")

    print("Loading UV coordinates...")
    uv_coords = load_uv_coordinates(args.model_dir)
    print(f"  Loaded {len(uv_coords)} UV coordinates")

    print("Loading face indices...")
    faces_vert = load_faces_vert(args.model_dir)
    faces_tex = load_faces_tex(args.model_dir)
    print(f"  Loaded {len(faces_vert)} faces")

    # Output paths
    obj_basename = os.path.basename(args.output)
    mtl_basename = obj_basename.replace('.obj', '.mtl')
    mtl_path = args.output.replace('.obj', '.mtl')

    # Copy texture to output directory
    texture_basename = os.path.basename(args.texture)
    texture_dest = os.path.join(output_dir, texture_basename) if output_dir else texture_basename
    if args.texture != texture_dest:
        shutil.copy(args.texture, texture_dest)
        print(f"Copied texture: {texture_dest}")

    # Create MTL file
    create_mtl_file(mtl_path, texture_basename)

    # Export OBJ with UV
    export_obj_with_uv(
        vertices=vertices,
        uv_coords=uv_coords,
        faces_vert=faces_vert,
        faces_tex=faces_tex,
        output_path=args.output,
        mtl_filename=mtl_basename,
    )

    print("\n" + "="*50)
    print("Export complete!")
    print("="*50)
    print(f"\nFiles created:")
    print(f"  - {args.output}")
    print(f"  - {mtl_path}")
    print(f"  - {texture_dest}")
    print(f"\nBlender import:")
    print(f"  File > Import > Wavefront (.obj) > Select {obj_basename}")


if __name__ == '__main__':
    main()
