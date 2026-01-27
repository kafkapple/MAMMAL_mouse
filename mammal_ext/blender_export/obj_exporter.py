"""
OBJ File Export Utilities

Create OBJ + MTL files with UV texture mapping for Blender import.
"""

import os
import shutil
import numpy as np
from typing import Optional


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


def export_obj_with_uv(
    vertices: np.ndarray,
    uv_coords: np.ndarray,
    faces_vert: np.ndarray,
    faces_tex: np.ndarray,
    output_path: str,
    mtl_filename: Optional[str] = None,
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
        f.write("# Exported for Blender visualization\n")
        f.write(f"# Vertices: {len(vertices)}, UV coords: {len(uv_coords)}, Faces: {len(faces_vert)}\n\n")

        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n\n")

        f.write("# Vertices\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")

        f.write("# Texture coordinates\n")
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("\n")

        if mtl_filename:
            f.write("usemtl mouse_material\n\n")

        f.write("# Faces (v/vt format)\n")
        for fv, ft in zip(faces_vert, faces_tex):
            v1, v2, v3 = fv[0] + 1, fv[1] + 1, fv[2] + 1
            t1, t2, t3 = ft[0] + 1, ft[1] + 1, ft[2] + 1
            f.write(f"f {v1}/{t1} {v2}/{t2} {v3}/{t3}\n")


def export_single_frame(
    mesh_path: str,
    texture_path: str,
    output_path: str,
    model_dir: str = "mouse_model/mouse_txt",
    transform: str = "mammal_to_blender",
    center: bool = True,
    scale_to_meters: bool = True,
):
    """
    Export a single frame mesh with texture for Blender.

    Args:
        mesh_path: Input OBJ mesh file
        texture_path: UV texture PNG
        output_path: Output OBJ path
        model_dir: Body model UV data directory
        transform: Coordinate transform to apply
        center: Center at origin
        scale_to_meters: Convert mm to meters
    """
    from .coordinate_transform import transform_vertices

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load and transform vertices
    vertices = parse_obj_vertices(mesh_path)
    vertices = transform_vertices(
        vertices,
        transform=transform,
        center=center,
        scale_to_meters=scale_to_meters,
    )

    # Load UV data
    uv_coords = load_uv_coordinates(model_dir)
    faces_vert = load_faces_vert(model_dir)
    faces_tex = load_faces_tex(model_dir)

    # Output paths
    obj_basename = os.path.basename(output_path)
    mtl_basename = obj_basename.replace('.obj', '.mtl')
    mtl_path = output_path.replace('.obj', '.mtl')

    # Copy texture
    texture_basename = os.path.basename(texture_path)
    texture_dest = os.path.join(output_dir, texture_basename) if output_dir else texture_basename
    if os.path.abspath(texture_path) != os.path.abspath(texture_dest):
        shutil.copy(texture_path, texture_dest)

    # Create MTL
    create_mtl_file(mtl_path, texture_basename)

    # Export OBJ
    export_obj_with_uv(
        vertices=vertices,
        uv_coords=uv_coords,
        faces_vert=faces_vert,
        faces_tex=faces_tex,
        output_path=output_path,
        mtl_filename=mtl_basename,
    )

    return output_path
