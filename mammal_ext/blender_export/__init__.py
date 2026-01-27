"""
Blender Export Module for MAMMAL Mouse

Batch export mesh sequences with UV textures and render 6-view grids.

Usage:
    # Single frame export
    from mammal_ext.blender_export import export_obj_with_uv, transform_vertices

    # Batch export all frames
    python -m mammal_ext.blender_export.batch_export \
        --result_dir results/fitting/<experiment> \
        --texture texture_final.png \
        --output_dir exports/

    # 6-view grid video
    python -m mammal_ext.blender_export.sequence_renderer \
        --result_dir results/fitting/<experiment> \
        --texture texture_final.png \
        --output_dir exports/
"""

from .coordinate_transform import (
    MAMMAL_TO_BLENDER,
    transform_vertices,
)
from .obj_exporter import (
    export_obj_with_uv,
    create_mtl_file,
    parse_obj_vertices,
    load_uv_coordinates,
    load_faces_tex,
    load_faces_vert,
)

__all__ = [
    'MAMMAL_TO_BLENDER',
    'transform_vertices',
    'export_obj_with_uv',
    'create_mtl_file',
    'parse_obj_vertices',
    'load_uv_coordinates',
    'load_faces_tex',
    'load_faces_vert',
]
