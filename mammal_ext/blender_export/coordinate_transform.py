"""
Coordinate System Transforms

MAMMAL (-Y up, +X forward, +Z right) -> Blender World (+Z up, +Y forward, +X right)

See docs/coordinates/coordinate_systems_reference.md for details.
"""

import numpy as np


# Rx(+90°): (x, y, z)_MAMMAL -> (x, z, -y)_Blender
MAMMAL_TO_BLENDER = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=np.float64)


def transform_vertices(
    vertices: np.ndarray,
    transform: str = "mammal_to_blender",
    center: bool = True,
    scale_to_meters: bool = True,
) -> np.ndarray:
    """
    Transform vertices from MAMMAL coordinate system to target.

    Args:
        vertices: (N, 3) vertex positions in MAMMAL coords (mm)
        transform: "mammal_to_blender" or "none"
        center: Center mesh at origin before transform
        scale_to_meters: Convert mm -> meters (MAMMAL uses mm)

    Returns:
        Transformed vertices (N, 3)
    """
    v = vertices.copy()

    if center:
        v -= v.mean(axis=0)

    if scale_to_meters:
        v *= 0.001  # mm -> meters

    if transform == "mammal_to_blender":
        v = v @ MAMMAL_TO_BLENDER.T
    elif transform == "none":
        pass
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return v
