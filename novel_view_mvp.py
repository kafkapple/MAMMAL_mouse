#!/usr/bin/env python3
"""[DEPRECATED — use `scripts/novel_view_render.py`]

This untextured MVP was superseded by `novel_view_render.py` (textured +
batch + video, 2026-04-17). Retained for MVP validation history. Remove
after batch run confirmed in production.

MVP: Render MAMMAL mesh from FaceLift v3 6-view (260306 reference).

Purpose
-------
Empirical validation of coord transform: MAMMAL mm → FaceLift gs-lrm
normalized → OpenGL camera. The v3 6-view is the user-approved reference
from `GoogleDrive:AMILab_my/_Results/FaceLift/260306_2nd_phase/_novel_view_rendering/novel_6view_temporal_v3/novel_6view_grid.png`.

Transform stages (per Fact-check C4 finding in 260417 audit):
  1. `mammal_to_gslrm(verts)` — translate + scale only (no axis swap)
  2. Camera extrinsics: `get_turntable_cameras()` returns OpenCV c2w
  3. OpenGL flip for pyrender: `c2w_gl = c2w_cv @ diag(1, -1, -1, 1)` —
     applied to CAMERA, not mesh

v3 camera config (from 260306 grid image labels, user-confirmed):
  Row 1: Top(e=+80,a=270), Front-high(e=+40,a=270), Right(e=+20,a=0)
  Row 2: Bottom(e=-85,a=270), Back-high(e=+40,a=90), Left(e=+20,a=180)

Usage
-----
    # Single frame, single OBJ test
    CUDA_VISIBLE_DEVICES=4 python scripts/novel_view_mvp.py \
        --obj results/fitting/production_3600_canon/obj/step_2_frame_001800.obj \
        --output results/novel_view_mvp/frame_001800

The script is a manual validation tool, not a pipeline. Output:
  - per-view PNG (6 files)
  - 2×3 grid PNG
  - extrinsics.json with all 6 c2w + intrinsics
"""

import argparse
import json
import os
import sys

import numpy as np


V3_NOVEL_6VIEWS = [
    # (name, elevation_deg, azimuth_deg, grid_row, grid_col)
    ("Top",        +80.0, 270.0, 0, 0),
    ("Front-high", +40.0, 270.0, 0, 1),
    ("Right",      +20.0,   0.0, 0, 2),
    ("Bottom",     -85.0, 270.0, 1, 0),
    ("Back-high",  +40.0,  90.0, 1, 1),
    ("Left",       +20.0, 180.0, 1, 2),
]

# FaceLift M5 canonical (from coordinate_utils.py)
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])  # mm
M5_DISTANCE_SCALE = 2.7 / 307.785  # ≈ 0.008781

# pyrender: OpenGL convention (Y-up, Z-back)
# Received c2w is OpenCV (Y-down, Z-forward) → flip
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def mammal_to_gslrm(xyz_mm: np.ndarray) -> np.ndarray:
    """MAMMAL mm → FaceLift GS-LRM normalized. Translate + scale only."""
    return (xyz_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def load_obj_verts_faces(path: str):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                p = line.split()[1:]
                idx = [int(t.split("/")[0]) - 1 for t in p[:3]]
                faces.append(idx)
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def spherical_c2w_opencv(elev_deg: float, azim_deg: float,
                          radius: float = 2.7,
                          center: np.ndarray = None,
                          up_vector: np.ndarray = None) -> np.ndarray:
    """Reimplementation of FaceLift get_turntable_cameras single view.

    Returns OpenCV c2w 4x4 matrix (X-right, Y-down, Z-forward).
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    if up_vector is None:
        up_vector = np.array([0.0, 0.0, 1.0])

    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    z = radius * np.sin(elev)
    base = radius * np.cos(elev)
    x = base * np.cos(azim)
    y = base * np.sin(azim)
    cam_pos = np.array([x, y, z]) + center

    forward = center - cam_pos
    forward = forward / np.linalg.norm(forward)
    # Degenerate: forward // up
    right = np.cross(forward, up_vector)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        fallback_up = np.array([0.0, 1.0, 0.0]) if abs(up_vector[2]) > 0.5 else np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, fallback_up)
        rn = np.linalg.norm(right)
    right = right / rn
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    R = np.stack((right, -up, forward), axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos
    return c2w


def render_view(verts_gslrm: np.ndarray, faces: np.ndarray,
                c2w_cv: np.ndarray, w: int, h: int, fov_deg: float = 50.0,
                bg=(1.0, 1.0, 1.0)) -> np.ndarray:
    import pyrender
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts_gslrm, faces=faces, process=False)
    # Per-vertex gray color so mesh visible without texture
    if mesh.visual.vertex_colors is None or len(mesh.visual.vertex_colors) == 0:
        mesh.visual.vertex_colors = np.full((len(verts_gslrm), 4), [180, 180, 180, 255], dtype=np.uint8)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    scene = pyrender.Scene(bg_color=list(bg) + [1.0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pr_mesh)

    # Camera: pyrender expects OpenGL c2w
    c2w_gl = c2w_cv @ CV_TO_GL
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov_deg), aspectRatio=w/h)
    scene.add(cam, pose=c2w_gl)

    # Directional light along camera forward
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=c2w_gl)

    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, _ = r.render(scene)
    r.delete()
    return color  # (H, W, 3) uint8


def make_grid_2x3(images, labels, tile_w=None, tile_h=None):
    import cv2
    if tile_w is None: tile_w = images[0].shape[1]
    if tile_h is None: tile_h = images[0].shape[0]
    cols, rows = 3, 2
    canvas = np.full((tile_h * rows, tile_w * cols, 3), 255, dtype=np.uint8)
    for idx, (img, label) in enumerate(zip(images, labels)):
        r, c = idx // cols, idx % cols
        y0, x0 = r * tile_h, c * tile_w
        canvas[y0:y0+tile_h, x0:x0+tile_w] = img
        # Label
        cv2.putText(canvas, label, (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--radius", type=float, default=2.7)
    ap.add_argument("--fov", type=float, default=50.0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading: {args.obj}")
    verts_mm, faces = load_obj_verts_faces(args.obj)
    print(f"  verts={verts_mm.shape}, faces={faces.shape}")
    print(f"  MAMMAL bounds: min={verts_mm.min(0)}, max={verts_mm.max(0)}")
    print(f"  centroid={verts_mm.mean(0)}")

    verts_g = mammal_to_gslrm(verts_mm)
    print(f"  GSLRM bounds: min={verts_g.min(0)}, max={verts_g.max(0)}")
    print(f"  centroid={verts_g.mean(0)}  (should be near origin if center correct)")

    cameras = {}
    images = []
    labels = []
    for name, elev, azim, row, col in V3_NOVEL_6VIEWS:
        c2w = spherical_c2w_opencv(elev, azim, radius=args.radius)
        cameras[name] = {
            "elevation_deg": elev, "azimuth_deg": azim,
            "c2w_opencv": c2w.tolist(),
            "radius": args.radius, "fov_deg": args.fov,
            "w": args.w, "h": args.h,
        }
        print(f"  [{name}] e={elev:+.0f}° a={azim:+.0f}° cam_pos={c2w[:3,3]}")
        img = render_view(verts_g, faces, c2w, args.w, args.h, fov_deg=args.fov)
        import imageio.v3 as iio
        iio.imwrite(os.path.join(args.output, f"{name}.png"), img)
        images.append(img)
        labels.append(f"{name} (e={elev:+.0f} a={azim:+.0f})")

    grid = make_grid_2x3(images, labels)
    import imageio.v3 as iio
    iio.imwrite(os.path.join(args.output, "grid_2x3.png"), grid)

    # Metadata
    metadata = {
        "source_obj": args.obj,
        "transform_pipeline": "mammal_to_gslrm(verts) + c2w_opencv → c2w_gl (diag(1,-1,-1,1))",
        "M5_SCENE_CENTER_mm": M5_SCENE_CENTER.tolist(),
        "M5_DISTANCE_SCALE": float(M5_DISTANCE_SCALE),
        "gslrm_bounds": {"min": verts_g.min(0).tolist(), "max": verts_g.max(0).tolist()},
        "cameras": cameras,
        "render": {"w": args.w, "h": args.h, "fov_deg": args.fov, "radius": args.radius},
    }
    with open(os.path.join(args.output, "extrinsics.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\nSaved to {args.output}/")
    print("  grid_2x3.png + 6 per-view PNGs + extrinsics.json")


if __name__ == "__main__":
    main()
