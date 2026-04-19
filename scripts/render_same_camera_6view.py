#!/usr/bin/env python3
"""Render mesh from same 6 cameras as GT capture → GT vs render comparison.

Unlike novel_view_render.py (which uses v3 6-view novel cameras), this uses
the actual 6 data cameras (new_cam.pkl) for direct GT alignment.

Usage:
    python scripts/render_same_camera_6view.py --frame 2700 \
        --obj results/fitting/production_3600_canon/obj/step_2_frame_002700.obj \
        --output results/belly_dent_investigation/frame_002700_6view/
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


def load_obj(p):
    verts, faces = [], []
    with open(p) as fh:
        for ln in fh:
            if ln.startswith("v "):
                parts = ln.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif ln.startswith("f "):
                parts = ln.split()[1:]
                idx = [int(t.split("/")[0]) - 1 for t in parts[:3]]
                faces.append(idx)
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def compute_vertex_colors_from_uv(faces, n_verts, textures_path, faces_tex_path, texture_img_path):
    """Same as novel_view_render.py:compute_vertex_colors_from_texture."""
    uv_coords = np.loadtxt(textures_path)
    faces_tex = np.loadtxt(faces_tex_path, dtype=np.int64)
    tex_bgr = cv2.imread(texture_img_path)
    if tex_bgr is None:
        raise FileNotFoundError(f"texture image not found: {texture_img_path}")
    tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    vuv = np.zeros((n_verts, 2), dtype=np.float32)
    vcnt = np.zeros(n_verts, dtype=np.int32)
    for fi in range(len(faces)):
        for i in range(3):
            vid = faces[fi, i]
            if vid < 0 or vid >= n_verts:
                continue
            tid = faces_tex[fi, i]
            vuv[vid] += uv_coords[tid]
            vcnt[vid] += 1
    valid = vcnt > 0
    vuv[valid] /= vcnt[valid, None]
    H, W = tex_rgb.shape[:2]
    px = np.clip((vuv[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip(((1 - vuv[:, 1]) * H).astype(int), 0, H - 1)
    rgb = (tex_rgb[py, px] * 255).astype(np.uint8)
    colors = np.concatenate([rgb, np.full((n_verts, 1), 255, dtype=np.uint8)], axis=1)
    colors[~valid] = [180, 180, 180, 255]
    return colors


def render_with_cam(verts, faces, K, R, T, H, W, vert_colors_rgba=None):
    """Render mesh with OpenCV-style camera (K, R, T world→cam)."""
    import pyrender
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if vert_colors_rgba is not None:
        mesh.visual.vertex_colors = vert_colors_rgba
    else:
        mesh.visual.vertex_colors = np.tile([180, 160, 140, 255], (verts.shape[0], 1))

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pr_mesh)

    # Convert OpenCV extrinsic (K, R, T) → pyrender GL pose.
    # OpenCV: x_cam = R x_world + T. c2w_cv = inv([R|T]) = [R^T | -R^T @ T]
    R = np.asarray(R); T = np.asarray(T).reshape(-1)
    Rwc = R.T
    twc = -Rwc @ T
    c2w_cv = np.eye(4); c2w_cv[:3, :3] = Rwc; c2w_cv[:3, 3] = twc
    # CV → GL: flip y, z
    c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])

    # Intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=1.0, zfar=10000.0)
    scene.add(cam, pose=c2w_gl)

    # Lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light, pose=c2w_gl)
    r = pyrender.OffscreenRenderer(W, H)
    color, _ = r.render(scene)
    r.delete()
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--texture-img",
                    default="results/sweep/production_p0/texture_final.png")
    ap.add_argument("--textures-txt", default="mouse_model/mouse_txt/textures.txt")
    ap.add_argument("--faces-tex-txt", default="mouse_model/mouse_txt/faces_tex.txt")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Load cameras + mesh
    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)
    verts, faces = load_obj(args.obj)

    # Compute vertex colors from production P0 UV texture
    print(f"[texture] loading {args.texture_img}")
    vcolors = compute_vertex_colors_from_uv(faces, len(verts),
                                            args.textures_txt, args.faces_tex_txt,
                                            args.texture_img)

    # Render from each camera
    renders = []
    gts = []
    for vid in range(6):
        cap = cv2.VideoCapture(os.path.join(args.data_dir, "videos_undist", f"{vid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, gt_bgr = cap.read(); cap.release()
        if not ok:
            print(f"view {vid}: GT read failed")
            continue
        H, W = gt_bgr.shape[:2]
        K = cams[vid]["K"]; R = cams[vid]["R"]; T = cams[vid]["T"]
        img = render_with_cam(verts, faces, K, R, T, H, W,
                              vert_colors_rgba=vcolors)
        # Save individual
        cv2.imwrite(str(out / f"gt_v{vid}.png"), gt_bgr)
        cv2.imwrite(str(out / f"render_v{vid}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        gts.append(gt_bgr)
        renders.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"view {vid}: rendered ({W}x{H})")

    # Grid: 2 rows × 6 cols (GT top, render bottom)
    # Resize all to consistent 400x300
    tW, tH = 400, 300
    bar = 28
    grid = np.full((2 * (tH + bar), 6 * tW, 3), 255, np.uint8)
    for vid in range(6):
        if vid < len(gts):
            g = cv2.resize(gts[vid], (tW, tH))
            r = cv2.resize(renders[vid], (tW, tH))
            # Row 1: GT
            cv2.putText(grid, f"GT v{vid}", (vid * tW + 8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
            grid[bar:bar + tH, vid * tW:(vid + 1) * tW] = g
            # Row 2: Render
            y2 = tH + bar
            cv2.putText(grid, f"Render v{vid}", (vid * tW + 8, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
            grid[y2 + bar:y2 + bar + tH, vid * tW:(vid + 1) * tW] = r

    gp = out / f"frame_{args.frame:06d}_6view_grid.png"
    cv2.imwrite(str(gp), grid)
    print(f"\nsaved grid: {gp}")


if __name__ == "__main__":
    sys.exit(main())
