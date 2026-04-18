#!/usr/bin/env python3
"""P2.5: Occlusion-aware direct vertex color from multi-view GT.

Improves P2 (which had bg bleed → ΔE 100+) by adding SimpleClick foreground
mask check: a vertex only samples color from a view if its projection lands
on a foreground pixel (mouse body), not background.

Optionally uses depth test via mesh rasterization (pytorch3d or pyrender) for
stricter occlusion handling. First version uses SimpleClick mask only.

Outputs:
    results/texture_experiment_v1/p25_occlusion/
        p25_vertex_colors.npy  (14522×4 RGBA)
        p25_{Top,Right,Bottom,Front-high}.png
        p25_stats.json  (per-vertex valid-view count)
"""
import argparse
import json
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


def p25_occlusion_aware_vertex_color(obj_path, data_dir, frame_id, views=range(6),
                                     mask_threshold=127):
    """Sample per-vertex color with SimpleClick foreground mask check."""
    verts, faces = load_obj(obj_path)
    n_verts = len(verts)

    with open(os.path.join(data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)

    vertex_colors_per_view = np.full((6, n_verts, 3), 128, dtype=np.uint8)
    valid_mask = np.zeros((6, n_verts), dtype=bool)
    n_rejected_bg = np.zeros(6, dtype=int)
    n_rejected_oob = np.zeros(6, dtype=int)

    for vid in views:
        rgb_path = os.path.join(data_dir, "videos_undist", f"{vid}.mp4")
        mask_path = os.path.join(data_dir, "simpleclick_undist", f"{vid}.mp4")
        cap_rgb = cv2.VideoCapture(rgb_path); cap_mk = cv2.VideoCapture(mask_path)
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, frame_id); cap_mk.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok1, img_bgr = cap_rgb.read(); ok2, mask_bgr = cap_mk.read()
        cap_rgb.release(); cap_mk.release()
        if not (ok1 and ok2):
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fg_mask = mask_bgr[:, :, 0] > mask_threshold  # foreground
        H, W = img_rgb.shape[:2]

        K = cams[vid]["K"]; R = cams[vid]["R"]; T = cams[vid]["T"].reshape(-1)
        Pc = (R @ verts.T).T + T
        depth = Pc[:, 2]
        p2h = (K @ Pc.T).T
        uv = p2h[:, :2] / (p2h[:, 2:] + 1e-8)
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)
        n_rejected_oob[vid] = int((~in_img).sum())

        # Occlusion filter: SimpleClick foreground mask at projected pixel
        valid = in_img.copy()
        valid[in_img] = fg_mask[v[in_img], u[in_img]]
        n_rejected_bg[vid] = int(in_img.sum() - valid.sum())

        valid_mask[vid] = valid
        vertex_colors_per_view[vid, valid] = img_rgb[v[valid], u[valid]]

    # Median across valid views
    vert_colors = np.full((n_verts, 3), 128, dtype=np.uint8)
    for vi in range(n_verts):
        valid_views = [vertex_colors_per_view[v, vi] for v in range(6) if valid_mask[v, vi]]
        if len(valid_views) >= 2:
            vert_colors[vi] = np.median(np.stack(valid_views), axis=0).astype(np.uint8)

    vert_colors_rgba = np.concatenate(
        [vert_colors, np.full((n_verts, 1), 255, dtype=np.uint8)], axis=1
    )
    stats = {
        "n_verts": n_verts,
        "per_view_rejected_oob": n_rejected_oob.tolist(),
        "per_view_rejected_bg_via_mask": n_rejected_bg.tolist(),
        "per_view_valid_count": valid_mask.sum(axis=1).tolist(),
        "n_verts_with_0_valid_views": int((valid_mask.sum(axis=0) == 0).sum()),
        "n_verts_with_ge2_valid_views": int((valid_mask.sum(axis=0) >= 2).sum()),
    }
    return verts, faces, vert_colors_rgba, stats


def render_mesh_colored(verts_gslrm, faces, vcolors_rgba, cam_elev, cam_azim, w=512, h=512):
    import pyrender
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts_gslrm, faces=faces, process=False,
                           vertex_colors=vcolors_rgba)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pr_mesh)
    elev = np.deg2rad(cam_elev); azim = np.deg2rad(cam_azim)
    radius = 2.7
    z = radius * np.sin(elev); base = radius * np.cos(elev)
    x = base * np.cos(azim); y = base * np.sin(azim)
    cam_pos = np.array([x, y, z])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    R_cv = np.stack((right, -up, forward), axis=1)
    c2w_cv = np.eye(4); c2w_cv[:3, :3] = R_cv; c2w_cv[:3, 3] = cam_pos
    c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(50.0), aspectRatio=w / h)
    scene.add(cam, pose=c2w_gl)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=c2w_gl)
    r = pyrender.OffscreenRenderer(w, h)
    color, _ = r.render(scene)
    r.delete()
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    print(f"[P2.5] occlusion-aware multi-view projection, frame {args.frame}")
    verts_mm, faces, vcolors, stats = p25_occlusion_aware_vertex_color(
        args.obj, args.data_dir, args.frame)
    print(f"  stats: {json.dumps(stats, indent=2)}")

    with open(out / "p25_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    np.save(out / "p25_vertex_colors.npy", vcolors)

    # Transform mm → GSLRM for render
    M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
    M5_DISTANCE_SCALE = 2.7 / 307.785
    verts_g = (verts_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE

    from PIL import Image
    views = [("Top", 80, 270), ("Right", 20, 0), ("Front-high", 40, 270), ("Bottom", -85, 270)]
    for name, elev, azim in views:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        img = render_mesh_colored(verts_g, faces, vcolors, elev, azim)
        Image.fromarray(img).save(out / f"p25_{name}.png")
        print(f"  rendered: p25_{name}.png")

    print(f"\nDone. Output: {out}/")


if __name__ == "__main__":
    sys.exit(main())
