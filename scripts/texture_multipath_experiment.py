#!/usr/bin/env python3
"""P0 (gamma correction) + P2 (direct vertex color) texture experiments in parallel.

P0: Load sweep-9 texture, apply gamma darkening + histogram match to GT RGB → variant PNG
P2: Project multi-view GT RGB onto mesh → per-vertex median color → apply directly (no UV)

Compare: render each variant from v3 top view (same frame 1800) → side-by-side grid.

Usage:
    python scripts/texture_multipath_experiment.py \
        --obj results/fitting/production_3600_canon/obj/step_2_frame_001800.obj \
        --output results/texture_experiment_v1/
"""
import argparse
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def p0_gamma_correct(tex_path, gamma=2.2, hist_match_gt=None):
    """Apply gamma darkening + optional histogram matching."""
    tex = cv2.imread(tex_path)
    tex_rgb = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Keep white background (UV mask) — only darken non-white pixels
    mask_nonwhite = np.any(tex_rgb < 0.95, axis=-1)
    tex_gamma = np.power(tex_rgb, gamma)

    # Preserve bg white where UV is empty (avoid darkening fg pixels only)
    out = tex_rgb.copy()
    out[mask_nonwhite] = tex_gamma[mask_nonwhite]

    # Histogram match to GT RGB distribution if provided
    if hist_match_gt is not None:
        gt = cv2.imread(hist_match_gt)
        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        # Extract mouse-region pixels (exclude white background)
        h, w = gt_rgb.shape[:2]
        gt_gray = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY)
        gt_fg = gt_rgb[gt_gray < 100]  # dark pixels only = mouse body
        if len(gt_fg) > 100:
            gt_mean_rgb = gt_fg.mean(axis=0) / 255.0
            # Scale dark pixels of texture toward GT mean
            fg_mask = mask_nonwhite
            out_fg = out[fg_mask]
            current_mean = out_fg.mean(axis=0) + 1e-6
            scale = gt_mean_rgb / current_mean
            out[fg_mask] = np.clip(out_fg * scale, 0, 1)

    out_bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr


def p2_direct_vertex_color(obj_path, data_dir, frame_id, params_pkl_path=None):
    """Project multi-view GT RGB onto mesh vertices → per-vertex median color."""
    import pickle

    # Load mesh
    verts, faces = [], []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                p = line.split()[1:]
                idx = [int(t.split("/")[0]) - 1 for t in p[:3]]
                faces.append(idx)
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # Load cameras
    with open(os.path.join(data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)

    # Load GT RGB frames (6 views)
    n_verts = len(verts)
    vertex_colors_per_view = np.full((6, n_verts, 3), 128, dtype=np.uint8)
    valid_mask = np.zeros((6, n_verts), dtype=bool)

    for vid in range(6):
        video_path = os.path.join(data_dir, f"videos_undist/{vid}.mp4")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = cap.read()
        cap.release()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        K = cams[vid]["K"]
        R = cams[vid]["R"]
        T = cams[vid]["T"]

        # Project vertices: Pc = R @ Pw + T; p2d = K @ Pc / Pc.z
        Pc = (R @ verts.T).T + T
        d = Pc[:, 2]
        p2h = (K @ Pc.T).T
        uv = p2h[:, :2] / (p2h[:, 2:] + 1e-8)
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (d > 0)
        valid_mask[vid] = in_img
        vertex_colors_per_view[vid, in_img] = img_rgb[v[in_img], u[in_img]]

    # Median across valid views per vertex
    vert_colors = np.full((n_verts, 3), 128, dtype=np.uint8)
    for vi in range(n_verts):
        valid_views = [vertex_colors_per_view[v, vi] for v in range(6) if valid_mask[v, vi]]
        if len(valid_views) >= 2:
            vert_colors[vi] = np.median(np.stack(valid_views), axis=0).astype(np.uint8)

    vert_colors_rgba = np.concatenate(
        [vert_colors, np.full((n_verts, 1), 255, dtype=np.uint8)], axis=1
    )
    return verts, faces, vert_colors_rgba


def render_mesh_with_colors(verts_gslrm, faces, vert_colors_rgba, cam_elev, cam_azim,
                             w=512, h=512, fov=50.0, radius=2.7):
    """Minimal pyrender novel-view render with vertex colors."""
    import pyrender
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts_gslrm, faces=faces, process=False,
                           vertex_colors=vert_colors_rgba)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pr_mesh)

    # Camera spherical position, looking at origin
    elev = np.deg2rad(cam_elev); azim = np.deg2rad(cam_azim)
    z = radius * np.sin(elev)
    base = radius * np.cos(elev)
    x = base * np.cos(azim); y = base * np.sin(azim)
    cam_pos = np.array([x, y, z])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    R_cv = np.stack((right, -up, forward), axis=1)
    c2w_cv = np.eye(4); c2w_cv[:3, :3] = R_cv; c2w_cv[:3, 3] = cam_pos
    c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=w/h)
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
    ap.add_argument("--sweep9-tex", default="results/sweep/run_wild-sweep-9/texture_final.png")
    ap.add_argument("--gt-rgb-ref", default=None, help="GT RGB frame for histogram matching (optional)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # P0: gamma-corrected texture
    print("[P0] Gamma correction + optional hist match")
    tex_p0 = p0_gamma_correct(args.sweep9_tex, gamma=2.2, hist_match_gt=args.gt_rgb_ref)
    p0_path = os.path.join(args.output, "texture_p0_gamma.png")
    cv2.imwrite(p0_path, tex_p0)
    print(f"  saved: {p0_path}")

    # P2: direct vertex color
    print(f"[P2] Direct multi-view projection → per-vertex median color (frame {args.frame})")
    verts_mm, faces, vert_colors = p2_direct_vertex_color(args.obj, args.data_dir, args.frame)
    print(f"  verts shape: {verts_mm.shape}, color shape: {vert_colors.shape}")

    # Transform to GSLRM for novel view rendering
    M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
    M5_DISTANCE_SCALE = 2.7 / 307.785
    verts_gslrm = (verts_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE

    # Render P2 from multiple views (Top + Right)
    from PIL import Image
    views = [("Top", 80, 270), ("Right", 20, 0), ("Front-high", 40, 270), ("Bottom", -85, 270)]
    p2_renders = {}
    for name, elev, azim in views:
        img = render_mesh_with_colors(verts_gslrm, faces, vert_colors, elev, azim)
        p = os.path.join(args.output, f"p2_{name}.png")
        Image.fromarray(img).save(p)
        p2_renders[name] = p
        print(f"  P2 {name}: {p}")

    print(f"\nDone. Output in {args.output}")
    print(f"  P0 texture: texture_p0_gamma.png (use with novel_view_render.py --texture-img)")
    print(f"  P2 renders: p2_{{Top,Right,Front-high,Bottom}}.png (direct vertex color)")


if __name__ == "__main__":
    main()
