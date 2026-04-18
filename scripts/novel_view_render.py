#!/usr/bin/env python3
"""Render MAMMAL mesh from FaceLift v3 novel 6-view with texture.

Supports single-frame MVP and batch (full sequence → video).

Coord transform pipeline (verified in 260417 MVP):
  1. Vertices: mammal_to_gslrm (translate + scale, no axis swap)
  2. Camera: spherical → OpenCV c2w → OpenGL (diag(1,-1,-1,1))

v3 6-view config from 260306 reference image:
  Top(+80,270), Front-high(+40,270), Right(+20,0)
  Bottom(-85,270), Back-high(+40,90), Left(+20,180)

Usage:
    # Single frame textured
    PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=4 python scripts/novel_view_render.py \
        --obj-dir results/fitting/production_3600_canon/obj \
        --frame 1800 \
        --output results/novel_view_mvp/frame_1800_tex

    # Batch + video
    PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=4 python scripts/novel_view_render.py \
        --obj-dir results/fitting/production_3600_canon/obj \
        --start 0 --end 18000 --step 5 \
        --output results/novel_view_batch/canon_3600 \
        --make-video
"""

import argparse
import glob
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np


V3_NOVEL_6VIEWS = [
    # (name, elev_deg, azim_deg, grid_row, grid_col)
    ("Top",        +80.0, 270.0, 0, 0),
    ("Front-high", +40.0, 270.0, 0, 1),
    ("Right",      +20.0,   0.0, 0, 2),
    ("Bottom",     -85.0, 270.0, 1, 0),
    ("Back-high",  +40.0,  90.0, 1, 1),
    ("Left",       +20.0, 180.0, 1, 2),
]

M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def mammal_to_gslrm(xyz_mm):
    return (xyz_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def load_obj_verts_faces(path):
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


def compute_vertex_colors_from_texture(faces, n_verts, textures_path,
                                        faces_tex_path, texture_img_path):
    """Per-vertex color via mean UV → texture lookup. Returns (N, 4) RGBA uint8."""
    import cv2
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
    # Invalid verts (no texture) → gray
    colors[~valid] = [180, 180, 180, 255]
    return colors


def spherical_c2w_opencv(elev_deg, azim_deg, radius=2.7,
                          center=None, up_vector=None):
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
    right = np.cross(forward, up_vector)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        fallback = np.array([0.0, 1.0, 0.0]) if abs(up_vector[2]) > 0.5 else np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, fallback)
        rn = np.linalg.norm(right)
    right = right / rn
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    R = np.stack((right, -up, forward), axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos
    return c2w


def render_one_view(verts_g, faces, vert_colors, c2w_cv,
                     w, h, fov_deg, bg):
    import pyrender
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts_g, faces=faces, process=False,
                           vertex_colors=vert_colors)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=list(bg) + [1.0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(pr_mesh)
    c2w_gl = c2w_cv @ CV_TO_GL
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov_deg), aspectRatio=w / h)
    scene.add(cam, pose=c2w_gl)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=c2w_gl)
    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, _ = r.render(scene)
    r.delete()
    return color


def make_grid_2x3(images, labels):
    import cv2
    h, w = images[0].shape[:2]
    cols, rows = 3, 2
    canvas = np.full((h * rows, w * cols, 3), 255, dtype=np.uint8)
    for idx, (img, label) in enumerate(zip(images, labels)):
        r, c = idx // cols, idx % cols
        y0, x0 = r * h, c * w
        canvas[y0:y0 + h, x0:x0 + w] = img
        cv2.putText(canvas, label, (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return canvas


def render_frame(obj_path, out_dir, vert_colors_or_none, textures, w, h, fov, radius, bg, save_individual):
    verts_mm, faces = load_obj_verts_faces(obj_path)
    verts_g = mammal_to_gslrm(verts_mm)

    if vert_colors_or_none is None and textures is not None:
        vert_colors = compute_vertex_colors_from_texture(
            faces, len(verts_mm),
            textures["textures_txt"], textures["faces_tex_txt"], textures["texture_img"]
        )
    elif vert_colors_or_none is not None:
        vert_colors = vert_colors_or_none
    else:
        vert_colors = np.full((len(verts_mm), 4), [180, 180, 180, 255], dtype=np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    images = []
    labels = []
    cameras = {}
    for name, elev, azim, _, _ in V3_NOVEL_6VIEWS:
        c2w = spherical_c2w_opencv(elev, azim, radius=radius)
        cameras[name] = {"elevation_deg": elev, "azimuth_deg": azim,
                          "c2w_opencv": c2w.tolist()}
        img = render_one_view(verts_g, faces, vert_colors, c2w, w, h, fov, bg)
        images.append(img)
        labels.append(f"{name} (e={elev:+.0f} a={azim:+.0f})")
        if save_individual:
            import imageio.v3 as iio
            iio.imwrite(os.path.join(out_dir, f"{name}.png"), img)

    grid = make_grid_2x3(images, labels)
    import imageio.v3 as iio
    iio.imwrite(os.path.join(out_dir, "grid_2x3.png"), grid)
    return grid, cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", required=True)
    ap.add_argument("--frame", type=int, help="Single frame mode")
    ap.add_argument("--start", type=int, help="Batch: start video frame")
    ap.add_argument("--end", type=int, help="Batch: end video frame (exclusive)")
    ap.add_argument("--step", type=int, default=5, help="Batch: frame step (video frame units)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--radius", type=float, default=2.7)
    ap.add_argument("--no-texture", action="store_true", help="Disable texture (gray mesh)")
    ap.add_argument("--textures-txt", default="mouse_model/mouse_txt/textures.txt")
    ap.add_argument("--faces-tex-txt", default="mouse_model/mouse_txt/faces_tex.txt")
    ap.add_argument("--texture-img",
                    default="results/sweep/production_p0/texture_final.png",
                    help="Default: P0 gamma+hist canonical (ΔE 17.7 vs GT, dark-brown). "
                         "Fallback: results/sweep/run_wild-sweep-9/texture_final.png (raw-average, olive-gray). "
                         "See docs/reports/260418_texture_multipath_comparison.md.")
    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    textures = None if args.no_texture else {
        "textures_txt": args.textures_txt,
        "faces_tex_txt": args.faces_tex_txt,
        "texture_img": args.texture_img,
    }
    bg = (1.0, 1.0, 1.0)

    # Single frame mode
    if args.frame is not None:
        obj_path = os.path.join(args.obj_dir, f"step_2_frame_{args.frame:06d}.obj")
        if not os.path.exists(obj_path):
            raise FileNotFoundError(obj_path)
        print(f"Single frame render: {obj_path}")
        grid, cameras = render_frame(obj_path, args.output, None, textures,
                                      args.w, args.h, args.fov, args.radius, bg, True)
        meta = {"frame": args.frame, "source_obj": obj_path,
                "cameras": cameras,
                "transform": "mammal_to_gslrm + OpenCV→OpenGL flip",
                "M5_SCENE_CENTER": M5_SCENE_CENTER.tolist(),
                "M5_DISTANCE_SCALE": float(M5_DISTANCE_SCALE),
                "textured": textures is not None}
        with open(os.path.join(args.output, "metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"Saved: {args.output}/grid_2x3.png + 6 PNGs + metadata.json")
        return

    # Batch mode
    assert args.start is not None and args.end is not None, "batch mode needs --start/--end"
    frames = list(range(args.start, args.end, args.step))
    grids_dir = os.path.join(args.output, "grids")
    os.makedirs(grids_dir, exist_ok=True)
    print(f"Batch render: {len(frames)} frames, texture={'yes' if textures else 'no'}")

    # Reuse vert colors across frames (topology-constant, skinning via LBS)
    # Compute once from first frame's topology
    first_obj = os.path.join(args.obj_dir, f"step_2_frame_{frames[0]:06d}.obj")
    _, first_faces = load_obj_verts_faces(first_obj)
    if textures is not None:
        print("Computing per-vertex colors from texture (once)...")
        n_verts = 14522
        vert_colors = compute_vertex_colors_from_texture(
            first_faces, n_verts,
            textures["textures_txt"], textures["faces_tex_txt"], textures["texture_img"]
        )
    else:
        vert_colors = None

    cameras_ref = None
    for i, fi in enumerate(frames):
        obj_path = os.path.join(args.obj_dir, f"step_2_frame_{fi:06d}.obj")
        if not os.path.exists(obj_path):
            print(f"  skip missing: {obj_path}")
            continue
        frame_dir = os.path.join(args.output, f"frame_{fi:06d}")
        grid, cameras = render_frame(obj_path, frame_dir, vert_colors, textures,
                                      args.w, args.h, args.fov, args.radius, bg, False)
        # Copy grid to sequence dir
        import imageio.v3 as iio
        iio.imwrite(os.path.join(grids_dir, f"grid_{fi:06d}.png"), grid)
        if cameras_ref is None:
            cameras_ref = cameras
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(frames)}] frame {fi}")

    # Metadata
    meta = {"frames": frames, "n_rendered": len(frames),
            "cameras": cameras_ref,
            "transform": "mammal_to_gslrm + OpenCV→OpenGL flip",
            "M5_SCENE_CENTER": M5_SCENE_CENTER.tolist(),
            "M5_DISTANCE_SCALE": float(M5_DISTANCE_SCALE),
            "textured": textures is not None,
            "resolution": [args.w, args.h], "fov_deg": args.fov, "radius": args.radius}
    with open(os.path.join(args.output, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Video assembly
    if args.make_video:
        video_path = os.path.join(args.output, "grid_sequence.mp4")
        print(f"Assembling video: {video_path}")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-pattern_type", "glob", "-i", os.path.join(grids_dir, "grid_*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
            video_path
        ]
        subprocess.run(cmd, check=True)
        print(f"Video: {video_path}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
