#!/usr/bin/env python3
"""F1: F6g 3D visualization (real PNG).

Renders canon mesh with:
- Belly vertices colored RED (ventral, y > mesh_y_max - 20mm)
- Top-5 dominant joints as large GREEN spheres
- Other joints as small GRAY spheres
- Multiple views (Top, Right, Bottom) for spatial understanding

Inputs:
    mouse_model/mouse.pkl (t_pose_joints, skinning_weights)
    results/fitting/production_3600_canon/obj/step_2_frame_001800.obj

Output:
    results/belly_f6g/f6g_3d_viz_{Top,Right,Bottom}.png
    results/belly_f6g/f6g_3d_viz_grid.png
"""
import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np


def load_obj(obj_path):
    verts, faces = [], []
    with open(obj_path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif ln.startswith("f "):
                p = ln.split()[1:]
                idx = [int(t.split("/")[0]) - 1 for t in p[:3]]
                faces.append(idx)
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def render_view(verts, faces, vert_colors_rgba, joint_positions, joint_colors_rgba, joint_sizes,
                cam_elev, cam_azim, w=768, h=768, fov=55.0):
    import pyrender
    import trimesh
    from scipy.spatial.transform import Rotation

    # Mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, vertex_colors=vert_colors_rgba)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pr_mesh)

    # Add joint spheres
    for pos, col, size in zip(joint_positions, joint_colors_rgba, joint_sizes):
        sp = trimesh.creation.icosphere(radius=size, subdivisions=2)
        sp.visual.vertex_colors = np.tile(col, (sp.vertices.shape[0], 1))
        sp.apply_translation(pos)
        pr_sp = pyrender.Mesh.from_trimesh(sp, smooth=True)
        scene.add(pr_sp)

    # Center on mesh
    c = verts.mean(0)
    radius = float(np.linalg.norm(verts - c, axis=1).max()) * 2.4

    elev = np.deg2rad(cam_elev); azim = np.deg2rad(cam_azim)
    z = radius * np.sin(elev); base = radius * np.cos(elev)
    x = base * np.cos(azim); y = base * np.sin(azim)
    cam_pos = np.array([x, y, z]) + c
    forward = (c - cam_pos); forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    R_cv = np.stack((right, -up, forward), axis=1)
    c2w_cv = np.eye(4); c2w_cv[:3, :3] = R_cv; c2w_cv[:3, 3] = cam_pos
    c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=w / h)
    scene.add(cam, pose=c2w_gl)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=c2w_gl)
    r = pyrender.OffscreenRenderer(w, h)
    color, _ = r.render(scene)
    r.delete()
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="results/fitting/production_3600_canon/obj/step_2_frame_001800.obj")
    ap.add_argument("--model", default="mouse_model/mouse.pkl")
    ap.add_argument("--output", default="results/belly_f6g/")
    ap.add_argument("--belly-margin-mm", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # Load mesh + model
    verts, faces = load_obj(args.obj)
    print(f"mesh: {verts.shape}, faces: {faces.shape}")

    with open(args.model, "rb") as f:
        m = pickle.load(f)
    from scipy.sparse import issparse
    W = m["skinning_weights"]
    W = W.toarray() if issparse(W) else np.asarray(W)
    if W.shape[0] < W.shape[1]:
        W = W.T
    tj_tpose = m["t_pose_joints"]  # T-pose joint positions (need to transform to canon pose)

    # Belly vertices (canon coord: Y = head-to-tail, Z = vertical w/ ground at z≈0)
    # True belly = torso ventral = y∈[40,90] (not head y>90 / tail y<30) AND z_low (ventral)
    y = verts[:, 1]; z = verts[:, 2]
    z_thr = np.percentile(z, 25)  # lower quartile = ventral side
    y_min, y_max = 40.0, 90.0  # torso range (exclude head y>90, tail y<40)
    belly_mask = (y >= y_min) & (y <= y_max) & (z < z_thr)
    print(f"belly verts: {belly_mask.sum()}  (torso y[{y_min},{y_max}] AND z<{z_thr:.1f})")

    # Build vertex colors: gray base, RED for belly
    vc = np.full((verts.shape[0], 4), [200, 200, 200, 255], dtype=np.uint8)
    vc[belly_mask] = [220, 30, 30, 255]  # red

    # Top-5 dominant joints on (correctly-identified) belly — from D5 re-run 2026-04-19
    # Previous [123,134,130,138,137] were for head (mis-labeled); corrected:
    top5 = [23, 46, 42, 38, 19]

    # Transform T-pose joints to canon. Approximation: use LBS to get joint positions.
    # Simpler proxy: scale T-pose joints from [-0.5, 0.9] to canon mesh scale
    # Robust: skin weights * verts (weighted by each joint) gives rough joint center in canon
    # For visualization purposes, use weighted centroid of verts for each joint
    joint_canon = np.zeros((140, 3), dtype=np.float32)
    for j in range(140):
        w = W[:, j]
        if w.sum() > 1e-6:
            joint_canon[j] = (verts * w[:, None]).sum(0) / w.sum()
        else:
            joint_canon[j] = verts.mean(0)

    print("top-5 joint positions in canon (weighted centroid):")
    for j in top5:
        p = joint_canon[j]
        print(f"  joint {j}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    # Sphere sizes: top5 big, others small
    all_joint_positions = []
    all_joint_colors = []
    all_joint_sizes = []
    for j in range(140):
        if j in top5:
            all_joint_positions.append(joint_canon[j])
            all_joint_colors.append([30, 220, 30, 255])  # green
            all_joint_sizes.append(2.5)  # mm
        else:
            all_joint_positions.append(joint_canon[j])
            all_joint_colors.append([80, 80, 220, 180])  # blue small
            all_joint_sizes.append(0.8)

    # Render 3 views
    views = [("Top", 80, 270), ("Right", 15, 0), ("Bottom", -80, 270)]
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    renders = {}
    for name, elev, azim in views:
        img = render_view(verts, faces, vc,
                          all_joint_positions, all_joint_colors, all_joint_sizes,
                          elev, azim)
        p = out / f"f6g_3d_viz_{name}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"saved: {p}")
        renders[name] = img

    # Grid
    H = max(r.shape[0] for r in renders.values())
    W_ = max(r.shape[1] for r in renders.values())
    bar = 28
    grid = np.full((H + bar, W_ * 3, 3), 255, np.uint8)
    for i, name in enumerate([v[0] for v in views]):
        cv2.putText(grid, f"{name} (belly=RED, top5 joints=GREEN)", (i * W_ + 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        grid[bar:bar + H, i * W_:(i + 1) * W_] = renders[name]
    gp = out / "f6g_3d_viz_grid.png"
    cv2.imwrite(str(gp), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"saved: {gp}")


if __name__ == "__main__":
    main()
