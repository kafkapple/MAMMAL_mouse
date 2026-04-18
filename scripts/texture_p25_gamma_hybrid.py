#!/usr/bin/env python3
"""G2: P2.5 + gamma hybrid — occlusion-aware vertex color with gamma pre-darken.

Motivation: P2.5 sampled correct L*=27 vertex colors from GT but pyrender
lighting lifted L* to 100+ in render. Fix: apply gamma darkening to the
sampled vertex colors (pre-compensate lighting gain) before rendering.

Sweep gamma ∈ {2.0, 3.0, 4.0, 5.0} to find best render ΔE.
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


def p25_sample_occlusion_aware(obj_path, data_dir, frame_id, mask_threshold=127):
    verts, faces = load_obj(obj_path)
    n_verts = len(verts)
    with open(os.path.join(data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)
    vcol_pv = np.full((6, n_verts, 3), 128, dtype=np.uint8)
    vmask = np.zeros((6, n_verts), dtype=bool)
    for vid in range(6):
        cap_r = cv2.VideoCapture(os.path.join(data_dir, "videos_undist", f"{vid}.mp4"))
        cap_m = cv2.VideoCapture(os.path.join(data_dir, "simpleclick_undist", f"{vid}.mp4"))
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, frame_id); cap_m.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok1, ibgr = cap_r.read(); ok2, mbgr = cap_m.read()
        cap_r.release(); cap_m.release()
        if not (ok1 and ok2): continue
        irgb = cv2.cvtColor(ibgr, cv2.COLOR_BGR2RGB)
        fg = mbgr[:, :, 0] > mask_threshold
        H, W = irgb.shape[:2]
        K, R, T = cams[vid]["K"], cams[vid]["R"], cams[vid]["T"].reshape(-1)
        Pc = (R @ verts.T).T + T
        uv = (K @ Pc.T).T[:, :2] / (Pc[:, 2:] + 1e-8)
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Pc[:, 2] > 0)
        valid = in_img.copy()
        valid[in_img] = fg[v[in_img], u[in_img]]
        vmask[vid] = valid
        vcol_pv[vid, valid] = irgb[v[valid], u[valid]]
    vcol = np.full((n_verts, 3), 128, dtype=np.uint8)
    for vi in range(n_verts):
        cvs = [vcol_pv[v, vi] for v in range(6) if vmask[v, vi]]
        if len(cvs) >= 2:
            vcol[vi] = np.median(np.stack(cvs), axis=0).astype(np.uint8)
    return verts, faces, vcol


def apply_gamma(vcol, gamma):
    v = vcol.astype(np.float32) / 255.0
    vg = np.power(v, gamma)
    return np.clip(vg * 255, 0, 255).astype(np.uint8)


def render(verts_mm, faces, vcol_rgba, elev, azim, w=512, h=512):
    import pyrender, trimesh
    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False, vertex_colors=vcol_rgba)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pr_mesh)
    import numpy as _np
    radius = 2.7
    elev_r = _np.deg2rad(elev); azim_r = _np.deg2rad(azim)
    z = radius * _np.sin(elev_r); base = radius * _np.cos(elev_r)
    x = base * _np.cos(azim_r); y = base * _np.sin(azim_r)
    cam_pos = _np.array([x, y, z])
    fw = -cam_pos / _np.linalg.norm(cam_pos)
    up_w = _np.array([0.0, 0.0, 1.0])
    rt = _np.cross(fw, up_w); rt /= _np.linalg.norm(rt)
    up = _np.cross(rt, fw); up /= _np.linalg.norm(up)
    R = _np.stack((rt, -up, fw), axis=1)
    c2w = _np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = cam_pos
    c2w_gl = c2w @ _np.diag([1.0, -1.0, -1.0, 1.0])
    cam = pyrender.PerspectiveCamera(yfov=_np.deg2rad(50.0), aspectRatio=w / h)
    scene.add(cam, pose=c2w_gl)
    light = pyrender.DirectionalLight(color=_np.ones(3), intensity=3.0)
    scene.add(light, pose=c2w_gl)
    r = pyrender.OffscreenRenderer(w, h)
    color, _ = r.render(scene); r.delete()
    return color


def body_lab(p, wt=0.95):
    img = cv2.imread(p)
    if img is None: return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = np.all(rgb < wt, axis=-1) & (rgb.sum(-1) > 0.1)
    if mask.sum() == 0: return None
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[mask].mean(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="results/fitting/production_3600_canon/obj/step_2_frame_001800.obj")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--gt-ref", default="/tmp/gt_rgb_f1800_v0.png")
    ap.add_argument("--output", default="results/texture_experiment_v1/p25_gamma/")
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # GT Lab
    gt = cv2.imread(args.gt_ref)
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m = (gray < 100) & (gray > 10)
    gt_lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[m].mean(0)
    print(f"GT body Lab: L={gt_lab[0]:.1f}")

    # Sample occlusion-aware colors once
    print(f"[P2.5] sampling occlusion-aware vertex colors (frame {args.frame})")
    verts_mm, faces, vcol = p25_sample_occlusion_aware(args.obj, args.data_dir, args.frame)

    # Transform for render
    M5_C = np.array([59.672, 51.517, 107.099])
    M5_S = 2.7 / 307.785
    verts_g = (verts_mm - M5_C) * M5_S

    # Sweep gamma
    results = []
    for g in args.gammas:
        vc = apply_gamma(vcol, g)
        vc_rgba = np.concatenate([vc, np.full((len(vc), 1), 255, dtype=np.uint8)], axis=1)
        img = render(verts_g, faces, vc_rgba, 20, 0)  # Right view
        p = out / f"p25_gamma{g:.1f}_Right.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        lab = body_lab(str(p))
        if lab is None: continue
        dE = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab, gt_lab))))
        results.append({"gamma": g, "dE": round(dE, 2), "L": round(lab[0], 1),
                        "render": str(p)})
        print(f"  gamma={g:.1f}  L={lab[0]:5.1f}  ΔE={dE:6.2f}")

    best = min(results, key=lambda r: r["dE"])
    with open(out / "p25_gamma_dE.json", "w") as f:
        json.dump({"gt_body_Lab": gt_lab.tolist(), "results": results, "best": best}, f, indent=2)
    print(f"\nBEST P2.5+gamma: γ={best['gamma']:.1f}  ΔE={best['dE']:.2f}  L={best['L']}")


if __name__ == "__main__":
    sys.exit(main())
