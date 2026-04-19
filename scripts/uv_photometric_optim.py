#!/usr/bin/env python3
"""L3: UV photometric optimization (MoReMouse-inspired) using pytorch3d.

Differentiable rendering + L1 + SSIM + LPIPS loss on multi-view GT photos.
Optimizes a single shared UV texture starting from P0 init.

Usage:
    python scripts/uv_photometric_optim.py \
        --frames 1800 5400 10800 \
        --init-tex results/sweep/production_p0/texture_final.png \
        --output results/uv_photometric_opt_v1/
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def load_obj(p):
    verts, faces = [], []
    faces_tex = []
    with open(p) as fh:
        for ln in fh:
            if ln.startswith("v "):
                parts = ln.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif ln.startswith("f "):
                parts = ln.split()[1:]
                vidx = []
                tidx = []
                for t in parts[:3]:
                    s = t.split("/")
                    vidx.append(int(s[0]) - 1)
                    if len(s) > 1 and s[1]:
                        tidx.append(int(s[1]) - 1)
                    else:
                        tidx.append(-1)
                faces.append(vidx)
                faces_tex.append(tidx)
    return (np.array(verts, dtype=np.float32),
            np.array(faces, dtype=np.int64),
            np.array(faces_tex, dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[1800, 5400, 10800])
    ap.add_argument("--obj-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--init-tex", default="results/sweep/production_p0/texture_final.png")
    ap.add_argument("--textures-txt", default="mouse_model/mouse_txt/textures.txt")
    ap.add_argument("--output", default="results/uv_photometric_opt_v1/")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=0.2)
    ap.add_argument("--w-tv", type=float, default=0.0)
    ap.add_argument("--tex-size", type=int, default=512)
    ap.add_argument("--render-h", type=int, default=512)
    ap.add_argument("--render-w", type=int, default=576)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    import torch
    from pytorch3d.renderer import (
        PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, TexturesUV, AmbientLights, PointLights)
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}, torch={torch.__version__}")

    # ===== Load mesh + UV (shared across frames) =====
    obj_p = os.path.join(args.obj_dir, f"step_2_frame_{args.frames[0]:06d}.obj")
    verts_np, faces_np, faces_tex_np = load_obj(obj_p)
    uv_np = np.loadtxt(args.textures_txt)  # (15399, 2)
    print(f"mesh: V={len(verts_np)} F={len(faces_np)}, UV={len(uv_np)}")

    # Init texture (H, W, 3) in [0, 1]
    tex_bgr = cv2.imread(args.init_tex)
    tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if tex_rgb.shape[:2] != (args.tex_size, args.tex_size):
        tex_rgb = cv2.resize(tex_rgb, (args.tex_size, args.tex_size))
    # Optimizable parameter
    texture_param = torch.tensor(tex_rgb, dtype=torch.float32, device=device, requires_grad=True)

    uv_t = torch.tensor(uv_np, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces_np, dtype=torch.int64, device=device)
    faces_uv_t = torch.tensor(faces_tex_np, dtype=torch.int64, device=device)

    # ===== Load cameras =====
    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams = pickle.load(f)

    # Pre-load GT RGB + mask per frame × view
    from collections import defaultdict
    train_data = []  # list of (frame, view, verts_tensor, gt_img_tensor, fg_mask, K_pt, R_pt, T_pt, H, W)
    for fid in args.frames:
        obj_p = os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj")
        if not os.path.exists(obj_p): continue
        verts_f, _, _ = load_obj(obj_p)
        verts_t = torch.tensor(verts_f, dtype=torch.float32, device=device)
        for vid in range(6):
            cap_rgb = cv2.VideoCapture(os.path.join(args.data_dir, "videos_undist", f"{vid}.mp4"))
            cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, img_bgr = cap_rgb.read(); cap_rgb.release()
            if not ok: continue
            cap_m = cv2.VideoCapture(os.path.join(args.data_dir, "simpleclick_undist", f"{vid}.mp4"))
            cap_m.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok2, mask_bgr = cap_m.read(); cap_m.release()
            if not ok2: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            fg = (mask_bgr[:, :, 0] > 127).astype(np.float32)
            H, W = img_rgb.shape[:2]
            train_data.append({
                "frame": fid, "view": vid,
                "verts": verts_t,
                "img": torch.tensor(img_rgb, dtype=torch.float32, device=device),
                "mask": torch.tensor(fg, dtype=torch.float32, device=device),
                "K": cams[vid]["K"], "R": cams[vid]["R"], "T": cams[vid]["T"].reshape(-1),
                "H": H, "W": W,
            })
    print(f"loaded {len(train_data)} (frame, view) samples")

    # ===== Render helper =====
    def render_view(verts_t, K_np, R_np, T_np, H, W, tex_hwc):
        # Use pytorch3d.utils.cameras_from_opencv_projection for proper OpenCV→P3D conversion
        R_pt = torch.tensor(R_np, dtype=torch.float32, device=device).unsqueeze(0)
        T_pt = torch.tensor(T_np, dtype=torch.float32, device=device).unsqueeze(0)
        K_pt = torch.tensor(K_np, dtype=torch.float32, device=device).unsqueeze(0)
        img_size_pt = torch.tensor([[H, W]], dtype=torch.float32, device=device)
        cam = cameras_from_opencv_projection(R=R_pt, tvec=T_pt, camera_matrix=K_pt,
                                              image_size=img_size_pt)
        cam = cam.to(device)

        # Mesh + texture
        faces_uv_batch = faces_uv_t.unsqueeze(0)
        uv_batch = uv_t.unsqueeze(0)
        tex_map = tex_hwc.unsqueeze(0)  # (1, H, W, 3)
        textures = TexturesUV(maps=tex_map, faces_uvs=faces_uv_batch, verts_uvs=uv_batch)
        mesh = Meshes(verts=[verts_t], faces=[faces_t], textures=textures)

        raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
                                                max_faces_per_bin=100000)
        rasterizer = MeshRasterizer(cameras=cam, raster_settings=raster_settings)
        lights = AmbientLights(device=device)
        shader = SoftPhongShader(device=device, cameras=cam, lights=lights)
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        image = renderer(mesh)[0, ..., :3]  # (H, W, 3)
        return image

    # ===== SSIM loss =====
    def ssim_loss(pred, gt, data_range=1.0, window_size=7):
        # Simple approximation: compute on windowed patches
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        from torch.nn.functional import avg_pool2d
        # (H, W, 3) → (1, 3, H, W)
        p = pred.permute(2, 0, 1).unsqueeze(0)
        g = gt.permute(2, 0, 1).unsqueeze(0)
        mu_p = avg_pool2d(p, window_size, stride=1)
        mu_g = avg_pool2d(g, window_size, stride=1)
        mu_p_sq = mu_p * mu_p
        mu_g_sq = mu_g * mu_g
        mu_pg = mu_p * mu_g
        sig_p_sq = avg_pool2d(p * p, window_size, stride=1) - mu_p_sq
        sig_g_sq = avg_pool2d(g * g, window_size, stride=1) - mu_g_sq
        sig_pg = avg_pool2d(p * g, window_size, stride=1) - mu_pg
        ssim_map = ((2 * mu_pg + C1) * (2 * sig_pg + C2)) / \
                   ((mu_p_sq + mu_g_sq + C1) * (sig_p_sq + sig_g_sq + C2))
        return (1.0 - ssim_map).mean()

    # ===== Optimization loop =====
    optimizer = torch.optim.Adam([texture_param], lr=args.lr)

    # Debug: save iter-0 render for visual verification
    with torch.no_grad():
        d0 = train_data[0]
        r0 = render_view(d0["verts"], d0["K"], d0["R"], d0["T"], d0["H"], d0["W"],
                         texture_param.clamp(0, 1))
        r0_np = (r0.detach().cpu().numpy() * 255).astype(np.uint8)
        debug = np.concatenate([
            (d0["img"].cpu().numpy() * 255).astype(np.uint8),
            r0_np
        ], axis=1)
        cv2.imwrite(str(out / "debug_iter0_gt_vs_render.png"),
                    cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
        print(f"[debug] saved iter-0 GT|render: {out}/debug_iter0_gt_vs_render.png")

    history = []
    t0 = time.time()
    for it in range(args.iters):
        # Mini-batch: 1 sample per iter (or average over batch)
        idx = it % len(train_data)
        d = train_data[idx]
        tex_clamped = torch.sigmoid(texture_param) if False else texture_param.clamp(0, 1)
        try:
            render = render_view(d["verts"], d["K"], d["R"], d["T"], d["H"], d["W"], tex_clamped)
        except Exception as e:
            print(f"[iter {it}] render failed: {e}")
            continue

        # Mask render (render has white bg from shader); composite with GT-mask region
        mask_3 = d["mask"].unsqueeze(-1)
        # Compute loss only where GT mask is foreground
        render_fg = render * mask_3
        gt_fg = d["img"] * mask_3

        l1 = (render_fg - gt_fg).abs().sum() / (mask_3.sum() * 3 + 1e-6)
        ssim_l = ssim_loss(render_fg, gt_fg)
        loss = args.w_l1 * l1 + args.w_ssim * ssim_l
        if args.w_tv > 0:
            tv = (tex_clamped[1:] - tex_clamped[:-1]).abs().mean() + \
                 (tex_clamped[:, 1:] - tex_clamped[:, :-1]).abs().mean()
            loss = loss + args.w_tv * tv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0 or it == args.iters - 1:
            print(f"[iter {it:4d}] L1={l1.item():.4f} SSIM_l={ssim_l.item():.4f} "
                  f"total={loss.item():.4f}  elapsed={time.time()-t0:.1f}s")
            history.append({"iter": it, "l1": float(l1), "ssim_loss": float(ssim_l),
                            "total": float(loss), "sample": f"f{d['frame']}v{d['view']}"})

    # Save optimized texture
    tex_final = texture_param.detach().clamp(0, 1).cpu().numpy()
    tex_out_bgr = cv2.cvtColor((tex_final * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / "texture_optimized.png"), tex_out_bgr)

    with open(out / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)

    print(f"\nSaved: {out}/texture_optimized.png")


if __name__ == "__main__":
    sys.exit(main())
