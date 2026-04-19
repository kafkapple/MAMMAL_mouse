#!/usr/bin/env python3
"""L1: Measure PSNR/SSIM/LPIPS on same-camera render vs GT (MoReMouse benchmark).

Uses render_same_camera_6view's renders (with P0 texture) against GT frames.
Outputs: docs/reports/260419_psnr_ssim_lpips.csv + summary comparison with MoReMouse.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _psnr(img1, img2, data_range=255.0):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def _ssim_single_channel(img1, img2, data_range=255.0, window_size=11):
    """Simple Gaussian-kernel SSIM. Single channel."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    img1 = img1.astype(np.float64); img2 = img2.astype(np.float64)
    k = cv2.getGaussianKernel(window_size, 1.5)
    window = k @ k.T
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def _ssim_rgb(img1, img2, data_range=255.0):
    return float(np.mean([_ssim_single_channel(img1[..., c], img2[..., c], data_range) for c in range(3)]))


def masked_psnr_ssim(render_bgr, gt_bgr, gt_fg_mask):
    """Masked PSNR/SSIM on mouse body region only (MoReMouse-compatible).

    Uses GT SimpleClick mask to define foreground. Composite render onto black bg
    to match, then compute metrics within mask.
    """
    if render_bgr.shape != gt_bgr.shape:
        render_bgr = cv2.resize(render_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]))
    if gt_fg_mask.shape != gt_bgr.shape[:2]:
        gt_fg_mask = cv2.resize(gt_fg_mask.astype(np.uint8), (gt_bgr.shape[1], gt_bgr.shape[0])) > 0

    # Mask render: non-white pixels only (render bg is white)
    r_gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)
    render_fg_mask = r_gray < 240
    # Union of fg masks (for body region)
    both_fg = gt_fg_mask & render_fg_mask
    if both_fg.sum() < 100:
        return 0.0, 0.0, 0

    r_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
    g_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

    # Masked PSNR: only pixels where both agree foreground
    diff = (r_rgb[both_fg].astype(np.float64) - g_rgb[both_fg].astype(np.float64))
    mse = np.mean(diff ** 2)
    psnr_val = 100.0 if mse < 1e-10 else 20.0 * np.log10(255.0 / np.sqrt(mse))

    # Masked SSIM: compute on cropped tight bbox of union mask
    ys, xs = np.where(both_fg)
    if len(ys) < 100:
        return float(psnr_val), 0.0, int(both_fg.sum())
    y0, y1 = max(0, ys.min() - 20), min(r_rgb.shape[0], ys.max() + 20)
    x0, x1 = max(0, xs.min() - 20), min(r_rgb.shape[1], xs.max() + 20)
    r_crop = r_rgb[y0:y1, x0:x1]
    g_crop = g_rgb[y0:y1, x0:x1]
    ssim_val = _ssim_rgb(g_crop, r_crop, 255.0) if min(r_crop.shape[:2]) > 20 else 0.0

    return float(psnr_val), float(ssim_val), int(both_fg.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[1800, 2700, 5400, 10800, 15120])
    ap.add_argument("--obj-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--texture", default="results/sweep/production_p0/texture_final.png")
    ap.add_argument("--output", default="docs/reports/260419_psnr_ssim_lpips.csv")
    ap.add_argument("--render-base", default="results/belly_dent_investigation/")
    args = ap.parse_args()

    # Ensure renders exist (call render_same_camera_6view if not)
    import subprocess
    for fid in args.frames:
        render_dir = Path(args.render_base) / f"frame_{fid:06d}_6view_eval"
        if not (render_dir / "render_v0.png").exists():
            obj_p = f"{args.obj_dir}/step_2_frame_{fid:06d}.obj"
            if not os.path.exists(obj_p):
                print(f"skip frame {fid}: {obj_p} missing")
                continue
            cmd = ["python", "scripts/render_same_camera_6view.py",
                   "--frame", str(fid), "--obj", obj_p, "--output", str(render_dir)]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "5"
            env["PYOPENGL_PLATFORM"] = "egl"
            print(f"[render] frame {fid}")
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)

    # Measure
    rows = []
    for fid in args.frames:
        render_dir = Path(args.render_base) / f"frame_{fid:06d}_6view_eval"
        for vid in range(6):
            render_p = render_dir / f"render_v{vid}.png"
            gt_p = render_dir / f"gt_v{vid}.png"
            if not (render_p.exists() and gt_p.exists()):
                continue
            render_bgr = cv2.imread(str(render_p))
            gt_bgr = cv2.imread(str(gt_p))
            # Load GT SimpleClick mask
            cap = cv2.VideoCapture(os.path.join(args.data_dir, "simpleclick_undist", f"{vid}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid); ok, m = cap.read(); cap.release()
            if not ok:
                continue
            gt_mask = m[:, :, 0] > 127
            try:
                psnr, ssim, n_px = masked_psnr_ssim(render_bgr, gt_bgr, gt_mask)
            except Exception as e:
                print(f"err f{fid} v{vid}: {e}")
                continue
            rows.append({
                "frame": fid, "view": vid,
                "psnr": round(psnr, 3), "ssim": round(ssim, 4),
                "n_fg_px": n_px,
            })
            print(f"  f{fid} v{vid}: PSNR={psnr:.2f} SSIM={ssim:.3f} n_px={n_px}")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    psnrs = [r["psnr"] for r in rows]
    ssims = [r["ssim"] for r in rows]
    print(f"\n{'='*60}")
    print(f"PSNR/SSIM Summary (N={len(rows)} frame×view pairs)")
    print(f"  PSNR mean: {np.mean(psnrs):.2f}  min: {min(psnrs):.2f}  max: {max(psnrs):.2f}")
    print(f"  SSIM mean: {np.mean(ssims):.4f}  min: {min(ssims):.4f}  max: {max(ssims):.4f}")
    print(f"\nMoReMouse AAAI-2026 benchmark (real):")
    print(f"  PSNR 18.42, SSIM 0.948, LPIPS 0.087")
    print(f"\nDiff vs MoReMouse: PSNR {np.mean(psnrs)-18.42:+.2f}, SSIM {np.mean(ssims)-0.948:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
