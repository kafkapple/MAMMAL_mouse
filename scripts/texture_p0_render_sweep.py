#!/usr/bin/env python3
"""G1: P0 render-space sweep — find gamma × hist_alpha minimizing RENDERED ΔE (not UV).

Prior E1 sweep optimized UV-space ΔE (got 0.59 but rendered = 73).
Lesson: pyrender ambient(0.5)+directional(3.0) lights lift L* by ~3-4x.
This sweep renders each variant + measures ΔE in render space directly.

Outputs:
    results/texture_experiment_v1/p0_render_sweep/
        variant_gamma*_alpha*/Right.png (one render per variant)
        render_dE.json   (all variants + winner)
        comparison_grid.png
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def gt_body_lab(gt_path, gray_lo=10, gray_hi=100):
    gt = cv2.imread(gt_path)
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (gray < gray_hi) & (gray > gray_lo)
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[mask].mean(0)


def make_variant(tex_path, gamma, alpha, gt_lab, out_path):
    tex = cv2.imread(tex_path)
    tex_rgb = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask_fg = np.any(tex_rgb < 0.95, axis=-1)
    tex_g = np.power(tex_rgb, gamma)
    out = tex_rgb.copy(); out[mask_fg] = tex_g[mask_fg]
    # Hist match in Lab
    out_lab = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    fg = mask_fg
    current_mean = out_lab[fg].mean(0)
    shift = (gt_lab - current_mean) * alpha
    out_lab[fg] += shift
    out_lab = np.clip(out_lab, 0, 255)
    out_rgb = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    cv2.imwrite(str(out_path), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))


def body_lab_from_render(p, wt=0.95):
    img = cv2.imread(p)
    if img is None: return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = np.all(rgb < wt, axis=-1) & (rgb.sum(-1) > 0.1)
    if mask.sum() == 0: return None
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[mask].mean(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--frame", type=int, default=1800)
    ap.add_argument("--source-tex", default="results/sweep/run_wild-sweep-9/texture_final.png")
    ap.add_argument("--gt-ref", default="/tmp/gt_rgb_f1800_v0.png")
    ap.add_argument("--output", default="results/texture_experiment_v1/p0_render_sweep/")
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 4.0])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--views", nargs="+", default=["Right"])
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    gt_lab = gt_body_lab(args.gt_ref)
    print(f"GT body Lab: L={gt_lab[0]:.1f} a={gt_lab[1]-128:.1f} b={gt_lab[2]-128:.1f}")

    variants = []
    for g in args.gammas:
        for a in args.alphas:
            variants.append((g, a))
    print(f"N variants: {len(variants)}")

    results = []
    for g, a in variants:
        name = f"gamma{g:.1f}_alpha{a:.2f}"
        vdir = out / f"variant_{name}"
        vdir.mkdir(exist_ok=True)
        tex_path = vdir / "tex.png"
        make_variant(args.source_tex, g, a, gt_lab, tex_path)

        # Render at Right view (single frame)
        cmd = [
            "python", "scripts/novel_view_render.py",
            "--obj-dir", args.obj_dir, "--frame", str(args.frame),
            "--output", str(vdir), "--texture-img", str(tex_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "5"
        env["PYOPENGL_PLATFORM"] = "egl"
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)

        # Read render Right.png, measure ΔE
        rp = vdir / "Right.png"
        lab = body_lab_from_render(str(rp))
        if lab is None:
            results.append({"gamma": g, "alpha": a, "dE": None}); continue
        dE = float(np.sqrt(sum((x - y) ** 2 for x, y in zip(lab, gt_lab))))
        results.append({
            "gamma": g, "alpha": a,
            "render_L": round(lab[0], 1),
            "render_a": round(lab[1] - 128, 1),
            "render_b": round(lab[2] - 128, 1),
            "dE_render": round(dE, 2),
            "tex_path": str(tex_path),
            "render_path": str(rp),
        })
        print(f"  γ={g:.1f} α={a:.2f}  L={lab[0]:5.1f}  ΔE_render={dE:6.2f}")

    valid = [r for r in results if r.get("dE_render") is not None]
    best = min(valid, key=lambda r: r["dE_render"])
    report = {"gt_body_Lab": gt_lab.tolist(), "variants": results, "best": best}
    with open(out / "render_dE.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBEST: γ={best['gamma']:.1f} α={best['alpha']:.2f}  ΔE={best['dE_render']:.2f}  L={best['render_L']}")
    print(f"Saved: {out}/render_dE.json")

    # Build comparison grid of best few
    top5 = sorted(valid, key=lambda r: r["dE_render"])[:5]
    tiles = []
    for r in top5:
        img = cv2.imread(r["render_path"])
        cv2.putText(img, f"g={r['gamma']:.1f} a={r['alpha']:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"dE={r['dE_render']:.1f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        tiles.append(img)
    grid = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(out / "top5_comparison.png"), grid)
    print(f"Saved: {out}/top5_comparison.png")


if __name__ == "__main__":
    sys.exit(main())
