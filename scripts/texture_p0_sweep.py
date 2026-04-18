#!/usr/bin/env python3
"""P0 variants sweep — gamma × hist_match_alpha, find minimum ΔE config.

CPU-only post-hoc operation on existing UV texture. No GPU needed.
Produces: sweep_results.json with ΔE per (gamma, alpha) combination.

Caveat: ΔE is measured on the UV texture itself (dark pixel mean vs GT body mean).
Final verification requires rendering each variant and re-measuring on rendered output,
but UV-space ΔE is a strong proxy.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def gt_body_lab(gt_path: str, gray_lo: int = 10, gray_hi: int = 100) -> np.ndarray:
    gt = cv2.imread(gt_path)
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (gray < gray_hi) & (gray > gray_lo)
    lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[mask].mean(0)


def p0_variant(tex_path: str, gamma: float, hist_alpha: float, gt_lab: np.ndarray) -> dict:
    tex = cv2.imread(tex_path)
    tex_rgb = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Gamma darkening on non-white pixels (UV foreground)
    mask_fg = np.any(tex_rgb < 0.95, axis=-1)
    tex_g = np.power(tex_rgb, gamma)
    out = tex_rgb.copy()
    out[mask_fg] = tex_g[mask_fg]

    # Histogram match toward GT body mean Lab, blended by hist_alpha
    out_lab = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    fg = mask_fg
    current_mean = out_lab[fg].mean(0)
    shift = (gt_lab - current_mean) * hist_alpha
    out_lab[fg] += shift
    out_lab = np.clip(out_lab, 0, 255)
    out_rgb = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # Body mean Lab → ΔE
    final_lab = cv2.cvtColor((out_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    body_lab_mean = final_lab[fg].mean(0)
    dE = float(np.sqrt(np.sum((body_lab_mean - gt_lab) ** 2)))

    return {"gamma": gamma, "hist_alpha": hist_alpha,
            "body_Lab": body_lab_mean.tolist(), "dE_uv": dE,
            "rgb_out": out_rgb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="results/sweep/run_wild-sweep-9/texture_final.png",
                    help="Source UV texture (sweep-9 raw-average)")
    ap.add_argument("--gt", default="/tmp/gt_rgb_f1800_v0.png",
                    help="GT RGB frame for histogram target")
    ap.add_argument("--output", default="results/texture_experiment_v1/p0_sweep/")
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.6, 2.0, 2.2, 2.4])
    ap.add_argument("--hist-alphas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_lab = gt_body_lab(args.gt)
    print(f"GT body Lab: L={gt_lab[0]:.1f} a={gt_lab[1] - 128:.1f} b={gt_lab[2] - 128:.1f}")

    results = []
    best = {"dE_uv": 1e9}
    for g in args.gammas:
        for a in args.hist_alphas:
            r = p0_variant(args.tex, g, a, gt_lab)
            # Save texture PNG
            name = f"gamma{g:.1f}_alpha{a:.1f}"
            tex_out = (r["rgb_out"] * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"tex_{name}.png"), cv2.cvtColor(tex_out, cv2.COLOR_RGB2BGR))
            r_entry = {k: v for k, v in r.items() if k != "rgb_out"}
            r_entry["file"] = f"tex_{name}.png"
            results.append(r_entry)
            print(f"gamma={g:.1f} alpha={a:.2f}  L={r['body_Lab'][0]:.1f}  ΔE_uv={r['dE_uv']:.2f}")
            if r["dE_uv"] < best["dE_uv"]:
                best = r_entry

    report = {
        "gt_body_Lab": gt_lab.tolist(),
        "best": best,
        "all": results,
        "note": "ΔE measured on UV texture body pixels. Final render ΔE may differ due to lighting/occlusion."
    }
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBest: {best['file']}  ΔE_uv={best['dE_uv']:.2f}")
    print(f"Saved: {out_dir}/sweep_results.json")


if __name__ == "__main__":
    main()
