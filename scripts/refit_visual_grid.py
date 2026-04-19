#!/usr/bin/env python3
"""Generate GT | paper_fast | accurate refit visual grid for refit comparison.

Usage:
    python scripts/refit_visual_grid.py \
        --frames 2700 5230 17670 13315 890 \
        --output ~/results/MAMMAL/260420_refit_grid/
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[2700, 5230, 17670, 13315, 890, 17755, 12095, 10055])
    ap.add_argument("--pre-obj-dir", default="results/fitting/production_3600_canon/obj/")
    ap.add_argument("--post-obj-dir", default="results/fitting/refit_outliers_152/obj/")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--output", default="results/refit_visual_comparison/")
    ap.add_argument("--texture", default="results/sweep/production_p0/texture_final.png")
    ap.add_argument("--render-view", type=int, default=0,
                    help="Which data camera to render from (0-5)")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    tiles_by_frame = {}
    for fid in args.frames:
        pre_obj = os.path.join(args.pre_obj_dir, f"step_2_frame_{fid:06d}.obj")
        post_obj = os.path.join(args.post_obj_dir, f"step_2_frame_{fid:06d}.obj")

        # 1. GT: extract from video
        cap = cv2.VideoCapture(os.path.join(args.data_dir, "videos_undist", f"{args.render_view}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, gt_bgr = cap.read(); cap.release()
        if not ok:
            print(f"skip f{fid}: GT read failed"); continue

        # 2. Render pre + post via render_same_camera_6view.py (single frame)
        tiles = {"GT": gt_bgr}
        for variant, obj_path in [("paper_fast", pre_obj), ("accurate", post_obj)]:
            if not os.path.exists(obj_path):
                print(f"skip f{fid} {variant}: {obj_path} missing")
                continue
            v_out = out / f"frame_{fid:06d}_{variant}"
            v_out.mkdir(exist_ok=True)
            cmd = ["python", "scripts/render_same_camera_6view.py",
                   "--frame", str(fid), "--obj", obj_path,
                   "--output", str(v_out), "--texture-img", args.texture]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "5"
            try:
                subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"render f{fid} {variant} failed: {e}"); continue
            render_p = v_out / f"render_v{args.render_view}.png"
            if render_p.exists():
                tiles[variant] = cv2.imread(str(render_p))
        if len(tiles) < 3:
            print(f"skip f{fid}: only {len(tiles)} variants available"); continue
        tiles_by_frame[fid] = tiles
        print(f"f{fid}: 3 variants ready")

    if not tiles_by_frame:
        print("No complete frame triplets. Exiting."); return 1

    # Grid: each row = 1 frame, 3 columns (GT, paper_fast, accurate)
    tW, tH = 400, 360
    bar = 32
    n_rows = len(tiles_by_frame)
    grid = np.full((n_rows * (tH + bar), 3 * tW, 3), 255, np.uint8)
    order = ["GT", "paper_fast", "accurate"]
    for r, (fid, tiles) in enumerate(sorted(tiles_by_frame.items())):
        yy = r * (tH + bar)
        cv2.putText(grid, f"Frame {fid}", (8, yy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 2)
        for c, key in enumerate(order):
            if key not in tiles: continue
            img = cv2.resize(tiles[key], (tW, tH))
            cv2.putText(grid, key, (c * tW + 8, yy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 2)
            grid[yy + bar:yy + bar + tH, c * tW:(c + 1) * tW] = img

    gp = out / "refit_comparison_grid.png"
    cv2.imwrite(str(gp), grid)
    print(f"\nSaved: {gp}")


if __name__ == "__main__":
    sys.exit(main())
