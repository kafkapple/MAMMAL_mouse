#!/usr/bin/env python3
"""Evaluate pre/post refit with rendered raster masks.

This is stricter than the older projected-vertex convex hull metric. It renders
the mesh from each calibrated camera, thresholds the rendered object silhouette,
and compares it against SimpleClick foreground masks.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RenderContext:
    renderer: object
    pyrender: object
    trimesh: object


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                toks = line.split()[1:4]
                faces.append([int(tok.split("/")[0]) - 1 for tok in toks])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def make_context(width: int, height: int) -> RenderContext:
    import pyrender
    import trimesh

    renderer = pyrender.OffscreenRenderer(width, height)
    return RenderContext(renderer=renderer, pyrender=pyrender, trimesh=trimesh)


def render_mask(
    ctx: RenderContext,
    verts: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    pyrender = ctx.pyrender
    trimesh = ctx.trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile([40, 40, 40, 255], (verts.shape[0], 1))
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pr_mesh)

    R = np.asarray(R)
    T = np.asarray(T).reshape(-1)
    Rwc = R.T
    twc = -Rwc @ T
    c2w_cv = np.eye(4)
    c2w_cv[:3, :3] = Rwc
    c2w_cv[:3, 3] = twc
    c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])

    cam = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        znear=1.0,
        zfar=10000.0,
    )
    scene.add(cam, pose=c2w_gl)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(light, pose=c2w_gl)

    _, depth = ctx.renderer.render(scene)
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth > 0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / max(union, 1))


def boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (mask_u8 - eroded).astype(bool)


def chamfer_px(pred: np.ndarray, gt: np.ndarray) -> float | None:
    bp = boundary(pred)
    bg = boundary(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return None
    # distanceTransform measures distance to zero pixels, so invert boundary maps.
    dt_gt = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (dt_gt[bp].mean() + dt_pred[bg].mean()))


def read_frames(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-dir", default="results/fitting/production_3600_canon/obj/")
    parser.add_argument("--post-dir", default="results/fitting/refit_outliers_152/obj/")
    parser.add_argument("--frame-list", default="conf/frames/outlier_severe_152.txt")
    parser.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    parser.add_argument("--comparison-csv", default="docs/reports/260420_refit_comparison.csv")
    parser.add_argument("--output", default="docs/reports/260520_refit_raster_eval_152.csv")
    parser.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    data_dir = Path(args.data_dir)
    with (data_dir / "new_cam.pkl").open("rb") as fh:
        cams = pickle.load(fh)

    frames = read_frames(Path(args.frame_list))
    if args.limit > 0:
        frames = frames[: args.limit]

    mask_caps = {
        view: cv2.VideoCapture(str(data_dir / "simpleclick_undist" / f"{view}.mp4"))
        for view in args.views
    }
    # Infer image size from first valid mask frame.
    first_cap = next(iter(mask_caps.values()))
    first_cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    ok, first = first_cap.read()
    if not ok:
        raise RuntimeError("failed to read first mask frame")
    height, width = first.shape[:2]
    ctx = make_context(width, height)

    belly_by_frame: dict[int, dict[str, str]] = {}
    comp_path = Path(args.comparison_csv)
    if comp_path.exists():
        for row in csv.DictReader(comp_path.open()):
            belly_by_frame[int(row["frame"])] = row

    rows: list[dict[str, object]] = []
    variants = {
        "pre": Path(args.pre_dir),
        "post": Path(args.post_dir),
    }

    try:
        for idx, frame in enumerate(frames, start=1):
            gt_masks: dict[int, np.ndarray] = {}
            for view, cap in mask_caps.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, mask_bgr = cap.read()
                if ok:
                    gt_masks[view] = mask_bgr[:, :, 0] > 127

            metrics: dict[str, dict[str, float | int | None]] = {}
            for variant, obj_dir in variants.items():
                obj_path = obj_dir / f"step_2_frame_{frame:06d}.obj"
                if not obj_path.exists():
                    continue
                verts, faces = load_obj(obj_path)
                ious: list[float] = []
                chamfers: list[float] = []
                for view in args.views:
                    if view not in gt_masks:
                        continue
                    pred = render_mask(
                        ctx,
                        verts,
                        faces,
                        cams[view]["K"],
                        cams[view]["R"],
                        cams[view]["T"],
                        height,
                        width,
                    )
                    gt = gt_masks[view]
                    ious.append(iou(pred, gt))
                    ch = chamfer_px(pred, gt)
                    if ch is not None:
                        chamfers.append(ch)
                metrics[variant] = {
                    "iou_mean": float(np.mean(ious)) if ious else None,
                    "iou_min": float(np.min(ious)) if ious else None,
                    "chamfer_mean": float(np.mean(chamfers)) if chamfers else None,
                    "n_views": len(ious),
                }

            pre = metrics.get("pre", {})
            post = metrics.get("post", {})
            old = belly_by_frame.get(frame, {})
            row = {
                "frame": frame,
                "pre_raster_iou": pre.get("iou_mean"),
                "post_raster_iou": post.get("iou_mean"),
                "d_raster_iou": (
                    post.get("iou_mean") - pre.get("iou_mean")
                    if post.get("iou_mean") is not None and pre.get("iou_mean") is not None
                    else None
                ),
                "pre_raster_iou_min": pre.get("iou_min"),
                "post_raster_iou_min": post.get("iou_min"),
                "pre_chamfer_px": pre.get("chamfer_mean"),
                "post_chamfer_px": post.get("chamfer_mean"),
                "d_chamfer_px": (
                    post.get("chamfer_mean") - pre.get("chamfer_mean")
                    if post.get("chamfer_mean") is not None and pre.get("chamfer_mean") is not None
                    else None
                ),
                "n_views": post.get("n_views", pre.get("n_views", 0)),
                "pre_convex_iou": old.get("pre_iou_global", ""),
                "post_convex_iou": old.get("post_iou_global", ""),
                "d_convex_iou": old.get("d_iou_global", ""),
                "pre_belly_iou": old.get("pre_iou_belly", ""),
                "post_belly_iou": old.get("post_iou_belly", ""),
                "d_belly_iou": old.get("d_iou_belly", ""),
            }
            rows.append(row)
            if idx % 10 == 0 or idx == len(frames):
                print(f"[{idx}/{len(frames)}] frame {frame}", flush=True)
    finally:
        ctx.renderer.delete()
        for cap in mask_caps.values():
            cap.release()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["d_raster_iou"] is not None]
    if valid:
        d_iou = np.asarray([float(r["d_raster_iou"]) for r in valid])
        d_ch = np.asarray([float(r["d_chamfer_px"]) for r in valid if r["d_chamfer_px"] is not None])
        print(f"Saved: {out} ({len(rows)} rows)")
        print(
            "Raster IoU mean: "
            f"{np.mean([float(r['pre_raster_iou']) for r in valid]):.4f} -> "
            f"{np.mean([float(r['post_raster_iou']) for r in valid]):.4f} "
            f"(d {d_iou.mean():+.4f}); improved {(d_iou > 0).sum()}/{len(d_iou)}"
        )
        if len(d_ch):
            print(f"Boundary Chamfer d mean: {d_ch.mean():+.4f}px; improved {(d_ch < 0).sum()}/{len(d_ch)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
