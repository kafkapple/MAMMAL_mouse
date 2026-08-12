#!/usr/bin/env python3
"""Low-dimensional residual deformation search on top of an accurate refit.

This is intentionally conservative:
- use the already validated pyrender raster pipeline;
- perturb a small belly/flank vertex region with a low-dimensional basis;
- score each candidate by rendered raster IoU and boundary Chamfer;
- keep the best candidate and export a comparison grid.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path

import cv2
import numpy as np


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


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w") as fh:
        for v in verts:
            fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces + 1:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")


def select_region(verts: np.ndarray, region: str) -> np.ndarray:
    y, z = verts[:, 1], verts[:, 2]
    if region == "belly":
        z_med = float(np.percentile(z, 50))
        return np.where((y >= 40.0) & (y <= 90.0) & (z >= 5.0) & (z <= z_med))[0]
    if region == "belly_lower":
        z_q25 = float(np.percentile(z, 25))
        z_med = float(np.percentile(z, 50))
        return np.where((y >= 40.0) & (y <= 90.0) & (z >= 5.0) & (z <= z_q25))[0]
    if region == "lower_body":
        return np.where((y >= 20.0) & (y <= 110.0))[0]
    return np.arange(len(verts))


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts, dtype=np.float32)
    for f in faces:
        a, b, c = map(int, f)
        v0, v1, v2 = verts[a], verts[b], verts[c]
        n = np.cross(v1 - v0, v2 - v0)
        normals[a] += n
        normals[b] += n
        normals[c] += n
    norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals / norm


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
    dt_gt = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (dt_gt[bp].mean() + dt_pred[bg].mean()))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / max(union, 1))


def make_context(width: int, height: int):
    import pyrender
    import trimesh

    renderer = pyrender.OffscreenRenderer(width, height)
    return renderer, pyrender, trimesh


def render_mask(renderer, pyrender, trimesh, verts, faces, K, R, T, height, width):
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

    _, depth = renderer.render(scene)
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth > 0


def render_mask_single(verts, faces, K, R, T, height, width):
    renderer, pyrender, trimesh = make_context(width, height)
    try:
        return render_mask(renderer, pyrender, trimesh, verts, faces, K, R, T, height, width)
    finally:
        renderer.delete()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--input-obj", required=True)
    ap.add_argument("--output-dir", default="results/fitting/residual_basis_search/")
    ap.add_argument("--cam-path", default="data/raw/markerless_mouse_1_nerf/new_cam.pkl")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--region", default="belly", choices=["belly", "belly_lower", "lower_body", "all"])
    ap.add_argument("--mode", default="normal", choices=["normal", "axis"])
    ap.add_argument("--z-grid", type=float, nargs="+", default=[-8, -4, 0, 4, 8, 12])
    ap.add_argument("--y-grid", type=float, nargs="+", default=[-4, 0, 4])
    ap.add_argument("--normal-grid", type=float, nargs="+", default=[-6, -3, 0, 3, 6, 9])
    ap.add_argument("--score-chamfer-weight", type=float, default=0.001)
    args = ap.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.cam_path).open("rb") as fh:
        cams = pickle.load(fh)

    verts, faces = load_obj(Path(args.input_obj))
    region_idx = select_region(verts, args.region)
    region_mask = np.zeros(len(verts), dtype=bool)
    region_mask[region_idx] = True

    basis_n = np.zeros_like(verts)
    basis_z = np.zeros_like(verts)
    basis_y = np.zeros_like(verts)
    basis_z[region_mask, 2] = 1.0
    basis_y[region_mask, 1] = 1.0
    if args.mode == "normal":
        basis_n[region_mask] = vertex_normals(verts, faces)[region_mask]

    mask_caps = {}
    gt_masks: dict[int, np.ndarray] = {}
    first_h = first_w = None
    for vid in args.views:
        cap = cv2.VideoCapture(str(Path(args.data_dir) / "simpleclick_undist" / f"{vid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, mask_bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"failed to read target mask for view {vid}")
        gt = mask_bgr[:, :, 0] > 127
        gt_masks[vid] = gt
        if first_h is None:
            first_h, first_w = gt.shape
    renderer, pyrender, trimesh = make_context(first_w, first_h)

    candidates = []
    if args.mode == "axis":
        for dz in args.z_grid:
            for dy in args.y_grid:
                candidates.append((dz, dy))
    else:
        for dn in args.normal_grid:
            candidates.append((dn, 0.0))

    rows = []
    best = None
    best_score = -1e9

    try:
        for a, b in candidates:
            if args.mode == "axis":
                dz, dy = a, b
                v = verts + dz * basis_z + dy * basis_y
            else:
                dn = a
                v = verts + dn * basis_n
            per_view = []
            for vid in args.views:
                cam = cams[vid]
                pred = render_mask(
                    renderer,
                    pyrender,
                    trimesh,
                    v,
                    faces,
                    cam["K"],
                    cam["R"],
                    cam["T"],
                    first_h,
                    first_w,
                )
                giou = iou(pred, gt_masks[vid])
                ch = chamfer_px(pred, gt_masks[vid])
                per_view.append((vid, giou, ch))
            ious = [x[1] for x in per_view]
            chs = [x[2] for x in per_view if x[2] is not None]
            mean_iou = float(np.mean(ious))
            mean_ch = float(np.mean(chs)) if chs else float("nan")
            score = mean_iou - args.score_chamfer_weight * mean_ch
            if args.mode == "axis":
                row = {
                    "dz": dz,
                    "dy": dy,
                    "mean_raster_iou": mean_iou,
                    "mean_boundary_chamfer_px": mean_ch,
                    "score": score,
                }
                msg = f"dz={dz:+.1f} dy={dy:+.1f}"
            else:
                row = {
                    "dn": dn,
                    "mean_raster_iou": mean_iou,
                    "mean_boundary_chamfer_px": mean_ch,
                    "score": score,
                }
                msg = f"dn={dn:+.1f}"
            rows.append(row)
            print(f"{msg} iou={mean_iou:.4f} chamfer={mean_ch:.2f} score={score:.4f}")
            if score > best_score:
                best_score = score
                best = (a, b, v, per_view, mean_iou, mean_ch, score)
    finally:
        renderer.delete()

    if best is None:
        raise RuntimeError("no candidate evaluated")

    a_best, b_best, v_best, per_view_best, mean_iou_best, mean_ch_best, score_best = best
    best_obj = out_dir / f"best_residual_frame_{args.frame:06d}.obj"
    save_obj(best_obj, v_best, faces)

    # Baseline for comparison is the provided input object.
    base_rows = []
    for vid in args.views:
        cam = cams[vid]
        cap = cv2.VideoCapture(str(Path(args.data_dir) / "simpleclick_undist" / f"{vid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, mask_bgr = cap.read()
        cap.release()
        if not ok:
            continue
        gt = mask_bgr[:, :, 0] > 127
        if first_h is None:
            first_h, first_w = gt.shape
        base_pred = render_mask_single(verts, faces, cam["K"], cam["R"], cam["T"], first_h, first_w)
        base_rows.append({
            "view": vid,
            "base_iou": iou(base_pred, gt),
            "base_chamfer_px": chamfer_px(base_pred, gt),
            "best_iou": per_view_best[vid][1],
            "best_chamfer_px": per_view_best[vid][2],
        })

    csv_path = out_dir / f"residual_basis_search_frame_{args.frame:06d}.csv"
    with csv_path.open("w", newline="") as fh:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    compare_path = out_dir / f"residual_basis_compare_frame_{args.frame:06d}.csv"
    with compare_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "view", "base_iou", "base_chamfer_px", "best_iou", "best_chamfer_px"
        ])
        writer.writeheader()
        writer.writerows(base_rows)

    # Save a simple summary grid for the first view.
    vid0 = args.views[0]
    cap = cv2.VideoCapture(str(Path(args.data_dir) / "simpleclick_undist" / f"{vid0}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, mask_bgr = cap.read()
    cap.release()
    gt = mask_bgr[:, :, 0] > 127
    cam = cams[vid0]
    base_pred = render_mask_single(verts, faces, cam["K"], cam["R"], cam["T"], first_h, first_w)
    best_pred = render_mask_single(v_best, faces, cam["K"], cam["R"], cam["T"], first_h, first_w)
    grid = np.zeros((first_h * 2, first_w * 2, 3), dtype=np.uint8)
    grid[:first_h, :first_w] = np.stack([gt * 255] * 3, axis=-1)
    grid[:first_h, first_w:] = np.stack([base_pred * 255] * 3, axis=-1)
    grid[first_h:, :first_w] = np.stack([best_pred * 255] * 3, axis=-1)
    delta = (best_pred.astype(np.int8) - base_pred.astype(np.int8))
    delta_vis = np.zeros((first_h, first_w, 3), dtype=np.uint8)
    delta_vis[..., 1] = (delta > 0).astype(np.uint8) * 255
    delta_vis[..., 2] = (delta < 0).astype(np.uint8) * 255
    grid[first_h:, first_w:] = delta_vis
    cv2.imwrite(str(out_dir / f"residual_basis_grid_frame_{args.frame:06d}.png"), grid)

    if args.mode == "axis":
        print(f"best dz={a_best:+.1f} dy={b_best:+.1f} mean_iou={mean_iou_best:.4f} mean_chamfer={mean_ch_best:.2f}")
    else:
        print(f"best dn={a_best:+.1f} mean_iou={mean_iou_best:.4f} mean_chamfer={mean_ch_best:.2f}")
    print(f"saved {best_obj}")
    print(f"saved {csv_path}")
    print(f"saved {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
