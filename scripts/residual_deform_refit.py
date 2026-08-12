#!/usr/bin/env python3
"""Residual deformation pilot on top of an existing accurate refit OBJ.

This is intentionally small-scope:
- start from a fitted OBJ mesh;
- optimize a bounded per-vertex residual on a selected region;
- fit against the same 6-view foreground masks;
- report silhouette IoU and boundary Chamfer before/after.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

from articulation_th import ArticulationTorch  # only for face topology reuse
from mammal_ext.preprocessing.silhouette_renderer import (
    SilhouetteLoss,
    SilhouetteRenderer,
    load_target_mask,
    visualize_silhouette_comparison,
)


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
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


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


def select_region(verts: np.ndarray, region: str) -> np.ndarray:
    y, z = verts[:, 1], verts[:, 2]
    if region == "belly":
        z_med = float(np.percentile(z, 50))
        return np.where((y >= 40.0) & (y <= 90.0) & (z >= 5.0) & (z <= z_med))[0]
    if region == "lower_body":
        return np.where((y >= 20.0) & (y <= 110.0))[0]
    return np.arange(len(verts))


def build_vertex_neighbors(faces: np.ndarray, n_verts: int) -> list[list[int]]:
    neigh = [set() for _ in range(n_verts)]
    for f in faces:
        a, b, c = map(int, f)
        neigh[a].update([b, c])
        neigh[b].update([a, c])
        neigh[c].update([a, b])
    return [sorted(list(s)) for s in neigh]


def laplacian_loss(delta: torch.Tensor, neighbors: list[list[int]], selected: torch.Tensor) -> torch.Tensor:
    # delta: (1, V, 3)
    losses = []
    for vid in selected.tolist():
        ns = neighbors[vid]
        if not ns:
            continue
        center = delta[:, vid:vid + 1, :]
        mean_nb = delta[:, ns, :].mean(dim=1, keepdim=True)
        losses.append((center - mean_nb).pow(2).mean())
    return torch.stack(losses).mean() if losses else delta.pow(2).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--input-obj", required=True)
    ap.add_argument("--output-dir", default="results/fitting/residual_deform_pilot/")
    ap.add_argument("--region", default="belly", choices=["belly", "lower_body", "all"])
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--cam-path", default="data/raw/markerless_mouse_1_nerf/new_cam.pkl")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--mask-prefix", default="simpleclick_undist")
    ap.add_argument("--save-prefix", default="residual")
    args = ap.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.cam_path, "rb") as fh:
        cams = pickle.load(fh)

    verts_np, faces_np = load_obj(Path(args.input_obj))
    selected = select_region(verts_np, args.region)
    selected_t = torch.from_numpy(selected).long().to(device)
    neighbors = build_vertex_neighbors(faces_np, len(verts_np))

    # Infer image size from first GT mask.
    cap = cv2.VideoCapture(str(Path(args.data_dir) / args.mask_prefix / f"{args.views[0]}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("failed to read first target mask")
    height, width = first.shape[:2]

    renderer = SilhouetteRenderer(
        image_size=(height, width),
        device=device,
        bin_size=0,
        max_faces_per_bin=200000,
    )

    gt_masks: dict[int, torch.Tensor] = {}
    for vid in args.views:
        gt_masks[vid] = load_target_mask(
            str(Path(args.data_dir) / args.mask_prefix / f"{vid}.mp4"),
            args.frame,
            device=device,
        )

    faces = torch.from_numpy(faces_np).long().to(device)
    verts = torch.from_numpy(verts_np).float().to(device).unsqueeze(0)
    delta = torch.zeros_like(verts, requires_grad=True)
    delta.data[:, selected_t, :] = 0.0

    # Cameras
    camera_cache = {}
    for vid in args.views:
        cam = cams[vid]
        # Match the existing OpenCV -> PyTorch3D convention used elsewhere in the repo.
        R = torch.from_numpy(cam["R"].T).float().unsqueeze(0).to(device)
        T = torch.from_numpy(cam["T"]).float().reshape(1, 3).to(device)
        K = torch.from_numpy(cam["K"].T).float().unsqueeze(0).to(device)
        camera_cache[vid] = cameras_from_opencv_projection(
            R=R,
            tvec=T,
            camera_matrix=K,
            image_size=torch.tensor([[height, width]], device=device, dtype=torch.float32),
        )

    opt = torch.optim.Adam([delta], lr=args.lr)

    def render_current(v: torch.Tensor, vid: int) -> torch.Tensor:
        return renderer.render_from_vertices_faces(v, faces, camera_cache[vid])

    with torch.no_grad():
        base_metrics = {}
        for vid in args.views:
            pred = render_current(verts, vid)
            iou = 1.0 - SilhouetteLoss.iou_loss(pred, gt_masks[vid]).item()
            base_metrics[vid] = (pred, iou)

    best_score = -1e9
    best_delta = None
    history = []

    for it in range(args.iters):
        opt.zero_grad()
        v_cur = verts + delta
        pred_losses = []
        pred_ious = []
        pred_sils = []
        for vid in args.views:
            pred = render_current(v_cur, vid)
            loss_iou = SilhouetteLoss.iou_loss(pred, gt_masks[vid])
            pred_losses.append(loss_iou)
            pred_ious.append(float((1.0 - loss_iou.detach()).item()))
            pred_sils.append(pred.detach())

        loss_iou = torch.stack(pred_losses).mean()
        loss_delta = delta[:, selected_t, :].pow(2).mean()
        loss_lap = laplacian_loss(delta, neighbors, selected_t)
        loss = loss_iou + 0.05 * loss_delta + 0.2 * loss_lap
        loss.backward()
        opt.step()
        delta.data[:, [i for i in range(delta.shape[1]) if i not in selected.tolist()], :] = 0.0

        score = float(np.mean(pred_ious))
        if score > best_score:
            best_score = score
            best_delta = delta.detach().clone()

        if it % 20 == 0 or it == args.iters - 1:
            history.append({
                "iter": it,
                "loss": float(loss.item()),
                "iou_mean": score,
                "loss_iou": float(loss_iou.item()),
                "loss_delta": float(loss_delta.item()),
                "loss_lap": float(loss_lap.item()),
            })
            print(f"[iter {it:03d}] loss={loss.item():.4f} iou={score:.4f}")

    if best_delta is None:
        best_delta = delta.detach().clone()

    # Final metrics and save outputs.
    v_final = (verts + best_delta).detach()
    final_metrics = []
    for vid in args.views:
        pred = render_current(v_final, vid)
        pred_np = pred[0].detach().cpu().numpy() > 0.5
        gt_np = gt_masks[vid][0].detach().cpu().numpy() > 0.5
        iou = 1.0 - SilhouetteLoss.iou_loss(pred, gt_masks[vid]).item()
        ch = chamfer_px(pred_np, gt_np)
        final_metrics.append((vid, iou, ch))

    # Save visualizations for first view.
    vid0 = args.views[0]
    pred0 = render_current(v_final, vid0)
    pred_base = base_metrics[vid0][0]
    visualize_silhouette_comparison(pred_base[0], gt_masks[vid0][0], save_path=str(out_dir / f"{args.save_prefix}_base_v{vid0}.png"))
    visualize_silhouette_comparison(pred0[0], gt_masks[vid0][0], save_path=str(out_dir / f"{args.save_prefix}_final_v{vid0}.png"))
    cv2.imwrite(str(out_dir / f"{args.save_prefix}_base_sil_v{vid0}.png"), (pred_base[0].detach().cpu().numpy() * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / f"{args.save_prefix}_final_sil_v{vid0}.png"), (pred0[0].detach().cpu().numpy() * 255).astype(np.uint8))

    # Save OBJ-like deformed vertices for inspection.
    obj_path = out_dir / f"{args.save_prefix}_frame_{args.frame:06d}.obj"
    with obj_path.open("w") as fh:
        for v in v_final[0].detach().cpu().numpy():
            fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces_np + 1:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")

    csv_path = out_dir / f"{args.save_prefix}_frame_{args.frame:06d}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["view", "iou", "boundary_chamfer_px"])
        writer.writerows(final_metrics)

    hist_path = out_dir / f"{args.save_prefix}_history.json"
    hist_path.write_text(str(history))

    print(f"Saved: {obj_path}")
    print(f"Saved: {csv_path}")
    for vid, iou, ch in final_metrics:
        print(f"view {vid}: iou={iou:.4f} chamfer={ch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
