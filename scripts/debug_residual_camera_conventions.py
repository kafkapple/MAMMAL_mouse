#!/usr/bin/env python3
"""Debug PyTorch3D camera conventions for residual deformation fitting."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch3d.utils import cameras_from_opencv_projection

from mammal_ext.preprocessing.silhouette_renderer import (
    SilhouetteLoss,
    SilhouetteRenderer,
    load_target_mask,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=1450)
    ap.add_argument(
        "--obj",
        default="results/fitting/refit_outliers_152/obj/step_2_frame_001450.obj",
    )
    ap.add_argument("--cam-path", default="data/raw/markerless_mouse_1_nerf/new_cam.pkl")
    ap.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os = __import__("os")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    obj_path = Path(args.obj)

    with Path(args.cam_path).open("rb") as fh:
        cams = pickle.load(fh)
    cam = cams[args.view]

    verts_np, faces_np = load_obj(obj_path)
    verts = torch.from_numpy(verts_np).float().to(device).unsqueeze(0)
    faces = torch.from_numpy(faces_np).long().to(device)

    cap = cv2.VideoCapture(str(data_dir / "simpleclick_undist" / f"{args.view}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("failed to load a target mask frame")
    height, width = first.shape[:2]

    renderer = SilhouetteRenderer(
        image_size=(height, width),
        device=device,
        bin_size=0,
        max_faces_per_bin=200000,
    )
    gt = load_target_mask(
        str(data_dir / "simpleclick_undist" / f"{args.view}.mp4"),
        args.frame,
        device=device,
    )

    combos = {
        "R/K": (cam["R"], cam["K"]),
        "Rt/K": (cam["R"].T, cam["K"]),
        "R/Kt": (cam["R"], cam["K"].T),
        "Rt/Kt": (cam["R"].T, cam["K"].T),
    }

    for name, (R_np, K_np) in combos.items():
        R = torch.from_numpy(np.asarray(R_np)).float().unsqueeze(0).to(device)
        K = torch.from_numpy(np.asarray(K_np)).float().unsqueeze(0).to(device)
        T = torch.from_numpy(np.asarray(cam["T"])).float().reshape(1, 3).to(device)
        cam_p3d = cameras_from_opencv_projection(
            R=R,
            tvec=T,
            camera_matrix=K,
            image_size=torch.tensor([[height, width]], device=device, dtype=torch.float32),
        )
        with torch.no_grad():
            pred = renderer.render_from_vertices_faces(verts, faces, cam_p3d)
            iou = 1.0 - SilhouetteLoss.iou_loss(pred, gt).item()
        print(f"{name}: iou={iou:.4f} pred_mean={pred.mean().item():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
