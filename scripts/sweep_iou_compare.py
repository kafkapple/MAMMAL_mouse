#!/usr/bin/env python3
"""Compare IoU across sweep configs on worst 5 frames (CPU only, no GPU needed)."""
import os, sys, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAMES = [9480, 9360, 5520, 1320, 8400]
VIEW = 3

import pickle
with open("data/raw/markerless_mouse_1_nerf/new_cam.pkl", "rb") as f:
    cams = pickle.load(f)
K, R, T = cams[VIEW]["K"], cams[VIEW]["R"], cams[VIEW]["T"]

from articulation_th import ArticulationTorch
faces = ArticulationTorch().faces_vert_np

def load_verts(path):
    v = []
    with open(path) as f:
        for l in f:
            if l.startswith("v "):
                p = l.split()
                v.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(v, dtype=np.float32)

def render_mask(verts):
    pc = (R @ verts.T).T + T
    p2h = (K @ pc.T).T
    d = p2h[:, 2]
    p2 = p2h[:, :2] / p2h[:, 2:]
    m = np.zeros((1024, 1152), dtype=np.uint8)
    for fi in np.argsort(-d[faces].mean(axis=1)):
        f = faces[fi]
        if (d[f] < 0).any(): continue
        cv2.fillPoly(m, [p2[f].astype(np.int32).reshape(-1,1,2)], 255)
    return m.astype(np.float32) / 255.0

def iou(pred, gt):
    p, g = (pred > 0.5).astype(float), (gt > 0.5).astype(float)
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float(inter / union) if union > 0 else 0

# Cache GT masks
gt = {}
for fid in FRAMES:
    cap = cv2.VideoCapture("data/raw/markerless_mouse_1_nerf/simpleclick_undist/3.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    _, fr = cap.read()
    cap.release()
    gt[fid] = fr[:,:,0].astype(np.float32) / 255.0

configs = [
    "sweep_s1_100_m_1000", "sweep_s1_100_m_3000", "sweep_s1_100_m_5000",
    "sweep_s1_200_m_1000", "sweep_s1_200_m_3000", "sweep_s1_200_m_5000",
    "sweep_s1_400_m_1000", "sweep_s1_400_m_3000", "sweep_s1_400_m_5000",
]

print(f"{'Config':<25} {'Mean':>7} {'Min':>7}  9480  9360  5520  1320  8400")
print("-" * 80)

results = {}
for cfg in configs:
    ious = []
    for fid in FRAMES:
        obj = f"results/fitting/{cfg}/obj/step_2_frame_{fid:06d}.obj"
        if not os.path.exists(obj):
            ious.append(0); continue
        ious.append(iou(render_mask(load_verts(obj)), gt[fid]))
    results[cfg] = {"mean": np.mean(ious), "ious": ious}
    s = " ".join(f"{x:.3f}" for x in ious)
    print(f"{cfg:<25} {np.mean(ious):>7.3f} {np.min(ious):>7.3f}  {s}")

print()
# Baselines
for label, obj_dir in [("fast_baseline", "/home/joon/data/synthetic/textured_obj"),
                        ("accurate_E2", "results/fitting/refit_accurate_23/obj")]:
    ious = []
    for fid in FRAMES:
        obj = f"{obj_dir}/step_2_frame_{fid:06d}.obj"
        if not os.path.exists(obj):
            ious.append(0); continue
        ious.append(iou(render_mask(load_verts(obj)), gt[fid]))
    s = " ".join(f"{x:.3f}" for x in ious)
    print(f"{label:<25} {np.mean(ious):>7.3f} {np.min(ious):>7.3f}  {s}")

os.makedirs("results/comparison/sweep", exist_ok=True)
with open("results/comparison/sweep/sweep_iou.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/comparison/sweep/sweep_iou.json")
