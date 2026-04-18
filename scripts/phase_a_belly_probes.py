#!/usr/bin/env python3
"""Phase A: Belly empirical probes — correlation analysis.

Tests:
  H-A1: belly_iou severity ↔ joint 49 (belly_stretch) |θ|
  H-A2: belly_iou severity ↔ bone_length[13] extreme value
  H-A3: belly_iou severity ↔ rearing (spine vector heuristic)

Decision gate:
  r(belly_iou, rearing) > 0.6 → F6b strong → Phase B (rearing init)
  r(belly_iou, theta49) > 0.5 → F6a structural → Phase C (deformer impl)
  All r < 0.3 → F6d (GT mask) or unknown

Usage:
  python scripts/phase_a_belly_probes.py \
    --params-dir results/fitting/production_900_merged/params/ \
    --belly-iou-csv results/reports/belly_iou_paperfast_interp.csv \
    --output results/reports/260418_phase_a_correlations.csv
"""
import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_pkl_thetas_bone(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    thetas = d["thetas"][0].detach().cpu().numpy() if hasattr(d["thetas"], "detach") else d["thetas"][0]
    bone_lengths = d["bone_lengths"][0].detach().cpu().numpy() if hasattr(d["bone_lengths"], "detach") else d["bone_lengths"][0]
    return thetas, bone_lengths


def spine_angle_deg(kp3d_neck, kp3d_pelvis, mammal_up=np.array([0, -1, 0])):
    """Spine vector from pelvis to neck. Rearing = large +up angle."""
    v = kp3d_neck - kp3d_pelvis
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        return 0.0
    v_unit = v / v_norm
    # Cos angle with MAMMAL up (-Y direction)
    cos_up = np.dot(v_unit, mammal_up)
    return np.rad2deg(np.arccos(np.clip(cos_up, -1.0, 1.0)))


def extract_kp22_from_params(pkl_path, articulator=None):
    """Extract 22 keypoints from fitted params via MAMMAL forward.

    If articulator is None, returns proxy: uses joint positions from pkl if available.
    Otherwise runs forward pass.
    """
    # Simpler proxy: use joints from the BodyModel forward
    from articulation_th import ArticulationTorch
    if articulator is None:
        articulator = ArticulationTorch()

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    dev = articulator.device
    thetas = d["thetas"].to(dev)
    bone_lengths = d["bone_lengths"].to(dev)
    R_ = d["rotation"].to(dev)
    T = d["trans"].to(dev)
    s = d["scale"].to(dev)
    chest = d["chest_deformer"].to(dev)

    with torch.no_grad():
        verts, joints = articulator(thetas, bone_lengths, R_, T, s, chest)
    kp22 = articulator.forward_keypoints22()
    return kp22[0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-dir", required=True)
    ap.add_argument("--belly-iou-csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    # Load belly IoU (per-frame, per-view)
    belly_data = {}  # frame -> list of per-view belly_iou
    with open(args.belly_iou_csv) as fh:
        for row in csv.DictReader(fh):
            fid = int(row["frame"])
            belly_data.setdefault(fid, []).append(float(row["belly_iou"]))
    belly_per_frame = {fid: float(np.mean(vals)) for fid, vals in belly_data.items()}
    frames = sorted(belly_per_frame)[:args.max_frames]
    print(f"Frames with belly_iou: {len(frames)} (using {len(frames)})")

    # For each frame, find nearest keyframe pkl
    pkls = sorted(glob.glob(os.path.join(args.params_dir, "step_2_frame_*.pkl")))
    kf_ids = [int(os.path.basename(p).split("_")[-1].replace(".pkl", "")) for p in pkls]
    kf_map = dict(zip(kf_ids, pkls))

    # Extract features
    print("Extracting features from nearest keyframe pkl...")
    from articulation_th import ArticulationTorch
    articulator = ArticulationTorch()

    rows = []
    for fi in frames:
        # Nearest keyframe (≤)
        lower_kf = max([k for k in kf_ids if k <= fi], default=None)
        if lower_kf is None:
            continue
        pkl_path = kf_map[lower_kf]
        try:
            thetas, bone_lengths = load_pkl_thetas_bone(pkl_path)
        except Exception as e:
            print(f"  skip {fi}: {e}")
            continue

        theta49_mag = float(np.linalg.norm(thetas[49]))
        bone13_raw = float(bone_lengths[13])
        bone13_actual = 1.0 / (1.0 + np.exp(-bone13_raw)) + 0.5
        bone13_extreme = min(bone13_actual - 0.5, 1.5 - bone13_actual)

        # Spine angle via kp22
        try:
            kp22 = extract_kp22_from_params(pkl_path, articulator)
            # kp3 = neck_stretch, kp5 = lumbar_vertebrae_0 / tail_0 mean
            # Use kp2 (V mean, likely trunk) and kp3 (J neck)
            neck = kp22[3]
            pelvis = kp22[5]
            spine_ang = spine_angle_deg(neck, pelvis)
        except Exception as e:
            spine_ang = np.nan

        rows.append({
            "frame": fi,
            "kf": lower_kf,
            "belly_iou": belly_per_frame[fi],
            "theta49_mag": theta49_mag,
            "bone13_actual": bone13_actual,
            "bone13_extreme": bone13_extreme,
            "spine_angle_deg": spine_ang,
        })

    # Correlations
    if len(rows) < 3:
        print("Not enough samples for correlation")
        return

    b = np.array([r["belly_iou"] for r in rows])
    t = np.array([r["theta49_mag"] for r in rows])
    be = np.array([r["bone13_extreme"] for r in rows])
    sp = np.array([r["spine_angle_deg"] for r in rows])
    valid_sp = ~np.isnan(sp)

    def pearson(x, y):
        if len(x) < 3:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    print(f"\nN={len(rows)} samples")
    print(f"Belly IoU: mean={b.mean():.4f}, std={b.std():.4f}, range=[{b.min():.4f}, {b.max():.4f}]")
    print(f"θ49 mag:   mean={t.mean():.3f}, range=[{t.min():.3f}, {t.max():.3f}]")
    print(f"bone13 extreme dist to bound: mean={be.mean():.3f}")
    print(f"Spine angle (from MAMMAL up): mean={sp[valid_sp].mean():.1f}°, range=[{sp[valid_sp].min():.1f}, {sp[valid_sp].max():.1f}]")

    print("\n=== Correlations (Pearson r) ===")
    print(f"r(belly_iou, -θ49)         = {pearson(b, -t):+.3f}  (large |θ49| → low belly IoU?)")
    print(f"r(belly_iou, -bone13_extr) = {pearson(b, -be):+.3f}  (large bone13 extr → low belly IoU? note: lower value = farther from middle)")
    if valid_sp.sum() >= 3:
        print(f"r(belly_iou, -spine_ang)   = {pearson(b[valid_sp], -sp[valid_sp]):+.3f}  (rearing → low belly IoU?)")

    # Decision gate
    print("\n=== Decision Gate ===")
    r_theta = pearson(b, -t)
    r_spine = pearson(b[valid_sp], -sp[valid_sp]) if valid_sp.sum() >= 3 else 0
    if abs(r_spine) > 0.6:
        print(f"🔴 F6b strong (|r_spine|={r_spine:.3f} > 0.6) → Phase B rearing init pilot")
    if abs(r_theta) > 0.5:
        print(f"🟡 F6a structural (|r_θ49|={r_theta:.3f} > 0.5) → Phase C deformer impl")
    if abs(r_spine) < 0.3 and abs(r_theta) < 0.3:
        print(f"⚪ No strong signal (r_spine={r_spine:.3f}, r_θ49={r_theta:.3f}) → F6d GT mask")

    # Save CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
