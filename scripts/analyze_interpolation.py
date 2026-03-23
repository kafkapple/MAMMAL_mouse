#!/usr/bin/env python3
"""
Analyze interpolation quality from dense fitting results.
Simulates keyframe + interpolation at various intervals and measures error.

Usage:
    # After E4 dense fitting completes:
    CUDA_VISIBLE_DEVICES=5 python scripts/analyze_interpolation.py

    # Custom params dir:
    python scripts/analyze_interpolation.py --params-dir results/fitting/dense_accurate_0_100/params/

    # Use keypoints-only (fast baseline, no params needed):
    python scripts/analyze_interpolation.py --mode keypoints
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_params(params_dir):
    """Load all step_2 params from a fitting result directory."""
    files = sorted(glob.glob(os.path.join(params_dir, "step_2_frame_*.pkl")))
    if not files:
        files = sorted(glob.glob(os.path.join(params_dir, "step_1_frame_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No param files in {params_dir}")

    frames = []
    params_list = []
    for f in files:
        fid = int(f.split("frame_")[1].split(".")[0])
        p = pickle.load(open(f, "rb"))
        # Convert torch tensors to numpy
        d = {}
        for k, v in p.items():
            if hasattr(v, "detach"):
                d[k] = v.detach().cpu().numpy()
            else:
                d[k] = np.array(v)
        frames.append(fid)
        params_list.append(d)

    return np.array(frames), params_list


def interpolate_params(params_a, params_b, alpha=0.5):
    """Linear interpolation of MAMMAL parameters."""
    result = {}
    for key in params_a:
        a, b = params_a[key], params_b[key]
        if key in ("rotation", "thetas"):
            # For axis-angle rotations, linear interp is approximate but fast
            # Slerp would be more accurate for large rotations
            result[key] = a * (1 - alpha) + b * alpha
        else:
            result[key] = a * (1 - alpha) + b * alpha
    return result


def params_to_vertices(params, body_model):
    """Forward pass: params → mesh vertices using MAMMAL body model."""
    import torch
    device = next(body_model.parameters()).device if hasattr(body_model, 'parameters') else 'cpu'

    thetas = torch.from_numpy(params["thetas"]).float().to(device)
    bone_lengths = torch.from_numpy(params["bone_lengths"]).float().to(device)
    rotation = torch.from_numpy(params["rotation"]).float().to(device)
    trans = torch.from_numpy(params["trans"]).float().to(device)
    scale = torch.from_numpy(params["scale"]).float().to(device)
    chest_deformer = torch.from_numpy(params["chest_deformer"]).float().to(device)

    V, _ = body_model.forward(thetas, bone_lengths, rotation, trans, scale, chest_deformer)
    return V[0].detach().cpu().numpy()


def analyze_keypoints_mode(output_dir):
    """Analyze interpolation using existing 3600-frame keypoints (no params needed)."""
    npz_path = "results/fitting/baseline_fast_3600/keypoints_22_3d.npz"
    d = np.load(npz_path)
    kps = d["keypoints"]  # (3600, 22, 3)
    body_len = np.linalg.norm(kps[:, 2] - kps[:, 4], axis=1).mean()

    intervals = [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]
    results = {}

    print(f"Keypoints: {kps.shape[0]} frames, body length ref: {body_len:.1f}mm")
    print(f"\n{'Interval':>8} {'Gap_s':>6} {'Mean_mm':>8} {'P95_mm':>8} {'Mean%':>7} {'P95%':>7} {'<3mm':>6} {'KF':>5}")
    print("-" * 65)

    for N in intervals:
        errors = []
        for i in range(N, len(kps) - N, N):
            gt = kps[i]
            interp = (kps[i - N] + kps[min(i + N, len(kps) - 1)]) / 2.0
            err = np.linalg.norm(gt - interp, axis=1)
            errors.append(err.mean())
        errors = np.array(errors)
        gap = N * 5 / 100.0
        mean = errors.mean()
        p95 = np.percentile(errors, 95)
        pct_ok = 100 * (errors < 3).sum() / len(errors)
        n_kf = len(kps) // N

        results[N] = {
            "gap_sec": gap,
            "mean_mm": float(mean),
            "p95_mm": float(p95),
            "mean_pct": float(mean / body_len * 100),
            "p95_pct": float(p95 / body_len * 100),
            "pct_under_3mm": float(pct_ok),
            "keyframes": n_kf,
        }
        print(f"{N:>8} {gap:>6.2f} {mean:>8.2f} {p95:>8.2f} {mean/body_len*100:>6.1f}% {p95/body_len*100:>6.1f}% {pct_ok:>5.1f}% {n_kf:>5}")

    return results, body_len


def analyze_params_mode(params_dir, output_dir):
    """Analyze interpolation using fitted params → vertices comparison."""
    frames, params_list = load_params(params_dir)
    print(f"Loaded {len(frames)} frames from {params_dir}")
    print(f"Frame range: {frames[0]} - {frames[-1]}, step={frames[1]-frames[0] if len(frames)>1 else 'N/A'}")

    # Load body model for vertex generation
    from articulation_th import ArticulationTorch
    body_model = ArticulationTorch()

    # Generate vertices for all frames
    print("Generating vertices...")
    vertices_all = []
    for i, p in enumerate(params_list):
        v = params_to_vertices(p, body_model)
        vertices_all.append(v)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(params_list)}")
    vertices_all = np.array(vertices_all)  # (N, V, 3)

    # Body length reference
    # Use nose (joint approx) - body center (vertex centroid difference)
    body_sizes = np.linalg.norm(
        vertices_all[:, :100].mean(axis=1) - vertices_all[:, -100:].mean(axis=1), axis=1
    )
    body_len = body_sizes.mean()

    intervals = [2, 3, 4, 6, 8, 12]
    results = {}

    step = frames[1] - frames[0] if len(frames) > 1 else 5

    print(f"\nBody size ref: {body_len:.1f}mm")
    print(f"\n{'Interval':>8} {'Gap_s':>6} {'Mean_mm':>8} {'P95_mm':>8} {'Max_mm':>8} {'Mean%':>7} {'P95%':>7}")
    print("-" * 65)

    for N in intervals:
        if N >= len(vertices_all) // 2:
            continue
        errors = []
        for i in range(N, len(vertices_all) - N, 1):
            gt = vertices_all[i]
            interp = (vertices_all[i - N] + vertices_all[min(i + N, len(vertices_all) - 1)]) / 2.0
            err = np.linalg.norm(gt - interp, axis=1)  # per-vertex error
            errors.append(err.mean())

        errors = np.array(errors)
        gap = N * step / 100.0
        mean = errors.mean()
        p95 = np.percentile(errors, 95)
        mx = errors.max()

        results[N] = {
            "gap_sec": float(gap),
            "mean_mm": float(mean),
            "p95_mm": float(p95),
            "max_mm": float(mx),
            "mean_pct": float(mean / body_len * 100),
            "p95_pct": float(p95 / body_len * 100),
        }
        print(f"{N:>8} {gap:>6.2f} {mean:>8.2f} {p95:>8.2f} {mx:>8.2f} {mean/body_len*100:>6.1f}% {p95/body_len*100:>6.1f}%")

    return results, body_len


def main():
    parser = argparse.ArgumentParser(description="Interpolation quality analysis")
    parser.add_argument("--mode", choices=["keypoints", "params"], default="params",
                        help="Analysis mode: keypoints (fast baseline) or params (vertex-level)")
    parser.add_argument("--params-dir", nargs="+",
                        default=["results/fitting/dense_accurate_0_100/params/",
                                 "results/fitting/dense_accurate_100_200/params/"],
                        help="Params directories for params mode")
    parser.add_argument("--output", default="results/comparison/interpolation/",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.mode == "keypoints":
        results, body_len = analyze_keypoints_mode(args.output)
        label = "keypoints_fast_baseline"
    else:
        # Merge params from multiple dirs
        all_results = {}
        for d in args.params_dir:
            if os.path.exists(d):
                r, body_len = analyze_params_mode(d, args.output)
                all_results.update(r)
            else:
                print(f"WARNING: {d} not found, skipping")
        results = all_results
        label = "params_accurate"

    # Save results
    json_path = os.path.join(args.output, f"interpolation_{label}.json")
    with open(json_path, "w") as f:
        json.dump({"body_len_mm": body_len, "intervals": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
