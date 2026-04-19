#!/usr/bin/env python3
"""F6g: Belly vertex skinning weight visualization.

Check whether belly vertices are correctly bound to ventral/abdominal joints.
Expected: ventral joints dominate belly vertex weights.
If mid-spine or distant joints dominate → rigging defect (F6g confirmed).

Inputs:
    mouse_model/mouse.pkl — MAMMAL body model (LBS weights + joint hierarchy)
    results/fitting/production_3600_canon/obj/step_2_frame_001800.obj — canon frame

Outputs:
    results/belly_f6g/
      belly_vertex_weights.png — heatmap of top joints' weights on belly
      belly_vertex_top_joints.json — top-K joint weights per belly vertex
      f6g_summary.md — one-page findings
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_obj_verts(obj_path: str) -> np.ndarray:
    verts = []
    with open(obj_path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(verts, dtype=np.float32)


def identify_belly_vertices(verts: np.ndarray, margin_mm: float = 20.0) -> np.ndarray:
    """Canon mesh convention (verified via iterative 3D viz 2026-04-19):
    Y = head-to-tail (head y=124, tail y=2)
    Z = vertical (ground z≈0, dorsal z≈51)

    True belly = torso y∈[40,90] AND z∈[5, z_median] (above paws z<5, below body center).
    Prior z<z25 (=4mm) caught paws instead of belly.
    """
    y = verts[:, 1]; z = verts[:, 2]
    z_med = float(np.percentile(z, 50))
    mask = (y >= 40.0) & (y <= 90.0) & (z >= 5.0) & (z <= z_med)
    return mask  # (N,) bool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pkl", default="mouse_model/mouse.pkl")
    ap.add_argument("--canon-obj", default="results/fitting/production_3600_canon/obj/step_2_frame_001800.obj")
    ap.add_argument("--output", default="results/belly_f6g/")
    ap.add_argument("--belly-margin-mm", type=float, default=20.0)
    ap.add_argument("--top-k-joints", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # Load body model
    with open(args.model_pkl, "rb") as f:
        model = pickle.load(f)
    print(f"Model keys: {list(model.keys())[:20]}")

    # MAMMAL: 'skinning_weights' is scipy csc_matrix. Actual shape is (J, N) = (140, 14522)
    from scipy.sparse import issparse
    weights = None
    for key in ["weights", "skinning_weights", "lbs_weights", "W"]:
        if key in model:
            w = model[key]
            weights = w.toarray() if issparse(w) else np.asarray(w)
            # Normalize to (N, J)
            if weights.shape[0] < weights.shape[1]:
                weights = weights.T
            print(f"Using weights from key='{key}'  shape (N_verts, J_joints)={weights.shape}")
            break
    if weights is None:
        print(f"ERROR: no weight key found. Keys: {list(model.keys())}")
        return 1

    # Joint names
    jnames = None
    for key in ["joint_names", "bone_names", "joints"]:
        if key in model:
            v = model[key]
            if isinstance(v, (list, tuple)) and isinstance(v[0] if len(v) else None, str):
                jnames = list(v); break
    if jnames is None:
        n_joints = weights.shape[1]
        jnames = [f"joint_{i:03d}" for i in range(n_joints)]

    # Canon verts
    verts = load_obj_verts(args.canon_obj)
    n_verts = verts.shape[0]
    print(f"Canon mesh verts: {n_verts}  weights: {weights.shape}")

    if weights.shape[0] != n_verts:
        print(f"WARNING: weight N={weights.shape[0]} != verts N={n_verts}. Sub-mesh may be used.")
        # Truncate or skip
        weights = weights[:n_verts] if weights.shape[0] >= n_verts else weights

    belly_mask = identify_belly_vertices(verts, args.belly_margin_mm)
    n_belly = int(belly_mask.sum())
    print(f"Belly vertices ({args.belly_margin_mm}mm from ventral-max): {n_belly}")

    belly_w = weights[belly_mask]  # (n_belly, J)

    # Per-vertex top-K joint indices + weights
    order = np.argsort(-belly_w, axis=1)[:, :args.top_k_joints]
    per_v = []
    for i, idx in enumerate(order):
        entry = {
            "vert_id": int(np.where(belly_mask)[0][i]),
            "pos_xyz": verts[np.where(belly_mask)[0][i]].tolist(),
            "top_joints": [{"idx": int(j), "name": jnames[j], "weight": float(belly_w[i, j])} for j in idx],
        }
        per_v.append(entry)

    with open(out / "belly_vertex_top_joints.json", "w") as f:
        json.dump(per_v, f, indent=2)

    # Aggregate: for each joint, sum of weight on belly vertices
    joint_belly_weight = belly_w.sum(axis=0)
    ranked = np.argsort(-joint_belly_weight)[:20]
    print("\nTop 20 joints by total belly-vertex weight:")
    rows = []
    for j in ranked:
        rows.append({"idx": int(j), "name": jnames[j], "total_weight": float(joint_belly_weight[j]),
                     "n_verts_where_top1": int((order[:, 0] == j).sum())})
        print(f"  [{j:3d}] {jnames[j]:30s} total={joint_belly_weight[j]:8.2f}  top1_count={int((order[:, 0] == j).sum())}")

    with open(out / "belly_joint_ranking.json", "w") as f:
        json.dump({"n_belly_verts": n_belly, "ranking": rows}, f, indent=2)

    # Simple text summary
    summary = f"""# F6g Belly Skinning Weight Analysis

- Canon frame: {args.canon_obj}
- Belly vertices: {n_belly} (margin {args.belly_margin_mm} mm from ventral-max)
- Body model: {args.model_pkl}

## Top 5 joints dominating belly vertices (by top1 count)
"""
    top1_counts = [(int((order[:, 0] == j).sum()), int(j), jnames[j]) for j in range(weights.shape[1])]
    top1_counts.sort(reverse=True)
    for c, j, name in top1_counts[:5]:
        summary += f"- [{j:3d}] {name}: {c} vertices ({100*c/n_belly:.1f}%)\n"

    summary += "\n## Interpretation\n"
    summary += "If ventral/abdomen-related joints dominate → skinning is correct → F6g NOT the cause.\n"
    summary += "If mid-spine/hip/shoulder joints dominate → skinning defect → F6g is cause of belly-dent.\n"

    with open(out / "f6g_summary.md", "w") as f:
        f.write(summary)
    print(f"\nSaved: {out}/")
    return 0


if __name__ == "__main__":
    exit(main())
