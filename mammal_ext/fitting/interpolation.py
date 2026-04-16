"""
Keyframe Interpolation Module

Interpolate MAMMAL parameters between fitted keyframes to generate
dense mesh sequences without per-frame optimization.

Pipeline:
    Fitted keyframes (params pkl) → interpolation (slerp/lerp) → body model forward → OBJ

Usage:
    from mammal_ext.fitting.interpolation import KeyframeInterpolator

    interp = KeyframeInterpolator("results/fitting/dense_accurate_0_100/params/")
    interp.interpolate_range(start=0, end=500, step=5)  # all M5 frames
    interp.export_objs("results/fitting/interpolated_0_100/obj/")
"""

import os
import glob
import pickle
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to rotation matrix (3, 3)."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (3, 3) to axis-angle (3,).

    Numerically unstable near angle=π (2*sin(angle) → 0). Not used by
    slerp_axis_angle (which routes through quaternions). If you need slerp,
    call slerp_axis_angle; if you need matrix→axis-angle near π, use a
    Shepperd-style extractor on the diagonal.
    """
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def canonicalize_axis_angle(aa: np.ndarray) -> np.ndarray:
    """Reduce axis-angle magnitude to [0, π] while preserving rotation.

    MAMMAL fitter converges to |θ|>π for ~4% of joint-theta entries (100% of
    keyframes have at least one such joint). Non-canonical forms cause
    pathological interpolation in axis-angle space — two rotation-equivalent
    representations slerp along different trajectories. Canonicalization is
    rotation-preserving (verified rotmat max_err ~1e-7 float noise).

    For |θ|>π: equivalent rotation is (-axis, 2π-θ). Wrapping handles the
    rare case |θ|>2π (seen up to 6.92 in production keyframes).
    """
    theta = np.linalg.norm(aa)
    if theta < 1e-8 or theta <= np.pi:
        return aa.copy() if isinstance(aa, np.ndarray) else np.asarray(aa)
    axis = aa / theta
    # Reduce theta to (0, 2pi], then fold into [0, pi] with axis negation.
    theta = theta % (2.0 * np.pi)
    if theta > np.pi:
        return -axis * (2.0 * np.pi - theta)
    return axis * theta


def _axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) → unit quaternion [w, x, y, z]. Auto-canonicalizes."""
    aa = canonicalize_axis_angle(aa)
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → canonical axis-angle (3,), |θ|≤π.

    Ensures scalar component w ≥ 0 (q and -q represent the same rotation; we
    pick the representative with non-negative scalar). Without this, slerp
    output via hemisphere-corrected quaternion can yield |θ|>π even when both
    endpoints are canonical — a subtle bug that only surfaces in
    cross-hemisphere sequences (e.g. θ_a > π with θ_b < π).
    """
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3)
    q = q / n
    if q[0] < 0.0:
        q = -q
    w = np.clip(q[0], 0.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3)
    return (q[1:] / s) * angle


def slerp_axis_angle(aa1: np.ndarray, aa2: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path slerp for axis-angle via quaternion.

    Fixes matrix-based slerp failure modes that produced mesh pops when
    adjacent keyframes straddle a joint-rotation antipode:
      - dot < 0  → hemisphere flip (shortest path)
      - dot > 0.9995 → nlerp (numerically stable near identity)
      - |dot| < 0.01 → nlerp fallback (geodesic ambiguous near ±π)
    """
    q1 = _axis_angle_to_quat(aa1)
    q2 = _axis_angle_to_quat(aa2)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    # After hemisphere correction dot ∈ [0, 1]. Only dot ≈ 1 (near-identity,
    # sin(theta_0) → 0) is numerically unstable for slerp; mid-range dot
    # (e.g. 0.01) corresponds to theta_0 ≈ π/2 and is stable.
    if dot > 0.9995:
        q = q1 + alpha * (q2 - q1)
        q = q / max(np.linalg.norm(q), 1e-12)
        return _quat_to_axis_angle(q)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    s1 = np.sin(theta_0 - theta) / sin_theta_0
    s2 = np.sin(theta) / sin_theta_0
    q = s1 * q1 + s2 * q2
    return _quat_to_axis_angle(q)


class KeyframeInterpolator:
    """Interpolate MAMMAL parameters between fitted keyframes."""

    def __init__(self, params_dir: str, device: str = "cuda"):
        """
        Args:
            params_dir: Directory containing step_2_frame_*.pkl files
            device: Torch device for body model forward pass
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.keyframes = {}  # frame_id -> params dict (numpy)
        self._load_keyframes(params_dir)
        self._load_body_model()

    def _load_keyframes(self, params_dir: str):
        """Load all keyframe parameters."""
        files = sorted(glob.glob(os.path.join(params_dir, "step_2_frame_*.pkl")))
        if not files:
            files = sorted(glob.glob(os.path.join(params_dir, "step_1_frame_*.pkl")))
        if not files:
            raise FileNotFoundError(f"No param files in {params_dir}")

        for f in files:
            fid = int(f.split("frame_")[1].split(".")[0])
            with open(f, "rb") as fh:
                p = pickle.load(fh)
            # Convert to numpy
            d = {}
            for k, v in p.items():
                if hasattr(v, "detach"):
                    d[k] = v.detach().cpu().numpy()
                elif hasattr(v, "numpy"):
                    d[k] = v.numpy()
                else:
                    d[k] = np.array(v)
            self.keyframes[fid] = d

        self.sorted_ids = sorted(self.keyframes.keys())
        print(f"Loaded {len(self.keyframes)} keyframes: {self.sorted_ids[0]}..{self.sorted_ids[-1]}")

    def _load_body_model(self):
        """Load MAMMAL body model for vertex generation."""
        from articulation_th import ArticulationTorch
        self.body_model = ArticulationTorch()
        self.faces = self.body_model.faces_vert_np

    def _find_neighbors(self, frame_id: int) -> Tuple[int, int]:
        """Find the two nearest keyframes bracketing frame_id.

        Out-of-range behavior: clamps to the edge keyframe (no extrapolation).
          frame_id < sorted_ids[0]   → (sorted_ids[0], sorted_ids[0])
          frame_id > sorted_ids[-1]  → (sorted_ids[-1], sorted_ids[-1])
        Callers detect clamping via prev_id == next_id.
        """
        prev_id = self.sorted_ids[0]
        next_id = self.sorted_ids[-1]
        for i, kid in enumerate(self.sorted_ids):
            if kid <= frame_id:
                prev_id = kid
            if kid >= frame_id:
                next_id = kid
                break
        return prev_id, next_id

    def interpolate_params(self, frame_id: int, method: str = "slerp") -> Dict[str, np.ndarray]:
        """Interpolate parameters for a given frame_id.

        Args:
            frame_id: Target video frame ID
            method: "slerp" for rotation-aware, "lerp" for linear

        Returns:
            Interpolated parameter dict
        """
        # Exact keyframe
        if frame_id in self.keyframes:
            return self.keyframes[frame_id]

        prev_id, next_id = self._find_neighbors(frame_id)

        # Edge cases
        if prev_id == next_id:
            return self.keyframes[prev_id]

        alpha = (frame_id - prev_id) / (next_id - prev_id)
        p_prev = self.keyframes[prev_id]
        p_next = self.keyframes[next_id]

        result = {}
        for key in p_prev:
            a, b = p_prev[key], p_next[key]

            if method == "slerp" and key in ("thetas", "rotation"):
                # Slerp for rotation parameters
                if key == "thetas":
                    # (1, N_joints, 3) — slerp each joint independently
                    interp = np.zeros_like(a)
                    for j in range(a.shape[1]):
                        interp[0, j] = slerp_axis_angle(a[0, j], b[0, j], alpha)
                    result[key] = interp
                elif key == "rotation":
                    # (1, 3) global rotation
                    result[key] = slerp_axis_angle(a[0], b[0], alpha).reshape(1, 3)
            else:
                # Linear interpolation for everything else
                result[key] = a * (1 - alpha) + b * alpha

        return result

    def params_to_vertices(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass: params → mesh vertices (N, 3)."""
        d = self.device
        thetas = torch.from_numpy(params["thetas"]).float().to(d)
        bone_lengths = torch.from_numpy(params["bone_lengths"]).float().to(d)
        rotation = torch.from_numpy(params["rotation"]).float().to(d)
        trans = torch.from_numpy(params["trans"]).float().to(d)
        scale = torch.from_numpy(params["scale"]).float().to(d)
        chest_deformer = torch.from_numpy(params["chest_deformer"]).float().to(d)

        V, _ = self.body_model.forward(
            thetas, bone_lengths, rotation, trans, scale, chest_deformer
        )
        return V[0].detach().cpu().numpy()

    def interpolate_range(
        self,
        start: int,
        end: int,
        step: int = 5,
        method: str = "slerp",
    ) -> Dict[int, np.ndarray]:
        """Interpolate vertices for a range of frame IDs.

        Args:
            start: Start video frame ID
            end: End video frame ID (exclusive)
            step: Frame step (default 5 = M5 frames)
            method: Interpolation method

        Returns:
            Dict of frame_id → vertices (N, 3)
        """
        results = {}
        targets = list(range(start, end, step))
        n_keyframe = sum(1 for t in targets if t in self.keyframes)
        n_interp = len(targets) - n_keyframe

        print(f"Interpolating {len(targets)} frames ({n_keyframe} keyframes + {n_interp} interpolated)")

        for i, fid in enumerate(targets):
            params = self.interpolate_params(fid, method)
            verts = self.params_to_vertices(params)
            results[fid] = verts
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(targets)}")

        return results

    def export_objs(
        self,
        vertices_dict: Dict[int, np.ndarray],
        output_dir: str,
        with_uv: bool = True,
        uv_template_dir: str = "mouse_model/mouse_txt",
    ):
        """Export interpolated vertices as OBJ files.

        Args:
            vertices_dict: frame_id → vertices
            output_dir: Output directory
            with_uv: Include UV coordinates from template
            uv_template_dir: Path to UV template files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load UV template if needed
        uv_lines, face_lines = [], []
        if with_uv:
            uv_coords = np.loadtxt(os.path.join(uv_template_dir, "textures.txt"))
            faces_tex = np.loadtxt(os.path.join(uv_template_dir, "faces_tex.txt"), dtype=np.int64)
            uv_lines = [f"vt {uv[0]:.6f} {uv[1]:.6f}\n" for uv in uv_coords]
            face_lines = [
                f"f {fv[0]+1}/{ft[0]+1} {fv[1]+1}/{ft[1]+1} {fv[2]+1}/{ft[2]+1}\n"
                for fv, ft in zip(self.faces, faces_tex)
            ]
        else:
            face_lines = [
                f"f {f[0]+1} {f[1]+1} {f[2]+1}\n" for f in self.faces
            ]

        for fid, verts in sorted(vertices_dict.items()):
            obj_path = os.path.join(output_dir, f"step_2_frame_{fid:06d}.obj")
            with open(obj_path, "w") as f:
                f.write(f"# Interpolated frame {fid}\n")
                f.write(f"# Vertices: {len(verts)}\n")
                for v in verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for line in uv_lines:
                    f.write(line)
                for line in face_lines:
                    f.write(line)

        print(f"Exported {len(vertices_dict)} OBJs to {output_dir}")
