"""
Debug script: Visualize mesh projection to verify camera parameters.

Compares current (potentially buggy) vs fixed projection.
"""

import os
import sys
import pickle
import numpy as np
import torch
import cv2
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from articulation_th import ArticulationTorch


def load_camera(data_dir: str, view_id: int = 0):
    """Load camera parameters."""
    cam_path = os.path.join(data_dir, 'new_cam.pkl')
    with open(cam_path, 'rb') as f:
        cams = pickle.load(f)
    return cams[view_id]


def load_mesh(result_dir: str, frame_idx: int = 0):
    """Load fitted mesh vertices."""
    param_path = os.path.join(result_dir, 'params', f'step_2_frame_{frame_idx:06d}.pkl')
    with open(param_path, 'rb') as f:
        params = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.tensor(v, dtype=torch.float32, device=device)

    body_model = ArticulationTorch()
    V, J = body_model.forward(
        params['thetas'], params['bone_lengths'],
        params['rotation'], params['trans'] / 1000,
        params['scale'] / 1000,
        params.get('chest_deformer', torch.zeros(1, 1, device=device)),
    )
    return V[0].detach().cpu().numpy()


def load_image(data_dir: str, view_id: int, frame_idx: int):
    """Load video frame."""
    video_path = os.path.join(data_dir, 'videos_undist', f'{view_id}.mp4')
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame


def project_current(vertices: np.ndarray, cam: dict) -> np.ndarray:
    """
    Current projection (texture_sampler.py style - potentially buggy).

    Stores K.T, R.T, then uses K.T and R.T again.
    """
    K = cam['K'].T  # Store transpose
    R = cam['R'].T  # Store transpose
    T = cam['T'].squeeze()

    # Project (current code uses .T again)
    vertices_cam = vertices @ R.T + T  # = vertices @ original_R
    vertices_proj = vertices_cam @ K.T  # = vertices_cam @ original_K

    proj_2d = vertices_proj[:, :2] / (vertices_proj[:, 2:3] + 1e-8)
    return proj_2d


def project_fixed(vertices: np.ndarray, cam: dict) -> np.ndarray:
    """
    Fixed projection (no double transpose).

    Stores K.T, R.T, then uses K and R directly.
    """
    K = cam['K'].T  # Store transpose
    R = cam['R'].T  # Store transpose
    T = cam['T'].squeeze()

    # Project (fixed: don't transpose again)
    vertices_cam = vertices @ R + T  # = vertices @ original_R.T (correct!)
    vertices_proj = vertices_cam @ K  # = vertices_cam @ original_K.T (correct!)

    proj_2d = vertices_proj[:, :2] / (vertices_proj[:, 2:3] + 1e-8)
    return proj_2d


def project_fitter_style(vertices: np.ndarray, cam: dict) -> np.ndarray:
    """
    Projection matching fitter_articulation.py style.

    Uses R @ X + T format (column vectors).
    """
    K = cam['K']
    R = cam['R']
    T = cam['T'].reshape(3, 1)

    # Column vector style: R @ X + T
    vertices_cam = (R @ vertices.T + T).T  # (3,3)@(3,N) + (3,1) -> (3,N) -> (N,3)
    vertices_proj = (K @ vertices_cam.T).T  # (3,3)@(3,N) -> (3,N) -> (N,3)

    proj_2d = vertices_proj[:, :2] / (vertices_proj[:, 2:3] + 1e-8)
    return proj_2d


def visualize_projection(
    image: np.ndarray,
    proj_2d: np.ndarray,
    title: str,
    color: tuple = (0, 255, 0),
    sample_every: int = 10,
) -> np.ndarray:
    """Draw projected vertices on image."""
    vis = image.copy()
    H, W = image.shape[:2]

    # Count in-bounds
    in_bounds = (
        (proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < W) &
        (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < H)
    )
    n_in = in_bounds.sum()
    n_total = len(proj_2d)

    # Draw points
    for i in range(0, len(proj_2d), sample_every):
        x, y = int(proj_2d[i, 0]), int(proj_2d[i, 1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(vis, (x, y), 2, color, -1)

    # Add text
    text = f"{title}: {n_in}/{n_total} in-bounds ({100*n_in/n_total:.1f}%)"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    # Print projection range
    print(f"\n{title}:")
    print(f"  X range: [{proj_2d[:, 0].min():.1f}, {proj_2d[:, 0].max():.1f}] (image W={W})")
    print(f"  Y range: [{proj_2d[:, 1].min():.1f}, {proj_2d[:, 1].max():.1f}] (image H={H})")
    print(f"  In-bounds: {n_in}/{n_total} ({100*n_in/n_total:.1f}%)")

    return vis


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Debug mesh projection')
    parser.add_argument('--result_dir', type=str,
                       default='results/fitting/markerless_mouse_1_nerf_v012345_sparse3_20241218_213052')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--view', type=int, default=0)
    parser.add_argument('--output', type=str, default='outputs/debug_projection.png')
    args = parser.parse_args()

    # Get data_dir from config
    import yaml
    config_path = os.path.join(args.result_dir, 'config.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_dir = cfg['data']['data_dir']
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_dir)

    print(f"Data dir: {data_dir}")
    print(f"Result dir: {args.result_dir}")
    print(f"Frame: {args.frame}, View: {args.view}")

    # Load data
    cam = load_camera(data_dir, args.view)
    vertices = load_mesh(args.result_dir, args.frame)
    image = load_image(data_dir, args.view, args.frame)

    print(f"\nCamera K:\n{cam['K']}")
    print(f"\nCamera R:\n{cam['R']}")
    print(f"\nCamera T: {cam['T'].flatten()}")
    print(f"\nVertices shape: {vertices.shape}")
    print(f"Vertices range: X=[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}], "
          f"Y=[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}], "
          f"Z=[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

    # Project with different methods
    proj_current = project_current(vertices, cam)
    proj_fixed = project_fixed(vertices, cam)
    proj_fitter = project_fitter_style(vertices, cam)

    # Visualize
    vis_current = visualize_projection(image, proj_current, "Current (buggy?)", (0, 0, 255))
    vis_fixed = visualize_projection(image, proj_fixed, "Fixed (no double T)", (0, 255, 0))
    vis_fitter = visualize_projection(image, proj_fitter, "Fitter style (R@X)", (255, 0, 0))

    # Stack horizontally
    combined = np.hstack([vis_current, vis_fixed, vis_fitter])

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, combined)
    print(f"\nSaved: {args.output}")

    # Also show projection comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print("Current method uses double transpose -> wrong projection")
    print("Fixed method stores K.T, R.T and uses them directly -> correct")
    print("Fitter style uses R@X format (column vectors) -> reference")


if __name__ == '__main__':
    main()
