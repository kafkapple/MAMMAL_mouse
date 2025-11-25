"""
Silhouette-based Mesh Fitting for Cropped Frames
Uses SAM masks to fit 3D mouse model
"""
import torch
import numpy as np
import cv2
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from articulation_th import ArticulationTorch
from preprocessing_utils.silhouette_renderer import (
    SilhouetteRenderer, SilhouetteLoss, visualize_silhouette_comparison
)
from pytorch3d.utils import cameras_from_opencv_projection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_camera(image_size, device):
    """
    Create a simple orthographic camera for single-view fitting

    Args:
        image_size: (height, width)
        device: torch device

    Returns:
        PyTorch3D camera
    """
    from pytorch3d.renderer import OrthographicCameras

    # Simple camera at origin looking down -Z
    R = torch.eye(3, device=device).unsqueeze(0)  # Identity rotation
    T = torch.tensor([[0.0, 0.0, 0.0]], device=device)  # At origin

    # Focal length and principal point (centered)
    focal_length = torch.tensor([[1.0, 1.0]], device=device)
    principal_point = torch.tensor([[0.0, 0.0]], device=device)

    camera = OrthographicCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R,
        T=T,
        device=device,
        image_size=torch.tensor([image_size], device=device)
    )

    return camera


def initialize_params(device):
    """Initialize neutral pose parameters"""
    params = {
        'thetas': torch.zeros(1, 140, 3, device=device, requires_grad=True),
        'bone_lengths': torch.ones(1, 28, device=device) * 10.0,  # Changed from 20 to 28
        'rotation': torch.zeros(1, 3, device=device, requires_grad=True),
        'translation': torch.tensor([[0.0, 0.0, 500.0]], device=device, requires_grad=True),
        'scale': torch.tensor([[2.0]], device=device, requires_grad=True),  # Larger initial scale
        'chest_deformer': torch.zeros(1, 1, device=device, requires_grad=True),
    }
    params['bone_lengths'].requires_grad = True
    params['chest_deformer'].requires_grad = True
    return params


def fit_single_frame(bodymodel, renderer, camera, target_mask,
                     iterations=200):
    """
    Fit model to a single frame mask (translation and scale only)

    Note: Fitting full pose from silhouette only is ill-posed without keypoints.
    This version fits translation and scale to get approximate alignment.

    Args:
        bodymodel: Mouse body model
        renderer: Silhouette renderer
        camera: PyTorch3D camera
        target_mask: Target silhouette (H, W) tensor
        iterations: Number of optimization iterations

    Returns:
        Optimized parameters
    """
    device = target_mask.device

    # Initialize parameters
    params = initialize_params(device)

    # Optimizer (translation and scale only)
    optimizer = torch.optim.Adam([
        {'params': [params['translation']], 'lr': 0.1},
        {'params': [params['scale']], 'lr': 0.05},
    ])

    # Get faces
    faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(device)

    # Optimization loop
    best_loss = float('inf')
    best_params = None

    for iter_idx in range(iterations):
        optimizer.zero_grad()

        # Forward pass
        vertices, _ = bodymodel.forward(
            thetas=params['thetas'],
            bone_lengths_core=params['bone_lengths'],
            R=params['rotation'],
            T=params['translation'],
            s=params['scale'],
            chest_deformer=params['chest_deformer']
        )

        # Render silhouette
        rendered_mask = renderer.render_from_vertices_faces(vertices, faces, camera)

        # Compute loss
        loss = SilhouetteLoss.iou_loss(rendered_mask, target_mask.unsqueeze(0))

        # Backward
        loss.backward()
        optimizer.step()

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {k: v.detach().clone() for k, v in params.items()}

        # Log progress
        if iter_idx % 50 == 0:
            logger.info(f"  Iter {iter_idx:3d}: Loss = {loss.item():.6f}")

    logger.info(f"  Final Loss: {best_loss:.6f}")

    return best_params, best_loss


def process_cropped_frames(frames_dir, output_dir, max_frames=None):
    """
    Process all cropped frames and fit mouse model

    Args:
        frames_dir: Directory containing cropped frames
        output_dir: Output directory for results
        max_frames: Maximum number of frames to process (None = all)
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Find cropped frames
    cropped_files = sorted(frames_dir.glob('*_cropped.png'))
    if max_frames:
        cropped_files = cropped_files[:max_frames]

    logger.info(f"Found {len(cropped_files)} cropped frames")

    if len(cropped_files) == 0:
        logger.error("No cropped frames found!")
        return

    # Initialize body model
    logger.info("Loading mouse body model...")
    bodymodel = ArticulationTorch()
    bodymodel.to(device)

    # Process each frame
    results = []

    for i, frame_file in enumerate(tqdm(cropped_files, desc="Fitting frames")):
        logger.info(f"\n{'='*60}")
        logger.info(f"Frame {i}: {frame_file.name}")
        logger.info(f"{'='*60}")

        # Load cropped frame and mask
        # frame_file.stem = "frame_000000_cropped", we need "frame_000000"
        frame_idx = '_'.join(frame_file.stem.split('_')[:-1])  # Remove "_cropped"
        mask_file = frames_dir / f"{frame_idx}_mask.png"

        if not mask_file.exists():
            logger.warning(f"Mask not found: {mask_file}")
            continue

        # Load mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Failed to load mask: {mask_file}")
            continue

        # Convert to tensor and normalize
        mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
        image_size = mask.shape  # (H, W)

        # Create camera and renderer for this frame size
        camera = create_simple_camera(image_size, device)
        renderer = SilhouetteRenderer(image_size=image_size, device=device)

        # Fit model
        logger.info(f"Image size: {image_size}")
        params, loss = fit_single_frame(
            bodymodel, renderer, camera, mask_tensor,
            iterations=200
        )

        # Render final result
        with torch.no_grad():
            vertices, _ = bodymodel.forward(
                thetas=params['thetas'],
                bone_lengths_core=params['bone_lengths'],
                R=params['rotation'],
                T=params['translation'],
                s=params['scale'],
                chest_deformer=params['chest_deformer']
            )

            faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(device)
            rendered_mask = renderer.render_from_vertices_faces(vertices, faces, camera)

        # Save results
        frame_result_dir = output_dir / frame_idx
        frame_result_dir.mkdir(exist_ok=True)

        # Save parameters
        params_dict = {k: v.cpu().numpy().tolist() for k, v in params.items()}
        params_dict['loss'] = float(loss)

        with open(frame_result_dir / 'params.json', 'w') as f:
            json.dump(params_dict, f, indent=2)

        # Save visualization
        visualize_result(
            mask,
            rendered_mask[0].cpu().numpy(),  # rendered_mask is (B, H, W)
            frame_file,
            frame_result_dir / 'comparison.png'
        )

        results.append({
            'frame': frame_idx,
            'loss': float(loss),
            'params_file': str(frame_result_dir / 'params.json')
        })

    # Save summary
    summary = {
        'total_frames': len(cropped_files),
        'processed_frames': len(results),
        'results': results
    }

    with open(output_dir / 'fitting_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"  Processed: {len(results)} frames")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'='*60}")

    return results


def visualize_result(target_mask, rendered_mask, frame_path, output_path):
    """Create visualization comparing target and rendered masks"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Target
    axes[0].imshow(target_mask, cmap='gray')
    axes[0].set_title('Target Mask (SAM)')
    axes[0].axis('off')

    # Rendered
    axes[1].imshow(rendered_mask, cmap='gray')
    axes[1].set_title('Rendered Silhouette')
    axes[1].axis('off')

    # Overlay
    overlay = np.stack([target_mask/255.0, rendered_mask, np.zeros_like(target_mask)], axis=-1)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red=Target, Green=Rendered)')
    axes[2].axis('off')

    plt.suptitle(f"Frame: {frame_path.stem}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fit mouse model to cropped frames")
    parser.add_argument('frames_dir', type=str,
                       help='Directory containing cropped frames and masks')
    parser.add_argument('--output-dir', type=str, default='fitting_results',
                       help='Output directory for fitting results')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (for testing)')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Silhouette-based Mesh Fitting")
    logger.info("="*60)
    logger.info(f"Input: {args.frames_dir}")
    logger.info(f"Output: {args.output_dir}")

    results = process_cropped_frames(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )

    logger.info("\nFitting complete!")


if __name__ == "__main__":
    main()
