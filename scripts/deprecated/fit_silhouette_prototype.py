"""
2-Stage Silhouette-based Fitting Prototype

Stage 1: Global Alignment (translation + scale)
Stage 2: Pose Refinement (add pose parameters)
"""
import torch
import numpy as np
import pickle
import cv2
from articulation_th import ArticulationTorch
from preprocessing_utils.silhouette_renderer import (
    SilhouetteRenderer, SilhouetteLoss, load_target_mask, visualize_silhouette_comparison
)
from pytorch3d.utils import cameras_from_opencv_projection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = torch.device('cuda')
IMAGE_SIZE = (480, 640)
CAM_PATH = "data/preprocessed_shank3_sam/new_cam.pkl"
MASK_VIDEO = "data/preprocessed_shank3_sam/simpleclick_undist/0.mp4"
FRAME_IDX = 0

# Stage 1 params
STAGE1_ITERATIONS = 100
STAGE1_LR_TRANS = 0.1
STAGE1_LR_SCALE = 0.01

# Stage 2 params
STAGE2_ITERATIONS = 200
STAGE2_LR_TRANS = 0.01
STAGE2_LR_SCALE = 0.01
STAGE2_LR_POSE = 0.001


def initialize_neutral_pose(bodymodel):
    """Initialize neutral pose parameters"""
    # Neutral pose (all zeros except initial values)
    thetas = torch.zeros(1, 140, 3, device=DEVICE, requires_grad=True)

    # Neutral bone lengths (from body model default)
    bone_lengths = torch.ones(1, 20, device=DEVICE) * 10.0
    bone_lengths.requires_grad = True

    # Initial rotation (identity)
    rotation = torch.zeros(1, 3, device=DEVICE, requires_grad=True)

    # Initial translation (start at camera distance)
    translation = torch.tensor([[0.0, 0.0, 500.0]], device=DEVICE, requires_grad=True)

    # Initial scale
    scale = torch.tensor([[1.0]], device=DEVICE, requires_grad=True)

    # Chest deformer
    chest_deformer = torch.zeros(1, 1, device=DEVICE, requires_grad=True)

    return {
        'thetas': thetas,
        'bone_lengths': bone_lengths,
        'rotation': rotation,
        'translation': translation,
        'scale': scale,
        'chest_deformer': chest_deformer,
    }


def stage1_global_alignment(bodymodel, renderer, camera, target_mask, init_params):
    """
    Stage 1: Optimize translation and scale only
    Keep pose fixed
    """
    logger.info("="*60)
    logger.info("STAGE 1: Global Alignment (Translation + Scale)")
    logger.info("="*60)

    # Fixed parameters (no gradients)
    thetas = init_params['thetas'].detach().clone()
    bone_lengths = init_params['bone_lengths'].detach().clone()
    rotation = init_params['rotation'].detach().clone()
    chest_deformer = init_params['chest_deformer'].detach().clone()

    # Optimizable parameters
    translation = init_params['translation'].detach().clone()
    translation.requires_grad = True
    scale = init_params['scale'].detach().clone()
    scale.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': [translation], 'lr': STAGE1_LR_TRANS},
        {'params': [scale], 'lr': STAGE1_LR_SCALE},
    ])

    # Get faces
    faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(DEVICE)

    # Optimization loop
    best_iou = 0.0
    best_params = None

    for iter_idx in range(STAGE1_ITERATIONS):
        optimizer.zero_grad()

        # Forward pass
        V, J = bodymodel.forward(
            thetas, bone_lengths, rotation, translation, scale, chest_deformer
        )

        # Render silhouette
        pred_silhouette = renderer.render_from_vertices_faces(V, faces, camera)

        # Compute loss
        iou_loss = SilhouetteLoss.iou_loss(pred_silhouette, target_mask)
        bce_loss = SilhouetteLoss.bce_loss(pred_silhouette, target_mask)

        # Combined loss
        total_loss = iou_loss + 0.1 * bce_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # Track best
        current_iou = 1.0 - iou_loss.item()
        if current_iou > best_iou:
            best_iou = current_iou
            best_params = {
                'translation': translation.detach().clone(),
                'scale': scale.detach().clone(),
            }

        # Log progress
        if (iter_idx + 1) % 20 == 0:
            logger.info(f"Iter {iter_idx+1}/{STAGE1_ITERATIONS}: "
                       f"IoU={current_iou:.4f}, Loss={total_loss.item():.4f}, "
                       f"Trans={translation[0].detach().cpu().numpy()}, "
                       f"Scale={scale[0, 0].item():.3f}")

    logger.info(f"Stage 1 Complete! Best IoU: {best_iou:.4f}")

    return {
        'thetas': thetas,
        'bone_lengths': bone_lengths,
        'rotation': rotation,
        'translation': best_params['translation'],
        'scale': best_params['scale'],
        'chest_deformer': chest_deformer,
    }


def stage2_pose_refinement(bodymodel, renderer, camera, target_mask, init_params):
    """
    Stage 2: Refine pose parameters
    Optimize all parameters together
    """
    logger.info("="*60)
    logger.info("STAGE 2: Pose Refinement (All Parameters)")
    logger.info("="*60)

    # All parameters optimizable
    thetas = init_params['thetas'].detach().clone()
    thetas.requires_grad = True

    bone_lengths = init_params['bone_lengths'].detach().clone()
    bone_lengths.requires_grad = True

    rotation = init_params['rotation'].detach().clone()
    rotation.requires_grad = True

    translation = init_params['translation'].detach().clone()
    translation.requires_grad = True

    scale = init_params['scale'].detach().clone()
    scale.requires_grad = True

    chest_deformer = init_params['chest_deformer'].detach().clone()
    chest_deformer.requires_grad = True

    # Optimizer with different learning rates
    optimizer = torch.optim.Adam([
        {'params': [translation], 'lr': STAGE2_LR_TRANS},
        {'params': [scale], 'lr': STAGE2_LR_SCALE},
        {'params': [rotation], 'lr': STAGE2_LR_POSE},
        {'params': [thetas], 'lr': STAGE2_LR_POSE},
        {'params': [bone_lengths], 'lr': STAGE2_LR_POSE * 0.1},
        {'params': [chest_deformer], 'lr': STAGE2_LR_POSE * 0.1},
    ])

    # Get faces
    faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(DEVICE)

    # Optimization loop
    best_iou = 0.0
    best_params = None

    for iter_idx in range(STAGE2_ITERATIONS):
        optimizer.zero_grad()

        # Forward pass
        V, J = bodymodel.forward(
            thetas, bone_lengths, rotation, translation, scale, chest_deformer
        )

        # Render silhouette
        pred_silhouette = renderer.render_from_vertices_faces(V, faces, camera)

        # Silhouette loss
        iou_loss = SilhouetteLoss.iou_loss(pred_silhouette, target_mask)
        bce_loss = SilhouetteLoss.bce_loss(pred_silhouette, target_mask)

        # Pose prior (regularization)
        pose_prior = (thetas ** 2).mean()
        bone_prior = ((bone_lengths - 10.0) ** 2).mean()

        # Combined loss
        total_loss = (
            iou_loss +
            0.1 * bce_loss +
            0.01 * pose_prior +
            0.001 * bone_prior
        )

        # Backward
        total_loss.backward()
        optimizer.step()

        # Track best
        current_iou = 1.0 - iou_loss.item()
        if current_iou > best_iou:
            best_iou = current_iou
            best_params = {
                'thetas': thetas.detach().clone(),
                'bone_lengths': bone_lengths.detach().clone(),
                'rotation': rotation.detach().clone(),
                'translation': translation.detach().clone(),
                'scale': scale.detach().clone(),
                'chest_deformer': chest_deformer.detach().clone(),
            }

        # Log progress
        if (iter_idx + 1) % 40 == 0:
            logger.info(f"Iter {iter_idx+1}/{STAGE2_ITERATIONS}: "
                       f"IoU={current_iou:.4f}, Loss={total_loss.item():.4f}, "
                       f"PosePrior={pose_prior.item():.4f}")

    logger.info(f"Stage 2 Complete! Best IoU: {best_iou:.4f}")

    return best_params


def main():
    logger.info("="*60)
    logger.info("2-Stage Silhouette Fitting Prototype")
    logger.info("="*60)

    # Load body model
    logger.info("\n[1] Loading body model...")
    bodymodel = ArticulationTorch()
    logger.info("Body model loaded")

    # Load camera
    logger.info("\n[2] Loading camera...")
    with open(CAM_PATH, 'rb') as f:
        cam_dict = pickle.load(f)

    cam = cam_dict[0]
    K = cam['K']
    R_cam = cam['R']
    T_cam = cam['T']

    # Create PyTorch3D camera
    R = torch.from_numpy(R_cam).float().unsqueeze(0).to(DEVICE)
    T = torch.from_numpy(T_cam).float().squeeze().unsqueeze(0).to(DEVICE)
    K_tensor = torch.from_numpy(K).float().unsqueeze(0).to(DEVICE)

    camera = cameras_from_opencv_projection(
        R=R, tvec=T, camera_matrix=K_tensor,
        image_size=torch.tensor([IMAGE_SIZE]).to(DEVICE)
    )
    logger.info("Camera created")

    # Initialize renderer
    logger.info("\n[3] Initializing silhouette renderer...")
    renderer = SilhouetteRenderer(image_size=IMAGE_SIZE, device=DEVICE)
    logger.info("Renderer initialized")

    # Load target mask
    logger.info("\n[4] Loading target mask...")
    target_mask = load_target_mask(MASK_VIDEO, FRAME_IDX, device=DEVICE)
    logger.info(f"Target mask loaded: coverage={target_mask.mean():.2%}")

    # Initialize parameters
    logger.info("\n[5] Initializing neutral pose...")
    init_params = initialize_neutral_pose(bodymodel)
    logger.info("Parameters initialized")

    # Test initial state
    logger.info("\n[6] Testing initial state...")
    with torch.no_grad():
        V, J = bodymodel.forward(
            init_params['thetas'], init_params['bone_lengths'],
            init_params['rotation'], init_params['translation'],
            init_params['scale'], init_params['chest_deformer']
        )
        faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(DEVICE)
        pred_sil = renderer.render_from_vertices_faces(V, faces, camera)
        initial_iou = 1.0 - SilhouetteLoss.iou_loss(pred_sil, target_mask).item()

    logger.info(f"Initial IoU: {initial_iou:.4f}")

    # Stage 1: Global Alignment
    logger.info("\n[7] Running Stage 1...")
    stage1_params = stage1_global_alignment(
        bodymodel, renderer, camera, target_mask, init_params
    )

    # Visualize Stage 1
    with torch.no_grad():
        V, J = bodymodel.forward(
            stage1_params['thetas'], stage1_params['bone_lengths'],
            stage1_params['rotation'], stage1_params['translation'],
            stage1_params['scale'], stage1_params['chest_deformer']
        )
        pred_sil = renderer.render_from_vertices_faces(V, faces, camera)
        stage1_iou = 1.0 - SilhouetteLoss.iou_loss(pred_sil, target_mask).item()

        visualize_silhouette_comparison(
            pred_sil[0], target_mask[0],
            save_path="fit_silhouette_stage1.png"
        )

    logger.info(f"Stage 1 IoU: {stage1_iou:.4f}")
    logger.info("Saved: fit_silhouette_stage1.png")

    # Stage 2: Pose Refinement
    logger.info("\n[8] Running Stage 2...")
    stage2_params = stage2_pose_refinement(
        bodymodel, renderer, camera, target_mask, stage1_params
    )

    # Visualize Stage 2
    with torch.no_grad():
        V, J = bodymodel.forward(
            stage2_params['thetas'], stage2_params['bone_lengths'],
            stage2_params['rotation'], stage2_params['translation'],
            stage2_params['scale'], stage2_params['chest_deformer']
        )
        pred_sil = renderer.render_from_vertices_faces(V, faces, camera)
        stage2_iou = 1.0 - SilhouetteLoss.iou_loss(pred_sil, target_mask).item()

        visualize_silhouette_comparison(
            pred_sil[0], target_mask[0],
            save_path="fit_silhouette_stage2.png"
        )

    logger.info(f"Stage 2 IoU: {stage2_iou:.4f}")
    logger.info("Saved: fit_silhouette_stage2.png")

    # Save final params
    output_path = "fit_silhouette_result.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(stage2_params, f)
    logger.info(f"\nSaved final params: {output_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Initial IoU:  {initial_iou:.4f}")
    logger.info(f"Stage 1 IoU:  {stage1_iou:.4f} (improvement: {stage1_iou-initial_iou:+.4f})")
    logger.info(f"Stage 2 IoU:  {stage2_iou:.4f} (improvement: {stage2_iou-stage1_iou:+.4f})")
    logger.info(f"Total gain:   {stage2_iou-initial_iou:+.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
