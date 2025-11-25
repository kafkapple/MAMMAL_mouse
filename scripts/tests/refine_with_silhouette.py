"""
Silhouette-based Refinement from Existing Fitting Results

Takes existing keypoint-based fitting and refines with silhouette loss
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
INIT_PARAMS_PATH = "mouse_fitting_result/results_preprocessed_shank3_20251104_010358/params/param0.pkl"
FRAME_IDX = 0

# Refinement params
ITERATIONS = 300
LR_TRANS = 0.5
LR_SCALE = 0.05
LR_ROTATION = 0.01
LR_POSE = 0.0001  # Very small for pose


def refine_with_silhouette(bodymodel, renderer, camera, target_mask, init_params):
    """
    Refine existing fitting using silhouette loss
    """
    logger.info("="*60)
    logger.info("Silhouette-based Refinement")
    logger.info("="*60)

    # Load params as optimizable (detach first to break computation graph)
    translation = init_params['trans'].detach().clone().to(DEVICE)
    translation.requires_grad = True

    scale = init_params['scale'].detach().clone().to(DEVICE)
    scale.requires_grad = True

    rotation = init_params['rotation'].detach().clone().to(DEVICE)
    rotation.requires_grad = True

    # Pose params (optimize slightly)
    thetas = init_params['thetas'].detach().clone().to(DEVICE)
    thetas.requires_grad = True

    bone_lengths = init_params['bone_lengths'].detach().clone().to(DEVICE)
    bone_lengths.requires_grad = True

    chest_deformer = init_params['chest_deformer'].detach().clone().to(DEVICE)
    chest_deformer.requires_grad = True

    # Optimizer with prioritized learning rates
    optimizer = torch.optim.Adam([
        {'params': [translation], 'lr': LR_TRANS},
        {'params': [scale], 'lr': LR_SCALE},
        {'params': [rotation], 'lr': LR_ROTATION},
        {'params': [thetas], 'lr': LR_POSE},
        {'params': [bone_lengths], 'lr': LR_POSE * 0.1},
        {'params': [chest_deformer], 'lr': LR_POSE * 0.1},
    ])

    # Get faces
    faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(DEVICE)

    # Track best
    best_iou = 0.0
    best_params = None

    # Optimization loop
    for iter_idx in range(ITERATIONS):
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

        # Pose regularization (keep close to initial)
        pose_reg = ((thetas - init_params['thetas'].to(DEVICE)) ** 2).mean()
        bone_reg = ((bone_lengths - init_params['bone_lengths'].to(DEVICE)) ** 2).mean()

        # Combined loss
        total_loss = (
            iou_loss +
            0.1 * bce_loss +
            0.001 * pose_reg +
            0.0001 * bone_reg
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
                'trans': translation.detach().clone(),
                'scale': scale.detach().clone(),
                'chest_deformer': chest_deformer.detach().clone(),
            }

        # Log progress
        if (iter_idx + 1) % 50 == 0:
            logger.info(f"Iter {iter_idx+1}/{ITERATIONS}: "
                       f"IoU={current_iou:.4f}, Loss={total_loss.item():.4f}, "
                       f"Coverage={pred_silhouette.mean():.2%}")

    logger.info(f"Refinement Complete! Best IoU: {best_iou:.4f}")

    return best_params


def main():
    logger.info("="*60)
    logger.info("Silhouette-based Refinement")
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

    # Load initial params
    logger.info("\n[5] Loading initial params from previous fitting...")
    with open(INIT_PARAMS_PATH, 'rb') as f:
        init_params = pickle.load(f)
    logger.info("Initial params loaded")

    # Test initial quality
    logger.info("\n[6] Testing initial quality...")
    with torch.no_grad():
        V, J = bodymodel.forward(
            init_params['thetas'], init_params['bone_lengths'],
            init_params['rotation'], init_params['trans'],
            init_params['scale'], init_params['chest_deformer']
        )
        faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(DEVICE)
        pred_sil = renderer.render_from_vertices_faces(V, faces, camera)
        initial_iou = 1.0 - SilhouetteLoss.iou_loss(pred_sil, target_mask).item()

        # Save initial visualization
        visualize_silhouette_comparison(
            pred_sil[0], target_mask[0],
            save_path="refine_initial.png"
        )

    logger.info(f"Initial IoU: {initial_iou:.4f}")
    logger.info("Saved: refine_initial.png")

    # Refine with silhouette
    logger.info("\n[7] Running silhouette-based refinement...")
    refined_params = refine_with_silhouette(
        bodymodel, renderer, camera, target_mask, init_params
    )

    # Test refined quality
    logger.info("\n[8] Testing refined quality...")
    with torch.no_grad():
        V, J = bodymodel.forward(
            refined_params['thetas'], refined_params['bone_lengths'],
            refined_params['rotation'], refined_params['trans'],
            refined_params['scale'], refined_params['chest_deformer']
        )
        pred_sil = renderer.render_from_vertices_faces(V, faces, camera)
        refined_iou = 1.0 - SilhouetteLoss.iou_loss(pred_sil, target_mask).item()

        # Save refined visualization
        visualize_silhouette_comparison(
            pred_sil[0], target_mask[0],
            save_path="refine_final.png"
        )

    logger.info(f"Refined IoU: {refined_iou:.4f}")
    logger.info("Saved: refine_final.png")

    # Save refined params
    output_path = "refined_params_silhouette.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(refined_params, f)
    logger.info(f"\nSaved refined params: {output_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Initial IoU:  {initial_iou:.4f}")
    logger.info(f"Refined IoU:  {refined_iou:.4f}")
    logger.info(f"Improvement:  {refined_iou-initial_iou:+.4f} ({(refined_iou/initial_iou-1)*100:+.1f}%)")
    logger.info("="*60)


if __name__ == "__main__":
    main()
