"""
Simplified silhouette renderer test using saved params
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

print("="*60)
print("Testing Silhouette Renderer (Simplified)")
print("="*60)

device = torch.device('cuda')
print(f"Device: {device}")

# Load body model
print("\n[1] Loading body model...")
bodymodel = ArticulationTorch()
print("Body model loaded")

# Load saved params from previous fitting
print("\n[2] Loading saved params...")
param_file = "mouse_fitting_result/results_preprocessed_shank3_20251104_010358/params/param0.pkl"
with open(param_file, 'rb') as f:
    params = pickle.load(f)

print(f"Params keys: {params.keys()}")

# Load camera
print("\n[3] Loading camera...")
cam_path = "data/preprocessed_shank3_sam/new_cam.pkl"
with open(cam_path, 'rb') as f:
    cam_dict = pickle.load(f)

cam = cam_dict[0]
K = cam['K']
R_cam = cam['R']
T_cam = cam['T']
image_size = (480, 640)

# Create PyTorch3D camera
R = torch.from_numpy(R_cam).float().unsqueeze(0).to(device)
T = torch.from_numpy(T_cam).float().squeeze().unsqueeze(0).to(device)  # Shape: (1, 3)
K_tensor = torch.from_numpy(K).float().unsqueeze(0).to(device)

camera = cameras_from_opencv_projection(
    R=R,
    tvec=T,
    camera_matrix=K_tensor,
    image_size=torch.tensor([image_size]).to(device)
)
print("Camera created")

# Initialize renderer
print("\n[4] Initializing silhouette renderer...")
renderer = SilhouetteRenderer(image_size=image_size, device=device)
print("Renderer initialized")

# Get mesh from params
print("\n[5] Getting mesh from saved params...")
V, J = bodymodel.forward(
    params['thetas'],
    params['bone_lengths'],
    params['rotation'],
    params['trans'],
    params['scale'],
    params['chest_deformer']
)

print(f"Vertices shape: {V.shape}")
print(f"Vertices range: [{V.min():.2f}, {V.max():.2f}]")

# Get faces
faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(device)
print(f"Faces shape: {faces.shape}")

# Render silhouette
print("\n[6] Rendering silhouette...")
pred_silhouette = renderer.render_from_vertices_faces(V, faces, camera)
print(f"Silhouette shape: {pred_silhouette.shape}")
print(f"Silhouette range: [{pred_silhouette.min():.4f}, {pred_silhouette.max():.4f}]")
print(f"Silhouette coverage: {(pred_silhouette > 0.5).float().mean():.2%}")

# Load target mask
print("\n[7] Loading target mask...")
target_mask = load_target_mask(
    "data/preprocessed_shank3_sam/simpleclick_undist/0.mp4",
    frame_idx=0,
    device=device
)
print(f"Target mask shape: {target_mask.shape}")
print(f"Target coverage: {target_mask.mean():.2%}")

# Compute losses
print("\n[8] Computing losses...")
iou_loss = SilhouetteLoss.iou_loss(pred_silhouette, target_mask)
bce_loss = SilhouetteLoss.bce_loss(pred_silhouette, target_mask)
dice_loss = SilhouetteLoss.dice_loss(pred_silhouette, target_mask)

print(f"IoU Loss: {iou_loss.item():.4f} (lower is better)")
print(f"BCE Loss: {bce_loss.item():.4f}")
print(f"Dice Loss: {dice_loss.item():.4f}")

# Compute actual IoU
iou_value = 1.0 - iou_loss.item()
print(f"Actual IoU: {iou_value:.4f}")

# Visualize
print("\n[9] Creating visualization...")
vis = visualize_silhouette_comparison(
    pred_silhouette[0],
    target_mask[0],
    save_path="test_silhouette_comparison.png"
)
print("Saved visualization: test_silhouette_comparison.png")
print("  - Green: Target (SAM mask)")
print("  - Red: Predicted (Rendered silhouette)")
print("  - Yellow: Overlap")

# Save rendered silhouette
sil_np = (pred_silhouette[0].detach().cpu().numpy() * 255).astype(np.uint8)
cv2.imwrite("test_silhouette_rendered.png", sil_np)
print("Saved rendered silhouette: test_silhouette_rendered.png")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
print(f"\nCurrent fitting quality: IoU = {iou_value:.4f}")
print("This will improve significantly with silhouette-based fitting!")
