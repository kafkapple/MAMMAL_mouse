"""
Test silhouette renderer with neutral mouse pose
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
print("Testing Silhouette Renderer")
print("="*60)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Initialize body model
print("\n[1] Loading body model...")
bodymodel = ArticulationTorch()
print("Body model loaded")

# Load camera
print("\n[2] Loading camera...")
cam_path = "data/preprocessed_shank3_sam/new_cam.pkl"
with open(cam_path, 'rb') as f:
    cam_dict = pickle.load(f)

cam = cam_dict[0]
print(f"Camera loaded: {cam.keys()}")

# Create PyTorch3D camera
K = cam['K']  # Intrinsics (3, 3)
R_cam = cam['R']  # Rotation (3, 3)
T_cam = cam['T']  # Translation (3,)
image_size = (480, 640)  # (H, W) - default for this dataset

print(f"Image size: {image_size}")
print(f"K shape: {K.shape}")
print(f"R shape: {R_cam.shape}")
print(f"T shape: {T_cam.shape}")

# Convert to PyTorch3D camera format
R = torch.from_numpy(R_cam).float().unsqueeze(0).to(device)
T = torch.from_numpy(T_cam).float().unsqueeze(0).to(device)
K_tensor = torch.from_numpy(K).float().unsqueeze(0).to(device)

# Create camera
camera = cameras_from_opencv_projection(
    R=R,
    tvec=T,
    camera_matrix=K_tensor,
    image_size=torch.tensor([image_size]).to(device)
)

print("Camera created")

# Initialize renderer
print("\n[3] Initializing silhouette renderer...")
renderer = SilhouetteRenderer(image_size=image_size, device=device)
print("Renderer initialized")

# Get neutral pose mesh
print("\n[4] Getting neutral pose mesh...")
# Initialize with default params (neutral pose)
params = {
    'thetas': torch.zeros((1, bodymodel.num_q), dtype=torch.float32).to(device),
    'bone_lengths': torch.ones((1, 20), dtype=torch.float32).to(device) * 30.0,
    'rotation': torch.zeros((1, 3), dtype=torch.float32).to(device),
    'trans': torch.tensor([[0., 0., 500.]]).float().to(device),  # 500mm back from camera
    'scale': torch.tensor([[1.0]]).float().to(device),
    'chest_deformer': torch.zeros((1,), dtype=torch.float32).to(device)
}

# Forward pass
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
faces = torch.from_numpy(bodymodel.faces).long().to(device)
print(f"Faces shape: {faces.shape}")

# Render silhouette
print("\n[5] Rendering silhouette...")
pred_silhouette = renderer.render_from_vertices_faces(V, faces, camera)
print(f"Silhouette shape: {pred_silhouette.shape}")
print(f"Silhouette range: [{pred_silhouette.min():.4f}, {pred_silhouette.max():.4f}]")
print(f"Silhouette coverage: {(pred_silhouette > 0.5).float().mean():.2%}")

# Load target mask
print("\n[6] Loading target mask...")
try:
    target_mask = load_target_mask(
        "data/preprocessed_shank3_sam/simpleclick_undist/0.mp4",
        frame_idx=0,
        device=device
    )
    print(f"Target mask shape: {target_mask.shape}")
    print(f"Target coverage: {target_mask.mean():.2%}")
except Exception as e:
    print(f"Warning: Could not load target mask: {e}")
    target_mask = None

# Compute losses
if target_mask is not None:
    print("\n[7] Computing losses...")
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
    print("\n[8] Creating visualization...")
    vis = visualize_silhouette_comparison(
        pred_silhouette[0],
        target_mask[0],
        save_path="test_silhouette_neutral.png"
    )
    print("Saved visualization: test_silhouette_neutral.png")
    print("  - Green: Target (SAM mask)")
    print("  - Red: Predicted (Rendered silhouette)")
    print("  - Yellow: Overlap")

# Save rendered silhouette
print("\n[9] Saving rendered silhouette...")
sil_np = (pred_silhouette[0].detach().cpu().numpy() * 255).astype(np.uint8)
cv2.imwrite("test_silhouette_rendered.png", sil_np)
print("Saved rendered silhouette: test_silhouette_rendered.png")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
print("\nNext steps:")
print("1. Check visualizations:")
print("   - test_silhouette_neutral.png")
print("   - test_silhouette_rendered.png")
print("2. If rendering looks correct, proceed to Step 2 (fitting prototype)")
