"""
Visualize silhouette fitting results from saved pkl
"""
import torch
import numpy as np
import pickle
import cv2
from articulation_th import ArticulationTorch
from preprocessing_utils.silhouette_renderer import (
    SilhouetteRenderer, load_target_mask, visualize_silhouette_comparison
)
from pytorch3d.utils import cameras_from_opencv_projection
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda')
IMAGE_SIZE = (480, 640)
CAM_PATH = "data/preprocessed_shank3_sam/new_cam.pkl"
MASK_VIDEO = "data/preprocessed_shank3_sam/simpleclick_undist/0.mp4"
FRAME_IDX = 0

print("="*60)
print("Silhouette Results Visualization")
print("="*60)

# Load body model
bodymodel = ArticulationTorch()

# Load camera
with open(CAM_PATH, 'rb') as f:
    cam_dict = pickle.load(f)

cam = cam_dict[0]
K, R_cam, T_cam = cam['K'], cam['R'], cam['T']

R = torch.from_numpy(R_cam).float().unsqueeze(0).to(DEVICE)
T = torch.from_numpy(T_cam).float().squeeze().unsqueeze(0).to(DEVICE)
K_tensor = torch.from_numpy(K).float().unsqueeze(0).to(DEVICE)

camera = cameras_from_opencv_projection(
    R=R, tvec=T, camera_matrix=K_tensor,
    image_size=torch.tensor([IMAGE_SIZE]).to(DEVICE)
)

# Initialize renderer
renderer = SilhouetteRenderer(image_size=IMAGE_SIZE, device=DEVICE)

# Load target mask
target_mask = load_target_mask(MASK_VIDEO, FRAME_IDX, device=DEVICE)

# Get faces
faces = torch.from_numpy(bodymodel.faces_vert_np).long().to(DEVICE)

# Load all params
params_files = {
    'Original Keypoint': 'mouse_fitting_result/results_preprocessed_shank3_20251104_010358/params/param0.pkl',
    'Silhouette Refined': 'refined_params_silhouette.pkl',
}

results = {}

for name, path in params_files.items():
    print(f"\n[{name}]")
    with open(path, 'rb') as f:
        params = pickle.load(f)

    # Render
    with torch.no_grad():
        V, J = bodymodel.forward(
            params['thetas'], params['bone_lengths'],
            params['rotation'], params['trans'],
            params['scale'], params['chest_deformer']
        )
        sil = renderer.render_from_vertices_faces(V, faces, camera)

    # Compute metrics
    from preprocessing_utils.silhouette_renderer import SilhouetteLoss
    iou_loss = SilhouetteLoss.iou_loss(sil, target_mask)
    iou = 1.0 - iou_loss.item()
    coverage = sil.mean().item()

    print(f"  IoU: {iou:.4f}")
    print(f"  Coverage: {coverage:.2%}")

    results[name] = {
        'silhouette': sil[0].detach().cpu().numpy(),
        'iou': iou,
        'coverage': coverage
    }

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Silhouette Fitting Results Comparison', fontsize=16)

# Row 1: Individual silhouettes
target_np = target_mask[0].detach().cpu().numpy()

axes[0, 0].imshow(target_np, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title(f'SAM Target Mask\nCoverage: {target_np.mean():.1%}')
axes[0, 0].axis('off')

axes[0, 1].imshow(results['Original Keypoint']['silhouette'], cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f"Original (Keypoint)\nIoU: {results['Original Keypoint']['iou']:.4f}")
axes[0, 1].axis('off')

axes[0, 2].imshow(results['Silhouette Refined']['silhouette'], cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f"Refined (Silhouette)\nIoU: {results['Silhouette Refined']['iou']:.4f}")
axes[0, 2].axis('off')

# Row 2: Overlays
def create_overlay(pred, target):
    """Green=target, Red=pred, Yellow=overlap"""
    vis = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    vis[:, :, 1] = (target * 255).astype(np.uint8)  # Green
    vis[:, :, 2] = (pred * 255).astype(np.uint8)    # Red
    return vis

# Original overlay
overlay_orig = create_overlay(
    results['Original Keypoint']['silhouette'],
    target_np
)
axes[1, 0].imshow(overlay_orig)
axes[1, 0].set_title('Original vs Target\n(Green=Target, Red=Mesh, Yellow=Match)')
axes[1, 0].axis('off')

# Refined overlay
overlay_refined = create_overlay(
    results['Silhouette Refined']['silhouette'],
    target_np
)
axes[1, 1].imshow(overlay_refined)
axes[1, 1].set_title('Refined vs Target\n(Green=Target, Red=Mesh, Yellow=Match)')
axes[1, 1].axis('off')

# Improvement visualization
improvement = results['Silhouette Refined']['silhouette'] - results['Original Keypoint']['silhouette']
axes[1, 2].imshow(improvement, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
axes[1, 2].set_title('Improvement Map\n(Green=Added, Red=Removed)')
axes[1, 2].axis('off')
plt.colorbar(axes[1, 2].imshow(improvement, cmap='RdYlGn', vmin=-0.5, vmax=0.5),
             ax=axes[1, 2], fraction=0.046)

plt.tight_layout()
plt.savefig('silhouette_results_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: silhouette_results_comparison.png")

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Target SAM Mask Coverage: {target_np.mean():.2%}")
print(f"\nOriginal (Keypoint-based):")
print(f"  IoU: {results['Original Keypoint']['iou']:.4f}")
print(f"  Coverage: {results['Original Keypoint']['coverage']:.2%}")
print(f"\nRefined (Silhouette-based):")
print(f"  IoU: {results['Silhouette Refined']['iou']:.4f}")
print(f"  Coverage: {results['Silhouette Refined']['coverage']:.2%}")
print(f"\nImprovement:")
iou_gain = results['Silhouette Refined']['iou'] - results['Original Keypoint']['iou']
print(f"  IoU: {iou_gain:+.4f} ({iou_gain/results['Original Keypoint']['iou']*100:+.1f}%)")
print("="*60)
