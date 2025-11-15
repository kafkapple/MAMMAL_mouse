"""
SAM with Point Prompt - Manually specify mouse center point
"""
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

print("="*60)
print("SAM Point Prompt for Mouse Detection")
print("="*60)

# Load SAM model
SAM_CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n[1] Loading SAM model ({MODEL_TYPE})...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
print("✅ SAM model loaded")

# Load original video frame
video_path = "data/preprocessed_shank3_sam/videos_undist/0.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to load video")
    exit(1)

print(f"\n[2] Loaded video frame: {frame.shape}")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Set image for predictor
print("\n[3] Setting image for SAM predictor...")
predictor.set_image(image_rgb)
print("✅ Image set")

# Manual point specification
# Frame size: 640x480 (WxH)
# Mouse is typically in center-ish area
# Let's try multiple points

H, W = frame.shape[:2]
print(f"\n[4] Frame size: {W}x{H}")

# Define multiple candidate points (x, y)
# These are rough guesses where mouse might be
candidate_points = [
    (W // 2, H // 2),           # Center
    (W // 2, H // 2 + 50),      # Slightly down
    (W // 2, H // 2 - 50),      # Slightly up
    (W // 2 + 50, H // 2),      # Slightly right
    (W // 2 - 50, H // 2),      # Slightly left
]

print(f"\n[5] Testing {len(candidate_points)} candidate points:")

results = []

for idx, (point_x, point_y) in enumerate(candidate_points):
    print(f"\n  Point {idx + 1}: ({point_x}, {point_y})")

    # Create input point (positive prompt)
    input_point = np.array([[point_x, point_y]])
    input_label = np.array([1])  # 1 = foreground point

    # Predict masks
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,  # Get 3 masks with different quality/coverage
    )

    # SAM returns 3 masks sorted by quality score
    print(f"    Generated {len(masks)} masks")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        coverage = mask.sum() / mask.size * 100
        print(f"    Mask {i + 1}: Score={score:.3f}, Coverage={coverage:.2f}%")

    # Use best mask (highest score)
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    results.append({
        'point': (point_x, point_y),
        'mask': best_mask,
        'score': best_score,
        'coverage': best_mask.sum() / best_mask.size * 100
    })

# Select overall best result
best_result = max(results, key=lambda x: x['score'])
print(f"\n✅ Best result: Point {best_result['point']}, Score={best_result['score']:.3f}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('SAM Point Prompt Results', fontsize=16, fontweight='bold')

# Show original image with all test points
original_with_points = image_rgb.copy()
for res in results:
    px, py = res['point']
    cv2.circle(original_with_points, (px, py), 10, (255, 0, 0), -1)  # Red dots
    cv2.putText(original_with_points, f"{res['score']:.2f}",
               (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

axes[0, 0].imshow(original_with_points)
axes[0, 0].set_title('Original with Test Points\n(Red dots = SAM prompts)', fontsize=12)
axes[0, 0].axis('off')

# Show best mask
axes[0, 1].imshow(best_result['mask'], cmap='gray')
axes[0, 1].set_title(f"Best SAM Mask\nScore={best_result['score']:.3f}, "
                     f"Coverage={best_result['coverage']:.1f}%", fontsize=12)
axes[0, 1].axis('off')

# Show overlay
overlay = image_rgb.copy()
overlay[best_result['mask']] = overlay[best_result['mask']] * 0.5 + np.array([0, 255, 0]) * 0.5
axes[0, 2].imshow(overlay.astype(np.uint8))
axes[0, 2].set_title('Overlay (Green=Detected Region)', fontsize=12)
axes[0, 2].axis('off')

# Show extracted region
extracted = image_rgb.copy()
extracted[~best_result['mask']] = 255  # White background
axes[1, 0].imshow(extracted)
axes[1, 0].set_title('Extracted Region', fontsize=12)
axes[1, 0].axis('off')

# Show mask boundary
boundary = image_rgb.copy()
contours, _ = cv2.findContours(best_result['mask'].astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(boundary, contours, -1, (0, 255, 0), 3)
axes[1, 1].imshow(boundary)
axes[1, 1].set_title('Mask Boundary', fontsize=12)
axes[1, 1].axis('off')

# Show comparison with all results
comparison = np.zeros((H, W * 3, 3), dtype=np.uint8)
comparison[:, :W] = image_rgb
comparison[:, W:W*2] = np.stack([best_result['mask']*255]*3, axis=-1)
overlay_comp = image_rgb.copy()
overlay_comp[best_result['mask']] = [0, 255, 0]
comparison[:, W*2:] = overlay_comp

axes[1, 2].imshow(comparison)
axes[1, 2].set_title('Side-by-side: Original | Mask | Overlay', fontsize=12)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('sam_point_prompt_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: sam_point_prompt_results.png")

# Save best mask
mask_output = (best_result['mask'] * 255).astype(np.uint8)
cv2.imwrite('sam_point_prompt_best_mask.png', mask_output)
print("✅ Saved: sam_point_prompt_best_mask.png")

# Save as binary mask for fitting
np.save('sam_point_prompt_mask.npy', best_result['mask'])
print("✅ Saved: sam_point_prompt_mask.npy")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best prompt point: {best_result['point']}")
print(f"SAM confidence score: {best_result['score']:.3f}")
print(f"Mask coverage: {best_result['coverage']:.2f}%")
print(f"\nFiles created:")
print("  - sam_point_prompt_results.png (visualization)")
print("  - sam_point_prompt_best_mask.png (binary mask)")
print("  - sam_point_prompt_mask.npy (numpy array for fitting)")
print("="*60)
