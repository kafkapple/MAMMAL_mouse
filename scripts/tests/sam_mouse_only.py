"""
SAM with precise point + negative prompts to select ONLY the mouse
"""
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

print("="*60)
print("SAM Mouse-Only Detection (Arena Excluded)")
print("="*60)

SAM_CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM
print(f"\n[1] Loading SAM...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# Load frame
video_path = "data/preprocessed_shank3_sam/videos_undist/0.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
H, W = frame.shape[:2]

predictor.set_image(image_rgb)
print(f"✅ Image set: {W}x{H}")

# Strategy: Use negative prompts to exclude arena floor
# Positive: Center of mouse (dark spot)
# Negative: Arena floor corners

print(f"\n[2] Detecting mouse with negative prompts...")

# Try different strategies
strategies = []

# Strategy 1: Center point + arena corners as negative
mouse_center = (W // 2, H // 2)  # Approximate mouse location
arena_negatives = [
    (W // 2, H // 2 - 100),  # Top of arena
    (W // 2, H // 2 + 100),  # Bottom of arena
    (W // 2 - 150, H // 2),  # Left
    (W // 2 + 150, H // 2),  # Right
]

input_points = np.array([mouse_center] + arena_negatives)
input_labels = np.array([1] + [0] * len(arena_negatives))  # 1=foreground, 0=background

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)

print(f"\nStrategy 1: Center + Arena negatives")
for i, (mask, score) in enumerate(zip(masks, scores)):
    coverage = mask.sum() / mask.size * 100
    print(f"  Mask {i + 1}: Score={score:.3f}, Coverage={coverage:.2f}%")

    # Find if this looks like a mouse (small, compact object)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        print(f"           Contours={len(contours)}, Circularity={circularity:.3f}")

    strategies.append({
        'name': f'Strategy1_Mask{i + 1}',
        'mask': mask,
        'score': score,
        'coverage': coverage
    })

# Strategy 2: Multiple small objects detection (looking for mouse-sized objects)
# Use automatic mask generation to find all objects, then filter by size
print(f"\nStrategy 2: Size-based filtering (5-15% coverage)")

# Test points in a grid, looking for small objects
test_points = []
for y in range(H // 4, 3 * H // 4, 50):
    for x in range(W // 4, 3 * W // 4, 50):
        test_points.append((x, y))

mouse_candidates = []

for px, py in test_points:
    input_point = np.array([[px, py]])
    input_label = np.array([1])

    masks_test, scores_test, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,  # Single mask
    )

    mask = masks_test[0]
    coverage = mask.sum() / mask.size * 100

    # Mouse should be 5-15% of frame (smaller than arena floor)
    if 5 <= coverage <= 15:
        mouse_candidates.append({
            'point': (px, py),
            'mask': mask,
            'coverage': coverage,
            'score': scores_test[0]
        })

if len(mouse_candidates) > 0:
    # Select best candidate
    best_candidate = max(mouse_candidates, key=lambda x: x['score'])
    print(f"  Found {len(mouse_candidates)} mouse-sized candidates")
    print(f"  Best: Point {best_candidate['point']}, "
          f"Score={best_candidate['score']:.3f}, "
          f"Coverage={best_candidate['coverage']:.2f}%")

    strategies.append({
        'name': 'Strategy2_SizeFiltered',
        'mask': best_candidate['mask'],
        'score': best_candidate['score'],
        'coverage': best_candidate['coverage']
    })

# Select best strategy
if len(strategies) > 0:
    # Prefer smaller coverage (more likely to be mouse, not arena)
    # But also consider score
    for s in strategies:
        s['combined_score'] = s['score'] - (s['coverage'] / 100)  # Penalize large coverage

    best_strategy = max(strategies, key=lambda x: x['combined_score'])

    print(f"\n✅ Selected: {best_strategy['name']}")
    print(f"   Score: {best_strategy['score']:.3f}")
    print(f"   Coverage: {best_strategy['coverage']:.2f}%")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SAM Mouse-Only Detection', fontsize=16, fontweight='bold')

    # Show all strategies
    for idx in range(min(6, len(strategies))):
        row = idx // 3
        col = idx % 3

        if idx < len(strategies):
            s = strategies[idx]
            overlay = image_rgb.copy()
            overlay[s['mask']] = overlay[s['mask']] * 0.5 + np.array([0, 255, 0]) * 0.5

            axes[row, col].imshow(overlay.astype(np.uint8))
            axes[row, col].set_title(f"{s['name']}\nScore={s['score']:.3f}, "
                                    f"Coverage={s['coverage']:.1f}%",
                                    fontsize=10)
        else:
            axes[row, col].axis('off')

        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('sam_mouse_only_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: sam_mouse_only_comparison.png")

    # Save best mask
    best_mask = (best_strategy['mask'] * 255).astype(np.uint8)
    cv2.imwrite('sam_mouse_only_best_mask.png', best_mask)
    print("✅ Saved: sam_mouse_only_best_mask.png")

    # Save overlay
    best_overlay = image_rgb.copy()
    best_overlay[best_strategy['mask']] = [0, 255, 0]
    cv2.imwrite('sam_mouse_only_overlay.png', cv2.cvtColor(best_overlay, cv2.COLOR_RGB2BGR))
    print("✅ Saved: sam_mouse_only_overlay.png")

else:
    print("\n❌ No suitable mouse mask found")

print("="*60)
