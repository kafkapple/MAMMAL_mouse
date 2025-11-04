"""
Quick test of SAM on a few frames
"""
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import os


def get_largest_mask(masks):
    """Get the largest mask from SAM output"""
    if len(masks) == 0:
        return None
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
    return masks_sorted[0]['segmentation']


def visualize_sam_results(frame, masks, output_path):
    """Visualize SAM results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')

    # All masks combined
    if len(masks) > 0:
        combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        for i, mask in enumerate(masks):
            combined[mask['segmentation']] = (i + 1) * (255 // max(len(masks), 1))
        axes[0, 1].imshow(combined, cmap='jet')
        axes[0, 1].set_title(f'All Masks ({len(masks)} detected)')
        axes[0, 1].axis('off')

        # Largest mask
        largest = get_largest_mask(masks)
        if largest is not None:
            axes[0, 2].imshow(largest, cmap='gray')
            axes[0, 2].set_title('Largest Mask (Mouse)')
            axes[0, 2].axis('off')

            # Overlay
            overlay = frame.copy()
            overlay[largest > 0] = overlay[largest > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
            axes[1, 0].imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Mask Overlay')
            axes[1, 0].axis('off')

            # Show mask statistics
            stats_text = f"Mask Stats:\n"
            stats_text += f"Area: {np.sum(largest)} pixels\n"
            stats_text += f"Coverage: {np.sum(largest) / largest.size * 100:.1f}%\n"
            stats_text += f"Total masks: {len(masks)}\n"
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            axes[1, 1].axis('off')

    else:
        for ax in axes.flat[1:]:
            ax.text(0.5, 0.5, 'No masks detected', ha='center', va='center')
            ax.axis('off')

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Testing SAM on shank3 video...")

    # Initialize SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    print("SAM loaded successfully!")

    # Load video
    video_path = "data/preprocessed_shank3/videos_undist/0.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Create output directory
    output_dir = "sam_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # Test on frames at different positions
    test_frames = [0, 100, 500, 1000, 2000]

    for frame_num in test_frames:
        if frame_num >= total_frames:
            continue

        print(f"\nProcessing frame {frame_num}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"  Error reading frame {frame_num}")
            continue

        # Run SAM
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("  Running SAM inference...")
        masks = mask_generator.generate(frame_rgb)
        print(f"  Detected {len(masks)} masks")

        # Visualize
        output_path = os.path.join(output_dir, f"sam_test_frame_{frame_num:06d}.png")
        visualize_sam_results(frame, masks, output_path)

    cap.release()

    print(f"\n{'='*50}")
    print(f"Test complete! Results saved to {output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
