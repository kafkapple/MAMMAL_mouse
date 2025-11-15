"""
SAM-based preprocessing for mouse video
Generates high-quality masks using Segment Anything Model
"""
import cv2
import numpy as np
import pickle
import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


def get_largest_mask(masks):
    """Get the largest mask from SAM output (likely the mouse)"""
    if len(masks) == 0:
        return None

    # Sort by area
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
    return masks_sorted[0]['segmentation']


def visualize_sam_masks(frame, masks, output_path):
    """Visualize all SAM masks for debugging"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original frame
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Frame')
    axes[0].axis('off')

    # All masks
    if len(masks) > 0:
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for i, mask in enumerate(masks):
            combined_mask[mask['segmentation']] = (i + 1) * (255 // len(masks))
        axes[1].imshow(combined_mask, cmap='jet')
        axes[1].set_title(f'All SAM Masks ({len(masks)} detected)')
        axes[1].axis('off')

        # Largest mask (mouse)
        largest_mask = get_largest_mask(masks)
        if largest_mask is not None:
            axes[2].imshow(largest_mask, cmap='gray')
            axes[2].set_title('Largest Mask (Mouse)')
            axes[2].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No masks detected', ha='center', va='center')
        axes[1].axis('off')
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def estimate_keypoints_from_mask(mask):
    """
    Estimate geometric keypoints from mask contour
    Returns 22 keypoints in MAMMAL format
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros((22, 3))

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute moments and centroid
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return np.zeros((22, 3))

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Fit ellipse
    if len(largest_contour) < 5:
        return np.zeros((22, 3))

    ellipse = cv2.fitEllipse(largest_contour)
    (ex, ey), (MA, ma), angle = ellipse

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Estimate 22 MAMMAL keypoints geometrically
    keypoints = np.zeros((22, 3))

    # Basic geometric placement (this is still naive, will be replaced with proper detection)
    # Head region (0-5)
    keypoints[0] = [x + w * 0.2, y + h * 0.2, 0.5]  # Nose
    keypoints[1] = [x + w * 0.25, y + h * 0.15, 0.3]  # Left ear
    keypoints[2] = [x + w * 0.15, y + h * 0.15, 0.3]  # Right ear
    keypoints[3] = [x + w * 0.2, y + h * 0.3, 0.4]  # Head center

    # Spine keypoints (6-13)
    for i in range(8):
        keypoints[6 + i] = [cx + (i - 4) * w / 16, cy, 0.4]

    # Limb keypoints (14-21)
    keypoints[14] = [x + w * 0.6, y + h * 0.7, 0.3]  # Left front
    keypoints[15] = [x + w * 0.4, y + h * 0.7, 0.3]  # Right front
    keypoints[16] = [x + w * 0.8, y + h * 0.8, 0.3]  # Left rear
    keypoints[17] = [x + w * 0.2, y + h * 0.8, 0.3]  # Right rear

    # Tail keypoints
    keypoints[18] = [x + w * 0.9, y + h * 0.9, 0.3]
    keypoints[19] = [x + w * 0.95, y + h * 0.95, 0.2]
    keypoints[20] = [x + w, y + h, 0.2]
    keypoints[21] = [cx, cy, 0.5]  # Centroid

    return keypoints


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main preprocessing with SAM"""

    print("="*50)
    print("SAM-based Preprocessing")
    print("="*50)

    # Paths
    video_dir = cfg.dataset.video_dir
    output_dir = cfg.dataset.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Output directories
    mask_dir = os.path.join(output_dir, "sam_masks")
    keypoint_dir = os.path.join(output_dir, "keypoints2d_undist")
    vis_dir = os.path.join(output_dir, "sam_visualizations")
    video_out_dir = os.path.join(output_dir, "videos_undist")
    mask_video_dir = os.path.join(output_dir, "sam_mask_video")

    for d in [mask_dir, keypoint_dir, vis_dir, video_out_dir, mask_video_dir]:
        os.makedirs(d, exist_ok=True)

    # Initialize SAM
    print("\nInitializing SAM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    # Process video
    video_path = os.path.join(video_dir, "0.mp4")
    print(f"\nProcessing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        os.path.join(video_out_dir, "0.mp4"),
        fourcc, fps, (frame_width, frame_height)
    )
    mask_writer = cv2.VideoWriter(
        os.path.join(mask_video_dir, "0.mp4"),
        fourcc, fps, (frame_width, frame_height), isColor=False
    )

    # Process frames
    all_keypoints = []
    frame_idx = 0

    print(f"\nProcessing {total_frames} frames...")
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for SAM
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Generate masks with SAM
        masks = mask_generator.generate(frame_rgb)

        # Get largest mask (mouse)
        if len(masks) > 0:
            largest_mask = get_largest_mask(masks)
            mask = (largest_mask * 255).astype(np.uint8)

            # Estimate keypoints from mask
            keypoints = estimate_keypoints_from_mask(largest_mask)
        else:
            # No masks detected
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            keypoints = np.zeros((22, 3))

        all_keypoints.append(keypoints)

        # Write videos
        video_writer.write(frame)
        mask_writer.write(mask)

        # Save visualization for sample frames
        if frame_idx % 500 == 0:
            vis_path = os.path.join(vis_dir, f"sam_vis_frame_{frame_idx:06d}.png")
            visualize_sam_masks(frame, masks, vis_path)
            print(f"\nSaved visualization: {vis_path}")

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    video_writer.release()
    mask_writer.release()

    # Save keypoints
    all_keypoints = np.array(all_keypoints)
    keypoint_path = os.path.join(keypoint_dir, "result_view_0.pkl")
    with open(keypoint_path, 'wb') as f:
        pickle.dump(all_keypoints, f)

    print(f"\n{'='*50}")
    print("Preprocessing Complete!")
    print(f"{'='*50}")
    print(f"Processed {frame_idx} frames")
    print(f"Keypoints shape: {all_keypoints.shape}")
    print(f"Output directory: {output_dir}")
    print(f"Mask video: {mask_video_dir}/0.mp4")
    print(f"Keypoints: {keypoint_path}")
    print(f"Visualizations: {vis_dir}/")


if __name__ == "__main__":
    main()
