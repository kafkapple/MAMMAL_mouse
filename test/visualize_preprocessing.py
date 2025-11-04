"""
Visualize preprocessing results to diagnose issues
"""
import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def visualize_preprocessing(data_dir, output_dir, num_frames=5):
    """Visualize mask and keypoints from preprocessing"""

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    video_path = os.path.join(data_dir, "videos_undist/0.mp4")
    mask_path = os.path.join(data_dir, "simpleclick_undist/0.mp4")
    keypoints_path = os.path.join(data_dir, "keypoints2d_undist/result_view_0.pkl")

    # Load keypoints
    with open(keypoints_path, 'rb') as f:
        keypoints = pickle.load(f)

    print(f"Keypoints shape: {keypoints.shape}")
    print(f"Sample keypoints (frame 0):\n{keypoints[0]}")

    # Open videos
    cap_video = cv2.VideoCapture(video_path)
    cap_mask = cv2.VideoCapture(mask_path)

    frame_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video size: {frame_width}x{frame_height}")
    print(f"Total frames: {total_frames}")

    # Sample frames evenly
    frame_indices = np.linspace(0, min(total_frames-1, 100), num_frames, dtype=int)

    for idx, frame_num in enumerate(frame_indices):
        # Read frame
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        cap_mask.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret1, frame = cap_video.read()
        ret2, mask = cap_mask.read()

        if not ret1 or not ret2:
            continue

        # Convert mask to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Get keypoints for this frame
        kpts = keypoints[frame_num]

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original frame
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Original Frame {frame_num}')
        axes[0, 0].axis('off')

        # Mask
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title(f'Mask Frame {frame_num}')
        axes[0, 1].axis('off')

        # Frame with keypoints
        frame_kpt = frame.copy()
        for i, kpt in enumerate(kpts):
            x, y, conf = kpt
            if conf > 0:
                color = (0, 255, 0) if conf > 0.5 else (255, 255, 0)
                cv2.circle(frame_kpt, (int(x), int(y)), 5, color, -1)
                cv2.putText(frame_kpt, str(i), (int(x)+10, int(y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        axes[1, 0].imshow(cv2.cvtColor(frame_kpt, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Keypoints (22 points)')
        axes[1, 0].axis('off')

        # Frame with mask overlay
        frame_overlay = frame.copy()
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[:,:,1] = mask  # Green channel
        frame_overlay = cv2.addWeighted(frame_overlay, 0.7, mask_color, 0.3, 0)

        axes[1, 1].imshow(cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Mask Overlay')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'preprocessing_vis_frame_{frame_num:04d}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization for frame {frame_num}")

        # Also save individual images
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_num:04d}_original.png'), frame)
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_num:04d}_mask.png'), mask)
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_num:04d}_keypoints.png'), frame_kpt)

    cap_video.release()
    cap_mask.release()

    print(f"\nVisualization complete! Saved to {output_dir}")
    print(f"Generated {len(frame_indices)} visualizations")

if __name__ == "__main__":
    data_dir = "data/preprocessed_shank3"
    output_dir = "preprocessing_debug"

    visualize_preprocessing(data_dir, output_dir, num_frames=5)
