"""
Test SuperAnimal-TopViewMouse keypoint detection
"""
import deeplabcut
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_keypoints(frame, keypoints, output_path):
    """Visualize detected keypoints"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Original frame
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Frame')
    axes[0].axis('off')

    # Frame with keypoints
    frame_kpt = frame.copy()

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:  # Only draw confident keypoints
            color = (0, 255, 0) if conf > 0.7 else (255, 255, 0)
            cv2.circle(frame_kpt, (int(x), int(y)), 4, color, -1)
            cv2.putText(frame_kpt, str(i), (int(x)+6, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    axes[1].imshow(cv2.cvtColor(frame_kpt, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'SuperAnimal Keypoints ({len(keypoints)} total)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("="*60)
    print("Testing SuperAnimal-TopViewMouse Keypoint Detection")
    print("="*60)

    # Create output directory
    output_dir = "superanimal_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    video_path = "data/preprocessed_shank3/videos_undist/0.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Test frames
    test_frames = [0, 100, 500, 1000, 2000]

    # Extract test frames to temp directory
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    for frame_num in test_frames:
        if frame_num >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            frame_path = os.path.join(temp_dir, f"frame_{frame_num:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append((frame_num, frame_path, frame))
            print(f"Extracted frame {frame_num}")

    cap.release()

    # Run SuperAnimal inference
    print("\n" + "="*60)
    print("Running SuperAnimal-TopViewMouse inference...")
    print("="*60)

    try:
        # Use video_inference_superanimal with correct parameters
        results = deeplabcut.video_inference_superanimal(
            [fp for _, fp, _ in frame_paths],
            superanimal_name='superanimal_topviewmouse',
            videotype='.png',
            scale_list=[250],  # Adjust based on your mouse size
            video_adapt=True
        )

        print(f"\nInference complete! Results: {results}")

        # Process results
        for frame_num, frame_path, frame in frame_paths:
            # Load predictions
            h5_file = frame_path.replace('.png', 'superanimal_topviewmouseDLC_dlcrnetms5_multi-mouseAug5shuffle1_40000.h5')

            if os.path.exists(h5_file):
                import pandas as pd
                df = pd.read_hdf(h5_file)

                # Extract keypoints (x, y, confidence)
                keypoints = []
                for col in df.columns.levels[1]:  # Body part names
                    x = df.iloc[0][('superanimal_topviewmouse', col, 'x')]
                    y = df.iloc[0][('superanimal_topviewmouse', col, 'y')]
                    conf = df.iloc[0][('superanimal_topviewmouse', col, 'likelihood')]
                    keypoints.append([x, y, conf])

                keypoints = np.array(keypoints)
                print(f"\nFrame {frame_num}:")
                print(f"  Detected {len(keypoints)} keypoints")
                print(f"  Mean confidence: {keypoints[:, 2].mean():.3f}")
                print(f"  High confidence (>0.7): {(keypoints[:, 2] > 0.7).sum()}")

                # Visualize
                output_path = os.path.join(output_dir, f"superanimal_frame_{frame_num:06d}.png")
                visualize_keypoints(frame, keypoints, output_path)
            else:
                print(f"Warning: H5 file not found for frame {frame_num}")

        print(f"\n{'='*60}")
        print(f"Test complete! Results saved to {output_dir}/")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
