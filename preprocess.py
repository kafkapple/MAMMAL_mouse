import cv2
import numpy as np
import os
import pickle
from omegaconf import DictConfig
import hydra

import hydra.utils # Added for path resolution

# Define the 22 keypoints for the mouse model (simplified for automatic generation)
# These are placeholders and will be geometrically derived from the mask.
# For actual anatomical accuracy, a dedicated pose estimator is required.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_paw", "right_paw", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_foot", "right_foot",
    "neck", "tail_base", "wither", "center", "tail_middle"
]

@hydra.main(config_path="./conf", config_name="config")
def preprocess_video(cfg: DictConfig):
    input_video_path = hydra.utils.to_absolute_path(cfg.preprocess.input_video_path)
    output_data_dir = hydra.utils.to_absolute_path(cfg.preprocess.output_data_dir)

    if input_video_path is None:
        print("Error: input_video_path is not specified in the config.")
        return

    if not os.path.exists(input_video_path):
        print(f"Error: Input video file not found at {input_video_path}")
        return

    os.makedirs(os.path.join(output_data_dir, "videos_undist"), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, "simpleclick_undist"), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, "keypoints2d_undist"), exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writers
    video_out = cv2.VideoWriter(
        os.path.join(output_data_dir, "videos_undist", "0.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height)
    )
    mask_out = cv2.VideoWriter(
        os.path.join(output_data_dir, "simpleclick_undist", "0.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), isColor=False
    )

    # Background subtractor for mask generation
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    all_keypoints_2d = []
    frame_idx = 0

    print(f"Preprocessing video: {input_video_path}")
    print(f"Total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Save original frame (as undistorted video)
        video_out.write(frame)

        # 2. Generate mask
        fgmask = fgbg.apply(frame)
        # Clean up mask: remove shadows (gray areas), apply morphological operations
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) # Remove shadows (values around 127)
        kernel = np.ones((5,5),np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) # Remove small noise
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) # Fill small holes
        mask_out.write(fgmask)

        # 3. Generate keypoints (simplified geometric approach)
        keypoints_frame = np.zeros((len(KEYPOINT_NAMES), 3), dtype=np.float32) # (x, y, confidence)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assume the largest contour is the mouse
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100: # Filter out very small contours
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center_kp = np.array([cx, cy, 1.0])
                else:
                    center_kp = np.array([0, 0, 0.0]) # No valid center

                # Bounding box for extreme points
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Simplified keypoint mapping (highly inaccurate, for pipeline completion)
                # nose, center, tail_base
                keypoints_frame[KEYPOINT_NAMES.index("center")] = center_kp
                keypoints_frame[KEYPOINT_NAMES.index("nose")] = np.array([x + w/2, y, 0.7]) # Top center of bounding box
                keypoints_frame[KEYPOINT_NAMES.index("tail_base")] = np.array([x + w/2, y + h, 0.7]) # Bottom center

                # Left/Right points (simplified)
                keypoints_frame[KEYPOINT_NAMES.index("left_shoulder")] = np.array([x, y + h/4, 0.5])
                keypoints_frame[KEYPOINT_NAMES.index("right_shoulder")] = np.array([x + w, y + h/4, 0.5])
                keypoints_frame[KEYPOINT_NAMES.index("left_hip")] = np.array([x, y + 3*h/4, 0.5])
                keypoints_frame[KEYPOINT_NAMES.index("right_hip")] = np.array([x + w, y + 3*h/4, 0.5])

                # Other keypoints as 0 for now, or derived from these basic ones
                # For a more complete set, more sophisticated contour analysis or a real model is needed.

        all_keypoints_2d.append(keypoints_frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    video_out.release()
    mask_out.release()

    # Save 2D keypoints
    keypoints_array = np.array(all_keypoints_2d)
    with open(os.path.join(output_data_dir, "keypoints2d_undist", "result_view_0.pkl"), 'wb') as f:
        pickle.dump(keypoints_array, f)

    # Generate dummy camera parameters (for a single view)
    dummy_cam_params = {
        0: {
            'K': np.array([[1000.0, 0.0, frame_width/2],
                           [0.0, 1000.0, frame_height/2],
                           [0.0, 0.0, 1.0]], dtype=np.float64),
            'R': np.eye(3, dtype=np.float64),
            'T': np.array([[0.0], [0.0], [1000.0]], dtype=np.float64) # Move camera back along Z-axis
        }
    }
    with open(os.path.join(output_data_dir, "new_cam.pkl"), 'wb') as f:
        pickle.dump(dummy_cam_params, f)

    print(f"Preprocessing complete. Data saved to {output_data_dir}")

if __name__ == "__main__":
    preprocess_video()
