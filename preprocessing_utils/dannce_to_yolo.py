"""
Convert DANNCE mouse dataset to YOLO Pose format for training

YOLO Pose label format:
<class_id> <x_center> <y_center> <width> <height> <x1> <y1> <v1> <x2> <y2> <v2> ... <x22> <y22> <v22>

All coordinates normalized to [0, 1]
Visibility: 2 = visible, 1 = occluded, 0 = not labeled

MAMMAL 22 keypoints:
0-5: Head (nose, left_ear, right_ear, left_eye, right_eye, head_center)
6-13: Spine (8 points from neck to tail_base)
14-17: Limbs (left_front_paw, right_front_paw, left_rear_paw, right_rear_paw)
18-20: Tail (tail_base, tail_mid, tail_tip)
21: Body centroid
"""

import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Use existing geometric keypoint estimator
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing_utils.keypoint_estimation import estimate_mammal_keypoints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DANNCEtoYOLOConverter:
    """
    Convert DANNCE mouse dataset to YOLO Pose format
    """

    def __init__(self, dannce_root, output_root):
        """
        Args:
            dannce_root: Path to DANNCE dataset root
            output_root: Path to output YOLO dataset
        """
        self.dannce_root = Path(dannce_root)
        self.output_root = Path(output_root)

        # YOLO dataset structure
        self.images_train = self.output_root / 'images' / 'train'
        self.images_val = self.output_root / 'images' / 'val'
        self.labels_train = self.output_root / 'labels' / 'train'
        self.labels_val = self.output_root / 'labels' / 'val'

        # Create directories
        for d in [self.images_train, self.images_val, self.labels_train, self.labels_val]:
            d.mkdir(parents=True, exist_ok=True)

    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        Convert bbox from [x_min, y_min, x_max, y_max] to YOLO format
        [x_center, y_center, width, height] normalized

        Args:
            bbox: [x_min, y_min, x_max, y_max]
            img_width: Image width
            img_height: Image height

        Returns:
            [x_center, y_center, width, height] normalized to [0, 1]
        """
        x_min, y_min, x_max, y_max = bbox

        # Clip bbox to image bounds
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(0, min(x_max, img_width - 1))
        y_max = max(0, min(y_max, img_height - 1))

        # Ensure min < max
        if x_min >= x_max:
            x_max = x_min + 1
        if y_min >= y_max:
            y_max = y_min + 1

        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalize
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        # Ensure all values are in [0, 1]
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        return [x_center_norm, y_center_norm, width_norm, height_norm]

    def convert_keypoints_to_yolo(self, keypoints, img_width, img_height):
        """
        Convert keypoints from [x, y, confidence] to YOLO format
        [x, y, visibility] normalized

        Args:
            keypoints: (22, 3) array of [x, y, confidence]
            img_width: Image width
            img_height: Image height

        Returns:
            List of 66 values: [x1, y1, v1, ..., x22, y22, v22]
        """
        yolo_kpts = []

        for i in range(22):
            x, y, conf = keypoints[i]

            # Normalize coordinates
            x_norm = x / img_width
            y_norm = y / img_height

            # Convert confidence to visibility
            # conf > 0.5 → visible (2), conf > 0.3 → occluded (1), else not visible (0)
            if conf > 0.5:
                visibility = 2
            elif conf > 0.3:
                visibility = 1
            else:
                visibility = 0

            yolo_kpts.extend([x_norm, y_norm, visibility])

        return yolo_kpts

    def process_single_image(self, rgb_path, mask_path, bbox_path, metadata_path,
                            output_image_path, output_label_path):
        """
        Process a single DANNCE image and generate YOLO label

        Args:
            rgb_path: Path to RGB image
            mask_path: Path to mask image
            bbox_path: Path to bbox txt file
            metadata_path: Path to metadata json
            output_image_path: Output image path
            output_label_path: Output label txt path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image and mask
            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                logger.warning(f"Failed to read RGB: {rgb_path}")
                return False

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to read mask: {mask_path}")
                return False

            img_height, img_width = rgb.shape[:2]

            # Read bbox
            # Format: frame_id x1 y1 x2 y2 cam_w cam_h ... (first 5 values, indices 1-4 are bbox)
            with open(bbox_path, 'r') as f:
                bbox_line = f.readline().strip()
                values = [float(x) for x in bbox_line.split()]
                # Extract bbox: x1, y1, x2, y2 (indices 1-4)
                if len(values) >= 5:
                    bbox = values[1:5]  # x1, y1, x2, y2
                else:
                    logger.warning(f"Invalid bbox format in {bbox_path}")
                    return False

            # Estimate keypoints using geometric method
            keypoints = estimate_mammal_keypoints(mask)  # (22, 3)

            # Convert to YOLO format
            bbox_yolo = self.convert_bbox_to_yolo(bbox, img_width, img_height)
            kpts_yolo = self.convert_keypoints_to_yolo(keypoints, img_width, img_height)

            # Write label file
            # Format: <class_id> <x_center> <y_center> <width> <height> <kpt1_x> <kpt1_y> <kpt1_v> ...
            class_id = 0  # Single class: mouse
            label_line = [class_id] + bbox_yolo + kpts_yolo

            with open(output_label_path, 'w') as f:
                f.write(' '.join(map(str, label_line)) + '\n')

            # Copy image
            shutil.copy(rgb_path, output_image_path)

            return True

        except Exception as e:
            logger.error(f"Error processing {rgb_path}: {e}")
            return False

    def convert_split(self, split='train', max_images=None):
        """
        Convert one split (train/val) of DANNCE dataset

        Args:
            split: 'train' or 'val'
            max_images: Maximum number of images to convert (for testing)

        Returns:
            Number of successfully converted images
        """
        split_dir = self.dannce_root / split

        if not split_dir.exists():
            logger.error(f"Split directory not found: {split_dir}")
            return 0

        # Determine output directories
        if split == 'train':
            output_images = self.images_train
            output_labels = self.labels_train
        else:
            output_images = self.images_val
            output_labels = self.labels_val

        # Collect all RGB images
        rgb_files = sorted(split_dir.glob('**/*_rgb.png'))

        if max_images is not None:
            rgb_files = rgb_files[:max_images]

        logger.info(f"Converting {split} split: {len(rgb_files)} images")

        success_count = 0

        for rgb_path in tqdm(rgb_files, desc=f"Converting {split}"):
            # Construct paths
            base_name = rgb_path.stem.replace('_rgb', '')
            parent_dir = rgb_path.parent

            mask_path = parent_dir / f"{base_name}_mask.png"
            bbox_path = parent_dir / f"{base_name}_box.txt"
            metadata_path = parent_dir / f"{base_name}_metadata.json"

            # Check if all files exist
            if not all(p.exists() for p in [mask_path, bbox_path, metadata_path]):
                logger.warning(f"Missing files for {base_name}, skipping")
                continue

            # Output paths
            output_image = output_images / f"{base_name}.png"
            output_label = output_labels / f"{base_name}.txt"

            # Process
            if self.process_single_image(rgb_path, mask_path, bbox_path, metadata_path,
                                         output_image, output_label):
                success_count += 1

        logger.info(f"Successfully converted {success_count}/{len(rgb_files)} images in {split}")
        return success_count

    def create_data_yaml(self, train_count, val_count):
        """
        Create data.yaml configuration file for YOLO training

        Args:
            train_count: Number of training images
            val_count: Number of validation images
        """
        yaml_content = f"""# MAMMAL Mouse Pose Dataset
# YOLO format for 22 keypoint mouse pose estimation

# Dataset paths
path: {self.output_root.absolute()}
train: images/train
val: images/val

# Number of classes
nc: 1

# Class names
names:
  0: mouse

# Keypoint configuration
kpt_shape: [22, 3]  # 22 keypoints, each with (x, y, visibility)

# Keypoint names (MAMMAL 22 keypoints)
keypoint_names:
  0: nose
  1: left_ear
  2: right_ear
  3: left_eye
  4: right_eye
  5: head_center
  6: spine_1
  7: spine_2
  8: spine_3
  9: spine_4
  10: spine_5
  11: spine_6
  12: spine_7
  13: spine_8
  14: left_front_paw
  15: right_front_paw
  16: left_rear_paw
  17: right_rear_paw
  18: tail_base
  19: tail_mid
  20: tail_tip
  21: centroid

# Keypoint flip indices for data augmentation
# (left-right symmetry)
flip_idx:
  - [1, 2]   # left_ear <-> right_ear
  - [3, 4]   # left_eye <-> right_eye
  - [14, 15] # left_front_paw <-> right_front_paw
  - [16, 17] # left_rear_paw <-> right_rear_paw

# Dataset statistics
train_images: {train_count}
val_images: {val_count}

# Source
source: DANNCE 6-view mouse dataset (converted to monocular)
date: 2025-11-14
"""

        yaml_path = self.output_root / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        logger.info(f"Created data.yaml at {yaml_path}")

    def convert(self, max_train=None, max_val=None):
        """
        Convert entire DANNCE dataset to YOLO format

        Args:
            max_train: Maximum training images (None = all)
            max_val: Maximum validation images (None = all)

        Returns:
            (train_count, val_count)
        """
        logger.info("Starting DANNCE to YOLO conversion")
        logger.info(f"DANNCE root: {self.dannce_root}")
        logger.info(f"Output root: {self.output_root}")

        # Convert train split
        train_count = self.convert_split('train', max_images=max_train)

        # Convert val split (if exists)
        val_count = 0
        if (self.dannce_root / 'val').exists():
            val_count = self.convert_split('val', max_images=max_val)
        else:
            logger.warning("No 'val' split found, skipping")

        # Create data.yaml
        self.create_data_yaml(train_count, val_count)

        logger.info(f"Conversion complete! Train: {train_count}, Val: {val_count}")
        return train_count, val_count


def main():
    """
    Example usage
    """
    # Paths
    dannce_root = "/home/joon/dev/data/3DAnimals/fauna_mouse/large_scale/mouse_dannce_6view"
    output_root = "/home/joon/dev/MAMMAL_mouse/data/yolo_mouse_pose"

    # Create converter
    converter = DANNCEtoYOLOConverter(dannce_root, output_root)

    # Convert dataset (start with small subset for testing)
    train_count, val_count = converter.convert(max_train=50, max_val=10)

    print(f"\n✅ Conversion complete!")
    print(f"   Train images: {train_count}")
    print(f"   Val images: {val_count}")
    print(f"   Output: {output_root}")
    print(f"\nNext steps:")
    print(f"   1. Review generated labels: {output_root}/labels/train/")
    print(f"   2. Train YOLO model: yolo task=pose mode=train data={output_root}/data.yaml")


if __name__ == '__main__':
    main()
