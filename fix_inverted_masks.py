"""
Fix inverted SAM masks by inverting all saved mask files
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def invert_mask_file(mask_path):
    """Invert a single mask file"""
    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error(f"Failed to read {mask_path}")
        return False

    # Invert
    inverted_mask = 255 - mask

    # Save back
    cv2.imwrite(str(mask_path), inverted_mask)
    return True


def main():
    # Find all mask files
    data_dir = Path("data/preprocessed_shank3_sam")
    mask_files = list(data_dir.rglob("mask_*.png"))

    logger.info(f"Found {len(mask_files)} mask files to invert")

    # Process each mask
    success_count = 0
    for mask_path in mask_files:
        if invert_mask_file(mask_path):
            success_count += 1
            if success_count % 10 == 0:
                logger.info(f"Progress: {success_count}/{len(mask_files)}")

    logger.info(f"✅ Successfully inverted {success_count}/{len(mask_files)} masks")

    # Verify first mask
    if len(mask_files) > 0:
        first_mask_path = sorted(mask_files)[0]
        mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
        coverage = (mask > 127).sum() / mask.size * 100
        logger.info(f"\nVerification (first mask):")
        logger.info(f"  Path: {first_mask_path.name}")
        logger.info(f"  Coverage: {coverage:.2f}%")
        logger.info(f"  Expected: ~80-85% (background + mouse, NOT arena)")


if __name__ == "__main__":
    main()
