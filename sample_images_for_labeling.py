#!/usr/bin/env python3
"""
Sample diverse images from DANNCE dataset for manual labeling
"""

import random
from pathlib import Path
import shutil
import argparse

def sample_images(input_dir, output_dir, n_samples=20, seed=42):
    """
    Sample diverse images from dataset

    Args:
        input_dir: DANNCE dataset directory
        output_dir: Output directory for sampled images
        n_samples: Number of images to sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    images_dir = output_path / 'images'
    masks_dir = output_path / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Find all RGB images
    rgb_files = sorted(list(input_path.rglob("*_rgb.png")))
    print(f"Found {len(rgb_files)} RGB images in dataset")

    if len(rgb_files) < n_samples:
        print(f"Warning: Only {len(rgb_files)} images available, sampling all")
        n_samples = len(rgb_files)

    # Sample evenly distributed indices
    indices = [int(i * len(rgb_files) / n_samples) for i in range(n_samples)]
    sampled_files = [rgb_files[i] for i in indices]

    print(f"\nSampling {n_samples} images...")

    for i, rgb_file in enumerate(sampled_files):
        # Copy RGB image
        out_name = f"sample_{i:03d}.png"
        rgb_out = images_dir / out_name
        shutil.copy(rgb_file, rgb_out)

        # Copy corresponding mask
        mask_file = rgb_file.parent / rgb_file.name.replace("_rgb.png", "_mask.png")
        if mask_file.exists():
            mask_out = masks_dir / out_name
            shutil.copy(mask_file, mask_out)
            status = "✓"
        else:
            status = "✗ (no mask)"

        print(f"  {i+1:2d}/{n_samples}: {rgb_file.name} → {out_name} {status}")

    print(f"\n✅ Sampled images saved to: {output_path}")
    print(f"   - RGB images: {images_dir}")
    print(f"   - Masks: {masks_dir}")

    # Create README
    readme = output_path / "README.md"
    with open(readme, 'w') as f:
        f.write(f"# Manual Labeling Dataset\n\n")
        f.write(f"Sampled {n_samples} images for manual keypoint labeling.\n\n")
        f.write(f"## Source\n")
        f.write(f"- Input: `{input_dir}`\n")
        f.write(f"- Total images: {len(rgb_files)}\n")
        f.write(f"- Sampling: Evenly distributed\n\n")
        f.write(f"## Structure\n")
        f.write(f"```\n")
        f.write(f"images/    - RGB images for labeling\n")
        f.write(f"masks/     - Binary masks (optional reference)\n")
        f.write(f"labels/    - YOLO pose labels (to be created)\n")
        f.write(f"```\n\n")
        f.write(f"## Next Steps\n\n")
        f.write(f"1. Label images using CVAT, Label Studio, or Roboflow\n")
        f.write(f"2. Export labels in YOLO pose format\n")
        f.write(f"3. Save to `labels/` directory\n")
        f.write(f"4. Run training: `python train_yolo_pose.py --data data.yaml`\n\n")
        f.write(f"See `docs/MANUAL_LABELING_GUIDE.md` for detailed instructions.\n")

    print(f"\n📄 Created README: {readme}")

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Sample images for manual labeling")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='DANNCE dataset directory')
    parser.add_argument('--output_dir', type=str, default='data/manual_labeling',
                       help='Output directory (default: data/manual_labeling)')
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of images to sample (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    sample_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
