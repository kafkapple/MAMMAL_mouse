#!/usr/bin/env python3
"""
Monocular MAMMAL Fitting Pipeline
Fits MAMMAL mouse model to single RGB images with masks

Usage:
    python fit_monocular.py --input_dir <path> --output_dir <path>
"""

import argparse
import cv2
import numpy as np
import pickle
import torch
import trimesh
from pathlib import Path
from tqdm import tqdm
import json
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from articulation_th import ArticulationTorch
from preprocessing_utils.keypoint_estimation import (
    estimate_mammal_keypoints,
    validate_keypoints,
    visualize_keypoints_on_frame
)
from preprocessing_utils.superanimal_detector import SuperAnimalDetector


class MonocularMAMMALFitter:
    """
    Fits MAMMAL parametric mouse model to monocular images
    """

    def __init__(self, device='cuda', detector='geometric', superanimal_model_path=None):
        """
        Args:
            device: 'cuda' or 'cpu'
            detector: 'geometric' or 'superanimal'
            superanimal_model_path: Path to SuperAnimal model (required if detector='superanimal')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load MAMMAL articulation model
        self.model = ArticulationTorch()
        self.model.init_params(batch_size=1)

        # Move model to correct device (ArticulationTorch defaults to CUDA)
        self.model.device = self.device
        self.model.to(self.device)
        print(f"MAMMAL model loaded: {self.model.v_template_th.shape[0]} vertices on {self.device}")

        # Initialize keypoint detector
        self.detector_type = detector
        if detector == 'superanimal':
            if superanimal_model_path is None:
                raise ValueError("superanimal_model_path required when detector='superanimal'")
            self.superanimal_detector = SuperAnimalDetector(
                model_path=superanimal_model_path,
                device=device
            )
            print(f"SuperAnimal detector loaded from: {superanimal_model_path}")
        else:
            self.superanimal_detector = None
            print("Using geometric keypoint detector")

        # Default camera parameters (monocular setup)
        self.default_camera_params = {
            'fx': 1000.0,
            'fy': 1000.0,
            'cx': 128.0,  # Half of 256
            'cy': 128.0,
            'image_width': 256,
            'image_height': 256
        }

    def extract_keypoints(self, mask, rgb=None):
        """
        Extract 2D keypoints from binary mask (and optionally RGB image)

        Args:
            mask: (H, W) binary mask
            rgb: (H, W, 3) RGB image (required for SuperAnimal detector)

        Returns:
            keypoints: (22, 3) array of [x, y, confidence]
        """
        if self.detector_type == 'superanimal':
            if rgb is None:
                raise ValueError("RGB image required for SuperAnimal detector")
            # SuperAnimal detector expects RGB in range [0, 255]
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            keypoints = self.superanimal_detector.detect(rgb)
        else:
            # Geometric method (mask-based)
            keypoints = estimate_mammal_keypoints(mask)
            keypoints = validate_keypoints(keypoints, mask.shape)

        return keypoints

    def initialize_pose_from_keypoints(self, keypoints_2d):
        """
        Initialize MAMMAL pose parameters from 2D keypoints

        Args:
            keypoints_2d: (22, 3) 2D keypoints [x, y, confidence]

        Returns:
            thetas: Joint angles (batch, n_joints, 3)
            bone_lengths: Bone length parameters (batch, 28)
            R: Global rotation (batch, 3)
            T: Global translation (batch, 3)
            s: Global scale (batch, 1)
            chest_deformer: Chest deformation (batch, 1)
        """
        batch_size = 1
        n_joints = self.model.jointnum

        # Initialize all joint angles to zero (T-pose)
        thetas = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=self.device)

        # Bone lengths initialized to default (will be optimized)
        bone_lengths = torch.zeros(batch_size, 28, dtype=torch.float32, device=self.device)

        # Global rotation (no rotation initially)
        R = torch.zeros(batch_size, 3, dtype=torch.float32, device=self.device)

        # Global translation based on keypoint center
        keypoints_center = keypoints_2d[:, :2].mean(axis=0)
        T = torch.tensor([[
            keypoints_center[0] - self.default_camera_params['cx'],
            keypoints_center[1] - self.default_camera_params['cy'],
            0.0  # Depth will be handled by scale
        ]], dtype=torch.float32, device=self.device)

        # Global scale (estimate from keypoint spread)
        keypoint_spread = np.std(keypoints_2d[:, :2], axis=0).mean()
        s = torch.tensor([[keypoint_spread / 50.0]], dtype=torch.float32, device=self.device)

        # Chest deformer (no deformation initially)
        chest_deformer = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)

        return thetas, bone_lengths, R, T, s, chest_deformer

    def optimize_pose_to_keypoints(self, keypoints_2d, thetas, bone_lengths, R, T, s, chest_deformer,
                                     n_iterations=50, lr=0.01):
        """
        Optimize MAMMAL parameters to match 2D keypoints

        Args:
            keypoints_2d: (22, 3) target 2D keypoints
            thetas, bone_lengths, R, T, s, chest_deformer: Initial parameters
            n_iterations: Number of optimization iterations
            lr: Learning rate

        Returns:
            Optimized parameters
        """
        # Make parameters optimizable
        thetas = thetas.clone().requires_grad_(True)
        T = T.clone().requires_grad_(True)
        s = s.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([thetas, T, s], lr=lr)

        # Target keypoints (only use confident ones)
        target_kpts = torch.tensor(keypoints_2d[:, :2], dtype=torch.float32, device=self.device)
        confidence = torch.tensor(keypoints_2d[:, 2], dtype=torch.float32, device=self.device)

        print(f"Optimizing MAMMAL parameters...")
        for iteration in range(n_iterations):
            optimizer.zero_grad()

            # Forward pass: Get 3D vertices and joints
            try:
                vertices, joints = self.model(thetas, bone_lengths, R, T, s, chest_deformer)

                # Get 22 keypoints from model
                keypoints_3d = self.model.forward_keypoints22()  # (batch, 22, 3)
            except Exception as e:
                print(f"Warning: Model forward failed at iter {iteration}: {e}")
                import traceback
                traceback.print_exc()
                break

            # Project 3D keypoints to 2D (simplified orthographic projection)
            keypoints_2d_pred = keypoints_3d[0, :, :2]  # (22, 2), take x,y only

            # Compute 2D reprojection loss
            diff = (keypoints_2d_pred - target_kpts) * confidence.unsqueeze(1)
            loss_2d = torch.sum(diff ** 2)

            # Regularization: Keep pose close to T-pose
            loss_pose_reg = 0.001 * torch.sum(thetas ** 2)

            # Total loss
            loss = loss_2d + loss_pose_reg

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: Loss={loss.item():.4f}, 2D={loss_2d.item():.4f}")

        return thetas.detach(), bone_lengths.detach(), R.detach(), T.detach(), s.detach(), chest_deformer.detach()

    def generate_mesh(self, thetas, bone_lengths, R, T, s, chest_deformer):
        """
        Generate 3D mesh from MAMMAL parameters

        Args:
            thetas, bone_lengths, R, T, s, chest_deformer: Model parameters

        Returns:
            trimesh.Trimesh object
        """
        with torch.no_grad():
            vertices, _ = self.model(thetas, bone_lengths, R, T, s, chest_deformer)

        vertices_np = vertices[0].cpu().numpy()  # Take first batch
        faces_np = self.model.faces_vert_np

        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        return mesh

    def fit_single_image(self, rgb_path, mask_path, visualize=True):
        """
        Fit MAMMAL model to a single image

        Args:
            rgb_path: Path to RGB image
            mask_path: Path to binary mask
            visualize: Whether to return visualization

        Returns:
            results: Dict containing mesh, params, keypoints, etc.
        """
        # Load image and mask
        rgb = cv2.imread(str(rgb_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if rgb is None or mask is None:
            raise ValueError(f"Failed to load image or mask: {rgb_path}, {mask_path}")

        # Binarize mask
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Convert BGR to RGB
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Step 1: Extract 2D keypoints (from mask or RGB depending on detector)
        keypoints_2d = self.extract_keypoints(mask_binary, rgb=rgb_rgb)

        # Step 2: Initialize MAMMAL parameters
        thetas, bone_lengths, R, T, s, chest_deformer = self.initialize_pose_from_keypoints(keypoints_2d)

        # Step 3: Optimize to fit keypoints
        thetas_opt, bone_lengths_opt, R_opt, T_opt, s_opt, chest_deformer_opt = self.optimize_pose_to_keypoints(
            keypoints_2d, thetas, bone_lengths, R, T, s, chest_deformer,
            n_iterations=50, lr=0.01
        )

        # Step 4: Generate final mesh
        mesh = self.generate_mesh(thetas_opt, bone_lengths_opt, R_opt, T_opt, s_opt, chest_deformer_opt)

        # Visualization
        vis_image = None
        if visualize:
            vis_image = visualize_keypoints_on_frame(rgb, keypoints_2d)

        results = {
            'mesh': mesh,
            'thetas': thetas_opt.cpu().numpy(),
            'bone_lengths': bone_lengths_opt.cpu().numpy(),
            'R': R_opt.cpu().numpy(),
            'T': T_opt.cpu().numpy(),
            's': s_opt.cpu().numpy(),
            'chest_deformer': chest_deformer_opt.cpu().numpy(),
            'keypoints_2d': keypoints_2d,
            'rgb': rgb,
            'mask': mask_binary,
            'vis_keypoints': vis_image
        }

        return results

    def process_directory(self, input_dir, output_dir, max_images=None):
        """
        Process all images in a directory

        Args:
            input_dir: Directory containing RGB and mask images
            output_dir: Output directory for results
            max_images: Maximum number of images to process (None = all)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all RGB images
        rgb_files = sorted(list(input_path.glob("*_rgb.png")))

        if max_images is not None:
            rgb_files = rgb_files[:max_images]

        print(f"Found {len(rgb_files)} RGB images")

        # Process each image
        for rgb_file in tqdm(rgb_files, desc="Processing images"):
            # Get corresponding mask file
            mask_file = rgb_file.parent / rgb_file.name.replace("_rgb.png", "_mask.png")

            if not mask_file.exists():
                print(f"Warning: Mask not found for {rgb_file.name}, skipping")
                continue

            try:
                # Fit MAMMAL model
                results = self.fit_single_image(rgb_file, mask_file, visualize=True)

                # Save results
                output_name = rgb_file.stem.replace("_rgb", "")

                # Save mesh
                mesh_path = output_path / f"{output_name}_mesh.obj"
                results['mesh'].export(str(mesh_path))

                # Save parameters
                params_path = output_path / f"{output_name}_params.pkl"
                with open(params_path, 'wb') as f:
                    pickle.dump({
                        'thetas': results['thetas'],
                        'bone_lengths': results['bone_lengths'],
                        'R': results['R'],
                        'T': results['T'],
                        's': results['s'],
                        'chest_deformer': results['chest_deformer'],
                        'keypoints_2d': results['keypoints_2d']
                    }, f)

                # Save visualization
                if results['vis_keypoints'] is not None:
                    vis_path = output_path / f"{output_name}_keypoints.png"
                    cv2.imwrite(str(vis_path), results['vis_keypoints'])

                print(f"  ✓ Processed {rgb_file.name}")

            except Exception as e:
                print(f"  ✗ Failed to process {rgb_file.name}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Monocular MAMMAL Fitting")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing RGB and mask images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--detector', type=str, default='geometric',
                       choices=['geometric', 'superanimal'],
                       help='Keypoint detector to use (default: geometric)')
    parser.add_argument('--superanimal_model', type=str,
                       default='models/superanimal_topviewmouse',
                       help='Path to SuperAnimal model directory')

    args = parser.parse_args()

    # Initialize fitter
    fitter = MonocularMAMMALFitter(
        device=args.device,
        detector=args.detector,
        superanimal_model_path=args.superanimal_model if args.detector == 'superanimal' else None
    )

    # Process directory
    fitter.process_directory(args.input_dir, args.output_dir,
                            max_images=args.max_images)

    print(f"\nProcessing complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
