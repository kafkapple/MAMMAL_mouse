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


# Keypoint group definitions
KEYPOINT_GROUPS = {
    'head': list(range(0, 6)),       # 0-5: nose, ears, eyes, head center
    'spine': list(range(6, 14)),     # 6-13: 8 spine points
    'limbs': list(range(14, 18)),    # 14-17: 4 paws
    'tail': list(range(18, 21)),     # 18-20: 3 tail points
    'centroid': [21]                 # 21: body centroid
}

def create_combined_visualization(rgb, mask, keypoints_2d, mesh_vertices=None,
                                   mesh_faces=None, camera_params=None):
    """
    Create combined overlay visualization: RGB + mesh + keypoints + mask

    Args:
        rgb: (H, W, 3) BGR image
        mask: (H, W) binary mask
        keypoints_2d: (22, 3) keypoints [x, y, confidence]
        mesh_vertices: (N, 3) 3D vertices (optional)
        mesh_faces: (F, 3) face indices (optional)
        camera_params: Camera parameters for projection (optional)

    Returns:
        combined: (H, W, 3) combined visualization
    """
    H, W = rgb.shape[:2]
    combined = rgb.copy()

    # 1. Draw mask outline (yellow)
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(combined, contours, -1, (0, 255, 255), 2)  # Yellow outline

    # 2. Draw keypoints with colors by group
    keypoint_colors = {
        'head': (255, 0, 0),        # Blue
        'spine': (0, 255, 0),       # Green
        'limbs': (0, 0, 255),       # Red
        'tail': (255, 255, 0),      # Cyan
        'centroid': (255, 0, 255)   # Magenta
    }

    for group_name, indices in KEYPOINT_GROUPS.items():
        color = keypoint_colors[group_name]
        for i in indices:
            x, y, conf = keypoints_2d[i]
            if conf > 0.3:
                radius = int(3 + conf * 4)
                cv2.circle(combined, (int(x), int(y)), radius, color, -1)
                cv2.circle(combined, (int(x), int(y)), radius + 1, (255, 255, 255), 1)

    # 3. Draw skeleton connections
    connections = [
        # Spine chain
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
        # Tail
        (13, 18), (18, 19), (19, 20),
        # Head connections
        (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
    ]

    for i, j in connections:
        if keypoints_2d[i, 2] > 0.3 and keypoints_2d[j, 2] > 0.3:
            pt1 = (int(keypoints_2d[i, 0]), int(keypoints_2d[i, 1]))
            pt2 = (int(keypoints_2d[j, 0]), int(keypoints_2d[j, 1]))
            cv2.line(combined, pt1, pt2, (200, 200, 200), 1)

    # 4. Add legend
    legend_y = 20
    for group_name, color in keypoint_colors.items():
        cv2.circle(combined, (15, legend_y), 6, color, -1)
        cv2.putText(combined, group_name, (30, legend_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 20

    return combined


def filter_keypoints_by_selection(keypoints_2d, selection='all'):
    """
    Filter keypoints based on selection

    Args:
        keypoints_2d: (22, 3) all keypoints
        selection: 'all', group names ('head', 'spine', etc), or comma-separated indices

    Returns:
        filtered_keypoints: (22, 3) with non-selected keypoints zeroed out
        selected_indices: List of selected keypoint indices
    """
    if selection == 'all':
        return keypoints_2d, list(range(22))

    selected_indices = []

    # Check if it's indices (comma-separated numbers)
    if selection.replace(',', '').replace(' ', '').isdigit():
        selected_indices = [int(x.strip()) for x in selection.split(',')]
    else:
        # Parse group names
        groups = [g.strip().lower() for g in selection.split(',')]
        for group in groups:
            if group in KEYPOINT_GROUPS:
                selected_indices.extend(KEYPOINT_GROUPS[group])

    # Filter keypoints
    filtered = keypoints_2d.copy()
    for i in range(22):
        if i not in selected_indices:
            filtered[i, 2] = 0  # Zero out confidence

    return filtered, selected_indices


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

    def fit_single_image(self, rgb_path, mask_path, visualize=True, keypoint_selection='all'):
        """
        Fit MAMMAL model to a single image

        Args:
            rgb_path: Path to RGB image
            mask_path: Path to binary mask
            visualize: Whether to return visualization
            keypoint_selection: Which keypoints to use ('all', 'head,spine', '0,5,6,21', etc)

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

        # Step 1.5: Filter keypoints if selection specified
        keypoints_filtered, selected_indices = filter_keypoints_by_selection(keypoints_2d, keypoint_selection)

        # Step 2: Initialize MAMMAL parameters
        thetas, bone_lengths, R, T, s, chest_deformer = self.initialize_pose_from_keypoints(keypoints_filtered)

        # Step 3: Optimize to fit keypoints (use filtered keypoints)
        thetas_opt, bone_lengths_opt, R_opt, T_opt, s_opt, chest_deformer_opt = self.optimize_pose_to_keypoints(
            keypoints_filtered, thetas, bone_lengths, R, T, s, chest_deformer,
            n_iterations=50, lr=0.01
        )

        # Step 4: Generate final mesh
        mesh = self.generate_mesh(thetas_opt, bone_lengths_opt, R_opt, T_opt, s_opt, chest_deformer_opt)

        # Visualization
        vis_keypoints = None
        vis_combined = None
        if visualize:
            # Original keypoint visualization
            vis_keypoints = visualize_keypoints_on_frame(rgb, keypoints_2d)

            # Combined overlay visualization (RGB + mask + keypoints)
            vis_combined = create_combined_visualization(
                rgb, mask_binary, keypoints_2d
            )

        results = {
            'mesh': mesh,
            'thetas': thetas_opt.cpu().numpy(),
            'bone_lengths': bone_lengths_opt.cpu().numpy(),
            'R': R_opt.cpu().numpy(),
            'T': T_opt.cpu().numpy(),
            's': s_opt.cpu().numpy(),
            'chest_deformer': chest_deformer_opt.cpu().numpy(),
            'keypoints_2d': keypoints_2d,
            'keypoints_filtered': keypoints_filtered,
            'selected_indices': selected_indices,
            'rgb': rgb,
            'mask': mask_binary,
            'vis_keypoints': vis_keypoints,
            'vis_combined': vis_combined
        }

        return results

    def process_directory(self, input_dir, output_dir, max_images=None, keypoint_selection='all'):
        """
        Process all images in a directory

        Args:
            input_dir: Directory containing RGB and mask images
            output_dir: Output directory for results
            max_images: Maximum number of images to process (None = all)
            keypoint_selection: Which keypoints to use ('all', 'head,spine', '0,5,6,21', etc)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all RGB images
        rgb_files = sorted(list(input_path.glob("*_rgb.png")))

        if max_images is not None:
            rgb_files = rgb_files[:max_images]

        print(f"Found {len(rgb_files)} RGB images")
        if keypoint_selection != 'all':
            print(f"Using keypoint selection: {keypoint_selection}")

        # Process each image
        for rgb_file in tqdm(rgb_files, desc="Processing images"):
            # Get corresponding mask file
            mask_file = rgb_file.parent / rgb_file.name.replace("_rgb.png", "_mask.png")

            if not mask_file.exists():
                print(f"Warning: Mask not found for {rgb_file.name}, skipping")
                continue

            try:
                # Fit MAMMAL model
                results = self.fit_single_image(
                    rgb_file, mask_file,
                    visualize=True,
                    keypoint_selection=keypoint_selection
                )

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
                        'keypoints_2d': results['keypoints_2d'],
                        'keypoints_filtered': results['keypoints_filtered'],
                        'selected_indices': results['selected_indices']
                    }, f)

                # Save keypoint visualization
                if results['vis_keypoints'] is not None:
                    vis_path = output_path / f"{output_name}_keypoints.png"
                    cv2.imwrite(str(vis_path), results['vis_keypoints'])

                # Save combined overlay visualization (RGB + mask + keypoints)
                if results['vis_combined'] is not None:
                    combined_path = output_path / f"{output_name}_overlay.png"
                    cv2.imwrite(str(combined_path), results['vis_combined'])

                print(f"  ✓ Processed {rgb_file.name}")

            except Exception as e:
                print(f"  ✗ Failed to process {rgb_file.name}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Monocular MAMMAL Fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keypoint Selection Examples:
  --keypoints all              Use all 22 keypoints (default)
  --keypoints head,spine       Use head (0-5) and spine (6-13) keypoints
  --keypoints spine,tail       Use spine and tail keypoints only
  --keypoints 0,5,6,13,21      Use specific keypoint indices

Keypoint Groups:
  head:     0-5   (nose, ears, eyes, head center)
  spine:    6-13  (8 spine points)
  limbs:    14-17 (4 paws)
  tail:     18-20 (3 tail points)
  centroid: 21    (body center)
        """
    )
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
    parser.add_argument('--keypoints', type=str, default='all',
                       help='Keypoint selection: all, group names (head,spine,limbs,tail,centroid), or indices (0,5,6,21)')

    args = parser.parse_args()

    # Initialize fitter
    fitter = MonocularMAMMALFitter(
        device=args.device,
        detector=args.detector,
        superanimal_model_path=args.superanimal_model if args.detector == 'superanimal' else None
    )

    # Process directory
    fitter.process_directory(
        args.input_dir, args.output_dir,
        max_images=args.max_images,
        keypoint_selection=args.keypoints
    )

    print(f"\nProcessing complete! Results saved to {args.output_dir}")
    print(f"Output files:")
    print(f"  - *_mesh.obj: 3D mesh (Blender compatible)")
    print(f"  - *_keypoints.png: Keypoint visualization")
    print(f"  - *_overlay.png: Combined overlay (RGB + mask + keypoints)")
    print(f"  - *_params.pkl: MAMMAL parameters")


if __name__ == "__main__":
    main()
