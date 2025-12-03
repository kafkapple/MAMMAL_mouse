# ===== GPU Configuration (MUST be before torch import) =====
import os
import socket

# Auto-detect server and set GPU
_hostname = socket.gethostname()
_gpu_defaults = {
    'gpu05': '1',   # gpu05: use GPU 1
    'bori': '0',    # bori: use GPU 0 (only 1 GPU)
}
_default_gpu = _gpu_defaults.get(_hostname.split('.')[0], '0')
_gpu_id = os.environ.get('GPU_ID', os.environ.get('CUDA_VISIBLE_DEVICES', _default_gpu))

os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_id
os.environ['EGL_DEVICE_ID'] = _gpu_id
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ''  # Disable X11 for headless rendering

import numpy as np
import math
import torch
import pickle 
from time import time 
import json 
from tqdm import tqdm 

import pyrender 
from pyrender.constants import RenderFlags 
import cv2 

import trimesh 
import torch.nn as nn 
import torch.functional as F
from articulation_th import ArticulationTorch 
from bodymodel_th import BodyModelTorch 
from data_seaker_video_new import DataSeakerDet
import copy 
from utils import *
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import hydra # Added for Hydra main decorator
from omegaconf import DictConfig # Added for MouseFitter type hinting
import hydra.utils # Added for path resolution

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    HardFlatShader,
    HardGouraudShader,
    AmbientLights,
    SoftSilhouetteShader
)
from pytorch3d.structures import Meshes
from torch.utils.tensorboard import SummaryWriter 
from pytorch3d.utils import cameras_from_opencv_projection
from omegaconf import DictConfig # Added for MouseFitter type hinting

class MouseFitter():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        mouse_path = "mouse_model/mouse.pkl"
        img_size = (1024, 1152) #H,W
        self.img_size = img_size
        self.device = torch.device('cuda')
        self.bodymodel = ArticulationTorch() 

        if self.cfg.fitter.with_render: 
            self.renderer = pyrender.OffscreenRenderer(viewport_width=img_size[1], viewport_height=img_size[0])
        self.reg_weights = np.loadtxt(hydra.utils.to_absolute_path("mouse_model/reg_weights.txt")).squeeze()

        # Initialize keypoint weights from config
        kw_cfg = getattr(self.cfg, 'keypoint_weights', None)
        sparse_indices = getattr(self.cfg.fitter, 'sparse_keypoint_indices', None)

        # Get default weight (1.0 for full keypoints, 0.0 for sparse mode)
        default_weight = getattr(kw_cfg, 'default', 1.0) if kw_cfg else 1.0
        self.keypoint_weight = np.ones(self.cfg.fitter.keypoint_num) * default_weight

        # If sparse mode with default=0, set specified indices to have weight
        if sparse_indices and default_weight == 0.0:
            print(f"Sparse keypoint mode: using indices {sparse_indices}")
            # Sparse indices will get weight from idx_* config or default to 1.0
            for idx in sparse_indices:
                self.keypoint_weight[idx] = 1.0  # Base weight, will be overridden by idx_* if set

        # Apply individual index weights from config (idx_0, idx_4, idx_18, etc.)
        if kw_cfg:
            for idx in range(self.cfg.fitter.keypoint_num):
                attr_name = f'idx_{idx}'
                if hasattr(kw_cfg, attr_name):
                    self.keypoint_weight[idx] = getattr(kw_cfg, attr_name)

        # Fallback: if no config, use original MAMMAL paper weights
        if kw_cfg is None:
            self.keypoint_weight[4] = 0.4
            self.keypoint_weight[11] = 0.9
            self.keypoint_weight[15] = 0.9
            self.keypoint_weight[5] = 2
            self.keypoint_weight[6] = 1.5
            self.keypoint_weight[7] = 1.5

        print(f"Keypoint weights: {self.keypoint_weight}")
        self.keypoint_weight = torch.from_numpy(self.keypoint_weight).reshape([1,-1,1]).to(self.device)
        
        bone_weight = np.ones(20) 
        bone_weight[11] = 1
        bone_weight[19] = 1
        self.bone_weight = torch.from_numpy(bone_weight).reshape([1,-1,1]).to(self.device) 
        self.data_loader = None 

        self.result_folder = ""

        # Load loss weights from config (with original MAMMAL paper values as defaults)
        lw_cfg = getattr(self.cfg, 'loss_weights', None)
        self.term_weights = {
            "theta": getattr(lw_cfg, 'theta', 3.0) if lw_cfg else 3.0,
            "3d": getattr(lw_cfg, '3d', 2.5) if lw_cfg else 2.5,
            "2d": getattr(lw_cfg, '2d', 0.2) if lw_cfg else 0.2,
            "bone": getattr(lw_cfg, 'bone', 0.5) if lw_cfg else 0.5,
            "scale": getattr(lw_cfg, 'scale', 0.5) if lw_cfg else 0.5,
            "mask": getattr(lw_cfg, 'mask', 10.0) if lw_cfg else 10.0,  # Original MAMMAL value
            "chest_deformer": getattr(lw_cfg, 'chest_deformer', 0.1) if lw_cfg else 0.1,
            "stretch": getattr(lw_cfg, 'stretch', 1.0) if lw_cfg else 1.0,
            "temp": getattr(lw_cfg, 'temp', 0.25) if lw_cfg else 0.25,
            "temp_d": getattr(lw_cfg, 'temp_d', 0.2) if lw_cfg else 0.2,
        }
        # Store step-specific mask weights
        self.mask_weight_step0 = getattr(lw_cfg, 'mask_step0', 0.0) if lw_cfg else 0.0
        self.mask_weight_step1 = getattr(lw_cfg, 'mask_step1', 0.0) if lw_cfg else 0.0
        self.mask_weight_step2 = getattr(lw_cfg, 'mask_step2', 3000.0) if lw_cfg else 3000.0

        # Store keypoint weights config for step2 adjustment
        kw_cfg = getattr(self.cfg, 'keypoint_weights', None)
        self.tail_weight_step2 = getattr(kw_cfg, 'tail_step2', 10.0) if kw_cfg else 10.0

        # Print loaded loss weights for verification
        print(f"Loss weights loaded: {self.term_weights}")
        print(f"Mask weights per step: step0={self.mask_weight_step0}, step1={self.mask_weight_step1}, step2={self.mask_weight_step2}")
        print(f"Keypoint weights: tail_step2={self.tail_weight_step2}")

        # Disable keypoint loss if use_keypoints is false
        if not getattr(self.cfg.fitter, 'use_keypoints', True):
            self.term_weights["2d"] = 0
            # Apply silhouette-specific weights from config
            sil_cfg = getattr(self.cfg, 'silhouette', None)
            if sil_cfg:
                self.term_weights["scale"] = getattr(sil_cfg, 'scale_weight', 50.0)
                self.term_weights["theta"] = getattr(sil_cfg, 'theta_weight', 10.0)
                self.term_weights["bone"] = getattr(sil_cfg, 'bone_weight', 2.0)
                self.silhouette_iter_multiplier = getattr(sil_cfg, 'iter_multiplier', 2.0)
                self.use_pca_init = getattr(sil_cfg, 'use_pca_init', True)
            else:
                self.term_weights["scale"] = 50.0
                self.term_weights["theta"] = 10.0
                self.term_weights["bone"] = 2.0
                self.silhouette_iter_multiplier = 2.0
                self.use_pca_init = True
            print(f"Silhouette-only mode: theta={self.term_weights['theta']}, bone={self.term_weights['bone']}, scale={self.term_weights['scale']}, iter_mult={self.silhouette_iter_multiplier}, pca_init={self.use_pca_init}")

        self.losses = {}
        self.loss_history = []  # For storing loss values over iterations
        self.param_history = []  # For storing parameter values over iterations

        ## init differentiable renderer
        sigma = 3e-5
        raster_settings_soft = RasterizationSettings(
            image_size=img_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            # blur_radius=0.01,
            faces_per_pixel=50,
            # bin_size = 56, 
            # max_faces_per_bin= 16
        )
        self.renderer_mask = MeshRenderer( 
            rasterizer=MeshRasterizer(raster_settings=raster_settings_soft),
            shader = SoftSilhouetteShader() 
        )
        faces_np = self.bodymodel.faces_vert_np.astype(np.int64) 
        self.faces_th = torch.from_numpy(faces_np).to(self.device).unsqueeze(0) 
        self.faces_th_reduced = torch.from_numpy(self.bodymodel.faces_reduced_7200).to(self.device).unsqueeze(0)

        self.mask_loss_func = torch.nn.MSELoss()

        ## data loader 
        self.cam_dict = [] 
        self.imgs = [] 
        self.id = None 
        
        self.last_params = None
        self.V_last = None
        self.J_last = None

        # Debug grid collector for compressed iteration images
        self.debug_collector = DebugGridCollector(
            thumbnail_size=(320, 240),
            grid_cols=5,
            jpeg_quality=85
        )

    def _log_iteration(self, step, iteration, total_loss, params):
        """Log iteration details with loss and parameter values."""
        # Store to history
        record = {
            'frame': self.id,
            'step': step,
            'iteration': iteration,
            'total_loss': total_loss,
            **self.losses,
            'scale': float(params['scale'].detach().cpu().mean()),
            'trans_norm': float(torch.norm(params['trans'].detach()).cpu()),
            'theta_norm': float(torch.norm(params['thetas'].detach()).cpu()),
        }
        self.loss_history.append(record)

        # Print detailed log to console
        loss_parts = ' | '.join([f"{k}:{v:.2f}" for k, v in self.losses.items()])
        print(f"  [{step}] iter {iteration:3d}: total={total_loss:.2f} | {loss_parts}")

    def save_loss_history(self, filepath):
        """Save loss history to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        print(f"Loss history saved to: {filepath}")

    def plot_loss_history(self, save_dir):
        """Generate and save loss visualization plots."""
        if not self.loss_history:
            print("No loss history to plot")
            return

        import pandas as pd
        df = pd.DataFrame(self.loss_history)

        # Plot 1: Total loss over iterations (all frames combined)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total loss by step
        ax = axes[0, 0]
        for step in ['Step0', 'Step1', 'Step2']:
            step_data = df[df['step'] == step]
            if not step_data.empty:
                ax.plot(step_data.index, step_data['total_loss'], label=step, alpha=0.7)
        ax.set_xlabel('Record Index')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss by Optimization Step')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Individual loss components
        ax = axes[0, 1]
        loss_cols = ['2d', 'mask', 'theta', 'bone', 'scale']
        available_cols = [c for c in loss_cols if c in df.columns]
        for col in available_cols:
            ax.plot(df.index, df[col], label=col, alpha=0.7)
        ax.set_xlabel('Record Index')
        ax.set_ylabel('Loss Value')
        ax.set_title('Individual Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss by frame (final values)
        ax = axes[1, 0]
        frames = df['frame'].unique()
        final_losses = []
        for frame in sorted(frames):
            frame_data = df[df['frame'] == frame]
            if not frame_data.empty:
                final_losses.append(frame_data['total_loss'].iloc[-1])
        ax.bar(range(len(final_losses)), final_losses)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Final Total Loss')
        ax.set_title('Final Loss per Frame')
        ax.grid(True, alpha=0.3)

        # Parameter evolution
        ax = axes[1, 1]
        if 'scale' in df.columns:
            ax.plot(df.index, df['scale'], label='scale', alpha=0.7)
        if 'trans_norm' in df.columns:
            ax.plot(df.index, df['trans_norm'], label='trans_norm', alpha=0.7)
        if 'theta_norm' in df.columns:
            ax.plot(df.index, df['theta_norm'], label='theta_norm', alpha=0.7)
        ax.set_xlabel('Record Index')
        ax.set_ylabel('Value')
        ax.set_title('Parameter Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'loss_history.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Loss plot saved to: {plot_path}")

        # Also save per-frame loss plots
        self._plot_per_frame_losses(df, save_dir)

    def _plot_per_frame_losses(self, df, save_dir):
        """Generate per-frame loss convergence plots."""
        frames = df['frame'].unique()
        if len(frames) <= 1:
            return

        n_frames = len(frames)
        n_cols = min(4, n_frames)
        n_rows = (n_frames + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = np.atleast_2d(axes)

        for idx, frame in enumerate(sorted(frames)):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            frame_data = df[df['frame'] == frame]
            for step in ['Step0', 'Step1', 'Step2']:
                step_data = frame_data[frame_data['step'] == step]
                if not step_data.empty:
                    ax.plot(step_data['iteration'], step_data['total_loss'],
                           label=step, marker='o', markersize=2)

            ax.set_title(f'Frame {frame}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(frames), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'loss_per_frame.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Per-frame loss plot saved to: {plot_path}")

    def set_cameras_dannce(self, cams):
        self.camN = len(cams)
        self.cam_dict = cams 
        self.cams_th = [] 
        self.Rs = [] 
        self.Ks = [] 
        self.Ts = [] 
        for cam in cams: 
            R = np.expand_dims(cam['R'].T, 0).astype(np.float32)
            K = np.expand_dims(cam['K'].T, 0).astype(np.float32) 
            T = cam['T'].astype(np.float32)
            
            # Fix T shape for PyTorch3D: ensure it's (1, 3) not (3, 1) or (1, 3, 1)
            if T.shape == (3, 1):
                T = T.T  # (3, 1) -> (1, 3)
            elif T.shape == (1, 3, 1):
                T = T.squeeze(-1)  # (1, 3, 1) -> (1, 3)
            elif T.shape == (3,):
                T = T.reshape(1, 3)  # (3,) -> (1, 3)
            
            img_size_np = np.expand_dims(np.asarray(self.img_size), 0).astype(np.float32)
            cam_th = self.build_opencv_camera(R, T, K, img_size_np)
            self.cams_th.append(cam_th)       
            self.Rs.append(torch.from_numpy(R).to(self.device))
            
            # For calc_2d_keypoint_loss, we need T in different shape
            T_original = cam['T'].astype(np.float32)
            if T_original.shape == (3, 1):
                T_for_projection = np.expand_dims(T_original, 0)  # (3, 1) -> (1, 3, 1)
            elif T_original.shape == (1, 3):
                T_for_projection = T_original.reshape(1, 3, 1)  # (1, 3) -> (1, 3, 1)
            elif T_original.shape == (3,):
                T_for_projection = T_original.reshape(1, 3, 1)  # (3,) -> (1, 3, 1)
            else:
                T_for_projection = T_original
                
            self.Ts.append(torch.from_numpy(T_for_projection).to(self.device))
            self.Ks.append(torch.from_numpy(K).to(self.device)) 

    # build camera from OpenCV camera format 
    # R: [N, 3,3], np.ndarray
    # T: [N, 3], np.ndarray
    # K: [N,3,3], np.ndarray
    # imgsize:[N, 2 ], np.ndarray, h,w
    def build_opencv_camera(self, R, T, K, imgsize): 
        R_tensor = torch.from_numpy(R).to(self.device) 
        tvec_tensor = torch.from_numpy(T).to(self.device) 
        K_tensor = torch.from_numpy(K).to(self.device) 
        imgsize_tensor = torch.from_numpy(imgsize).to(self.device)
        return cameras_from_opencv_projection(R=R_tensor,
            tvec=tvec_tensor, camera_matrix=K_tensor, image_size=imgsize_tensor)

    def init_params(self, batch_size):
        body_param = {
            "thetas": np.tile(self.bodymodel.init_joint_rotvec_np, [batch_size, 1, 1]),
            "trans": np.zeros([batch_size,3]),
            "scale": np.ones([batch_size,1]) * 115,
            "rotation": np.zeros([batch_size, 3]),
            "bone_lengths": np.zeros([batch_size, 20]),
            "chest_deformer": np.zeros([batch_size, 1])
        }

        self.init_thetas = torch.tensor(body_param["thetas"], dtype=torch.float32, device=self.device)

        return body_param

    def init_params_from_masks(self, masks, cam_dicts, batch_size=1):
        """
        Initialize parameters from mask silhouettes when keypoints are not available.
        Uses mask centroids and bounding boxes to estimate initial position and scale.

        Args:
            masks: dict of masks for each view {"mask0": tensor, "mask1": tensor, ...}
            cam_dicts: list of camera parameter dicts
            batch_size: batch size (default 1)

        Returns:
            body_param: initialized parameters dict
        """
        # Start with default params
        body_param = self.init_params(batch_size)

        # Collect 2D centroids and areas from all views
        centroids_2d = []
        areas = []
        valid_views = []

        for view_id in range(len(cam_dicts)):
            mask_key = f"mask{view_id}"
            if mask_key not in masks:
                continue

            mask = masks[mask_key]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = mask.squeeze()

            # Find mask region
            if mask_np.max() <= 0:
                continue

            # Binarize mask
            binary_mask = (mask_np > 0.5).astype(np.uint8)

            # Find contours or use moments
            moments = cv2.moments(binary_mask)
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                area = moments["m00"]
                centroids_2d.append([cx, cy])
                areas.append(area)
                valid_views.append(view_id)

        if len(valid_views) < 2:
            print("Warning: Not enough valid masks for initialization, using defaults")
            return body_param

        # Estimate 3D position using triangulation from multiple views
        # Simplified: use average back-projection from two views
        points_3d = []
        for i, view_id in enumerate(valid_views[:min(4, len(valid_views))]):
            cam = cam_dicts[view_id]
            K = cam['K'].T if cam['K'].shape[0] == 3 else cam['K']
            R = cam['R'].T if cam['R'].shape[0] == 3 else cam['R']
            T = cam['T'].squeeze() / 1000  # Convert to meters

            cx, cy = centroids_2d[i]

            # Back-project to ray
            K_inv = np.linalg.inv(K)
            ray_cam = K_inv @ np.array([cx, cy, 1.0])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)

            # Transform ray to world coordinates
            R_inv = R.T
            ray_world = R_inv @ ray_cam
            cam_center = -R_inv @ T

            # Estimate depth from mask area (rough approximation)
            # Larger area = closer to camera
            avg_area = np.mean(areas)
            estimated_depth = 0.3 * np.sqrt(100000 / max(avg_area, 1000))  # Heuristic
            estimated_depth = np.clip(estimated_depth, 0.1, 0.5)

            point_3d = cam_center + ray_world * estimated_depth
            points_3d.append(point_3d)

        # Average 3D position estimate
        if len(points_3d) > 0:
            estimated_trans = np.mean(points_3d, axis=0) * 1000  # Convert back to mm
            body_param["trans"] = estimated_trans.reshape([batch_size, 3])
            print(f"Mask-based init: estimated trans = {estimated_trans}")

        # Estimate scale from average mask area
        avg_area = np.mean(areas)
        # Heuristic: map area to scale (calibrated for mouse size)
        # Default scale is 115, so we use that as baseline
        estimated_scale = np.clip(np.sqrt(avg_area) * 0.2, 100, 130)
        # If estimation is uncertain, use default
        if avg_area < 5000:
            estimated_scale = 115  # Use default for small/uncertain masks
        body_param["scale"] = np.ones([batch_size, 1]) * estimated_scale
        print(f"Mask-based init: estimated scale = {estimated_scale} (avg_area={avg_area:.0f})")

        # PCA-based rotation estimation (if enabled)
        if getattr(self, 'use_pca_init', True):
            try:
                rotation_estimates = []
                for view_id in valid_views[:min(3, len(valid_views))]:
                    mask_key = f"mask{view_id}"
                    mask = masks[mask_key]
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.squeeze().cpu().numpy()
                    else:
                        mask_np = mask.squeeze()

                    binary_mask = (mask_np > 0.5).astype(np.uint8)

                    # Find contour points
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) == 0:
                        continue

                    # Get largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    points = largest_contour.reshape(-1, 2).astype(np.float32)

                    if len(points) < 10:
                        continue

                    # PCA on contour points
                    mean_pt = np.mean(points, axis=0)
                    centered = points - mean_pt
                    cov = np.cov(centered.T)
                    eigenvalues, eigenvectors = np.linalg.eig(cov)

                    # Principal axis (direction of largest variance = body axis)
                    idx = np.argmax(eigenvalues)
                    principal_axis = eigenvectors[:, idx]

                    # Convert 2D angle to rotation around Z axis
                    angle_2d = np.arctan2(principal_axis[1], principal_axis[0])
                    rotation_estimates.append(angle_2d)

                if len(rotation_estimates) > 0:
                    # Average rotation estimate
                    avg_rotation = np.mean(rotation_estimates)
                    # Convert to rotation vector (rotation around Z axis)
                    # Mouse typically lies flat, so main rotation is around vertical (Y or Z)
                    body_param["rotation"] = np.array([[0, avg_rotation, 0]], dtype=np.float32)
                    print(f"Mask-based init: estimated rotation (Y) = {np.degrees(avg_rotation):.1f} deg")
            except Exception as e:
                print(f"PCA rotation estimation failed: {e}, using default rotation")

        return body_param 

    def calc_2d_keypoint_loss(self, J3d, x2): 
        loss = 0 
        for camid in range(self.camN): 
            # Fix: proper matrix multiplication order for camera projection
            # J3d: (1, 22, 3), Rs: (1, 3, 3) -> need (1, 3, 3) @ (1, 3, 22) = (1, 3, 22)
            # Camera projection with corrected matrix operations
            J3d_t = J3d.transpose(1, 2)  # (1, 3, 22)
            rotated = self.Rs[camid] @ J3d_t  # (1, 3, 3) @ (1, 3, 22) = (1, 3, 22)
            
            # T vector broadcasting
            T_vec = self.Ts[camid]  # Should be (1, 3, 1)
            if T_vec.dim() == 2:
                T_vec = T_vec.unsqueeze(2)  # (1, 3) -> (1, 3, 1)
                
            J3d_cam = rotated + T_vec  # (1, 3, 22) + (1, 3, 1) = (1, 3, 22)
            J2d = self.Ks[camid] @ J3d_cam  # (1, 3, 3) @ (1, 3, 22) = (1, 3, 22)
            J2d = J2d.transpose(1, 2)  # (1, 22, 3)
            J2d = J2d / J2d[:,:,2:3]  # Normalize by z coordinate  
            J2d = J2d[:,:,0:2]  # Take only x,y coordinates: (1, 22, 2)
            
            # Fix keypoint_weight broadcasting: ensure it matches the tensor dimensions
            # keypoint_weight has shape (1, 22, 1), need to broadcast to match (1, 22, 2)
            diff = (J2d - x2[:,camid,:,0:2]) * x2[:,camid,:,2:]  # Shape: (1, 22, 2)
            weighted_diff = diff * self.keypoint_weight[..., [0,0]]  # Broadcast weight to last dim
            loss += torch.mean(torch.norm(weighted_diff, dim=-1) ) 
        return loss     

    def calc_3d_loss(self, x1, x2): 
        res = (x1 - x2) * self.keypoint_weight
        loss = torch.mean(torch.norm(res, dim=-1))
        return loss 

    ## both J_prev and J_curr are embed joints (140, 3)
    def calc_deformer_end_temporal_loss(self, J_prev, J_curr): 
        res = J_prev[:,[50,120]] - J_curr[:,[50,120]]
        loss  = torch.mean(torch.norm(res, dim=-1)) * 5 
        return loss 

    def calc_scale_loss(self, scale, target): 
        loss = torch.mean(torch.norm(scale-target, dim=-1))
        return loss 

    def calc_chest_deformer_loss(self, chest_deformer): 
        loss = torch.mean(torch.norm(chest_deformer, dim=-1))
        return loss 

    def calc_bone_length_constraint(self, bone_lengths): 
        loss = torch.norm(bone_lengths * self.bone_weight)
        return loss 

    def calc_temporal_term(self, params, V, J): 
        loss = 0
        for k, v in params.items():
            loss = loss + torch.mean(torch.norm(params[k] - self.last_params[k], dim=-1))
        loss += torch.mean(torch.norm(V-self.V_last, dim=-1))
        loss += torch.mean(torch.norm(J-self.J_last, dim=-1)) 
        return loss  

    def calc_stretch_to_constraints(self, joints): 
        dist1 = torch.mean(torch.norm(joints[:,50] - joints[:,121], dim=-1)) 
        dist2 = torch.mean(torch.norm(joints[:,65] - joints[:,123], dim=-1)) 
        dist3 = torch.mean(torch.norm(joints[:,72] - joints[:,134], dim=-1)) 
        dist4 = torch.mean(torch.norm(joints[:,97] - joints[:,134], dim=-1)) 
        return dist1 + dist2 + dist3 + dist4

    ## theta regularization. 
    def calc_theta_loss(self, thetas):
        weights = torch.from_numpy(self.reg_weights).reshape([1,-1,1]).type(torch.float32).to(self.device)
        loss_theta = torch.norm( (thetas - self.init_thetas) * weights)
        return loss_theta 

    def set_previous_frame(self, previous_params): 
        self.last_params = {} 
        for k,v in previous_params.items(): 
            self.last_params.update({
                k: v.detach().to(self.device) 
            })
        self.V_last, self.J_last = self.bodymodel.forward(
            self.last_params["thetas"], 
            self.last_params["bone_lengths"],
            self.last_params["rotation"],
            self.last_params["trans"],
            self.last_params["scale"], 
            self.last_params["chest_deformer"]
        )

    def gen_closure(self, optimizer, body_param, target):
        def closure():
            optimizer.zero_grad() 
            V,J = self.bodymodel.forward(body_param["thetas"], body_param["bone_lengths"], \
                body_param["rotation"], body_param["trans"], body_param["scale"], body_param["chest_deformer"])

            keypoints = self.bodymodel.forward_keypoints22() 
            # loss_3d = self.calc_3d_loss(keypoints, target["target_3d"])
            loss_2d = self.calc_2d_keypoint_loss(keypoints, target["target_2d"])
            
            loss_theta = self.calc_theta_loss(body_param["thetas"])

            loss_bone_length = self.calc_bone_length_constraint(body_param["bone_lengths"])

            loss_scale = self.calc_scale_loss(body_param["scale"], 115)

            loss_chest_deformer = self.calc_chest_deformer_loss(body_param["chest_deformer"])

            loss_stretch_to_constraints = self.calc_stretch_to_constraints(J) 


            loss_temp = 0 
            loss_deformer_temp = 0
            if self.last_params is not None: 
                loss_temp = self.calc_temporal_term(body_param, V, J)
                loss_deformer_temp = self.calc_deformer_end_temporal_loss(J_prev=self.J_last, J_curr = J)


            ## mask loss
            V_reduced = V[:,self.bodymodel.reduced_ids,:]
            mesh = Meshes(V_reduced, self.faces_th_reduced)
            loss_mask = torch.tensor(0.0, device=self.device)
            if self.term_weights["mask"] > 0:
                # Use position indices (0, 1, 2, ...) for cams_th and target masks
                for view_idx in range(self.camN):
                    mask = self.renderer_mask(mesh, cameras = self.cams_th[view_idx])[...,-1]
                    target_mask = target["mask"+str(view_idx)]
                    # Resize target mask to match rendered mask if needed
                    if mask.shape != target_mask.shape:
                        print(f"Mask shape mismatch: rendered {mask.shape}, target {target_mask.shape}. Skipping mask loss.")
                        continue
                    loss_mask += self.mask_loss_func(mask, target_mask) 

            self.losses.update({
                "theta": round(float(loss_theta.detach().cpu().numpy()), 2),
                "2d": round(float(loss_2d.detach().cpu().numpy()), 2), 
                "bone": round(float(loss_bone_length.detach().cpu().numpy()), 2), 
                "scale" : round(float(loss_scale.detach().cpu().numpy()), 2),
                "mask": round(float(loss_mask.detach().cpu().numpy()), 2),
                "chest_d": round(float(loss_chest_deformer.detach().cpu().numpy()), ), 
                "stretch": round(float(loss_stretch_to_constraints.detach().cpu().numpy()), 2)
            })
            if loss_temp > 0: 
                self.losses.update(
                    {
                        "temp": round(float(loss_temp.detach().cpu().numpy()), 2), 
                        "temp_d": round(float(loss_deformer_temp.detach().cpu().numpy()), 2)
                    }
                )

            loss_v = loss_2d * self.term_weights["2d"] \
                + loss_theta * self.term_weights["theta"] \
                + loss_bone_length * self.term_weights["bone"] \
                + loss_scale * self.term_weights["scale"] \
                + loss_mask * self.term_weights["mask"] \
                + loss_chest_deformer * self.term_weights["chest_deformer"] \
                + loss_stretch_to_constraints * self.term_weights["stretch"] \
                + loss_temp * self.term_weights["temp"] \
                + loss_deformer_temp * self.term_weights["temp_d"]

            loss_v.backward() 
            return loss_v 
        return closure

    def solve_step0(self, params, target, max_iters, pbar=None):
        """Step 0: Global positioning (trans, rotation, scale)"""
        tolerate = 1e-2
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        closure = self.gen_closure(optimizer, params, target)

        loss_prev = float('inf')
        self.keypoint_weight[:,16:19,:] = 1
        self.keypoint_weight[:,19:22,:] = 1
        if not getattr(self.cfg.fitter, 'use_keypoints', True):
            self.term_weights["mask"] = max(self.mask_weight_step0, 1000.0)
        else:
            self.term_weights["mask"] = self.mask_weight_step0
        self.term_weights["stretch"] = 0
        params["chest_deformer"].requires_grad_(False)
        params["thetas"].requires_grad_(False)
        params["bone_lengths"].requires_grad_(False)

        # Iteration progress bar (nested under frame progress)
        iter_pbar = tqdm(range(max_iters), desc="  Step0", leave=False,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        for i in iter_pbar:
            loss = optimizer.step(closure).item()
            # Show all loss components in progress bar
            loss_str = {k: f"{v:.1f}" for k, v in self.losses.items()}
            iter_pbar.set_postfix(total=f"{loss:.1f}", **loss_str)
            if pbar:
                pbar.set_postfix(step="S0", iter=f"{i+1}/{max_iters}", loss=f"{loss:.1f}")
            # Log detailed parameters every 10 iterations
            if i % 10 == 0:
                self._log_iteration("Step0", i, loss, params)
            if abs(loss-loss_prev) < tolerate:
                break
            loss_prev = loss
            if self.cfg.fitter.with_render:
                imgs = self.imgs.copy()
                # Render to memory and add to grid collector instead of saving individual files
                render_img = self.render(params, imgs, 0, None, self.cam_dict, step_name='Step0', return_image=True)
                if render_img is not None:
                    self.debug_collector.add_image('step0', i, render_img)
        iter_pbar.close()

        # Save Step0 debug images as compressed grid
        if self.cfg.fitter.with_render and self.debug_collector.images.get('step0'):
            grid_path = f"{self.result_folder}/render/debug/step0_frame_{self.id:06d}_grid.jpg"
            self.debug_collector._save_single_grid('step0', grid_path)

        params["chest_deformer"].requires_grad_(True)
        params["thetas"].requires_grad_(True)
        params["bone_lengths"].requires_grad_(True)
        return params

    def solve_step1(self, params, target, max_iters, pbar=None):
        """Step 1: Articulation fitting (thetas, bone_lengths)"""
        tolerate = 1e-4
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        closure = self.gen_closure(optimizer, params, target)

        loss_prev = float('inf')
        self.keypoint_weight[:,16:19,:] = 1
        self.keypoint_weight[:,19:22,:] = 1
        if not getattr(self.cfg.fitter, 'use_keypoints', True):
            self.term_weights["mask"] = max(self.mask_weight_step1, 1500.0)
        else:
            self.term_weights["mask"] = self.mask_weight_step1
        self.term_weights["stretch"] = 0
        params["chest_deformer"].requires_grad_(False)

        # Iteration progress bar
        iter_pbar = tqdm(range(max_iters), desc="  Step1", leave=False,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        for i in iter_pbar:
            loss = optimizer.step(closure).item()
            # Show all loss components in progress bar
            loss_str = {k: f"{v:.1f}" for k, v in self.losses.items()}
            iter_pbar.set_postfix(total=f"{loss:.1f}", **loss_str)
            if pbar:
                pbar.set_postfix(step="S1", iter=f"{i+1}/{max_iters}", loss=f"{loss:.1f}")
            # Log detailed parameters every 10 iterations
            if i % 10 == 0:
                self._log_iteration("Step1", i, loss, params)
            if abs(loss-loss_prev) < tolerate:
                break
            loss_prev = loss
            if self.id == 0 and self.cfg.fitter.with_render:
                imgs = self.imgs.copy()
                # Render to memory and add to grid collector instead of saving individual files
                render_img = self.render(params, imgs, 0, None, self.cam_dict, step_name='Step1', return_image=True)
                if render_img is not None:
                    self.debug_collector.add_image('step1', i, render_img)
        iter_pbar.close()

        # Save Step1 debug images as compressed grid
        if self.id == 0 and self.cfg.fitter.with_render and self.debug_collector.images.get('step1'):
            grid_path = f"{self.result_folder}/render/debug/step1_frame_{self.id:06d}_grid.jpg"
            self.debug_collector._save_single_grid('step1', grid_path)

        if self.cfg.fitter.with_render:
            imgs = self.imgs.copy()
            # Pass target_2d for keypoint overlay
            target_2d = target.get('target_2d', None) if getattr(self.cfg.fitter, 'use_keypoints', True) else None
            self.render(params, imgs, 0, self.result_folder + "/render/step_1_frame_{:06d}.png".format(self.id), self.cam_dict,
                       show_keypoints=True, step_name='Step1', target_2d=target_2d)
            if getattr(self.cfg.fitter, 'use_keypoints', True) and target_2d is not None:
                # Save keypoint comparison to render/ subfolder
                self.draw_keypoints_compare(params, imgs, 0, self.result_folder + "/render/keypoints/step_1_frame_{:06d}_keypoints.png".format(self.id), self.cam_dict, target_2d)
        with open(self.result_folder + "/params/step_1_frame_{:06d}.pkl".format(self.id), 'wb') as f:
            pickle.dump(params,f)
        params["chest_deformer"].requires_grad_(True)
        return params

    def solve_step2(self, params, target, max_iters, pbar=None):
        """Step 2: Silhouette refinement with mask loss"""
        tolerate = 1e-4
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        closure = self.gen_closure(optimizer, params, target)

        # Apply step2-specific weights from config
        self.keypoint_weight[:,16:19,:] = self.tail_weight_step2
        self.keypoint_weight[:,19:22,:] = self.tail_weight_step2
        self.term_weights["mask"] = self.mask_weight_step2
        self.term_weights["stretch"] = 0
        loss_prev = float('inf')
        optimizer.zero_grad()

        # Iteration progress bar
        iter_pbar = tqdm(range(max_iters), desc="  Step2", leave=False,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        for i in iter_pbar:
            loss = optimizer.step(closure).item()
            # Show all loss components in progress bar
            loss_str = {k: f"{v:.1f}" for k, v in self.losses.items()}
            iter_pbar.set_postfix(total=f"{loss:.1f}", **loss_str)
            if pbar:
                pbar.set_postfix(step="S2", iter=f"{i+1}/{max_iters}", loss=f"{loss:.1f}")
            # Log detailed parameters every 10 iterations
            if i % 10 == 0:
                self._log_iteration("Step2", i, loss, params)
            if abs(loss-loss_prev) < tolerate:
                break
            loss_prev = loss
        iter_pbar.close()

        if self.cfg.fitter.with_render:
            imgs = self.imgs.copy()
            self.render(params, imgs, 0, self.result_folder+"/render/step_2_frame_{:06d}.png".format(self.id), self.cam_dict,
                       show_keypoints=True, step_name='Step2')
        with open(self.result_folder + "/params/step_2_frame_{:06d}.pkl".format(self.id), 'wb') as f:
            pickle.dump(params,f)
        self.result = params

        # Save the final mesh as an .obj file
        V, _ = self.bodymodel.forward(
            params["thetas"], params["bone_lengths"],
            params["rotation"], params["trans"], params["scale"], params["chest_deformer"]
        )
        vertices = V[0].detach().cpu().numpy()
        faces = self.bodymodel.faces_vert_np
        obj_filename = os.path.join(self.result_folder, "obj/step_2_frame_{:06d}.obj".format(self.id))
        with open(obj_filename, 'w') as fp:
            for v in vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

        return params 
    def render(self, result, imgs, batch_id, filename, cams_dict,
               show_keypoints=False, step_name=None, target_2d=None, return_image=False):
        """
        Render mesh overlay on images for all available views.

        Args:
            result: body model parameters
            imgs: list of images (indexed by position in views_to_use)
            batch_id: batch index
            filename: output filename (None to skip saving)
            cams_dict: list of camera dicts (indexed by position in views_to_use)
            show_keypoints: if True, overlay keypoints with part-aware colors
            step_name: if provided, add step description text to image
            target_2d: GT 2D keypoints tensor for comparison overlay
            return_image: if True, always return the rendered image
        """
        V,J = self.bodymodel.forward(result["thetas"], result["bone_lengths"],
            result["rotation"], result["trans"] / 1000, result["scale"] / 1000, result["chest_deformer"])
        vertices = V[batch_id].detach().cpu().numpy()
        faces = self.bodymodel.faces_vert_np

        # Get keypoints if needed
        keypoints_2d = None
        if show_keypoints:
            keypoints = self.bodymodel.forward_keypoints22()
            joints_3d = keypoints[batch_id].detach().cpu().numpy()

        scene = pyrender.Scene()
        light_node = scene.add(pyrender.PointLight(color=np.ones(3), intensity=0.2))
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=np.array([0.8, 0.6, 0.4]))))
        color_maps = []
        # Iterate over all available views (position indices)
        num_views = len(imgs)
        for view_idx in range(num_views):
            cam_param = cams_dict[view_idx]
            K, R, T = cam_param['K'].T, cam_param['R'].T, cam_param['T'] / 1000
            # Fix T shape: ensure it's (3, 1) for matrix multiplication
            if T.shape == (1, 3):
                T = T.T
            elif T.shape == (3,):
                T = T.reshape(3, 1)
            elif T.shape == (1, 3, 1):
                T = T.squeeze().reshape(3, 1)
            elif T.shape == (3, 1, 1):
                T = T.squeeze()
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R.T
            camera_pose[:3, 3:4] = np.dot(-R.T, T)
            camera_pose[:, 1:3] = -camera_pose[:, 1:3]
            camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
            cam_node = scene.add(camera, name='cam', pose=camera_pose)
            light_node._matrix = camera_pose
            color, _ = self.renderer.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
            scene.remove_node(cam_node)
            img_i = imgs[view_idx]
            color = copy.deepcopy(color)

            # Resize rendered image to match input image size
            if color.shape[:2] != img_i.shape[:2]:
                color = cv2.resize(color, (img_i.shape[1], img_i.shape[0]))

            background_mask = color[:, :, :] == 255
            color[background_mask] = img_i[background_mask]

            # Overlay keypoints if requested
            if show_keypoints:
                # Project 3D keypoints to 2D
                T_flat = cam_param['T'] / 1000
                if len(T_flat.shape) > 1:
                    T_flat = T_flat.squeeze()
                data2d = (joints_3d @ cam_param['R'] + T_flat) @ cam_param['K']
                data2d = data2d[:, 0:2] / data2d[:, 2:]

                # Get sparse indices if in sparse mode
                sparse_indices = getattr(self.cfg.fitter, 'sparse_keypoint_indices', None)
                if sparse_indices:
                    indices_to_draw = list(sparse_indices)
                else:
                    indices_to_draw = list(range(22))

                # Draw predicted keypoints with part-aware colors (small circles)
                for idx in indices_to_draw:
                    if idx >= data2d.shape[0]:
                        continue
                    x, y = data2d[idx, 0], data2d[idx, 1]
                    if math.isnan(x) or math.isnan(y) or (x == 0 and y == 0):
                        continue
                    p = (int(x), int(y))
                    part_color = self.get_keypoint_color(idx)
                    cv2.circle(color, p, 5, part_color, -1)
                    cv2.circle(color, p, 5, (255, 255, 255), 1)  # White outline

                # Draw GT keypoints if available (larger circles with X marker)
                if target_2d is not None:
                    gt_data = target_2d[batch_id, view_idx].detach().cpu().numpy()
                    gt_xy = gt_data[:, :2]
                    gt_conf = gt_data[:, 2]
                    for idx in indices_to_draw:
                        if idx >= gt_xy.shape[0] or gt_conf[idx] < 0.25:
                            continue
                        x, y = gt_xy[idx, 0], gt_xy[idx, 1]
                        if math.isnan(x) or math.isnan(y) or (x == 0 and y == 0):
                            continue
                        p = (int(x), int(y))
                        part_color = self.get_keypoint_color(idx)
                        # Draw X marker for GT
                        cv2.drawMarker(color, p, part_color, cv2.MARKER_CROSS, 12, 2)

            color_maps.append(color)

        output = pack_images(color_maps)

        # Add step description if provided
        if step_name:
            output = self._add_step_label(output, step_name)

        if filename is not None:
            cv2.imwrite(filename, output)
        return output

    def _add_step_label(self, img, step_name):
        """Add step description label to the top of the image."""
        step_descriptions = {
            'Step0': 'Step 0: Global Positioning (trans, rotation, scale only)',
            'Step1': 'Step 1: Articulation Fitting (joint angles, bone lengths)',
            'Step2': 'Step 2: Silhouette Refinement (mask loss enabled)',
        }
        text = step_descriptions.get(step_name, step_name)

        # Add header bar
        h, w = img.shape[:2]
        header_h = 40
        new_img = np.zeros((h + header_h, w, 3), dtype=np.uint8)
        new_img[:header_h, :] = (40, 40, 40)  # Dark gray header
        new_img[header_h:, :] = img

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(new_img, text, (10, 28), font, 0.7, (255, 255, 255), 2)

        return new_img

    def create_step_summary(self, frame_id):
        """
        Create a summary image showing all 3 fitting steps side by side.
        Reads existing step images and combines them into a single comparison.
        """
        step_files = [
            (f"{self.result_folder}/render/debug/step_0_frame_{frame_id:06d}_iter_00000.png", "Step0"),
            (f"{self.result_folder}/render/step_1_frame_{frame_id:06d}.png", "Step1"),
            (f"{self.result_folder}/render/step_2_frame_{frame_id:06d}.png", "Step2"),
        ]

        images = []
        for filepath, step_name in step_files:
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    images.append((img, step_name))

        if len(images) < 2:
            return None

        # Get max dimensions
        max_h = max(img.shape[0] for img, _ in images)
        max_w = max(img.shape[1] for img, _ in images)

        # Create combined image with step labels
        header_h = 50
        gap = 10
        total_w = sum(img.shape[1] for img, _ in images) + gap * (len(images) - 1)
        combined = np.zeros((max_h + header_h, total_w, 3), dtype=np.uint8)
        combined[:header_h, :] = (30, 30, 30)

        step_descriptions = {
            'Step0': 'Step 0: Global Positioning\n(trans, rotation, scale)',
            'Step1': 'Step 1: Articulation\n(joint angles, bones)',
            'Step2': 'Step 2: Silhouette\n(mask refinement)',
        }

        x_offset = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for img, step_name in images:
            h, w = img.shape[:2]
            # Place image
            combined[header_h:header_h+h, x_offset:x_offset+w] = img

            # Add step label
            desc = step_descriptions.get(step_name, step_name)
            lines = desc.split('\n')
            for i, line in enumerate(lines):
                cv2.putText(combined, line, (x_offset + 10, 20 + i*18),
                           font, 0.5, (255, 255, 255), 1)

            x_offset += w + gap

        # Save summary
        summary_path = f"{self.result_folder}/render/step_summary_frame_{frame_id:06d}.png"
        cv2.imwrite(summary_path, combined)
        return summary_path

    # GT Keypoint labels (22 keypoints) - Based on mouse_22_defs.py
    # NOTE: GT annotation uses different order for head keypoints!
    # GT: 0=L_ear, 1=R_ear, 2=nose (from mouse_22_defs.py)
    # Model: 0=nose, 1=L_ear, 2=R_ear (from keypoint22_mapper.json)
    GT_KEYPOINT_LABELS = {
        0: 'L_ear', 1: 'R_ear', 2: 'nose',  # GT order!
        3: 'neck', 4: 'body',
        5: 'tail_root', 6: 'tail_mid', 7: 'tail_end',
        8: 'L_paw', 9: 'L_paw_end', 10: 'L_elbow', 11: 'L_shoulder',
        12: 'R_paw', 13: 'R_paw_end', 14: 'R_elbow', 15: 'R_shoulder',
        16: 'L_foot', 17: 'L_knee', 18: 'L_hip',
        19: 'R_foot', 20: 'R_knee', 21: 'R_hip'
    }

    # Model keypoint labels (for predicted keypoints)
    MODEL_KEYPOINT_LABELS = {
        0: 'nose', 1: 'L_ear', 2: 'R_ear',  # Model order!
        3: 'neck', 4: 'body',
        5: 'tail_base', 6: 'tail_mid', 7: 'tail_tip',
        8: 'L_fp_dig', 9: 'L_fpaw', 10: 'L_ulna', 11: 'L_humer',
        12: 'R_fp_dig', 13: 'R_fpaw', 14: 'R_ulna', 15: 'R_humer',
        16: 'L_hp_dig', 17: 'L_hpaw', 18: 'L_tibia',
        19: 'R_hp_dig', 20: 'R_hpaw', 21: 'R_tibia'
    }

    # Default to GT labels for visualization
    KEYPOINT_LABELS = GT_KEYPOINT_LABELS

    # Part-aware color scheme for keypoints (BGR format for OpenCV)
    # Consistent color coding across all visualizations
    KEYPOINT_COLORS = {
        'head': (0, 255, 255),       # Yellow - indices 0, 1, 2
        'body': (255, 0, 255),       # Magenta - indices 3, 4
        'tail': (0, 165, 255),       # Orange - indices 5, 6, 7
        'L_front': (255, 0, 0),      # Blue - indices 8, 9, 10, 11
        'R_front': (0, 255, 0),      # Green - indices 12, 13, 14, 15
        'L_hind': (255, 255, 0),     # Cyan - indices 16, 17, 18
        'R_hind': (0, 0, 255),       # Red - indices 19, 20, 21
    }

    # Index to body part mapping
    KEYPOINT_PART_MAP = {
        0: 'head', 1: 'head', 2: 'head',
        3: 'body', 4: 'body',
        5: 'tail', 6: 'tail', 7: 'tail',
        8: 'L_front', 9: 'L_front', 10: 'L_front', 11: 'L_front',
        12: 'R_front', 13: 'R_front', 14: 'R_front', 15: 'R_front',
        16: 'L_hind', 17: 'L_hind', 18: 'L_hind',
        19: 'R_hind', 20: 'R_hind', 21: 'R_hind',
    }

    def get_keypoint_color(self, idx):
        """Get color for keypoint index based on body part."""
        part = self.KEYPOINT_PART_MAP.get(idx, 'body')
        return self.KEYPOINT_COLORS.get(part, (255, 255, 255))

    def draw_keypoints_compare(self, result, imgs, batch_id, filename, cams_dict, target_2d=None):
        """
        Draw keypoints comparison visualization for all available views.
        In sparse mode, only draws the sparse keypoint indices.

        Saves three images:
        1. {filename} - Predicted keypoints only (backward compatible)
        2. {filename}_gt.png - GT keypoints only
        3. {filename}_compare.png - Side-by-side GT vs Predicted with legend and labels

        Args:
            result: body model parameters
            imgs: list of images (indexed by position in views_to_use)
            batch_id: batch index
            filename: output filename
            cams_dict: list of camera dicts (indexed by position in views_to_use)
            target_2d: GT 2D keypoints tensor [batch, views, keypoints, 3] (optional)
        """
        myimages = imgs.copy()
        V,J = self.bodymodel.forward(result["thetas"], result["bone_lengths"],
            result["rotation"], result["trans"] / 1000, result["scale"] / 1000, result["chest_deformer"])
        keypoints = self.bodymodel.forward_keypoints22()
        joints = keypoints[batch_id].detach().cpu().numpy()

        # Get sparse indices if in sparse mode
        sparse_indices = getattr(self.cfg.fitter, 'sparse_keypoint_indices', None)
        if sparse_indices:
            sparse_indices = list(sparse_indices)  # Ensure it's a list

        pred_images = []
        gt_images = []
        compare_images = []
        num_views = len(imgs)

        for view_idx in range(num_views):
            cam = cams_dict[view_idx]
            T = cam["T"] / 1000
            if len(T.shape) > 1:
                T = T.squeeze()

            # Predicted keypoints (project 3D to 2D)
            data2d_pred = (joints@cam['R'] + T)@cam["K"]
            data2d_pred = data2d_pred[:,0:2] / data2d_pred[:,2:]

            # GT keypoints (from target_2d if available)
            data2d_gt = None
            gt_conf = None
            if target_2d is not None:
                gt_data = target_2d[batch_id, view_idx].detach().cpu().numpy()  # [22, 3]
                data2d_gt = gt_data[:, :2]  # [22, 2] - xy only
                gt_conf = gt_data[:, 2]  # confidence

            # Determine which indices to draw
            if sparse_indices:
                indices_to_draw = sparse_indices
            else:
                indices_to_draw = list(range(22))

            # Draw predicted keypoints (green circles with labels)
            img_pred = myimages[view_idx].copy()
            img_pred = self._draw_keypoints_with_labels(
                img_pred, data2d_pred, indices_to_draw,
                color=(0, 255, 0), radius=7, show_labels=True
            )
            pred_images.append(img_pred)

            # Draw GT keypoints (red circles with labels) if available
            if data2d_gt is not None:
                img_gt = myimages[view_idx].copy()
                # Only draw GT points that have confidence > 0.25
                valid_gt_indices = [idx for idx in indices_to_draw if gt_conf[idx] > 0.25]
                img_gt = self._draw_keypoints_with_labels(
                    img_gt, data2d_gt, valid_gt_indices,
                    color=(0, 0, 255), radius=7, show_labels=True,
                    confidence=gt_conf  # Include confidence values
                )
                gt_images.append(img_gt)

                # Draw comparison (GT=red, Pred=green on same image) with legend
                img_compare = myimages[view_idx].copy()
                # Draw GT first (red, larger) with confidence
                img_compare = self._draw_keypoints_with_labels(
                    img_compare, data2d_gt, valid_gt_indices,
                    color=(0, 0, 255), radius=9, show_labels=True, label_prefix="GT:",
                    confidence=gt_conf  # Include confidence values
                )
                # Draw Pred on top (green, smaller)
                img_compare = self._draw_keypoints_with_labels(
                    img_compare, data2d_pred, indices_to_draw,
                    color=(0, 255, 0), radius=5, show_labels=False
                )
                # Add legend to comparison image
                img_compare = self._add_legend(img_compare, sparse_indices)
                compare_images.append(img_compare)

        # Save predicted keypoints image (backward compatible)
        outputimg = pack_images(pred_images)
        cv2.imwrite(filename, outputimg)

        # Save GT and comparison images if GT available
        if gt_images:
            base_name = filename.rsplit('.', 1)[0]

            # GT only
            gt_outputimg = pack_images(gt_images)
            cv2.imwrite(f"{base_name}_gt.png", gt_outputimg)

            # Comparison (GT red + Pred green)
            compare_outputimg = pack_images(compare_images)
            cv2.imwrite(f"{base_name}_compare.png", compare_outputimg)

    def _draw_keypoints_with_labels(self, img, proj, indices, color=(0, 255, 0), radius=9,
                                      show_labels=True, label_prefix="", confidence=None):
        """
        Draw keypoints with index:label text annotations and optional confidence values.

        Args:
            img: Image to draw on
            proj: Projected 2D keypoints [N, 2]
            indices: List of keypoint indices to draw
            color: BGR color tuple
            radius: Circle radius
            show_labels: Whether to show text labels
            label_prefix: Prefix for labels (e.g., "GT:")
            confidence: Optional confidence array [N] for each keypoint
        """
        for idx in indices:
            if idx >= proj.shape[0]:
                continue
            x, y = proj[idx, 0], proj[idx, 1]
            if math.isnan(x) or math.isnan(y) or (x == 0 and y == 0):
                continue

            p = (int(x), int(y))
            cv2.circle(img, p, radius, color, -1)

            if show_labels:
                label = self.KEYPOINT_LABELS.get(idx, str(idx))
                # Include confidence if available
                if confidence is not None and idx < len(confidence):
                    conf_val = confidence[idx]
                    text = f"{label_prefix}{idx}:{label}({conf_val:.2f})"
                else:
                    text = f"{label_prefix}{idx}:{label}"

                # Draw text with background for visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Position text above the point
                text_x = int(x) + 10
                text_y = int(y) - 10

                # Draw background rectangle
                cv2.rectangle(img,
                    (text_x - 2, text_y - text_h - 2),
                    (text_x + text_w + 2, text_y + 2),
                    (0, 0, 0), -1)

                # Draw text
                cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

        return img

    def _add_legend(self, img, sparse_indices=None):
        """
        Add a legend to the image explaining colors and sparse keypoints.

        Args:
            img: Image to add legend to
            sparse_indices: List of sparse keypoint indices (or None for all 22)
        """
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Legend box position (top-left)
        legend_x = 10
        legend_y = 20
        line_height = 20

        # Draw semi-transparent background
        legend_h = 80 if sparse_indices else 60
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (220, legend_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Draw legend items
        # Red = GT
        cv2.circle(img, (legend_x, legend_y), 6, (0, 0, 255), -1)
        cv2.putText(img, "RED = Ground Truth (GT)", (legend_x + 15, legend_y + 5),
                    font, font_scale, (255, 255, 255), thickness)

        # Green = Predicted
        legend_y += line_height
        cv2.circle(img, (legend_x, legend_y), 6, (0, 255, 0), -1)
        cv2.putText(img, "GREEN = Predicted", (legend_x + 15, legend_y + 5),
                    font, font_scale, (255, 255, 255), thickness)

        # Sparse mode info
        if sparse_indices:
            legend_y += line_height
            sparse_names = [f"{i}:{self.KEYPOINT_LABELS.get(i, '?')}" for i in sparse_indices]
            sparse_text = f"Sparse: {', '.join(sparse_names)}"
            cv2.putText(img, sparse_text, (legend_x, legend_y + 5),
                        font, font_scale, (255, 255, 0), thickness)

        return img

    def _draw_keypoints_color(self, img, proj, color=(0, 255, 0), radius=9):
        """Draw keypoints with specified color (legacy method for backward compatibility)."""
        for k in range(proj.shape[0]):
            if math.isnan(proj[k,0]) or (proj[k,0] == 0 and proj[k,1] == 0):
                continue
            p = (int(proj[k,0]), int(proj[k,1]))
            cv2.circle(img, p, radius, color, -1)
        return img

def preprocess_cli_args():
    """
    Convert argparse-style arguments to Hydra overrides for CLI consistency.

    Supports:
        --keypoints none  → fitter.use_keypoints=false
        --input_dir /path → data.data_dir=/path
        --output_dir /path → result_folder=/path
        --start_frame N   → fitter.start_frame=N
        --end_frame N     → fitter.end_frame=N
        --with_render     → fitter.with_render=true
    """
    import sys

    # Mapping from argparse-style to Hydra overrides
    arg_mapping = {
        '--keypoints': lambda v: 'fitter.use_keypoints=false' if v == 'none' else None,
        '--input_dir': lambda v: f'data.data_dir={v}',
        '--output_dir': lambda v: f'result_folder={v}',
        '--start_frame': lambda v: f'fitter.start_frame={v}',
        '--end_frame': lambda v: f'fitter.end_frame={v}',
        '--with_render': lambda v: 'fitter.with_render=true',
    }

    new_argv = [sys.argv[0]]
    i = 1
    converted = []

    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg in arg_mapping:
            # Get value if needed
            if arg == '--with_render':
                override = arg_mapping[arg](None)
                new_argv.append(override)
                converted.append(f"{arg} → {override}")
                i += 1
            elif i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                value = sys.argv[i + 1]
                override = arg_mapping[arg](value)
                if override:
                    new_argv.append(override)
                    converted.append(f"{arg} {value} → {override}")
                i += 2
            else:
                i += 1
        else:
            # Keep Hydra-style arguments as-is
            new_argv.append(arg)
            i += 1

    if converted:
        print("CLI arguments converted to Hydra format:")
        for c in converted:
            print(f"  {c}")

    sys.argv = new_argv


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def optim_single(cfg: DictConfig):
    data_loader = DataSeakerDet(cfg)
    device = torch.device('cuda')

    fitter = MouseFitter(cfg)
    fitter.set_cameras_dannce(data_loader.cams_dict_out)
    camN = len(data_loader.cams_dict_out)
    
    # Generate dynamic result folder name with dataset, views, keypoint info, and timestamp
    import datetime
    from omegaconf import OmegaConf

    dataset_name = os.path.basename(cfg.data.data_dir.rstrip('/'))  # Extract dataset name from path
    views_str = f"v{''.join(map(str, cfg.data.views_to_use))}"  # e.g., "v012345" or "v024"

    # Keypoint info string
    use_kp = getattr(cfg.fitter, 'use_keypoints', True)
    sparse_indices = getattr(cfg.fitter, 'sparse_keypoint_indices', None)
    if not use_kp:
        kp_str = "noKP"  # No keypoints (silhouette only)
    elif sparse_indices:
        kp_str = f"sparse{len(sparse_indices)}"  # e.g., "sparse3"
    else:
        kp_str = "kp22"  # Full 22 keypoints

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dynamic_result_folder = f"results/fitting/{dataset_name}_{views_str}_{kp_str}_{timestamp}"

    fitter.result_folder = hydra.utils.to_absolute_path(dynamic_result_folder)
    print(f"Results will be saved to: {fitter.result_folder}")
    os.makedirs(fitter.result_folder, exist_ok=True)
    subfolders = ["params", "render", "render/debug/", "render/keypoints/", "obj"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(fitter.result_folder, subfolder), exist_ok=True)

    # Save config to result folder for reproducibility
    config_path = os.path.join(fitter.result_folder, "config.yaml")
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Config saved to: {config_path}")

    print("camN: ", camN)
    targets = np.arange(cfg.fitter.start_frame, cfg.fitter.end_frame, cfg.fitter.interval).tolist()

    start = targets[0]
    total_frames = len(targets)
    start_time = time()

    # Progress bar with ETA
    pbar = tqdm(targets, desc="Fitting frames", unit="frame",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for index in pbar:
        pbar.set_postfix(frame=index)
        labels = data_loader.fetch(index, with_img = cfg.fitter.with_render)
        fitter.id = index
        fitter.imgs = labels["imgs"]

        # Build target dict with masks first (needed for mask-based init)
        target = {
            'target_2d': torch.FloatTensor(labels["label2d"]).reshape([-1,camN,cfg.fitter.keypoint_num,3]).to(device)
        }
        for viewid in range(camN):
            target.update({
                "mask{}".format(viewid): torch.from_numpy(np.expand_dims(labels["bgs"][viewid],axis=0)).to(device)
            })

        if index == start:
            if cfg.fitter.resume:
                with open(os.path.join(fitter.result_folder, "params/step_2_frame_{:06d}.pkl".format(index-1)), 'rb') as f:
                    init_params = pickle.load(f)
                fitter.set_previous_frame(init_params)
                params = init_params
            else:
                # Use mask-based initialization when keypoints are disabled
                if not getattr(cfg.fitter, 'use_keypoints', True):
                    print("Using mask-based initialization (silhouette-only mode)")
                    init_params = fitter.init_params_from_masks(target, data_loader.cams_dict_out, batch_size=1)
                else:
                    init_params = fitter.init_params(1)
                params = {k: torch.tensor(v, dtype=torch.float32, device=device).requires_grad_(True) for k, v in init_params.items()}

        # Apply iteration multiplier for silhouette-only mode
        iter_mult = getattr(fitter, 'silhouette_iter_multiplier', 1.0) if not getattr(cfg.fitter, 'use_keypoints', True) else 1.0
        step0_iters = int(cfg.optim.solve_step0_iters * iter_mult)
        step1_iters = int(cfg.optim.solve_step1_iters * iter_mult)
        step2_iters = int(cfg.optim.solve_step2_iters * iter_mult)

        if index == start:
            params = fitter.solve_step0(params=params, target=target, max_iters=step0_iters, pbar=pbar)
            params = fitter.solve_step1(params=params, target=target, max_iters=step1_iters, pbar=pbar)
            params = fitter.solve_step2(params=params, target=target, max_iters=step2_iters, pbar=pbar)
            # Create step-by-step summary for first frame (has all 3 steps)
            if cfg.fitter.with_render:
                fitter.create_step_summary(index)
        else:
            params = fitter.solve_step1(params=params, target=target, max_iters=step1_iters, pbar=pbar)
            params = fitter.solve_step2(params=params, target=target, max_iters=step2_iters, pbar=pbar)
        fitter.set_previous_frame(params)

    # Print total elapsed time
    elapsed = time() - start_time
    print(f"\n{'='*50}")
    print(f"Fitting complete: {total_frames} frames in {elapsed:.1f}s ({elapsed/total_frames:.2f}s/frame)")
    print(f"Results saved to: {fitter.result_folder}")
    print(f"{'='*50}")

    # Save loss history and generate plots
    fitter.save_loss_history(os.path.join(fitter.result_folder, 'loss_history.json'))
    fitter.plot_loss_history(fitter.result_folder) 

if __name__ == "__main__":
    # Convert argparse-style args to Hydra format for CLI consistency
    preprocess_cli_args()
    optim_single()
