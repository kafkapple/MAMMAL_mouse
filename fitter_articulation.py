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

    def solve_step0(self, params, target, max_iters):
        ## step1: optimize skeleton
        tolerate = 1e-2
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        # optimizer = torch.optim.Adam(params.values(), lr=0.001)
        closure = self.gen_closure(
                    optimizer, params, target
                )

        loss_prev = float('inf')
        self.keypoint_weight[:,16:19,:] = 1
        self.keypoint_weight[:,19:22,:] = 1
        # Set mask weight from config (silhouette-only mode uses higher weight)
        if not getattr(self.cfg.fitter, 'use_keypoints', True):
            self.term_weights["mask"] = max(self.mask_weight_step0, 1000.0)  # Use mask for guidance when no keypoints
        else:
            self.term_weights["mask"] = self.mask_weight_step0
        self.term_weights["stretch"] = 0
        params["chest_deformer"].requires_grad_(False)
        params["thetas"].requires_grad_(False)
        params["bone_lengths"].requires_grad_(False)
        for i in range(max_iters):
            loss = optimizer.step(closure).item()
            print(self.losses)
            if abs(loss-loss_prev) < tolerate:
                break
            else:
                print('iter ' + str(i) + ': ' + '%.2f'%loss + "  diff: " + "%.2f"%(loss-loss_prev))
            loss_prev = loss
            if self.cfg.fitter.with_render:
                imgs = self.imgs.copy()
                self.render(params, imgs, self.cfg.fitter.render_cameras, 0, self.result_folder + "/render/debug/fitting_{}_global_iter_{:05d}.png".format(self.id, i), self.cam_dict)
        params["chest_deformer"].requires_grad_(True)
        params["thetas"].requires_grad_(True)
        params["bone_lengths"].requires_grad_(True)
        return params

    def solve_step1(self, params, target, max_iters):
        ## step1: optimize skeleton
        tolerate = 1e-4
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        # optimizer = torch.optim.Adam(params.values(), lr=0.001)
        closure = self.gen_closure(
                    optimizer, params, target
                )

        loss_prev = float('inf')
        self.keypoint_weight[:,16:19,:] = 1
        self.keypoint_weight[:,19:22,:] = 1
        # Set mask weight from config (silhouette-only mode uses higher weight)
        if not getattr(self.cfg.fitter, 'use_keypoints', True):
            self.term_weights["mask"] = max(self.mask_weight_step1, 1500.0)  # Use mask for guidance when no keypoints
        else:
            self.term_weights["mask"] = self.mask_weight_step1
        self.term_weights["stretch"] = 0
        params["chest_deformer"].requires_grad_(False)
        for i in range(max_iters):
            loss = optimizer.step(closure).item()
            print(self.losses) 
            if abs(loss-loss_prev) < tolerate: 
                break 
            else: 
                print('iter ' + str(i) + ': ' + '%.2f'%loss + "  diff: " + "%.2f"%(loss-loss_prev))

            loss_prev = loss 
            if self.id == 0 and self.cfg.fitter.with_render: 
                imgs = self.imgs.copy() 
                self.render(params, imgs, self.cfg.fitter.render_cameras, 0, self.result_folder + "/render/debug/fitting_{}_debug_iter_{:05d}.png".format(self.id, i), self.cam_dict)
                

        if self.cfg.fitter.with_render:
            imgs = self.imgs.copy()
            self.render(params, imgs, self.cfg.fitter.render_cameras, 0, self.result_folder + "/render/fitting_{}.png".format(self.id), self.cam_dict)
            # Only draw keypoints visualization if keypoints are enabled
            if getattr(self.cfg.fitter, 'use_keypoints', True):
                self.draw_keypoints_compare(params, imgs, self.cfg.fitter.render_cameras, 0, self.result_folder + "/fitting_keypoints_{}.png".format(self.id), self.cam_dict)
        with open(self.result_folder + "/params/param{}.pkl".format(self.id), 'wb') as f:
            pickle.dump(params,f)
        params["chest_deformer"].requires_grad_(True)
        return params

    def solve_step2(self, params, target, max_iters): 
        ## step2: optim with mask 
        ## enlarge foot keypoint weight because mask on foot are bad. 
        tolerate = 1e-4
        optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
        # optimizer = torch.optim.Adam(params.values(), lr=0.001)
        closure = self.gen_closure(
                    optimizer, params, target
                )

        # Apply step2-specific weights from config
        self.keypoint_weight[:,16:19,:] = self.tail_weight_step2
        self.keypoint_weight[:,19:22,:] = self.tail_weight_step2
        self.term_weights["mask"] = self.mask_weight_step2
        self.term_weights["stretch"] = 0
        loss_prev = float('inf')
        optimizer.zero_grad() 
        for i in range(max_iters): 
            loss = optimizer.step(closure).item()  
            print(self.losses) 
            if abs(loss-loss_prev) < tolerate: 
                break 
            else: 
                print('iter ' + str(i) + ': ' + '%.2f'%loss + "  diff: " + "%.2f"%(loss-loss_prev))

            loss_prev = loss 
        if self.cfg.fitter.with_render: 
            imgs = self.imgs.copy() 
            self.render(params, imgs, self.cfg.fitter.render_cameras, 0, self.result_folder+"/render/fitting_{}_sil.png".format(self.id), self.cam_dict)
            # self.draw_keypoints_compare(params, imgs, self.cfg.fitter.render_cameras, 0, self.result_folder + "/fitting_keypoints_{}_sil.png".format(self.id), self.cam_dict)
        with open(self.result_folder + "/params/param{}_sil.pkl".format(self.id), 'wb') as f: 
            pickle.dump(params,f) 
        self.result = params

        # Save the final mesh as an .obj file
        V, _ = self.bodymodel.forward(
            params["thetas"], params["bone_lengths"],
            params["rotation"], params["trans"], params["scale"], params["chest_deformer"]
        )
        vertices = V[0].detach().cpu().numpy()
        faces = self.bodymodel.faces_vert_np
        obj_filename = os.path.join(self.result_folder, "obj/mesh_{:06d}.obj".format(self.id))
        with open(obj_filename, 'w') as fp:
            for v in vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

        return params 
    def render(self, result, imgs, views, batch_id, filename, cams_dict):
        """
        Render mesh overlay on images.

        Args:
            result: body model parameters
            imgs: list of images (indexed by position in views_to_use, NOT original camera ID)
            views: list of view indices to render (these are positions, not original camera IDs)
            batch_id: batch index
            filename: output filename
            cams_dict: list of camera dicts (indexed by position in views_to_use)
        """
        V,J = self.bodymodel.forward(result["thetas"], result["bone_lengths"],
            result["rotation"], result["trans"] / 1000, result["scale"] / 1000, result["chest_deformer"])
        vertices = V[batch_id].detach().cpu().numpy()
        faces = self.bodymodel.faces_vert_np
        scene = pyrender.Scene()
        light_node = scene.add(pyrender.PointLight(color=np.ones(3), intensity=0.2))
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=np.array([0.8, 0.6, 0.4]))))
        color_maps = []
        # views are now position indices (0, 1, 2, ...) matching imgs and cams_dict order
        for view_idx in range(len(views)):
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
            img_i = imgs[view_idx]  # Use position index
            color = copy.deepcopy(color)

            # Resize rendered image to match input image size
            if color.shape[:2] != img_i.shape[:2]:
                color = cv2.resize(color, (img_i.shape[1], img_i.shape[0]))

            background_mask = color[:, :, :] == 255
            color[background_mask] = img_i[background_mask]
            color_maps.append(color)
        output = pack_images(color_maps)
        if filename is not None:
            cv2.imwrite(filename, output)
        return output
        

    def draw_keypoints_compare(self, result, imgs, views, batch_id, filename, cams_dict):
        """
        Draw keypoints comparison visualization.

        Args:
            result: body model parameters
            imgs: list of images (indexed by position in views_to_use)
            views: list of view indices (positions, not original camera IDs)
            batch_id: batch index
            filename: output filename
            cams_dict: list of camera dicts (indexed by position in views_to_use)
        """
        myimages = imgs.copy()
        V,J = self.bodymodel.forward(result["thetas"], result["bone_lengths"],
            result["rotation"], result["trans"] / 1000, result["scale"] / 1000, result["chest_deformer"])
        keypoints = self.bodymodel.forward_keypoints22()
        joints = keypoints[batch_id].detach().cpu().numpy()
        all_drawn_images = []
        # Use position indices to access imgs and cams_dict
        for view_idx in range(len(views)):
            cam = cams_dict[view_idx]
            # Fix T shape for numpy broadcasting
            T = cam["T"] / 1000
            if len(T.shape) > 1:
                T = T.squeeze()
            data2d = (joints@cam['R'] + T)@cam["K"]
            data2d = data2d[:,0:2] / data2d[:,2:]
            img_drawn = draw_keypoints(myimages[view_idx], data2d, bones, is_draw_bone=True)
            all_drawn_images.append(img_drawn)
        outputimg = pack_images(all_drawn_images)
        cv2.imwrite(filename, outputimg)

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
    subfolders = ["params", "render", "render/debug/", "obj"]
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
    for index in targets:
        print("process ... ", index)
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
                with open(os.path.join(fitter.result_folder, "params/param{}_sil.pkl".format(index-1)), 'rb') as f:
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
            params = fitter.solve_step0(params = params, target=target, max_iters = step0_iters)
            params = fitter.solve_step1(params = params, target=target, max_iters = step1_iters)
            params = fitter.solve_step2(params = params, target=target, max_iters = step2_iters)
        else:
            params = fitter.solve_step1(params = params, target=target, max_iters = step1_iters)
            params = fitter.solve_step2(params = params, target=target, max_iters = step2_iters)
        fitter.set_previous_frame(params) 

if __name__ == "__main__":
    # Convert argparse-style args to Hydra format for CLI consistency
    preprocess_cli_args()
    optim_single()
