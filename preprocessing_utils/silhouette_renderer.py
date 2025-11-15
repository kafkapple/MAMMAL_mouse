"""
Silhouette Renderer using PyTorch3D

Renders mouse mesh as binary silhouette for direct mask-based fitting
"""
import torch
import torch.nn.functional as F
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
    PerspectiveCameras,
)
from pytorch3d.structures import Meshes
import numpy as np


class SilhouetteRenderer:
    """
    Differentiable silhouette renderer for mesh fitting
    """

    def __init__(self, image_size=(480, 640), device='cuda'):
        """
        Args:
            image_size: (H, W) tuple
            device: 'cuda' or 'cpu'
        """
        self.image_size = image_size
        self.device = torch.device(device)

        # Rasterization settings for silhouette
        # Smaller blur_radius = sharper silhouette
        # Larger faces_per_pixel = better quality
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-5,  # Soft rasterization
            faces_per_pixel=50,
            perspective_correct=True,
        )

        # Soft silhouette shader
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=self.raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

    def render(self, meshes, cameras):
        """
        Render mesh as silhouette

        Args:
            meshes: PyTorch3D Meshes object
            cameras: PyTorch3D Cameras object

        Returns:
            silhouette: (B, H, W) tensor of alpha values in [0, 1]
        """
        # Render RGBA
        images = self.renderer(meshes, cameras=cameras)

        # Extract alpha channel (silhouette)
        silhouette = images[..., 3]

        return silhouette

    def render_from_vertices_faces(self, vertices, faces, cameras):
        """
        Convenience method to render from vertices and faces

        Args:
            vertices: (B, V, 3) tensor
            faces: (F, 3) tensor
            cameras: PyTorch3D cameras

        Returns:
            silhouette: (B, H, W) tensor
        """
        # Ensure tensors are on correct device
        vertices = vertices.to(self.device)
        faces = faces.to(self.device)

        # Create mesh
        meshes = Meshes(verts=vertices, faces=faces.unsqueeze(0).expand(vertices.shape[0], -1, -1))

        return self.render(meshes, cameras)


class SilhouetteLoss:
    """
    Loss functions for silhouette-based fitting
    """

    @staticmethod
    def iou_loss(pred_silhouette, target_mask):
        """
        IoU (Intersection over Union) loss

        Args:
            pred_silhouette: (B, H, W) predicted silhouette in [0, 1]
            target_mask: (B, H, W) target binary mask in [0, 1]

        Returns:
            loss: scalar, 1 - IoU
        """
        # Flatten
        pred = pred_silhouette.view(pred_silhouette.shape[0], -1)
        target = target_mask.view(target_mask.shape[0], -1)

        # Intersection and union
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - intersection

        # IoU
        iou = intersection / (union + 1e-6)

        # Loss is 1 - IoU
        return (1.0 - iou).mean()

    @staticmethod
    def bce_loss(pred_silhouette, target_mask):
        """
        Binary Cross Entropy loss

        Args:
            pred_silhouette: (B, H, W) predicted silhouette in [0, 1]
            target_mask: (B, H, W) target binary mask in [0, 1]

        Returns:
            loss: scalar
        """
        return F.binary_cross_entropy(pred_silhouette, target_mask, reduction='mean')

    @staticmethod
    def combined_loss(pred_silhouette, target_mask, iou_weight=0.5, bce_weight=0.5):
        """
        Combination of IoU and BCE losses

        Args:
            pred_silhouette: (B, H, W) predicted silhouette
            target_mask: (B, H, W) target mask
            iou_weight: weight for IoU loss
            bce_weight: weight for BCE loss

        Returns:
            loss: scalar
        """
        iou = SilhouetteLoss.iou_loss(pred_silhouette, target_mask)
        bce = SilhouetteLoss.bce_loss(pred_silhouette, target_mask)

        return iou_weight * iou + bce_weight * bce

    @staticmethod
    def dice_loss(pred_silhouette, target_mask):
        """
        Dice coefficient loss (similar to IoU but smoother gradients)

        Args:
            pred_silhouette: (B, H, W) predicted silhouette
            target_mask: (B, H, W) target mask

        Returns:
            loss: scalar, 1 - Dice
        """
        pred = pred_silhouette.view(pred_silhouette.shape[0], -1)
        target = target_mask.view(target_mask.shape[0], -1)

        intersection = (pred * target).sum(dim=1)
        dice = (2.0 * intersection) / (pred.sum(dim=1) + target.sum(dim=1) + 1e-6)

        return (1.0 - dice).mean()


def load_target_mask(mask_path, frame_idx, device='cuda'):
    """
    Load target mask from video

    Args:
        mask_path: path to mask video (e.g., simpleclick_undist/0.mp4)
        frame_idx: frame index to load
        device: torch device

    Returns:
        mask: (1, H, W) tensor in [0, 1]
    """
    import cv2

    cap = cv2.VideoCapture(mask_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, mask = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from {mask_path}")

    # Convert to grayscale if needed
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Normalize to [0, 1]
    mask = mask.astype(np.float32) / 255.0

    # IMPORTANT: Invert mask (SAM preprocessing saved inverted masks)
    # Original: white=arena, black=mouse
    # After inversion: white=mouse, black=arena
    mask = 1.0 - mask

    # To tensor
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    return mask


def visualize_silhouette_comparison(pred_sil, target_mask, save_path=None):
    """
    Visualize predicted silhouette vs target mask

    Args:
        pred_sil: (H, W) predicted silhouette
        target_mask: (H, W) target mask
        save_path: optional path to save visualization

    Returns:
        vis_image: (H, W, 3) visualization
    """
    import cv2

    # Convert to numpy
    if torch.is_tensor(pred_sil):
        pred_sil = pred_sil.detach().cpu().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.detach().cpu().numpy()

    # Create RGB visualization
    H, W = pred_sil.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)

    # Target in green
    vis[:, :, 1] = (target_mask * 255).astype(np.uint8)

    # Prediction in red
    vis[:, :, 2] = (pred_sil * 255).astype(np.uint8)

    # Overlap in yellow (green + red)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis
