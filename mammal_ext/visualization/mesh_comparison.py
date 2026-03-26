# no-split: cohesive module — rendering + comparison + video export tightly coupled
"""
Mesh Comparison Module

Compare mesh fitting quality between two OBJ sets (e.g., fast vs accurate).
Renders silhouettes from dataset cameras, computes IoU with GT masks,
and produces comparison grid images.

Usage:
    from mammal_ext.visualization.mesh_comparison import MeshComparison

    comp = MeshComparison(data_dir="data/raw/markerless_mouse_1_nerf/")
    results = comp.compare(
        obj_dir_a="results/fitting/original/obj/",
        obj_dir_b="results/fitting/refit_accurate_23/obj/",
        frame_ids=[720, 1320],
        view_ids=[3],  # cam_003 for IoU
    )
    comp.save_grid(results, "outputs/refit_comparison/")
"""

import os
import pickle
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ComparisonConfig:
    """Configuration for mesh comparison."""
    image_size: Tuple[int, int] = (1152, 1024)  # (W, H) matching dataset
    iou_threshold: float = 0.7
    grid_cols: int = 4  # columns per row in summary grid
    label_font_scale: float = 1.0
    label_thickness: int = 2
    label_color: Tuple[int, int, int] = (255, 255, 255)
    overlay_alpha: float = 0.5
    silhouette_color_a: Tuple[int, int, int] = (80, 80, 255)   # bright red (BGR) for BEFORE
    silhouette_color_b: Tuple[int, int, int] = (80, 255, 80)   # bright green (BGR) for AFTER
    label_bar_height: int = 40  # height of colored label bar at top
    render_bg_color: Tuple[int, int, int] = (255, 255, 255)  # white background for mesh render


@dataclass
class FrameResult:
    """Result for a single frame comparison."""
    frame_id: int
    iou_a: Dict[int, float] = field(default_factory=dict)  # view_id -> IoU
    iou_b: Dict[int, float] = field(default_factory=dict)
    images: Dict[str, np.ndarray] = field(default_factory=dict)


class MeshComparison:
    """Compare mesh fitting quality between two OBJ sets."""

    def __init__(
        self,
        data_dir: str,
        config: Optional[ComparisonConfig] = None,
        device: str = "cuda",
        model_dir: str = "mouse_model/mouse_txt",
        texture_path: str = "exports/texture_final.png",
    ):
        self.data_dir = Path(data_dir)
        self.config = config or ComparisonConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.texture_path = Path(texture_path)

        # Load cameras
        self._load_cameras()
        # Load body model faces (topology is constant)
        self._load_faces()
        # Load UV template for transplant + textured rendering
        self._load_uv_template()
        # Setup PyTorch3D renderer
        self._setup_renderer()

    def _load_cameras(self):
        """Load camera parameters from dataset."""
        cam_pkl = self.data_dir / "new_cam.pkl"
        with open(cam_pkl, "rb") as f:
            cams_raw = pickle.load(f)
        # cams_raw is a list of dicts (one per camera view)
        # Apply same transforms as DataSeakerDet
        self.cams_dict = {}
        for camid, cam in enumerate(cams_raw):
            self.cams_dict[camid] = {
                "T": np.expand_dims(cam["T"], 0),
                "R": cam["R"].T,
                "K": cam["K"].T,
            }

    def _load_faces(self):
        """Load mesh faces from body model."""
        from articulation_th import ArticulationTorch
        model = ArticulationTorch()
        self.faces = model.faces_vert_np  # (F, 3)

    def _load_uv_template(self):
        """Load UV coordinates and face-texture indices from model template."""
        self.uv_coords = np.loadtxt(self.model_dir / "textures.txt")      # (M, 2)
        self.faces_tex = np.loadtxt(self.model_dir / "faces_tex.txt", dtype=np.int64)  # (F, 3)
        # Load texture image
        if self.texture_path.exists():
            tex_img = cv2.imread(str(self.texture_path))
            self.texture_image = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            self.texture_image = None
            print(f"WARNING: texture not found at {self.texture_path}")

    def _setup_renderer(self):
        """Setup PyTorch3D silhouette renderer."""
        from pytorch3d.renderer import (
            MeshRasterizer, RasterizationSettings, SoftSilhouetteShader, MeshRenderer
        )
        from pytorch3d.utils import cameras_from_opencv_projection

        H, W = self.config.image_size[1], self.config.image_size[0]
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1e-5,
            faces_per_pixel=50,
            bin_size=0,  # Use naive rasterization to avoid bin overflow
        )
        self.sil_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftSilhouetteShader(),
        )

        # Build PyTorch3D cameras for each view
        self.cameras_th = {}
        for camid, cam in self.cams_dict.items():
            R = np.expand_dims(cam["R"].T, 0).astype(np.float32)
            K = np.expand_dims(cam["K"].T, 0).astype(np.float32)
            T = cam["T"].astype(np.float32)
            if T.shape == (3, 1):
                T = T.T
            elif T.shape == (3,):
                T = T.reshape(1, 3)
            img_size = np.array([[H, W]], dtype=np.float32)
            self.cameras_th[camid] = cameras_from_opencv_projection(
                R=torch.from_numpy(R).to(self.device),
                tvec=torch.from_numpy(T).to(self.device),
                camera_matrix=torch.from_numpy(K).to(self.device),
                image_size=torch.from_numpy(img_size).to(self.device),
            )

    def _sample_vertex_colors(self, vertices: np.ndarray) -> np.ndarray:
        """Sample texture colors at each vertex UV coordinate.

        Returns:
            colors: (N, 4) RGBA uint8
        """
        if self.texture_image is None:
            return np.full((len(vertices), 4), 180, dtype=np.uint8)

        n_verts = len(vertices)
        vertex_uvs = np.zeros((n_verts, 2), dtype=np.float32)
        vertex_counts = np.zeros(n_verts, dtype=np.int32)

        for f_idx in range(len(self.faces)):
            for i in range(3):
                v_idx = self.faces[f_idx, i]
                uv_idx = self.faces_tex[f_idx, i]
                vertex_uvs[v_idx] += self.uv_coords[uv_idx]
                vertex_counts[v_idx] += 1

        valid = vertex_counts > 0
        vertex_uvs[valid] /= vertex_counts[valid, None]

        H, W = self.texture_image.shape[:2]
        u = vertex_uvs[:, 0]
        v = 1.0 - vertex_uvs[:, 1]
        px = np.clip((u * W).astype(np.int32), 0, W - 1)
        py = np.clip((v * H).astype(np.int32), 0, H - 1)
        rgb = (self.texture_image[py, px] * 255).astype(np.uint8)
        alpha = np.full((n_verts, 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=1)

    def _render_textured(self, vertices: np.ndarray, view_id: int) -> np.ndarray:
        """Render textured mesh from dataset camera using OpenCV projection.

        Projects mesh faces with vertex colors onto the image plane using
        the exact camera K, R, T from the dataset. This guarantees pixel-accurate
        alignment with GT images (verified via cv2.projectPoints).

        Returns:
            image: (H, W, 3) BGR uint8
        """
        cam = self.cams_dict[view_id]
        # Undo transposes from _load_cameras to get original OpenCV params
        K = cam["K"].T   # original K: [[fx,0,cx],[0,fy,cy],[0,0,1]]
        R = cam["R"].T   # original R: world-to-camera rotation
        T = cam["T"].flatten()  # world-to-camera translation

        W, H = self.config.image_size
        bg_color = getattr(self.config, 'render_bg_color', (255, 255, 255))

        # Project all vertices to 2D
        pts_cam = (R @ vertices.T).T + T  # (N, 3)
        pts_2d_h = (K @ pts_cam.T).T      # (N, 3)
        depths = pts_2d_h[:, 2].copy()
        pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:]  # (N, 2)

        # Get vertex colors from texture
        vert_colors = self._sample_vertex_colors(vertices)[:, :3]  # (N, 3) RGB

        # Rasterize faces using OpenCV fillPoly
        image = np.full((H, W, 3), bg_color, dtype=np.uint8)
        z_buffer = np.full((H, W), np.inf, dtype=np.float64)

        # Sort faces by mean depth (back-to-front painter's algorithm)
        face_depths = depths[self.faces].mean(axis=1)
        face_order = np.argsort(-face_depths)  # farthest first

        for fi in face_order:
            f = self.faces[fi]
            tri_2d = pts_2d[f].astype(np.int32)  # (3, 2)

            # Skip if any vertex is behind camera
            if (depths[f] < 0).any():
                continue
            # Skip if outside image
            if (tri_2d[:, 0] < -W).all() or (tri_2d[:, 0] > 2*W).all():
                continue
            if (tri_2d[:, 1] < -H).all() or (tri_2d[:, 1] > 2*H).all():
                continue

            # Average face color from vertex colors
            face_color = vert_colors[f].mean(axis=0).astype(np.uint8)
            # BGR for OpenCV
            color_bgr = (int(face_color[2]), int(face_color[1]), int(face_color[0]))

            cv2.fillPoly(image, [tri_2d.reshape(-1, 1, 2)], color_bgr)

        return image  # BGR uint8

    def _load_obj_vertices(self, obj_path: str) -> np.ndarray:
        """Load vertex positions from OBJ file."""
        vertices = []
        with open(obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(vertices, dtype=np.float32)

    def _render_silhouette(self, vertices: np.ndarray, view_id: int) -> np.ndarray:
        """Render mesh silhouette from given camera view.

        Returns:
            mask: (H, W) float32 in [0, 1]
        """
        from pytorch3d.structures import Meshes

        verts_th = torch.from_numpy(vertices).unsqueeze(0).float().to(self.device)
        faces_th = torch.from_numpy(self.faces).unsqueeze(0).long().to(self.device)
        mesh = Meshes(verts=verts_th, faces=faces_th)
        sil = self.sil_renderer(mesh, cameras=self.cameras_th[view_id])
        mask = sil[0, ..., -1].detach().cpu().numpy()
        return mask

    def _load_gt_mask(self, frame_id: int, view_id: int) -> np.ndarray:
        """Load GT segmentation mask for a frame/view.

        Returns:
            mask: (H, W) float32 in [0, 1]
        """
        mask_video = self.data_dir / "simpleclick_undist" / f"{view_id}.mp4"
        cap = cv2.VideoCapture(str(mask_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read mask frame {frame_id} from view {view_id}")
        return frame[:, :, 0].astype(np.float32) / 255.0

    def _load_gt_image(self, frame_id: int, view_id: int) -> np.ndarray:
        """Load GT RGB image for a frame/view.

        Returns:
            image: (H, W, 3) uint8
        """
        video = self.data_dir / "videos_undist" / f"{view_id}.mp4"
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read image frame {frame_id} from view {view_id}")
        return frame  # BGR uint8

    @staticmethod
    def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray, threshold: float = 0.5) -> float:
        """Compute IoU between predicted and GT masks."""
        pred_bin = (mask_pred > threshold).astype(np.float32)
        gt_bin = (mask_gt > threshold).astype(np.float32)
        intersection = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum() - intersection
        if union < 1:
            return 0.0
        return float(intersection / union)

    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay colored silhouette on RGB image."""
        out = image.copy()
        mask_bin = (mask > 0.5).astype(np.uint8)
        overlay = np.zeros_like(image)
        overlay[:, :] = color  # BGR
        mask_3ch = np.stack([mask_bin] * 3, axis=-1)
        out = np.where(mask_3ch, cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0), out)
        return out

    def _add_label(
        self,
        image: np.ndarray,
        text: str,
        bar_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Add colored label bar at top of image.

        Args:
            image: Input image
            text: Label text
            bar_color: Background color for label bar (default: dark gray)
        """
        H, W = image.shape[:2]
        bar_h = self.config.label_bar_height
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.config.label_font_scale
        thickness = self.config.label_thickness

        # Create label bar
        bg = bar_color if bar_color else (40, 40, 40)
        bar = np.full((bar_h, W, 3), bg, dtype=np.uint8)

        # Center text in bar
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        org = ((W - tw) // 2, (bar_h + th) // 2)
        cv2.putText(bar, text, org, font, scale, self.config.label_color, thickness, cv2.LINE_AA)

        # Stack bar on top of image
        return np.concatenate([bar, image], axis=0)

    def compare_frame(
        self,
        frame_id: int,
        obj_dir_a: str,
        obj_dir_b: str,
        view_ids: List[int],
        label_a: str = "fast",
        label_b: str = "accurate",
    ) -> FrameResult:
        """Compare two OBJ files for a single frame across specified views.

        Returns:
            FrameResult with IoU scores and comparison images.
        """
        result = FrameResult(frame_id=frame_id)

        obj_name = f"step_2_frame_{frame_id:06d}.obj"
        path_a = os.path.join(obj_dir_a, obj_name)
        path_b = os.path.join(obj_dir_b, obj_name)

        if not os.path.exists(path_a):
            print(f"  WARNING: {path_a} not found, skipping")
            return result
        if not os.path.exists(path_b):
            print(f"  WARNING: {path_b} not found, skipping")
            return result

        verts_a = self._load_obj_vertices(path_a)
        verts_b = self._load_obj_vertices(path_b)

        for vid in view_ids:
            gt_mask = self._load_gt_mask(frame_id, vid)
            gt_img = self._load_gt_image(frame_id, vid)

            sil_a = self._render_silhouette(verts_a, vid)
            sil_b = self._render_silhouette(verts_b, vid)

            iou_a = self.compute_iou(sil_a, gt_mask)
            iou_b = self.compute_iou(sil_b, gt_mask)
            result.iou_a[vid] = iou_a
            result.iou_b[vid] = iou_b

            # Create comparison images with colored overlays
            overlay_a = self._create_overlay(gt_img, sil_a, self.config.silhouette_color_a, self.config.overlay_alpha)
            overlay_b = self._create_overlay(gt_img, sil_b, self.config.silhouette_color_b, self.config.overlay_alpha)

            # Colored label bars matching overlay colors
            col_a = self.config.silhouette_color_a
            col_b = self.config.silhouette_color_b
            bar_a = (col_a[0] // 3, col_a[1] // 3, col_a[2] // 3)  # dimmed
            bar_b = (col_b[0] // 3, col_b[1] // 3, col_b[2] // 3)

            overlay_a = self._add_label(overlay_a, f"BEFORE ({label_a}) IoU={iou_a:.3f}", bar_color=bar_a)
            overlay_b = self._add_label(overlay_b, f"AFTER ({label_b}) IoU={iou_b:.3f}", bar_color=bar_b)

            # GT mask visualization
            gt_vis = (gt_mask * 255).astype(np.uint8)
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
            gt_vis = self._add_label(gt_vis, f"GT mask (view {vid})")

            # Side-by-side: [GT | BEFORE | AFTER]
            row = np.concatenate([gt_vis, overlay_a, overlay_b], axis=1)
            result.images[f"compare_v{vid}"] = row

            # Textured rendering: [GT | BEFORE mesh | AFTER mesh] — separate panels, no overlay
            if self.texture_image is not None:
                tex_a_bgr = self._render_textured(verts_a, vid)  # BGR on black bg
                tex_b_bgr = self._render_textured(verts_b, vid)  # BGR on black bg

                tex_a_labeled = self._add_label(tex_a_bgr, f"BEFORE ({label_a})", bar_color=bar_a)
                tex_b_labeled = self._add_label(tex_b_bgr, f"AFTER ({label_b})", bar_color=bar_b)
                gt_labeled = self._add_label(gt_img, f"GT (view {vid}, frame {frame_id})")

                tex_row = np.concatenate([gt_labeled, tex_a_labeled, tex_b_labeled], axis=1)
                result.images[f"textured_v{vid}"] = tex_row

        # Build 6-view grid if all 6 views requested
        if len(view_ids) == 6 and all(f"compare_v{v}" in result.images for v in view_ids):
            result.images["grid_6view"] = self._build_6view_grid(result, view_ids, label_a, label_b)
            # Also build textured 6-view grid
            if all(f"textured_v{v}" in result.images for v in view_ids):
                result.images["grid_6view_textured"] = self._build_6view_textured_grid(
                    result, view_ids, label_a, label_b)

        return result

    def _build_6view_textured_grid(
        self,
        result: FrameResult,
        view_ids: List[int],
        label_a: str,
        label_b: str,
    ) -> np.ndarray:
        """Build 6-view textured rendering grid: 3 rows (GT/BEFORE/AFTER) x 6 cols."""
        scale = 0.35

        gt_imgs, before_imgs, after_imgs = [], [], []
        for vid in view_ids:
            row_img = result.images[f"textured_v{vid}"]
            W = row_img.shape[1] // 3
            gt_imgs.append(cv2.resize(row_img[:, :W], None, fx=scale, fy=scale))
            before_imgs.append(cv2.resize(row_img[:, W:2*W], None, fx=scale, fy=scale))
            after_imgs.append(cv2.resize(row_img[:, 2*W:], None, fx=scale, fy=scale))

        # 3 sections x 2 rows of 3 views
        W_total = np.concatenate(gt_imgs[:3], axis=1).shape[1]
        header_h = 30
        col_a = self.config.silhouette_color_a
        col_b = self.config.silhouette_color_b

        def make_header(text, color):
            h = np.full((header_h, W_total, 3), color, dtype=np.uint8)
            cv2.putText(h, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            return h

        grid = np.concatenate([
            make_header(f"GT  |  Frame {result.frame_id}", (60, 60, 60)),
            np.concatenate(gt_imgs[:3], axis=1),
            np.concatenate(gt_imgs[3:], axis=1),
            make_header(f"BEFORE ({label_a})", (col_a[0]//3, col_a[1]//3, col_a[2]//3)),
            np.concatenate(before_imgs[:3], axis=1),
            np.concatenate(before_imgs[3:], axis=1),
            make_header(f"AFTER ({label_b})", (col_b[0]//3, col_b[1]//3, col_b[2]//3)),
            np.concatenate(after_imgs[:3], axis=1),
            np.concatenate(after_imgs[3:], axis=1),
        ], axis=0)

        return grid

    def _build_6view_grid(
        self,
        result: FrameResult,
        view_ids: List[int],
        label_a: str,
        label_b: str,
    ) -> np.ndarray:
        """Build a 6-view comparison grid: 2 rows (BEFORE/AFTER) x 6 columns (views).

        Layout:
            Row 0: BEFORE overlays for views 0-5
            Row 1: AFTER overlays for views 0-5

        Returns:
            grid: Combined image
        """
        scale = 0.35  # Scale down for 6 views to fit
        col_a = self.config.silhouette_color_a
        col_b = self.config.silhouette_color_b

        before_imgs = []
        after_imgs = []
        for vid in view_ids:
            key = f"compare_v{vid}"
            row_img = result.images[key]
            # row_img is [GT | BEFORE | AFTER], each with label bar
            # Split into 3 panels
            W = row_img.shape[1] // 3
            before_panel = row_img[:, W:2*W]
            after_panel = row_img[:, 2*W:]
            before_imgs.append(cv2.resize(before_panel, None, fx=scale, fy=scale))
            after_imgs.append(cv2.resize(after_panel, None, fx=scale, fy=scale))

        # 2 rows x 6 cols: top=BEFORE, bottom=AFTER
        # Split into 2 rows of 3
        row_before_1 = np.concatenate(before_imgs[:3], axis=1)
        row_before_2 = np.concatenate(before_imgs[3:], axis=1)
        row_after_1 = np.concatenate(after_imgs[:3], axis=1)
        row_after_2 = np.concatenate(after_imgs[3:], axis=1)

        # Add section headers
        W_total = row_before_1.shape[1]
        mean_iou_a = np.mean([result.iou_a.get(v, 0) for v in view_ids])
        mean_iou_b = np.mean([result.iou_b.get(v, 0) for v in view_ids])

        header_h = 30
        header_before = np.full((header_h, W_total, 3), (col_a[0]//3, col_a[1]//3, col_a[2]//3), dtype=np.uint8)
        header_after = np.full((header_h, W_total, 3), (col_b[0]//3, col_b[1]//3, col_b[2]//3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(header_before, f"BEFORE ({label_a}) mean IoU={mean_iou_a:.3f}  |  Frame {result.frame_id}",
                    (10, 22), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(header_after, f"AFTER ({label_b}) mean IoU={mean_iou_b:.3f}",
                    (10, 22), font, 0.7, (255,255,255), 2, cv2.LINE_AA)

        grid = np.concatenate([
            header_before, row_before_1, row_before_2,
            header_after, row_after_1, row_after_2,
        ], axis=0)

        return grid

    def compare(
        self,
        obj_dir_a: str,
        obj_dir_b: str,
        frame_ids: List[int],
        view_ids: List[int] = [3],
        label_a: str = "fast",
        label_b: str = "accurate",
    ) -> List[FrameResult]:
        """Compare two OBJ sets across multiple frames.

        Args:
            obj_dir_a: Directory with original (fast) OBJ files
            obj_dir_b: Directory with refit (accurate) OBJ files
            frame_ids: MAMMAL frame IDs to compare
            view_ids: Camera view IDs for comparison (default: cam_003)
            label_a: Label for set A
            label_b: Label for set B

        Returns:
            List of FrameResult objects
        """
        results = []
        print(f"Comparing {len(frame_ids)} frames, views {view_ids}")
        for i, fid in enumerate(frame_ids):
            print(f"  [{i+1}/{len(frame_ids)}] Frame {fid}...")
            result = self.compare_frame(fid, obj_dir_a, obj_dir_b, view_ids, label_a, label_b)
            results.append(result)
        return results

    def save_results(
        self,
        results: List[FrameResult],
        output_dir: str,
        view_id: int = 3,
        fps: int = 2,
    ) -> str:
        """Save comparison outputs: summary images + videos + IoU chart.

        Output structure:
            output_dir/
            ├── iou_chart.jpg                  # Bar chart (BEFORE vs AFTER per frame)
            ├── best_frame_6view_textured.jpg   # Best IoU frame, 6-view textured
            ├── worst_frame_6view_textured.jpg  # Worst IoU frame, 6-view textured
            ├── summary_silhouette.jpg          # All frames silhouette grid
            ├── video_silhouette.mp4            # Silhouette sequence
            ├── video_textured_6view.mp4        # 6-view textured sequence
            ├── iou_report.txt
            └── iou_report.json
        """
        import json
        os.makedirs(output_dir, exist_ok=True)

        # --- 1. IoU chart (bar chart, BEFORE vs AFTER) ---
        self._save_iou_chart(results, output_dir, view_id)

        # --- 2. Best/Worst frame 6-view textured (1 image each) ---
        results_with_iou = [(r, r.iou_b.get(view_id, 0)) for r in results if r.iou_b]
        if results_with_iou:
            results_with_iou.sort(key=lambda x: x[1])
            worst_r, worst_iou = results_with_iou[0]
            best_r, best_iou = results_with_iou[-1]

            for r, label, iou in [(worst_r, "worst", worst_iou), (best_r, "best", best_iou)]:
                img = r.images.get("grid_6view_textured")
                if img is not None:
                    path = os.path.join(output_dir, f"{label}_frame_{r.frame_id:06d}_6view.jpg")
                    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    print(f"  {label.upper()} frame {r.frame_id} (IoU={iou:.3f}): {path}")

        # --- 3. Summary silhouette grid ---
        sil_key = f"compare_v{view_id}"
        sil_rows = [r.images[sil_key] for r in results if sil_key in r.images]
        if sil_rows:
            scale = 0.5
            resized = [cv2.resize(r, None, fx=scale, fy=scale) for r in sil_rows]
            cols = self.config.grid_cols
            grid_rows = []
            for i in range(0, len(resized), cols):
                batch = resized[i:i + cols]
                while len(batch) < cols:
                    batch.append(np.zeros_like(batch[0]))
                grid_rows.append(np.concatenate(batch, axis=1))
            grid = np.concatenate(grid_rows, axis=0)
            path = os.path.join(output_dir, "summary_silhouette.jpg")
            cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Summary silhouette: {path}")

        # --- 4. Videos (per-frame sequences, proper multi-frame) ---
        if sil_rows:
            vid_path = os.path.join(output_dir, "video_silhouette.mp4")
            self._write_video(sil_rows, vid_path, fps)

        tex_6v = [r.images.get("grid_6view_textured") for r in results]
        tex_6v = [g for g in tex_6v if g is not None]
        if tex_6v:
            vid_path = os.path.join(output_dir, "video_textured_6view.mp4")
            self._write_video(tex_6v, vid_path, fps)

        sil_6v = [r.images.get("grid_6view") for r in results]
        sil_6v = [g for g in sil_6v if g is not None]
        if sil_6v:
            vid_path = os.path.join(output_dir, "video_silhouette_6view.mp4")
            self._write_video(sil_6v, vid_path, fps)

        # --- 5. Reports ---
        self._save_report(results, output_dir, view_id)
        self._save_report_json(results, output_dir)

        print(f"\n  Output: {output_dir} ({len(os.listdir(output_dir))} files)")
        return output_dir

    def _write_video(self, frames: List[np.ndarray], path: str, fps: int = 2):
        """Write list of images as MP4 video using ffmpeg pipe."""
        import subprocess, tempfile
        if not frames:
            return
        H, W = frames[0].shape[:2]
        # H264 requires even dimensions
        W = W if W % 2 == 0 else W - 1
        H = H if H % 2 == 0 else H - 1
        if not path.endswith(".mp4"):
            path = path.rsplit(".", 1)[0] + ".mp4"

        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "fast",
            path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            for frame in frames:
                resized = cv2.resize(frame, (W, H)) if frame.shape[:2] != (H, W) else frame
                proc.stdin.write(resized.tobytes())
            proc.stdin.close()
            ret = proc.wait()
            if ret == 0:
                print(f"  Video: {path} ({len(frames)} frames, {fps}fps, {W}x{H})")
            else:
                err = proc.stderr.read().decode()[-200:]
                print(f"  Video FAILED: {path} (ffmpeg exit {ret}: {err})")
        except FileNotFoundError:
            print(f"  WARNING: ffmpeg not found, skipping video {path}")

    def _save_iou_chart(self, results: List[FrameResult], output_dir: str, view_id: int):
        """Generate IoU bar chart comparing BEFORE vs AFTER."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  WARNING: matplotlib not available, skipping IoU chart")
            return

        frame_ids = []
        ious_a, ious_b = [], []
        for r in results:
            if view_id in r.iou_a and view_id in r.iou_b:
                frame_ids.append(r.frame_id)
                ious_a.append(r.iou_a[view_id])
                ious_b.append(r.iou_b[view_id])

        if not frame_ids:
            return

        fig, ax = plt.subplots(figsize=(max(10, len(frame_ids) * 0.5), 5))
        x = np.arange(len(frame_ids))
        w = 0.35
        bars_a = ax.bar(x - w/2, ious_a, w, label="BEFORE (fast)", color="#CC4444", alpha=0.8)
        bars_b = ax.bar(x + w/2, ious_b, w, label="AFTER (accurate)", color="#44AA44", alpha=0.8)
        ax.axhline(y=self.config.iou_threshold, color="orange", linestyle="--", linewidth=1, label=f"Threshold ({self.config.iou_threshold})")
        ax.set_xlabel("MAMMAL Frame ID")
        ax.set_ylabel("Silhouette IoU")
        ax.set_title(f"Mesh Fitting Quality: BEFORE vs AFTER (view {view_id})")
        ax.set_xticks(x)
        ax.set_xticklabels(frame_ids, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add mean annotations
        mean_a, mean_b = np.mean(ious_a), np.mean(ious_b)
        ax.text(0.98, 0.02, f"Mean: {mean_a:.3f} → {mean_b:.3f} (+{mean_b-mean_a:.3f})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout()
        path = os.path.join(output_dir, "iou_chart.jpg")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  IoU chart: {path}")

    def _save_report_json(self, results: List[FrameResult], output_dir: str):
        """Save machine-readable IoU report."""
        import json
        data = {}
        for r in results:
            data[r.frame_id] = {
                "iou_a": {str(k): float(v) for k, v in r.iou_a.items()},
                "iou_b": {str(k): float(v) for k, v in r.iou_b.items()},
            }
        path = os.path.join(output_dir, "iou_report.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # Keep old name as alias for backwards compatibility
    save_grid = save_results

    def _save_report(self, results: List[FrameResult], output_dir: str, view_id: int):
        """Save text report of IoU comparison."""
        report_path = os.path.join(output_dir, "iou_report.txt")
        with open(report_path, "w") as f:
            f.write(f"MAMMAL Mesh Refit Comparison (view {view_id})\n")
            f.write(f"{'='*60}\n")
            f.write(f"{'Frame':>8} {'IoU_fast':>10} {'IoU_accurate':>14} {'Delta':>8} {'Pass':>6}\n")
            f.write(f"{'-'*60}\n")

            ious_a, ious_b = [], []
            for r in results:
                iou_a = r.iou_a.get(view_id, -1)
                iou_b = r.iou_b.get(view_id, -1)
                if iou_a >= 0 and iou_b >= 0:
                    delta = iou_b - iou_a
                    passed = "YES" if iou_b >= self.config.iou_threshold else "NO"
                    f.write(f"{r.frame_id:>8} {iou_a:>10.4f} {iou_b:>14.4f} {delta:>+8.4f} {passed:>6}\n")
                    ious_a.append(iou_a)
                    ious_b.append(iou_b)

            f.write(f"{'-'*60}\n")
            if ious_a:
                f.write(f"{'Mean':>8} {np.mean(ious_a):>10.4f} {np.mean(ious_b):>14.4f} "
                        f"{np.mean(ious_b) - np.mean(ious_a):>+8.4f}\n")
                n_pass = sum(1 for b in ious_b if b >= self.config.iou_threshold)
                f.write(f"\nPassed (IoU >= {self.config.iou_threshold}): {n_pass}/{len(ious_b)}\n")

        print(f"IoU report saved: {report_path}")
