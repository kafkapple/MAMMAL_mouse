"""
Unified Annotation Tool: Mask + Keypoint
Integrates SAM-based mask annotation and semantic keypoint annotation

Features:
- Two modes: Mask Mode (SAM) and Keypoint Mode (semantic)
- Shared frame navigation and storage
- Modular design for easy extension
- Compatible output formats (MAMMAL, SAM)

Usage:
    python unified_annotator.py --input data/frames --output data/annotations --mode both
"""
import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import json
import argparse
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# Try to import SAM2 (optional)
try:
    sam2_available = True
    sam2_path = Path.home() / 'dev/segment-anything-2'
    if sam2_path.exists():
        sys.path.append(str(sam2_path))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    sam2_available = False
    print("⚠️  SAM2 not available. Mask annotation will be disabled.")


class AnnotationMode(Enum):
    """Annotation mode selection"""
    MASK = "mask"           # SAM-based mask annotation
    KEYPOINT = "keypoint"   # Semantic keypoint annotation
    BOTH = "both"           # Both modes available


@dataclass
class AnnotationConfig:
    """Configuration for unified annotator"""
    input_dir: Path
    output_dir: Path
    mode: AnnotationMode

    # SAM config
    sam_checkpoint: Optional[Path] = None
    sam_config: str = "sam2_hiera_l.yaml"
    sam_device: str = "cuda"

    # Keypoint config
    keypoint_names: List[str] = None

    # Display config
    point_size: int = 5
    mask_alpha: float = 0.3

    def __post_init__(self):
        # Default keypoint names
        if self.keypoint_names is None:
            self.keypoint_names = [
                'nose', 'neck', 'spine_mid', 'hip', 'tail_base',
                'left_ear', 'right_ear'
            ]


class UnifiedAnnotator:
    """
    Unified annotation tool combining SAM mask and semantic keypoint annotation

    Supports two annotation modes:
    1. Mask Mode: SAM-based foreground/background segmentation
    2. Keypoint Mode: Semantic keypoint annotation with visibility
    """

    # Keypoint definitions
    KEYPOINT_COLORS = {
        'nose': (255, 0, 0),        # Red
        'neck': (255, 165, 0),      # Orange
        'spine_mid': (255, 255, 0), # Yellow
        'hip': (0, 255, 0),         # Green
        'tail_base': (0, 0, 255),   # Blue
        'left_ear': (255, 0, 255),  # Magenta
        'right_ear': (0, 255, 255), # Cyan
    }

    VISIBILITY_LEVELS = {
        'visible': 1.0,
        'occluded': 0.5,
        'not_visible': 0.0
    }

    def __init__(self, config: AnnotationConfig):
        """Initialize unified annotator with config"""
        self.config = config
        self.config.input_dir = Path(config.input_dir).expanduser()
        self.config.output_dir = Path(config.output_dir).expanduser()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Find frames
        self.frame_files = sorted(list(self.config.input_dir.glob('*.png')))
        if len(self.frame_files) == 0:
            self.frame_files = sorted(list(self.config.input_dir.glob('*.jpg')))

        print(f"Found {len(self.frame_files)} frames in {self.config.input_dir}")

        # Current state
        self.current_frame_idx = 0
        self.current_frame = None
        self.current_frame_path = None

        # Mask mode state
        self.mask_points = []
        self.mask_labels = []
        self.current_mask = None
        self.mask_confidence = None

        # Keypoint mode state
        self.current_keypoints = {}  # {name: (x, y, visibility)}

        # SAM model (lazy loading)
        self.sam_predictor = None
        if config.mode in [AnnotationMode.MASK, AnnotationMode.BOTH]:
            if sam2_available and config.sam_checkpoint:
                self._load_sam()

        # Load existing annotations
        self.annotations = self._load_all_annotations()

    def _load_sam(self):
        """Load SAM2 model"""
        try:
            checkpoint_path = Path(self.config.sam_checkpoint).expanduser()
            if not checkpoint_path.exists():
                print(f"⚠️  SAM checkpoint not found: {checkpoint_path}")
                return

            print("Loading SAM2...")
            sam2_model = build_sam2(
                self.config.sam_config,
                str(checkpoint_path),
                device=self.config.sam_device
            )
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            print("✅ SAM2 loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load SAM2: {e}")
            self.sam_predictor = None

    def _load_all_annotations(self) -> Dict:
        """Load all existing annotations from output directory"""
        annotations = {}

        for frame_file in self.frame_files:
            frame_name = frame_file.stem
            annotation_file = self.config.output_dir / f'{frame_name}_annotation.json'

            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    annotations[frame_name] = json.load(f)

        print(f"Loaded {len(annotations)} existing annotations")
        return annotations

    # ===== Frame Navigation =====

    def load_frame(self, frame_idx: int) -> Tuple[np.ndarray, str]:
        """Load frame and existing annotations"""
        if frame_idx < 0 or frame_idx >= len(self.frame_files):
            return None, f"Invalid frame index (0-{len(self.frame_files)-1})"

        self.current_frame_idx = frame_idx
        self.current_frame_path = self.frame_files[frame_idx]

        # Load image
        img = cv2.imread(str(self.current_frame_path))
        if img is None:
            return None, f"Failed to load {self.current_frame_path}"

        self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load existing annotation
        frame_name = self.current_frame_path.stem
        if frame_name in self.annotations:
            annotation = self.annotations[frame_name]

            # Load mask data
            if 'mask' in annotation:
                self.mask_points = annotation['mask'].get('points', [])
                self.mask_labels = annotation['mask'].get('labels', [])
                self.mask_confidence = annotation['mask'].get('confidence')

                # Load mask image
                mask_file = self.config.output_dir / f'{frame_name}_mask.png'
                if mask_file.exists():
                    mask_uint8 = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    self.current_mask = (mask_uint8 > 0)
                else:
                    self.current_mask = None
            else:
                self.mask_points = []
                self.mask_labels = []
                self.current_mask = None
                self.mask_confidence = None

            # Load keypoint data
            if 'keypoints' in annotation:
                self.current_keypoints = {
                    name: (kp['x'], kp['y'], kp.get('visibility', 1.0))
                    for name, kp in annotation['keypoints'].items()
                }
            else:
                self.current_keypoints = {}
        else:
            # Fresh annotation
            self.mask_points = []
            self.mask_labels = []
            self.current_mask = None
            self.mask_confidence = None
            self.current_keypoints = {}

        # Visualize
        vis_frame = self._draw_annotations(self.current_frame.copy())

        # Status message
        num_mask_pts = len(self.mask_points)
        num_keypoints = len([k for k, v in self.current_keypoints.items() if v[2] > 0])
        total_keypoints = len(self.config.keypoint_names)

        msg = f"Frame {frame_idx + 1}/{len(self.frame_files)} | "
        msg += f"Mask: {num_mask_pts} points | "
        msg += f"Keypoints: {num_keypoints}/{total_keypoints}"

        return vis_frame, msg

    def _draw_annotations(self, image: np.ndarray) -> np.ndarray:
        """Draw all annotations on image"""
        # Draw mask
        if self.current_mask is not None:
            mask_overlay = np.zeros_like(image)
            mask_overlay[self.current_mask] = (0, 255, 0)  # Green
            image = cv2.addWeighted(image, 1 - self.config.mask_alpha,
                                   mask_overlay, self.config.mask_alpha, 0)

        # Draw mask points
        for (px, py), label in zip(self.mask_points, self.mask_labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green/Red
            cv2.circle(image, (px, py), self.config.point_size, color, -1)
            cv2.circle(image, (px, py), self.config.point_size + 2, (255, 255, 255), 2)

        # Draw keypoints
        for name, (x, y, vis) in self.current_keypoints.items():
            if vis == 0.0:
                continue

            color = self.KEYPOINT_COLORS.get(name, (255, 255, 255))

            # Filled if visible, hollow if occluded
            if vis == 0.5:
                cv2.circle(image, (int(x), int(y)), self.config.point_size, color, 2)
            else:
                cv2.circle(image, (int(x), int(y)), self.config.point_size, color, -1)

            # Label
            cv2.putText(image, name, (int(x) + 8, int(y) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return image

    # ===== Mask Mode Methods =====

    def add_mask_point(self, evt: gr.SelectData, point_type: str) -> Tuple[np.ndarray, str]:
        """Add foreground/background point for SAM"""
        if self.current_frame is None:
            return self.current_frame, "Load a frame first!"

        x, y = evt.index[0], evt.index[1]
        label = 1 if point_type == "Foreground" else 0

        self.mask_points.append([x, y])
        self.mask_labels.append(label)

        vis_frame = self._draw_annotations(self.current_frame.copy())
        msg = f"Added {'foreground' if label == 1 else 'background'} point at ({x}, {y})"

        return vis_frame, msg

    def generate_mask(self) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Generate mask using SAM"""
        if self.sam_predictor is None:
            return self.current_frame, None, "SAM not available!"

        if len(self.mask_points) == 0:
            return self.current_frame, None, "Add points first!"

        # SAM inference
        self.sam_predictor.set_image(self.current_frame)

        point_coords = np.array(self.mask_points, dtype=np.float32)
        point_labels = np.array(self.mask_labels, dtype=np.int32)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # Best mask
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx]
        self.mask_confidence = float(scores[best_idx])

        # Visualize
        vis_frame = self._draw_annotations(self.current_frame.copy())
        mask_binary = (self.current_mask * 255).astype(np.uint8)

        mask_area = self.current_mask.sum() / self.current_mask.size * 100
        msg = f"Mask generated! Confidence: {self.mask_confidence:.3f}, Area: {mask_area:.1f}%"

        return vis_frame, mask_binary, msg

    def clear_mask(self) -> Tuple[np.ndarray, str]:
        """Clear mask points and mask"""
        self.mask_points = []
        self.mask_labels = []
        self.current_mask = None
        self.mask_confidence = None

        vis_frame = self._draw_annotations(self.current_frame.copy())
        return vis_frame, "Mask cleared"

    # ===== Keypoint Mode Methods =====

    def add_keypoint(self, evt: gr.SelectData, keypoint_name: str,
                     visibility: str) -> Tuple[np.ndarray, str]:
        """Add semantic keypoint"""
        if self.current_frame is None:
            return self.current_frame, "Load a frame first!"

        x, y = evt.index[0], evt.index[1]
        vis_value = self.VISIBILITY_LEVELS[visibility]

        self.current_keypoints[keypoint_name] = (x, y, vis_value)

        vis_frame = self._draw_annotations(self.current_frame.copy())

        num_kps = len([k for k, v in self.current_keypoints.items() if v[2] > 0])
        total_kps = len(self.config.keypoint_names)

        msg = f"Added {keypoint_name} ({visibility}) at ({x}, {y}) | {num_kps}/{total_kps}"

        return vis_frame, msg

    def mark_keypoint_invisible(self, keypoint_name: str) -> Tuple[np.ndarray, str]:
        """Mark keypoint as not visible"""
        self.current_keypoints[keypoint_name] = (0, 0, 0.0)

        vis_frame = self._draw_annotations(self.current_frame.copy())
        return vis_frame, f"Marked {keypoint_name} as not visible"

    def remove_keypoint(self, keypoint_name: str) -> Tuple[np.ndarray, str]:
        """Remove keypoint"""
        if keypoint_name in self.current_keypoints:
            del self.current_keypoints[keypoint_name]
            vis_frame = self._draw_annotations(self.current_frame.copy())
            return vis_frame, f"Removed {keypoint_name}"
        else:
            return self._draw_annotations(self.current_frame.copy()), f"{keypoint_name} not found"

    def clear_keypoints(self) -> Tuple[np.ndarray, str]:
        """Clear all keypoints"""
        self.current_keypoints = {}
        vis_frame = self._draw_annotations(self.current_frame.copy())
        return vis_frame, "Keypoints cleared"

    # ===== Save/Load Methods =====

    def save_annotation(self) -> str:
        """Save current annotation (mask + keypoints)"""
        if self.current_frame is None:
            return "No frame loaded!"

        frame_name = self.current_frame_path.stem
        annotation = {
            'frame': str(self.current_frame_path),
            'frame_idx': self.current_frame_idx
        }

        # Save mask data
        if len(self.mask_points) > 0 or self.current_mask is not None:
            annotation['mask'] = {
                'points': self.mask_points,
                'labels': self.mask_labels,
                'has_mask': self.current_mask is not None
            }

            if self.current_mask is not None:
                annotation['mask']['confidence'] = self.mask_confidence
                annotation['mask']['mask_area_pct'] = float(
                    self.current_mask.sum() / self.current_mask.size * 100
                )

                # Save mask image
                mask_file = self.config.output_dir / f'{frame_name}_mask.png'
                mask_uint8 = (self.current_mask * 255).astype(np.uint8)
                cv2.imwrite(str(mask_file), mask_uint8)

        # Save keypoint data
        if len(self.current_keypoints) > 0:
            annotation['keypoints'] = {
                name: {
                    'x': float(x),
                    'y': float(y),
                    'visibility': float(vis)
                }
                for name, (x, y, vis) in self.current_keypoints.items()
            }

        # Save annotation JSON
        annotation_file = self.config.output_dir / f'{frame_name}_annotation.json'
        with open(annotation_file, 'w') as f:
            json.dump(annotation, f, indent=2)

        # Update cached annotations
        self.annotations[frame_name] = annotation

        num_mask = len(self.mask_points)
        num_kps = len([k for k, v in self.current_keypoints.items() if v[2] > 0])

        return f"✅ Saved to {annotation_file.name} | Mask pts: {num_mask}, Keypoints: {num_kps}"

    def get_summary(self) -> str:
        """Get annotation summary"""
        total_frames = len(self.frame_files)
        annotated_frames = len(self.annotations)

        frames_with_mask = sum(1 for a in self.annotations.values() if 'mask' in a)
        frames_with_kps = sum(1 for a in self.annotations.values() if 'keypoints' in a)

        summary = f"**Progress**: {annotated_frames}/{total_frames} frames\n\n"
        summary += f"- Frames with mask: {frames_with_mask}\n"
        summary += f"- Frames with keypoints: {frames_with_kps}\n\n"

        if self.current_keypoints:
            summary += "**Current Keypoints**:\n"
            for name in self.config.keypoint_names:
                if name in self.current_keypoints:
                    x, y, vis = self.current_keypoints[name]
                    if vis == 1.0:
                        summary += f"✅ {name}: ({x:.0f}, {y:.0f})\n"
                    elif vis == 0.5:
                        summary += f"⚠️ {name}: ({x:.0f}, {y:.0f}) occluded\n"
                    else:
                        summary += f"👁️ {name}: not visible\n"
                else:
                    summary += f"❌ {name}: Not set\n"

        return summary


def create_unified_ui(annotator: UnifiedAnnotator, config: AnnotationConfig):
    """Create unified Gradio UI"""

    max_frame_idx = max(0, len(annotator.frame_files) - 1)

    with gr.Blocks(title="Unified Mouse Annotator") as demo:
        gr.Markdown("# 🐭 Unified Mouse Annotation Tool")
        gr.Markdown("**Mask (SAM) + Keypoint annotation in one interface**")

        with gr.Row():
            # Left column: Image display
            with gr.Column(scale=2):
                image_display = gr.Image(
                    label="Frame (Click to annotate)",
                    type="numpy",
                    interactive=True
                )

                status_text = gr.Textbox(label="Status", interactive=False)

            # Right column: Controls
            with gr.Column(scale=1):
                gr.Markdown("### Frame Navigation")

                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=max_frame_idx,
                    value=0,
                    step=1,
                    label="Frame"
                )

                load_btn = gr.Button("📂 Load Frame", variant="primary")
                save_btn = gr.Button("💾 Save Annotation", variant="primary")

                gr.Markdown("---")

                # Mode tabs
                with gr.Tabs():
                    # Mask Mode Tab
                    with gr.Tab("🎯 Mask Mode"):
                        gr.Markdown("**SAM-based segmentation**")

                        mask_point_type = gr.Radio(
                            ["Foreground", "Background"],
                            value="Foreground",
                            label="Point Type"
                        )

                        generate_mask_btn = gr.Button("🎯 Generate Mask")
                        clear_mask_btn = gr.Button("🗑️ Clear Mask")

                        mask_display = gr.Image(label="Mask", type="numpy")

                    # Keypoint Mode Tab
                    with gr.Tab("📍 Keypoint Mode"):
                        gr.Markdown("**Semantic keypoint annotation**")

                        keypoint_selector = gr.Radio(
                            choices=config.keypoint_names,
                            value=config.keypoint_names[0],
                            label="Keypoint"
                        )

                        visibility_selector = gr.Radio(
                            choices=['visible', 'occluded', 'not_visible'],
                            value='visible',
                            label="Visibility"
                        )

                        mark_invisible_btn = gr.Button("👁️ Mark Not Visible")
                        remove_kp_btn = gr.Button("🗑️ Remove Keypoint")
                        clear_kps_btn = gr.Button("🗑️ Clear All Keypoints")

                gr.Markdown("---")

                summary_text = gr.Markdown(label="Summary")
                update_summary_btn = gr.Button("🔄 Update Summary")

        # Event handlers
        def load_and_update(frame_idx):
            vis_frame, msg = annotator.load_frame(frame_idx)
            summary = annotator.get_summary()
            return vis_frame, msg, summary

        load_btn.click(
            fn=load_and_update,
            inputs=[frame_slider],
            outputs=[image_display, status_text, summary_text]
        )

        # Mask mode events
        def handle_mask_click(evt, point_type):
            vis_frame, msg = annotator.add_mask_point(evt, point_type)
            return vis_frame, msg

        # Keypoint mode events
        def handle_keypoint_click(evt, kp_name, visibility):
            vis_frame, msg = annotator.add_keypoint(evt, kp_name, visibility)
            summary = annotator.get_summary()
            return vis_frame, msg, summary

        # Image click - determine mode from active tab
        # For now, we use separate handlers

        generate_mask_btn.click(
            fn=annotator.generate_mask,
            outputs=[image_display, mask_display, status_text]
        )

        clear_mask_btn.click(
            fn=annotator.clear_mask,
            outputs=[image_display, status_text]
        )

        mark_invisible_btn.click(
            fn=annotator.mark_keypoint_invisible,
            inputs=[keypoint_selector],
            outputs=[image_display, status_text]
        )

        remove_kp_btn.click(
            fn=annotator.remove_keypoint,
            inputs=[keypoint_selector],
            outputs=[image_display, status_text]
        )

        clear_kps_btn.click(
            fn=annotator.clear_keypoints,
            outputs=[image_display, status_text]
        )

        save_btn.click(
            fn=annotator.save_annotation,
            outputs=[status_text]
        )

        update_summary_btn.click(
            fn=annotator.get_summary,
            outputs=[summary_text]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Unified Mouse Annotation Tool")
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing frames')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for annotations')
    parser.add_argument('--mode', '-m', type=str, default='both',
                       choices=['mask', 'keypoint', 'both'],
                       help='Annotation mode')
    parser.add_argument('--sam-checkpoint', type=str, default=None,
                       help='Path to SAM2 checkpoint (for mask mode)')
    parser.add_argument('--port', '-p', type=int, default=7860,
                       help='Server port (default: 7860)')

    args = parser.parse_args()

    # Create config
    config = AnnotationConfig(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        mode=AnnotationMode(args.mode),
        sam_checkpoint=Path(args.sam_checkpoint) if args.sam_checkpoint else None
    )

    # Create annotator
    print("="*60)
    print("Unified Mouse Annotation Tool")
    print("="*60)
    print(f"Input:  {config.input_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Mode:   {config.mode.value}")
    print("="*60)

    annotator = UnifiedAnnotator(config)

    # Create UI
    demo = create_unified_ui(annotator, config)

    # Launch
    print(f"\n🚀 Launching server on port {args.port}...")
    print(f"📍 Local: http://localhost:{args.port}")
    print(f"📍 SSH tunnel: ssh -L {args.port}:localhost:{args.port} user@server\n")

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
