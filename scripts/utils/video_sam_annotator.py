"""
Video Frame Extraction + SAM Interactive Annotation
Gradio-based web UI for extracting frames and annotating mouse regions
"""
import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys
import json
import torch
from typing import Optional, Tuple, Dict, List

# SAM 2 imports
sam2_path = Path.home() / 'dev/segment-anything-2'
if sam2_path.exists():
    sys.path.insert(0, str(sam2_path))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
else:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 not found. Annotation features will be disabled.")


class VideoSAMAnnotator:
    """Interactive video frame extraction and SAM annotation"""

    def __init__(self,
                 video_path: str,
                 output_dir: str = 'video_annotations',
                 sam_checkpoint: Optional[str] = None,
                 sam_config: str = 'sam2_hiera_l.yaml',
                 device: str = 'cuda'):
        """
        Initialize annotator

        Args:
            video_path: Path to video file
            output_dir: Output directory for frames and annotations
            sam_checkpoint: Path to SAM 2 checkpoint
            sam_config: SAM 2 model config
            device: Device to use ('cuda' or 'cpu')
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Video info
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video loaded: {self.video_path.name}")
        print(f"  Frames: {self.total_frames}, FPS: {self.fps:.2f}")
        print(f"  Resolution: {self.width}x{self.height}")

        # Current state
        self.current_frame_idx = 0
        self.current_frame = None
        self.current_frame_rgb = None
        self.points = []
        self.labels = []
        self.current_mask = None
        self.current_bbox = None

        # Load SAM 2
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.predictor = None

        if SAM2_AVAILABLE:
            self._load_sam(sam_checkpoint, sam_config)

    def _load_sam(self, checkpoint_path: Optional[str], config: str):
        """Load SAM 2 model"""
        if checkpoint_path is None:
            # Try default location
            checkpoint_path = Path.home() / 'dev/segment-anything-2/checkpoints/sam2_hiera_large.pt'

        checkpoint_path = Path(checkpoint_path).expanduser()

        if not checkpoint_path.exists():
            print(f"Warning: SAM checkpoint not found at {checkpoint_path}")
            print("Annotation features disabled.")
            return

        print(f"Loading SAM 2...")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {self.device}")

        sam2_model = build_sam2(config, str(checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        print("SAM 2 loaded successfully!")

    def load_frame(self, frame_idx: int) -> Tuple[np.ndarray, str]:
        """
        Load a specific frame from video

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            frame_rgb: Frame in RGB format
            status: Status message
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return self.current_frame_rgb, f"Invalid frame index (0-{self.total_frames-1})"

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            return self.current_frame_rgb, f"Failed to read frame {frame_idx}"

        self.current_frame_idx = frame_idx
        self.current_frame = frame
        self.current_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reset annotation state
        self.points = []
        self.labels = []
        self.current_mask = None
        self.current_bbox = None

        timestamp = frame_idx / self.fps
        return self.current_frame_rgb, f"Loaded frame {frame_idx} (t={timestamp:.2f}s)"

    def add_point(self, evt: gr.SelectData, point_type: str) -> Tuple[np.ndarray, str]:
        """
        Add annotation point

        Args:
            evt: Click event with coordinates
            point_type: 'Foreground' or 'Background'

        Returns:
            frame_with_points: Frame with points visualized
            status: Status message
        """
        if self.current_frame_rgb is None:
            return None, "Load a frame first!"

        x, y = evt.index[0], evt.index[1]
        label = 1 if point_type == "Foreground" else 0

        self.points.append([x, y])
        self.labels.append(label)

        # Visualize points
        frame_vis = self.current_frame_rgb.copy()

        for (px, py), lbl in zip(self.points, self.labels):
            color = (0, 255, 0) if lbl == 1 else (255, 0, 0)  # Green=FG, Red=BG
            cv2.circle(frame_vis, (px, py), 8, color, -1)
            cv2.circle(frame_vis, (px, py), 10, (255, 255, 255), 2)

        point_type_str = 'foreground' if label == 1 else 'background'
        return frame_vis, f"Added {point_type_str} point at ({x}, {y}). Total: {len(self.points)} points"

    def generate_mask(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generate mask using SAM

        Returns:
            frame_with_overlay: Frame with mask overlay
            mask_binary: Binary mask image
            status: Status message
        """
        if self.predictor is None:
            return self.current_frame_rgb, None, "SAM not available!"

        if len(self.points) == 0:
            return self.current_frame_rgb, None, "Add points first!"

        # Set image for SAM
        self.predictor.set_image(self.current_frame_rgb)

        # Predict
        point_coords = np.array(self.points, dtype=np.float32)
        point_labels = np.array(self.labels, dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # Select best mask
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx]

        # Compute bounding box
        y_indices, x_indices = np.where(self.current_mask)
        if len(y_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            self.current_bbox = [int(x_min), int(y_min),
                                int(x_max - x_min), int(y_max - y_min)]
        else:
            self.current_bbox = None

        # Visualize
        frame_vis = self.current_frame_rgb.copy()

        # Mask overlay
        mask_overlay = np.zeros_like(frame_vis)
        mask_overlay[self.current_mask > 0] = [0, 255, 0]  # Green
        frame_vis = cv2.addWeighted(frame_vis, 0.7, mask_overlay, 0.3, 0)

        # Draw points
        for (px, py), lbl in zip(self.points, self.labels):
            color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
            cv2.circle(frame_vis, (px, py), 8, color, -1)
            cv2.circle(frame_vis, (px, py), 10, (255, 255, 255), 2)

        # Draw bounding box
        if self.current_bbox is not None:
            x, y, w, h = self.current_bbox
            cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Binary mask for display
        mask_binary = (self.current_mask * 255).astype(np.uint8)

        # Statistics
        mask_area = self.current_mask.sum()
        mask_pct = mask_area / self.current_mask.size * 100
        status = f"Mask generated! Confidence: {scores[best_idx]:.3f}, Area: {mask_pct:.1f}%"

        return frame_vis, mask_binary, status

    def save_annotation(self) -> str:
        """
        Save current annotation (points, mask, cropped frame)

        Returns:
            status: Status message
        """
        if len(self.points) == 0:
            return "No points to save!"

        frame_dir = self.output_dir / f"frame_{self.current_frame_idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save original frame
        original_path = frame_dir / "original.png"
        cv2.imwrite(str(original_path), self.current_frame)

        # Save annotation data
        annotation = {
            'video_path': str(self.video_path),
            'frame_idx': self.current_frame_idx,
            'timestamp': self.current_frame_idx / self.fps,
            'original_shape': [self.height, self.width],
            'points': self.points,
            'labels': self.labels,
        }

        # Save mask and cropped frame if available
        if self.current_mask is not None:
            # Save mask
            mask_path = frame_dir / "mask.png"
            mask_uint8 = (self.current_mask * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_uint8)

            # Crop frame using mask bbox with padding
            if self.current_bbox is not None:
                x, y, w, h = self.current_bbox
                padding = 50

                x_min = max(0, x - padding)
                y_min = max(0, y - padding)
                x_max = min(self.width, x + w + padding)
                y_max = min(self.height, y + h + padding)

                cropped_frame = self.current_frame[y_min:y_max, x_min:x_max]
                cropped_mask = self.current_mask[y_min:y_max, x_min:x_max]

                cropped_path = frame_dir / "cropped.png"
                cropped_mask_path = frame_dir / "cropped_mask.png"

                cv2.imwrite(str(cropped_path), cropped_frame)
                cv2.imwrite(str(cropped_mask_path), (cropped_mask * 255).astype(np.uint8))

                annotation.update({
                    'bbox': self.current_bbox,
                    'crop_coords': [x_min, y_min, x_max, y_max],
                    'cropped_shape': [y_max - y_min, x_max - x_min],
                    'mask_area': int(self.current_mask.sum()),
                })

        # Save annotation JSON
        annotation_path = frame_dir / "annotation.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        # Update global index
        index_path = self.output_dir / "annotation_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {'video_path': str(self.video_path), 'frames': []}

        # Add or update frame entry
        frame_entry = {
            'frame_idx': self.current_frame_idx,
            'annotation_dir': str(frame_dir.relative_to(self.output_dir)),
            'has_mask': self.current_mask is not None,
            'num_points': len(self.points)
        }

        # Remove existing entry if present
        index['frames'] = [f for f in index['frames'] if f['frame_idx'] != self.current_frame_idx]
        index['frames'].append(frame_entry)
        index['frames'].sort(key=lambda x: x['frame_idx'])

        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

        return f"Saved to {frame_dir}. Total annotated frames: {len(index['frames'])}"

    def clear_points(self) -> Tuple[np.ndarray, str]:
        """Clear all points and mask"""
        self.points = []
        self.labels = []
        self.current_mask = None
        self.current_bbox = None
        return self.current_frame_rgb, "Points and mask cleared"

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap'):
            self.cap.release()


def create_ui(annotator: VideoSAMAnnotator):
    """Create Gradio UI"""

    with gr.Blocks(title="Video SAM Annotator") as demo:
        gr.Markdown("# 🎬 Video Frame Extraction + SAM Annotation")
        gr.Markdown(f"**Video**: {annotator.video_path.name} | "
                   f"**Frames**: {annotator.total_frames} | "
                   f"**FPS**: {annotator.fps:.2f} | "
                   f"**Resolution**: {annotator.width}x{annotator.height}")

        with gr.Row():
            with gr.Column(scale=2):
                # Frame display
                image_display = gr.Image(
                    label="Video Frame (Click to add annotation points)",
                    type="numpy",
                    height=600
                )

                with gr.Row():
                    point_type = gr.Radio(
                        ["Foreground", "Background"],
                        value="Foreground",
                        label="Point Type",
                        info="Green=Foreground (mouse), Red=Background"
                    )

                status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                # Mask display
                mask_display = gr.Image(
                    label="Generated Mask",
                    type="numpy",
                    height=300
                )

                gr.Markdown("### Frame Navigation")

                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=annotator.total_frames - 1,
                    value=0,
                    step=1,
                    label="Frame Index"
                )

                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous")
                    load_btn = gr.Button("📂 Load Frame", variant="primary")
                    next_btn = gr.Button("➡️ Next")

                gr.Markdown("### Annotation")

                with gr.Row():
                    generate_btn = gr.Button("🎯 Generate Mask", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")

                save_btn = gr.Button("💾 Save Annotation", variant="primary")

                gr.Markdown("### Instructions")
                gr.Markdown("""
                1. Use slider or buttons to select frame
                2. Click **Load Frame**
                3. Select **Foreground** (mouse) or **Background**
                4. Click on image to add points
                   - Green circles: Foreground points
                   - Red circles: Background points
                5. Click **Generate Mask** to run SAM
                6. If satisfied, click **Save Annotation**
                7. Move to next frame and repeat

                **Output**: Saves original, cropped, mask, and annotation JSON
                """)

        # Event handlers
        def load_frame_handler(idx):
            return annotator.load_frame(int(idx))

        def prev_frame():
            new_idx = max(0, annotator.current_frame_idx - 1)
            frame, status = annotator.load_frame(new_idx)
            return frame, status, new_idx

        def next_frame():
            new_idx = min(annotator.total_frames - 1, annotator.current_frame_idx + 1)
            frame, status = annotator.load_frame(new_idx)
            return frame, status, new_idx

        load_btn.click(
            fn=load_frame_handler,
            inputs=[frame_slider],
            outputs=[image_display, status_text]
        )

        prev_btn.click(
            fn=prev_frame,
            outputs=[image_display, status_text, frame_slider]
        )

        next_btn.click(
            fn=next_frame,
            outputs=[image_display, status_text, frame_slider]
        )

        image_display.select(
            fn=annotator.add_point,
            inputs=[point_type],
            outputs=[image_display, status_text]
        )

        generate_btn.click(
            fn=annotator.generate_mask,
            outputs=[image_display, mask_display, status_text]
        )

        save_btn.click(
            fn=annotator.save_annotation,
            outputs=[status_text]
        )

        clear_btn.click(
            fn=annotator.clear_points,
            outputs=[image_display, status_text]
        )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Video Frame Extraction + SAM Annotation")
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output-dir', type=str, default='video_annotations',
                       help='Output directory')
    parser.add_argument('--sam-checkpoint', type=str, default=None,
                       help='Path to SAM 2 checkpoint')
    parser.add_argument('--sam-config', type=str, default='sam2_hiera_l.yaml',
                       help='SAM 2 model config')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Server port')
    parser.add_argument('--share', action='store_true',
                       help='Create public Gradio share link')

    args = parser.parse_args()

    print("="*80)
    print("Video SAM Annotator")
    print("="*80)

    # Create annotator
    annotator = VideoSAMAnnotator(
        video_path=args.video_path,
        output_dir=args.output_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        device=args.device
    )

    # Create and launch UI
    demo = create_ui(annotator)

    print("\nLaunching web interface...")
    print(f"  Local: http://localhost:{args.port}")
    print(f"  SSH tunnel: ssh -L {args.port}:localhost:{args.port} user@server")
    print("="*80)

    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
