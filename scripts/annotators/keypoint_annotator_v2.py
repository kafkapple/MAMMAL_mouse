"""
Enhanced Keypoint Annotation Tool with Zoom and Visibility Control
"""
import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


class KeypointAnnotator:
    """Enhanced keypoint annotation tool with zoom and visibility"""

    # Define essential keypoints for mouse
    KEYPOINT_NAMES = [
        'nose',
        'neck',
        'spine_mid',
        'hip',
        'tail_base',
        'left_ear',
        'right_ear',
    ]

    # Visibility states
    VISIBILITY = {
        'visible': 1.0,      # Clearly visible
        'occluded': 0.5,     # Partially visible/uncertain
        'not_visible': 0.0   # Not visible/skip
    }

    # Colors for each keypoint (RGB format for display)
    KEYPOINT_COLORS = {
        'nose': (255, 0, 0),        # Red
        'neck': (255, 165, 0),      # Orange
        'spine_mid': (255, 255, 0), # Yellow
        'hip': (0, 255, 0),         # Green
        'tail_base': (0, 0, 255),   # Blue
        'left_ear': (255, 0, 255),  # Magenta
        'right_ear': (0, 255, 255), # Cyan
    }

    def __init__(self, frames_dir: str, output_file: str = 'keypoints.json'):
        self.frames_dir = Path(frames_dir)
        self.output_file = Path(output_file)
        self.frame_files = sorted(self.frames_dir.glob('*_cropped.png'))
        self.annotations = self.load_annotations()

        # Current state
        self.current_frame_idx = 0
        self.current_image = None
        self.current_keypoints = {}  # {name: (x, y, visibility)}

        # Display settings
        self.zoom_level = 2.0  # Start zoomed in
        self.point_size = 3  # Smaller points

    def load_annotations(self) -> Dict:
        """Load existing annotations from file"""
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                return json.load(f)
        return {}

    def save_annotations(self):
        """Save all annotations to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

    def apply_zoom(self, image: np.ndarray, zoom: float) -> np.ndarray:
        """Apply zoom to image"""
        if zoom == 1.0:
            return image

        h, w = image.shape[:2]
        # Calculate new size
        new_h, new_w = int(h * zoom), int(w * zoom)
        # Resize
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return zoomed

    def load_frame(self, frame_idx: int, zoom: float, point_size: int) -> Tuple[np.ndarray, str]:
        """Load a frame with zoom applied"""
        if frame_idx < 0 or frame_idx >= len(self.frame_files):
            return None, f"Invalid frame index: {frame_idx}"

        self.current_frame_idx = frame_idx
        self.zoom_level = zoom
        self.point_size = point_size

        frame_file = self.frame_files[frame_idx]

        # Load image (BGR)
        img = cv2.imread(str(frame_file))
        if img is None:
            return None, f"Failed to load image: {frame_file}"

        # Convert BGR to RGB
        self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get frame name
        frame_name = frame_file.stem.replace('_cropped', '')

        # Load existing keypoints for this frame
        if frame_name in self.annotations:
            self.current_keypoints = {
                name: (kp['x'], kp['y'], kp.get('visibility', 1.0))
                for name, kp in self.annotations[frame_name].items()
            }
        else:
            self.current_keypoints = {}

        # Draw keypoints and apply zoom
        vis_image = self.draw_keypoints(self.current_image.copy(), zoom, point_size)

        num_kps = len([k for k, v in self.current_keypoints.items() if v[2] > 0])
        total_kps = len(self.KEYPOINT_NAMES)

        return vis_image, f"Frame {frame_idx + 1}/{len(self.frame_files)} - {frame_name} ({num_kps}/{total_kps} keypoints)"

    def draw_keypoints(self, image: np.ndarray, zoom: float, point_size: int) -> np.ndarray:
        """Draw current keypoints on image with zoom"""
        for name, (x, y, vis) in self.current_keypoints.items():
            if vis == 0.0:  # Not visible, skip drawing
                continue

            color = self.KEYPOINT_COLORS.get(name, (255, 255, 255))

            # Scale coordinates for zoom
            x_scaled, y_scaled = int(x * zoom), int(y * zoom)

            # Adjust sizes for zoom
            circle_size = max(2, int(point_size * zoom))
            border_size = max(1, int((point_size + 2) * zoom))
            font_scale = 0.3 * zoom
            text_thickness = max(1, int(zoom))

            # Draw circle
            if vis == 0.5:  # Occluded - hollow circle
                cv2.circle(image, (x_scaled, y_scaled), circle_size, color, 2)
            else:  # Visible - filled circle
                cv2.circle(image, (x_scaled, y_scaled), circle_size, color, -1)

            cv2.circle(image, (x_scaled, y_scaled), border_size, (0, 0, 0), max(1, int(zoom)))

            # Draw label
            text_offset = int(10 * zoom)
            cv2.putText(
                image, name,
                (x_scaled + text_offset, y_scaled - text_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                text_thickness,
                cv2.LINE_AA
            )

        # Apply zoom
        return self.apply_zoom(image, zoom)

    def add_keypoint(self, x: int, y: int, keypoint_name: str, visibility: str,
                    zoom: float, point_size: int) -> Tuple[np.ndarray, str]:
        """Add or update a keypoint"""
        if self.current_image is None:
            return None, "No frame loaded"

        # Adjust coordinates for zoom
        x_orig = x / zoom
        y_orig = y / zoom

        # Get visibility value
        vis_value = self.VISIBILITY.get(visibility, 1.0)

        # Add keypoint
        self.current_keypoints[keypoint_name] = (x_orig, y_orig, vis_value)

        # Redraw
        vis_image = self.draw_keypoints(self.current_image.copy(), zoom, point_size)

        num_kps = len([k for k, v in self.current_keypoints.items() if v[2] > 0])
        total_kps = len(self.KEYPOINT_NAMES)

        vis_label = {1.0: "visible", 0.5: "occluded", 0.0: "not visible"}.get(vis_value, "")
        return vis_image, f"Added {keypoint_name} ({vis_label}) at ({x_orig:.1f}, {y_orig:.1f}) - ({num_kps}/{total_kps})"

    def mark_not_visible(self, keypoint_name: str, zoom: float, point_size: int) -> Tuple[np.ndarray, str]:
        """Mark a keypoint as not visible"""
        # Add with visibility = 0.0 at dummy location
        self.current_keypoints[keypoint_name] = (0, 0, 0.0)

        vis_image = self.draw_keypoints(self.current_image.copy(), zoom, point_size)

        num_kps = len([k for k, v in self.current_keypoints.items() if v[2] > 0])
        total_kps = len(self.KEYPOINT_NAMES)

        return vis_image, f"Marked {keypoint_name} as NOT VISIBLE ({num_kps}/{total_kps})"

    def remove_keypoint(self, keypoint_name: str, zoom: float, point_size: int) -> Tuple[np.ndarray, str]:
        """Remove a keypoint"""
        if keypoint_name in self.current_keypoints:
            del self.current_keypoints[keypoint_name]
            vis_image = self.draw_keypoints(self.current_image.copy(), zoom, point_size)
            return vis_image, f"Removed {keypoint_name}"
        else:
            return self.draw_keypoints(self.current_image.copy(), zoom, point_size), f"{keypoint_name} not found"

    def save_current_frame(self) -> str:
        """Save keypoints for current frame"""
        if self.current_frame_idx >= len(self.frame_files):
            return "No frame loaded"

        frame_name = self.frame_files[self.current_frame_idx].stem.replace('_cropped', '')

        # Convert to dict format
        keypoints_dict = {
            name: {
                'x': float(x),
                'y': float(y),
                'visibility': float(vis)
            }
            for name, (x, y, vis) in self.current_keypoints.items()
        }

        self.annotations[frame_name] = keypoints_dict
        self.save_annotations()

        num_kps = len([k for k, v in self.current_keypoints.items() if v[2] > 0])
        return f"✅ Saved {num_kps} keypoints for {frame_name}"

    def get_keypoint_summary(self) -> str:
        """Get summary of current annotations"""
        total_frames = len(self.frame_files)
        annotated_frames = len(self.annotations)

        summary = f"**Progress**: {annotated_frames}/{total_frames} frames\n\n"
        summary += "**Current Frame Keypoints**:\n"

        for name in self.KEYPOINT_NAMES:
            if name in self.current_keypoints:
                x, y, vis = self.current_keypoints[name]
                if vis == 1.0:
                    summary += f"✅ {name}: ({x:.0f}, {y:.0f}) - visible\n"
                elif vis == 0.5:
                    summary += f"⚠️ {name}: ({x:.0f}, {y:.0f}) - occluded\n"
                else:
                    summary += f"👁️ {name}: not visible\n"
            else:
                summary += f"❌ {name}: Not set\n"

        return summary


def create_ui(annotator: KeypointAnnotator):
    """Create enhanced Gradio UI with zoom and visibility controls"""

    with gr.Blocks(title="Mouse Keypoint Annotator") as demo:
        gr.Markdown("# 🐭 Mouse Keypoint Annotation Tool (Enhanced)")
        gr.Markdown("📌 **New**: Zoom support, adjustable point size, visibility control")

        with gr.Row():
            with gr.Column(scale=2):
                # Main image display
                image_display = gr.Image(
                    label="🔍 Click to add keypoint (Zoomed view)",
                    type="numpy",
                    interactive=True
                )

                with gr.Row():
                    keypoint_selector = gr.Radio(
                        choices=annotator.KEYPOINT_NAMES,
                        value=annotator.KEYPOINT_NAMES[0],
                        label="Select Keypoint"
                    )

                    visibility_selector = gr.Radio(
                        choices=['visible', 'occluded', 'not_visible'],
                        value='visible',
                        label="Visibility"
                    )

                status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### Display Settings")

                zoom_slider = gr.Slider(
                    minimum=1.0,
                    maximum=4.0,
                    value=2.0,
                    step=0.5,
                    label="🔍 Zoom Level"
                )

                point_size_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=3,
                    step=1,
                    label="⚫ Point Size"
                )

                gr.Markdown("### Navigation")

                frame_idx = gr.Slider(
                    minimum=0,
                    maximum=max(0, len(annotator.frame_files) - 1),
                    value=0,
                    step=1,
                    label="Frame Index"
                )

                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous")
                    next_btn = gr.Button("Next ➡️")

                load_btn = gr.Button("📂 Reload View", variant="secondary")
                save_btn = gr.Button("💾 Save Keypoints", variant="primary")

                gr.Markdown("### Quick Actions")

                mark_invisible_btn = gr.Button("👁️ Mark Current as NOT VISIBLE")

                remove_selector = gr.Dropdown(
                    choices=annotator.KEYPOINT_NAMES,
                    label="Remove Keypoint"
                )
                remove_btn = gr.Button("🗑️ Remove Selected")

                gr.Markdown("### Summary")
                summary_text = gr.Markdown(annotator.get_keypoint_summary())

                gr.Markdown("### Instructions")
                gr.Markdown("""
                **Zoom**: Use slider to enlarge image (default 2x)
                **Point Size**: Adjust marker size

                **Visibility Options**:
                - ✅ **Visible**: Point clearly visible
                - ⚠️ **Occluded**: Uncertain/partially visible
                - 👁️ **Not Visible**: Cannot see (skip this keypoint)

                **Workflow**:
                1. Adjust zoom (2-3x recommended)
                2. Select keypoint + visibility
                3. Click precise location
                4. Repeat for all keypoints
                5. Mark invisible ones with button
                6. Save when done
                """)

        # Event handlers
        def on_image_click(img_data, evt: gr.SelectData):
            if img_data is None:
                return None, "Load a frame first"

            x, y = evt.index[0], evt.index[1]
            return annotator.add_keypoint(
                x, y,
                keypoint_selector.value,
                visibility_selector.value,
                zoom_slider.value,
                point_size_slider.value
            )

        def on_load_frame(idx, zoom, point_size):
            img, msg = annotator.load_frame(int(idx), zoom, point_size)
            summary = annotator.get_keypoint_summary()
            return img, msg, summary

        def on_save():
            msg = annotator.save_current_frame()
            summary = annotator.get_keypoint_summary()
            return msg, summary

        def on_remove(kp_name, zoom, point_size):
            img, msg = annotator.remove_keypoint(kp_name, zoom, point_size)
            summary = annotator.get_keypoint_summary()
            return img, msg, summary

        def on_mark_invisible(zoom, point_size):
            img, msg = annotator.mark_not_visible(keypoint_selector.value, zoom, point_size)
            summary = annotator.get_keypoint_summary()
            return img, msg, summary

        def on_prev(current_idx, zoom, point_size):
            new_idx = max(0, int(current_idx) - 1)
            return on_load_frame(new_idx, zoom, point_size) + (new_idx,)

        def on_next(current_idx, zoom, point_size):
            new_idx = min(len(annotator.frame_files) - 1, int(current_idx) + 1)
            return on_load_frame(new_idx, zoom, point_size) + (new_idx,)

        def on_zoom_change(idx, zoom, point_size):
            return on_load_frame(idx, zoom, point_size)

        # Connect events
        image_display.select(
            on_image_click,
            inputs=[image_display],
            outputs=[image_display, status_text]
        )

        load_btn.click(
            on_load_frame,
            inputs=[frame_idx, zoom_slider, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        frame_idx.change(
            on_load_frame,
            inputs=[frame_idx, zoom_slider, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        zoom_slider.change(
            on_zoom_change,
            inputs=[frame_idx, zoom_slider, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        point_size_slider.change(
            on_zoom_change,
            inputs=[frame_idx, zoom_slider, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        save_btn.click(
            on_save,
            outputs=[status_text, summary_text]
        )

        mark_invisible_btn.click(
            on_mark_invisible,
            inputs=[zoom_slider, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        remove_btn.click(
            on_remove,
            inputs=[remove_selector, zoom_slider, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        prev_btn.click(
            on_prev,
            inputs=[frame_idx, zoom_slider, point_size_slider],
            outputs=[image_display, status_text, summary_text, frame_idx]
        )

        next_btn.click(
            on_next,
            inputs=[frame_idx, zoom_slider, point_size_slider],
            outputs=[image_display, status_text, summary_text, frame_idx]
        )

        # Auto-load first frame
        demo.load(
            on_load_frame,
            inputs=[
                gr.Number(value=0, visible=False),
                gr.Number(value=2.0, visible=False),
                gr.Number(value=3, visible=False)
            ],
            outputs=[image_display, status_text, summary_text]
        )

    return demo


def launch_annotator(frames_dir: str, output_file: str = 'keypoints.json', port: int = 7861):
    """Launch the enhanced keypoint annotation tool"""
    annotator = KeypointAnnotator(frames_dir, output_file)
    demo = create_ui(annotator)

    demo.launch(
        server_name='0.0.0.0',
        server_port=port,
        share=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Keypoint Annotation Tool")
    parser.add_argument('frames_dir', type=str, help='Directory with cropped frames')
    parser.add_argument('--output', type=str, default='keypoints.json',
                       help='Output JSON file')
    parser.add_argument('--port', type=int, default=7861,
                       help='Server port')

    args = parser.parse_args()

    launch_annotator(args.frames_dir, args.output, args.port)
