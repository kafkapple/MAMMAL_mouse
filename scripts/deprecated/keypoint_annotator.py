"""
Simple Keypoint Annotation Tool using Gradio
Similar to SAM annotator but for keypoint annotation
"""
import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


class KeypointAnnotator:
    """Simple keypoint annotation tool"""

    # Define essential keypoints for mouse (11 keypoints total)
    KEYPOINT_NAMES = [
        # Head & Spine (5)
        'nose',
        'neck',
        'spine_mid',
        'hip',
        'tail_base',
        # Ears (2)
        'left_ear',
        'right_ear',
        # Limbs (4) - NEW!
        'left_front_paw',
        'right_front_paw',
        'left_hind_paw',
        'right_hind_paw',
    ]

    # Visibility states
    VISIBILITY = {
        'visible': 1.0,      # Clearly visible
        'occluded': 0.5,     # Partially visible/uncertain
        'not_visible': 0.0   # Not visible/skip
    }

    # Colors for each keypoint (RGB format for Gradio)
    KEYPOINT_COLORS = {
        # Head & Spine
        'nose': (255, 0, 0),           # Red
        'neck': (255, 165, 0),         # Orange
        'spine_mid': (255, 255, 0),    # Yellow
        'hip': (0, 255, 0),            # Green
        'tail_base': (0, 0, 255),      # Blue
        # Ears
        'left_ear': (255, 0, 255),     # Magenta
        'right_ear': (0, 255, 255),    # Cyan
        # Limbs (NEW!)
        'left_front_paw': (255, 192, 203),   # Pink
        'right_front_paw': (255, 192, 203),  # Pink
        'left_hind_paw': (147, 112, 219),    # Purple
        'right_hind_paw': (147, 112, 219),   # Purple
    }

    def __init__(self, frames_dir: str, output_file: str = 'keypoints.json'):
        """
        Initialize annotator

        Args:
            frames_dir: Directory containing cropped frames
            output_file: JSON file to save annotations
        """
        self.frames_dir = Path(frames_dir)
        self.output_file = Path(output_file)

        # Find all cropped frames
        self.frame_files = sorted(self.frames_dir.glob('*_cropped.png'))

        # Load existing annotations if any
        self.annotations = self.load_annotations()

        # Current state
        self.current_frame_idx = 0
        self.current_image = None
        self.current_keypoints = {}  # {name: (x, y, visibility)}
        self.current_keypoint_name = self.KEYPOINT_NAMES[0]

        # Display settings
        self.zoom_level = 1.0
        self.point_size = 3  # Default smaller size

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

    def load_frame(self, frame_idx: int, point_size: int = 3) -> Tuple[np.ndarray, str]:
        """
        Load a frame and its existing keypoints

        Args:
            frame_idx: Index of frame to load
            point_size: Size of keypoint markers

        Returns:
            image: Image with keypoints drawn
            message: Status message
        """
        if frame_idx < 0 or frame_idx >= len(self.frame_files):
            return None, f"Invalid frame index: {frame_idx}"

        self.current_frame_idx = frame_idx
        self.point_size = point_size
        frame_file = self.frame_files[frame_idx]

        # Load image
        self.current_image = cv2.imread(str(frame_file))
        if self.current_image is None:
            return None, f"Failed to load image: {frame_file}"

        # Convert BGR to RGB for Gradio
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

        # Get frame name
        frame_name = frame_file.stem.replace('_cropped', '')

        # Load existing keypoints for this frame
        if frame_name in self.annotations:
            self.current_keypoints = {
                name: (kp['x'], kp['y'])
                for name, kp in self.annotations[frame_name].items()
            }
        else:
            self.current_keypoints = {}

        # Draw keypoints
        vis_image = self.draw_keypoints(self.current_image.copy(), point_size)

        num_kps = len(self.current_keypoints)
        total_kps = len(self.KEYPOINT_NAMES)

        return vis_image, f"Loaded frame {frame_idx + 1}/{len(self.frame_files)} - {frame_name} ({num_kps}/{total_kps} keypoints)"

    def draw_keypoints(self, image: np.ndarray, point_size: int = 3) -> np.ndarray:
        """Draw current keypoints on image"""
        for name, (x, y) in self.current_keypoints.items():
            color = self.KEYPOINT_COLORS.get(name, (255, 255, 255))

            # Draw circle with adjustable size
            cv2.circle(image, (int(x), int(y)), point_size, color, -1)
            cv2.circle(image, (int(x), int(y)), point_size + 2, (0, 0, 0), 2)  # Black border

            # Draw label
            cv2.putText(
                image, name,
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA
            )

        return image

    def add_keypoint(self, x: int, y: int, keypoint_name: str, point_size: int = 3) -> Tuple[np.ndarray, str]:
        """
        Add or update a keypoint

        Args:
            x, y: Coordinates
            keypoint_name: Name of keypoint
            point_size: Size of keypoint markers

        Returns:
            image: Updated visualization
            message: Status message
        """
        if self.current_image is None:
            return None, "No frame loaded"

        # Add keypoint
        self.current_keypoints[keypoint_name] = (x, y)
        self.point_size = point_size

        # Redraw
        vis_image = self.draw_keypoints(self.current_image.copy(), point_size)

        num_kps = len(self.current_keypoints)
        total_kps = len(self.KEYPOINT_NAMES)

        return vis_image, f"Added {keypoint_name} at ({x}, {y}) - ({num_kps}/{total_kps})"

    def remove_keypoint(self, keypoint_name: str, point_size: int = 3) -> Tuple[np.ndarray, str]:
        """Remove a keypoint"""
        self.point_size = point_size
        if keypoint_name in self.current_keypoints:
            del self.current_keypoints[keypoint_name]
            vis_image = self.draw_keypoints(self.current_image.copy(), point_size)
            return vis_image, f"Removed {keypoint_name}"
        else:
            return self.draw_keypoints(self.current_image.copy(), point_size), f"{keypoint_name} not found"

    def save_current_frame(self) -> str:
        """Save keypoints for current frame"""
        if self.current_frame_idx >= len(self.frame_files):
            return "No frame loaded"

        frame_name = self.frame_files[self.current_frame_idx].stem.replace('_cropped', '')

        # Convert to dict format
        keypoints_dict = {
            name: {'x': float(x), 'y': float(y), 'visibility': 1.0}
            for name, (x, y) in self.current_keypoints.items()
        }

        self.annotations[frame_name] = keypoints_dict
        self.save_annotations()

        num_kps = len(self.current_keypoints)
        return f"✅ Saved {num_kps} keypoints for {frame_name}"

    def get_keypoint_summary(self) -> str:
        """Get summary of current annotations"""
        total_frames = len(self.frame_files)
        annotated_frames = len(self.annotations)

        summary = f"**Progress**: {annotated_frames}/{total_frames} frames\n\n"
        summary += "**Current Frame Keypoints**:\n"

        for name in self.KEYPOINT_NAMES:
            if name in self.current_keypoints:
                x, y = self.current_keypoints[name]
                summary += f"✅ {name}: ({x:.0f}, {y:.0f})\n"
            else:
                summary += f"❌ {name}: Not set\n"

        return summary


def create_ui(annotator: KeypointAnnotator):
    """Create Gradio UI"""

    with gr.Blocks(title="Mouse Keypoint Annotator") as demo:
        gr.Markdown("# 🐭 Mouse Keypoint Annotation Tool")
        gr.Markdown("Click on the image to mark keypoint locations")

        with gr.Row():
            with gr.Column(scale=2):
                # Main image display
                image_display = gr.Image(
                    label="Click to add keypoint",
                    type="numpy",
                    interactive=True
                )

                with gr.Row():
                    keypoint_selector = gr.Radio(
                        choices=annotator.KEYPOINT_NAMES,
                        value=annotator.KEYPOINT_NAMES[0],
                        label="Select Keypoint to Add"
                    )

                status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
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

                load_btn = gr.Button("📂 Load Frame", variant="primary")
                save_btn = gr.Button("💾 Save Keypoints", variant="primary")

                gr.Markdown("### Display Settings")

                point_size_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=3,
                    step=1,
                    label="⚫ Point Size"
                )

                gr.Markdown("### Quick Actions")

                remove_selector = gr.Dropdown(
                    choices=annotator.KEYPOINT_NAMES,
                    label="Remove Keypoint"
                )
                remove_btn = gr.Button("🗑️ Remove Selected")

                gr.Markdown("### Summary")
                summary_text = gr.Markdown(annotator.get_keypoint_summary())

                gr.Markdown("### Instructions")
                gr.Markdown("""
                1. **Load Frame**: Click load or use slider
                2. **Select Keypoint**: Choose from radio buttons
                3. **Click Image**: Click on mouse body part location
                4. **Repeat**: For all keypoints
                5. **Save**: Click save when done with frame
                6. **Next Frame**: Move to next frame

                **Essential Keypoints (11 total)**:

                *Core (5 - highest priority):*
                - 🔴 nose: Tip of nose/snout
                - 🟠 neck: Base of skull
                - 🟡 spine_mid: Middle of spine
                - 🟢 hip: Hip/pelvis region
                - 🔵 tail_base: Start of tail

                *Limbs (4 - high priority):*
                - 💗 left/right_front_paw: Front paws
                - 💜 left/right_hind_paw: Hind paws

                *Optional (2):*
                - 🟣 left/right_ear: Ear tips

                **Target: 7-9 keypoints per frame**
                """)

        # Event handlers
        def on_image_click(keypoint_name, point_size, img_data, evt: gr.SelectData):
            """Handle image click to add keypoint"""
            if img_data is None:
                return None, "Load a frame first", annotator.get_keypoint_summary()

            x, y = evt.index[0], evt.index[1]
            img, msg = annotator.add_keypoint(x, y, keypoint_name, int(point_size))
            summary = annotator.get_keypoint_summary()
            return img, msg, summary

        def on_load_frame(idx, point_size):
            """Load frame and update summary"""
            img, msg = annotator.load_frame(int(idx), int(point_size))
            summary = annotator.get_keypoint_summary()
            return img, msg, summary

        def on_save():
            """Save current frame"""
            msg = annotator.save_current_frame()
            summary = annotator.get_keypoint_summary()
            return msg, summary

        def on_remove(kp_name, point_size):
            """Remove keypoint"""
            img, msg = annotator.remove_keypoint(kp_name, int(point_size))
            summary = annotator.get_keypoint_summary()
            return img, msg, summary

        def on_prev(current_idx, point_size):
            """Go to previous frame"""
            new_idx = max(0, int(current_idx) - 1)
            return on_load_frame(new_idx, point_size) + (new_idx,)

        def on_next(current_idx, point_size):
            """Go to next frame"""
            new_idx = min(len(annotator.frame_files) - 1, int(current_idx) + 1)
            return on_load_frame(new_idx, point_size) + (new_idx,)

        def on_point_size_change(idx, point_size):
            """Handle point size change"""
            return on_load_frame(idx, point_size)

        # Connect events
        image_display.select(
            on_image_click,
            inputs=[keypoint_selector, point_size_slider, image_display],
            outputs=[image_display, status_text, summary_text]
        )

        load_btn.click(
            on_load_frame,
            inputs=[frame_idx, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        frame_idx.change(
            on_load_frame,
            inputs=[frame_idx, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        point_size_slider.change(
            on_point_size_change,
            inputs=[frame_idx, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        save_btn.click(
            on_save,
            outputs=[status_text, summary_text]
        )

        remove_btn.click(
            on_remove,
            inputs=[remove_selector, point_size_slider],
            outputs=[image_display, status_text, summary_text]
        )

        prev_btn.click(
            on_prev,
            inputs=[frame_idx, point_size_slider],
            outputs=[image_display, status_text, summary_text, frame_idx]
        )

        next_btn.click(
            on_next,
            inputs=[frame_idx, point_size_slider],
            outputs=[image_display, status_text, summary_text, frame_idx]
        )

        # Auto-load first frame
        demo.load(
            on_load_frame,
            inputs=[gr.Number(value=0, visible=False), gr.Number(value=3, visible=False)],
            outputs=[image_display, status_text, summary_text]
        )

    return demo


def launch_annotator(frames_dir: str, output_file: str = 'keypoints.json', port: int = 7861):
    """
    Launch the keypoint annotation tool

    Args:
        frames_dir: Directory containing cropped frames
        output_file: Where to save annotations
        port: Server port
    """
    annotator = KeypointAnnotator(frames_dir, output_file)
    demo = create_ui(annotator)

    demo.launch(
        server_name='0.0.0.0',
        server_port=port,
        share=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Keypoint Annotation Tool")
    parser.add_argument('frames_dir', type=str, help='Directory with cropped frames')
    parser.add_argument('--output', type=str, default='keypoints.json',
                       help='Output JSON file')
    parser.add_argument('--port', type=int, default=7861,
                       help='Server port')

    args = parser.parse_args()

    launch_annotator(args.frames_dir, args.output, args.port)
