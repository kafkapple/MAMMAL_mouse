#!/usr/bin/env python
"""
Interactive 3D Mesh Sequence Viewer for MAMMAL fitting results.

Features:
- Load mesh sequence from fitting results
- Interactive 3D visualization with rotation/zoom
- Frame-by-frame navigation (keyboard/slider)
- Play/pause animation
- Export to video

Usage:
    python scripts/interactive_mesh_viewer.py <results_folder>
    python scripts/interactive_mesh_viewer.py results/fitting/baseline_xxx --fps 15

Controls:
    - Space: Play/pause animation
    - Left/Right arrows: Previous/next frame
    - Home/End: First/last frame
    - R: Reset view
    - S: Save current frame as PNG
    - V: Export video
    - Q/Esc: Quit
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from articulation_th import ArticulationTorch

# Check for PyVista availability (preferred for interactive viewing)
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# Fallback to matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation


class MeshSequenceLoader:
    """Load and manage mesh sequence from fitting results."""

    def __init__(self, results_folder, device='cuda'):
        self.results_path = Path(results_folder)
        self.device = device

        # Find pkl files
        self.pkl_files = self._find_pkl_files()
        if not self.pkl_files:
            raise FileNotFoundError(f"No parameter files found in {results_folder}")

        print(f"Found {len(self.pkl_files)} frames")

        # Initialize MAMMAL model
        print("Loading MAMMAL model...")
        self.model = ArticulationTorch()
        self.model.init_params(batch_size=1)
        self.model.to(device)

        # Cache for generated meshes
        self.mesh_cache = {}

    def _find_pkl_files(self):
        """Find all parameter pkl files."""
        patterns = [
            'params/param*_sil.pkl',
            'params/param*.pkl',
        ]

        for pattern in patterns:
            files = sorted(self.results_path.glob(pattern))
            if files:
                # Prefer _sil files
                sil_files = [f for f in files if '_sil' in f.name]
                return sil_files if sil_files else files
        return []

    def __len__(self):
        return len(self.pkl_files)

    def get_frame_number(self, idx):
        """Get frame number from pkl filename."""
        stem = self.pkl_files[idx].stem
        return int(stem.replace('param', '').replace('_sil', ''))

    def load_params(self, idx):
        """Load parameters from pkl file."""
        with open(self.pkl_files[idx], 'rb') as f:
            return pickle.load(f)

    def get_mesh(self, idx):
        """Get mesh data for frame index."""
        if idx in self.mesh_cache:
            return self.mesh_cache[idx]

        params = self.load_params(idx)

        # Convert to tensors
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            elif isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                return torch.tensor(x, dtype=torch.float32, device=self.device)

        # Generate mesh
        with torch.no_grad():
            vertices, joints = self.model(
                to_tensor(params['thetas']),
                to_tensor(params['bone_lengths']),
                to_tensor(params['R']),
                to_tensor(params['T']),
                to_tensor(params['s']),
                to_tensor(params['chest_deformer'])
            )
            keypoints = self.model.forward_keypoints22()

        mesh_data = {
            'vertices': vertices[0].cpu().numpy(),
            'joints': joints[0].cpu().numpy(),
            'keypoints': keypoints[0].cpu().numpy(),
            'faces': self.model.faces_vert_np,
            'frame': self.get_frame_number(idx)
        }

        self.mesh_cache[idx] = mesh_data
        return mesh_data

    def preload_all(self, callback=None):
        """Preload all meshes into cache."""
        for i in range(len(self)):
            self.get_mesh(i)
            if callback:
                callback(i, len(self))


class PyVistaViewer:
    """Interactive viewer using PyVista (if available)."""

    def __init__(self, loader, fps=10):
        self.loader = loader
        self.fps = fps
        self.current_frame = 0
        self.playing = False

        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.add_key_event('space', self.toggle_play)
        self.plotter.add_key_event('Right', self.next_frame)
        self.plotter.add_key_event('Left', self.prev_frame)
        self.plotter.add_key_event('r', self.reset_view)

        # Add mesh actor
        self.mesh_actor = None
        self.keypoint_actor = None

        self.update_mesh()

    def update_mesh(self):
        """Update displayed mesh."""
        mesh_data = self.loader.get_mesh(self.current_frame)

        # Remove old actors
        if self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor)
        if self.keypoint_actor is not None:
            self.plotter.remove_actor(self.keypoint_actor)

        # Create PyVista mesh
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']

        # PyVista expects faces as [n, v0, v1, v2, ...]
        pv_faces = np.column_stack([
            np.full(len(faces), 3),
            faces
        ]).flatten()

        mesh = pv.PolyData(vertices, pv_faces)

        self.mesh_actor = self.plotter.add_mesh(
            mesh, color='lightblue', opacity=0.8, smooth_shading=True
        )

        # Add keypoints
        keypoints = mesh_data['keypoints']
        point_cloud = pv.PolyData(keypoints)
        self.keypoint_actor = self.plotter.add_mesh(
            point_cloud, color='red', point_size=15, render_points_as_spheres=True
        )

        self.plotter.add_text(
            f"Frame: {mesh_data['frame']} ({self.current_frame+1}/{len(self.loader)})",
            position='upper_left', font_size=12, name='frame_text'
        )

    def toggle_play(self):
        self.playing = not self.playing

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % len(self.loader)
        self.update_mesh()

    def prev_frame(self):
        self.current_frame = (self.current_frame - 1) % len(self.loader)
        self.update_mesh()

    def reset_view(self):
        self.plotter.reset_camera()

    def run(self):
        """Run the interactive viewer."""
        def callback(step):
            if self.playing:
                self.next_frame()
            return True

        self.plotter.add_callback(callback, interval=int(1000/self.fps))
        self.plotter.show()


class MatplotlibViewer:
    """Interactive viewer using Matplotlib (fallback)."""

    def __init__(self, loader, fps=10):
        self.loader = loader
        self.fps = fps
        self.current_frame = 0
        self.playing = False
        self.anim = None

        # Create figure
        self.fig = plt.figure(figsize=(14, 10))

        # 3D plot
        self.ax = self.fig.add_subplot(121, projection='3d')

        # Info panel
        self.ax_info = self.fig.add_subplot(122)
        self.ax_info.axis('off')

        # Slider
        ax_slider = plt.axes([0.15, 0.02, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, len(loader)-1,
                           valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        # Buttons
        ax_play = plt.axes([0.85, 0.02, 0.1, 0.03])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)

        # Key bindings
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_display()

    def update_display(self):
        """Update the display."""
        self.ax.clear()

        mesh_data = self.loader.get_mesh(self.current_frame)
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        keypoints = mesh_data['keypoints']

        # Plot mesh (simplified)
        step = max(1, len(faces) // 2000)
        mesh_collection = Poly3DCollection(
            vertices[faces[::step]],
            alpha=0.4,
            facecolor='lightblue',
            edgecolor='gray',
            linewidth=0.1
        )
        self.ax.add_collection3d(mesh_collection)

        # Plot keypoints
        self.ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
                       c='red', s=50, marker='o', label='Keypoints')

        # Labels for key points
        key_labels = {0: 'Nose', 21: 'Body', 18: 'Tail'}
        for idx, label in key_labels.items():
            self.ax.text(keypoints[idx, 0], keypoints[idx, 1], keypoints[idx, 2],
                        label, fontsize=8)

        # Set axis limits
        center = vertices.mean(axis=0)
        max_range = np.abs(vertices - center).max()
        self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
        self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
        self.ax.set_zlim(center[2] - max_range, center[2] + max_range)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"Frame {mesh_data['frame']} ({self.current_frame+1}/{len(self.loader)})")

        # Update info panel
        self.ax_info.clear()
        self.ax_info.axis('off')
        info_text = f"""
Frame: {mesh_data['frame']}
Index: {self.current_frame + 1} / {len(self.loader)}
Vertices: {len(vertices)}
Keypoints: {len(keypoints)}

Controls:
---------
Space: Play/Pause
Left/Right: Prev/Next frame
R: Reset view
S: Save screenshot
Q: Quit
        """
        self.ax_info.text(0.1, 0.5, info_text, fontsize=10,
                         verticalalignment='center', fontfamily='monospace')

        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        self.current_frame = int(val)
        self.update_display()

    def toggle_play(self, event=None):
        self.playing = not self.playing
        self.btn_play.label.set_text('Pause' if self.playing else 'Play')

        if self.playing:
            self.anim = animation.FuncAnimation(
                self.fig, self.animate, interval=1000//self.fps, blit=False
            )
        elif self.anim:
            self.anim.event_source.stop()

    def animate(self, frame):
        if self.playing:
            self.current_frame = (self.current_frame + 1) % len(self.loader)
            self.slider.set_val(self.current_frame)
        return []

    def on_key(self, event):
        if event.key == ' ':
            self.toggle_play()
        elif event.key == 'right':
            self.current_frame = (self.current_frame + 1) % len(self.loader)
            self.slider.set_val(self.current_frame)
        elif event.key == 'left':
            self.current_frame = (self.current_frame - 1) % len(self.loader)
            self.slider.set_val(self.current_frame)
        elif event.key == 'r':
            self.ax.view_init(elev=30, azim=45)
            self.update_display()
        elif event.key == 's':
            save_path = f"frame_{self.current_frame:05d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        elif event.key in ['q', 'escape']:
            plt.close()

    def run(self):
        """Run the interactive viewer."""
        plt.show()


def export_video(loader, output_path, fps=10):
    """Export mesh sequence as video."""
    import cv2

    print(f"Exporting video to {output_path}...")

    # Create figure for rendering
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    frames = []
    for i in range(len(loader)):
        ax.clear()

        mesh_data = loader.get_mesh(i)
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        keypoints = mesh_data['keypoints']

        # Plot mesh
        step = max(1, len(faces) // 2000)
        mesh_collection = Poly3DCollection(
            vertices[faces[::step]],
            alpha=0.4,
            facecolor='lightblue',
            edgecolor='gray',
            linewidth=0.1
        )
        ax.add_collection3d(mesh_collection)

        # Plot keypoints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
                  c='red', s=50, marker='o')

        # Set limits
        center = vertices.mean(axis=0)
        max_range = np.abs(vertices - center).max()
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_title(f"Frame {mesh_data['frame']}")

        # Render to array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        print(f"\rRendering frame {i+1}/{len(loader)}", end='')

    plt.close()

    # Write video
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"\nVideo saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Interactive mesh sequence viewer')
    parser.add_argument('results_folder', type=str, help='Path to results folder')
    parser.add_argument('--fps', type=int, default=10, help='Playback FPS')
    parser.add_argument('--export-video', type=str, default=None,
                       help='Export to video instead of interactive view')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for mesh generation')
    parser.add_argument('--preload', action='store_true',
                       help='Preload all meshes (slower start, smoother playback)')
    args = parser.parse_args()

    # Load mesh sequence
    loader = MeshSequenceLoader(args.results_folder, device=args.device)

    # Preload if requested
    if args.preload:
        print("Preloading meshes...")
        loader.preload_all(lambda i, n: print(f"\rLoading {i+1}/{n}", end=''))
        print("\nDone!")

    # Export video or run viewer
    if args.export_video:
        export_video(loader, args.export_video, args.fps)
    else:
        if PYVISTA_AVAILABLE:
            print("Using PyVista viewer (interactive)")
            viewer = PyVistaViewer(loader, args.fps)
        else:
            print("Using Matplotlib viewer (PyVista not available)")
            viewer = MatplotlibViewer(loader, args.fps)

        viewer.run()


if __name__ == '__main__':
    main()
