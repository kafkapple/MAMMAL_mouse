"""
Debug image grid compression utility.

Instead of saving individual high-resolution debug images per iteration,
this module collects images and saves them as a compact grid.
"""

import cv2
import numpy as np
from pathlib import Path


class DebugGridCollector:
    """Collects debug images and saves as compressed grid."""

    def __init__(self,
                 thumbnail_size=(320, 240),
                 grid_cols=5,
                 jpeg_quality=85):
        """
        Args:
            thumbnail_size: (width, height) for each thumbnail
            grid_cols: Number of columns in grid
            jpeg_quality: JPEG compression quality (0-100)
        """
        self.thumbnail_size = thumbnail_size
        self.grid_cols = grid_cols
        self.jpeg_quality = jpeg_quality
        self.images = {}  # {step_name: [(iter, image), ...]}

    def add_image(self, step_name: str, iteration: int, image: np.ndarray):
        """Add a debug image for later grid compilation.

        Args:
            step_name: 'Step0', 'Step1', etc.
            iteration: Iteration number
            image: BGR image (numpy array)
        """
        if step_name not in self.images:
            self.images[step_name] = []

        # Resize to thumbnail
        thumb = cv2.resize(image, self.thumbnail_size, interpolation=cv2.INTER_AREA)
        self.images[step_name].append((iteration, thumb))

    def save_grid(self, output_path: str, step_name: str = None):
        """Save collected images as grid(s).

        Args:
            output_path: Base path for output (e.g., 'debug/step_0_frame_000000')
            step_name: If specified, only save this step. Otherwise save all.
        """
        steps_to_save = [step_name] if step_name else list(self.images.keys())

        for step in steps_to_save:
            if step not in self.images or not self.images[step]:
                continue

            # Sort by iteration
            sorted_imgs = sorted(self.images[step], key=lambda x: x[0])

            # Create grid
            grid = self._create_grid(sorted_imgs)

            # Save as JPEG for compression
            step_lower = step.lower()
            path = f"{output_path}_{step_lower}_grid.jpg"

            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

            # Clear saved images to free memory
            self.images[step] = []

        return path

    def save_all_grids(self, base_path: str, frame_id: int):
        """Save all step grids for a frame.

        Args:
            base_path: Base directory (e.g., 'results/.../render/debug')
            frame_id: Frame number
        """
        saved_paths = []
        for step_name in list(self.images.keys()):
            if self.images[step_name]:
                path = f"{base_path}/{step_name.lower()}_frame_{frame_id:06d}_grid.jpg"
                self._save_single_grid(step_name, path)
                saved_paths.append(path)
        return saved_paths

    def _save_single_grid(self, step_name: str, output_path: str):
        """Save a single step's grid."""
        if step_name not in self.images or not self.images[step_name]:
            return None

        sorted_imgs = sorted(self.images[step_name], key=lambda x: x[0])
        grid = self._create_grid(sorted_imgs)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        # Clear to free memory
        self.images[step_name] = []
        return output_path

    def _create_grid(self, sorted_imgs):
        """Create grid image from sorted (iter, image) list."""
        n_images = len(sorted_imgs)
        n_cols = min(self.grid_cols, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        thumb_w, thumb_h = self.thumbnail_size
        header_h = 20  # Space for iteration label

        # Create canvas
        grid_w = n_cols * thumb_w
        grid_h = n_rows * (thumb_h + header_h)
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        grid[:] = (40, 40, 40)  # Dark gray background

        for idx, (iteration, img) in enumerate(sorted_imgs):
            row = idx // n_cols
            col = idx % n_cols

            x = col * thumb_w
            y = row * (thumb_h + header_h)

            # Draw iteration label
            label = f"iter {iteration}"
            cv2.putText(grid, label, (x + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Place thumbnail
            grid[y + header_h:y + header_h + thumb_h, x:x + thumb_w] = img

        return grid

    def clear(self):
        """Clear all collected images."""
        self.images = {}


def create_iteration_grid_from_folder(debug_folder: str,
                                       output_path: str,
                                       step_pattern: str = "step_*_frame_*_iter_*.png",
                                       thumbnail_size=(320, 240),
                                       grid_cols=5):
    """Create grid from existing debug images in a folder.

    Useful for post-processing existing results.

    Args:
        debug_folder: Path to debug folder with iteration images
        output_path: Output grid image path
        step_pattern: Glob pattern for finding images
        thumbnail_size: Size of each thumbnail
        grid_cols: Number of columns

    Returns:
        Path to created grid image, or None if no images found
    """
    import glob
    import re

    folder = Path(debug_folder)
    images = sorted(folder.glob(step_pattern))

    if not images:
        return None

    collector = DebugGridCollector(thumbnail_size, grid_cols)

    for img_path in images:
        # Extract iteration from filename
        match = re.search(r'iter_(\d+)', img_path.name)
        if match:
            iteration = int(match.group(1))
            # Extract step name
            step_match = re.search(r'(step_\d)', img_path.name)
            step_name = step_match.group(1) if step_match else "step"

            img = cv2.imread(str(img_path))
            if img is not None:
                collector.add_image(step_name, iteration, img)

    # Save grid
    for step_name in collector.images:
        if collector.images[step_name]:
            collector._save_single_grid(step_name, output_path.replace('.jpg', f'_{step_name}.jpg'))

    return output_path


def compress_existing_debug_folder(debug_folder: str,
                                    delete_originals: bool = False):
    """Compress all debug images in a folder to grids.

    Args:
        debug_folder: Path to debug folder
        delete_originals: If True, delete original PNG files after compression

    Returns:
        List of created grid paths
    """
    import glob
    import re
    from collections import defaultdict

    folder = Path(debug_folder)

    # Group by frame
    frame_images = defaultdict(lambda: defaultdict(list))

    for img_path in folder.glob("step_*_frame_*_iter_*.png"):
        # Extract frame and step
        frame_match = re.search(r'frame_(\d+)', img_path.name)
        step_match = re.search(r'(step_\d)', img_path.name)
        iter_match = re.search(r'iter_(\d+)', img_path.name)

        if frame_match and step_match and iter_match:
            frame_id = int(frame_match.group(1))
            step_name = step_match.group(1)
            iteration = int(iter_match.group(1))

            frame_images[frame_id][step_name].append((iteration, str(img_path)))

    created_grids = []
    files_to_delete = []

    for frame_id, steps in frame_images.items():
        collector = DebugGridCollector()

        for step_name, img_list in steps.items():
            for iteration, img_path in sorted(img_list):
                img = cv2.imread(img_path)
                if img is not None:
                    collector.add_image(step_name, iteration, img)
                    files_to_delete.append(img_path)

        # Save grids for this frame
        paths = collector.save_all_grids(str(folder), frame_id)
        created_grids.extend(paths)

    # Delete originals if requested
    if delete_originals and created_grids:
        for f in files_to_delete:
            Path(f).unlink(missing_ok=True)
        print(f"Deleted {len(files_to_delete)} original files")

    return created_grids
