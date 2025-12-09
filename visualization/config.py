"""
Visualization Configuration Module

Dataclasses for UV-textured mesh visualization settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class VisualizationConfig:
    """Configuration for UV-textured mesh visualization."""

    # ========== Input Paths ==========
    result_dir: str
    """Fitting result directory containing params/, obj/, uvmap/"""

    texture_path: Optional[str] = None
    """Path to UV texture image. Auto-detected if None (result_dir/uvmap/texture_final.png)"""

    model_dir: str = 'mouse_model/mouse_txt'
    """Directory containing mesh model files (vertices, faces, textures)"""

    # ========== Frame Selection ==========
    start_frame: int = 0
    """Starting frame index"""

    end_frame: int = -1
    """Ending frame index. -1 means all available frames"""

    frame_interval: int = 1
    """Process every N-th frame"""

    # ========== Rendering Settings ==========
    image_size: Tuple[int, int] = (1024, 1024)
    """Rendered image size (width, height)"""

    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Background color RGB in [0, 1]"""

    # ========== Camera/View Settings ==========
    view_modes: List[str] = field(default_factory=lambda: ['orbit'])
    """View modes to render: 'orbit', 'multiview', 'fixed', 'novel'"""

    # Orbit view settings
    orbit_frames: int = 120
    """Number of frames for 360-degree rotation"""

    orbit_elevation: float = 30.0
    """Camera elevation angle in degrees"""

    orbit_distance: float = 0.5
    """Distance from mesh center (relative to mesh scale)"""

    # Multi-view settings (original cameras from fitting)
    views_to_use: Optional[List[int]] = None
    """Camera indices to use for multi-view. None means all available"""

    # Fixed view settings
    fixed_views: List[str] = field(default_factory=lambda: ['front', 'side', 'top', 'diagonal'])
    """Fixed viewpoints: 'front', 'back', 'side', 'top', 'diagonal'"""

    # Novel view settings
    novel_azimuths: List[float] = field(default_factory=list)
    """Azimuth angles for novel views (degrees)"""

    novel_elevations: List[float] = field(default_factory=list)
    """Elevation angles for novel views (degrees)"""

    # ========== Output Settings ==========
    output_dir: Optional[str] = None
    """Output directory. Auto: result_dir/visualization/"""

    save_rrd: bool = True
    """Save Rerun RRD file"""

    save_video: bool = True
    """Save MP4 video"""

    save_frames: bool = False
    """Save individual PNG frames"""

    video_fps: int = 30
    """Video frame rate"""

    video_codec: str = 'mp4v'
    """Video codec for OpenCV"""

    # ========== Debug/Overlay Settings ==========
    show_wireframe: bool = False
    """Show mesh wireframe overlay"""

    show_keypoints: bool = False
    """Show 3D keypoints"""

    show_skeleton: bool = False
    """Show skeleton bone connections"""

    show_original_images: bool = True
    """Include original camera images in output"""

    # ========== Rerun Settings ==========
    rerun_app_name: str = 'mammal_mouse_visualization'
    """Rerun application name"""

    def __post_init__(self):
        """Validate and resolve paths."""
        self.result_dir = str(Path(self.result_dir).resolve())

        # Auto-detect texture path
        if self.texture_path is None:
            auto_path = Path(self.result_dir) / 'uvmap' / 'texture_final.png'
            if auto_path.exists():
                self.texture_path = str(auto_path)

        # Auto-set output directory
        if self.output_dir is None:
            self.output_dir = str(Path(self.result_dir) / 'visualization')

        # Resolve model directory
        if not Path(self.model_dir).is_absolute():
            # Try relative to MAMMAL_mouse root
            import os
            script_dir = Path(__file__).parent.parent
            full_path = script_dir / self.model_dir
            if full_path.exists():
                self.model_dir = str(full_path)

    @property
    def uvmap_dir(self) -> str:
        """UV map output directory."""
        return str(Path(self.result_dir) / 'uvmap')

    @property
    def params_dir(self) -> str:
        """Fitting parameters directory."""
        return str(Path(self.result_dir) / 'params')

    @property
    def obj_dir(self) -> str:
        """OBJ mesh files directory."""
        return str(Path(self.result_dir) / 'obj')

    def ensure_output_dir(self) -> Path:
        """Create output directory if it doesn't exist."""
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        return output


@dataclass
class CameraConfig:
    """Camera parameters for rendering."""

    position: Tuple[float, float, float]
    """Camera position in world coordinates"""

    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look-at target point"""

    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Camera up vector (Z-up for mouse mesh)"""

    fov: float = 45.0
    """Field of view in degrees"""

    near: float = 0.01
    """Near clipping plane"""

    far: float = 10.0
    """Far clipping plane"""

    name: str = 'camera'
    """Camera identifier"""


@dataclass
class RenderOutput:
    """Container for render outputs."""

    image: 'np.ndarray'
    """Rendered RGB image (H, W, 3)"""

    depth: Optional['np.ndarray'] = None
    """Depth map (H, W)"""

    mask: Optional['np.ndarray'] = None
    """Foreground mask (H, W)"""

    camera_name: str = ''
    """Camera identifier"""

    frame_idx: int = 0
    """Frame index"""
