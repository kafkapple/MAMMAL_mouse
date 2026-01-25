"""GPU configuration utilities.

This module handles automatic GPU detection and environment configuration.
Extracted from fitter_articulation.py to enable reuse across scripts.

Usage:
    # At the TOP of any script that uses GPU (before torch import):
    from mammal_ext.config import configure_gpu
    configure_gpu()

    import torch  # Now safe to import
"""

import os
import socket
from typing import Optional

# Server-specific GPU defaults
# Add new servers here as needed
GPU_DEFAULTS = {
    'gpu05': '1',   # gpu05: use GPU 1 (GPU 0 reserved)
    'bori': '0',    # bori: use GPU 0 (only 1 GPU available)
}


def get_default_gpu() -> str:
    """Get default GPU ID based on hostname.

    Returns:
        str: GPU ID to use ('0', '1', etc.)
    """
    hostname = socket.gethostname().split('.')[0]
    return GPU_DEFAULTS.get(hostname, '0')


def configure_gpu(gpu_id: Optional[str] = None) -> str:
    """Configure GPU environment variables for PyTorch and rendering.

    This function MUST be called before importing torch or any rendering
    libraries. It sets:
    - CUDA_VISIBLE_DEVICES: Which GPU PyTorch sees
    - EGL_DEVICE_ID: Which GPU EGL rendering uses
    - PYOPENGL_PLATFORM: Use EGL for headless rendering
    - DISPLAY: Disabled for headless mode

    Args:
        gpu_id: Explicit GPU ID to use. If None, checks:
            1. GPU_ID environment variable
            2. CUDA_VISIBLE_DEVICES environment variable
            3. Hostname-based default from GPU_DEFAULTS

    Returns:
        str: The configured GPU ID.

    Example:
        >>> from mammal_ext.config import configure_gpu
        >>> configure_gpu()  # Auto-detect
        '1'
        >>> configure_gpu('0')  # Force GPU 0
        '0'
    """
    if gpu_id is None:
        gpu_id = os.environ.get(
            'GPU_ID',
            os.environ.get('CUDA_VISIBLE_DEVICES', get_default_gpu())
        )

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['EGL_DEVICE_ID'] = gpu_id
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['DISPLAY'] = ''  # Disable X11 for headless rendering

    return gpu_id
