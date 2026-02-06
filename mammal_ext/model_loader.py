"""Centralized loader for the MAMMAL ArticulationTorch body model.

Isolates the reverse dependency on the original MAMMAL codebase
(articulation_th.py) into a single location. All mammal_ext modules
should import from here instead of directly from articulation_th.

Usage:
    from mammal_ext.model_loader import load_body_model
    body_model = load_body_model()
    V, J = body_model.forward(thetas, bone_lengths, rotation, trans, scale, ...)
"""

import os
import sys
from typing import Optional

_body_model_cache: Optional[object] = None


def load_body_model(use_cache: bool = True):
    """Load ArticulationTorch body model with project root auto-detection.

    Args:
        use_cache: If True, reuse previously loaded model instance.

    Returns:
        ArticulationTorch instance.

    Raises:
        ImportError: If articulation_th.py is not found in project root.
    """
    global _body_model_cache

    if use_cache and _body_model_cache is not None:
        return _body_model_cache

    # Ensure project root is in sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from articulation_th import ArticulationTorch
    except ImportError:
        raise ImportError(
            "articulation_th.py not found. Ensure mammal_ext is inside "
            "the MAMMAL_mouse project directory."
        )

    model = ArticulationTorch()

    if use_cache:
        _body_model_cache = model

    return model
