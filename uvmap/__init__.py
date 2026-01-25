"""
Backward compatibility wrapper for uvmap module.

The uvmap module has been moved to mammal_ext.uvmap.
This wrapper maintains backward compatibility with existing code.

Migration:
    # Old (deprecated)
    from uvmap import UVPipeline

    # New (recommended)
    from mammal_ext.uvmap import UVPipeline
"""

# Re-export from mammal_ext.uvmap
from mammal_ext.uvmap import *

import warnings
warnings.warn(
    "uvmap module has moved to mammal_ext.uvmap. "
    "Please update imports: from mammal_ext.uvmap import ...",
    DeprecationWarning,
    stacklevel=2
)
