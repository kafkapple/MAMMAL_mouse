"""
Backward compatibility wrapper for preprocessing_utils module.

The preprocessing_utils module has been moved to mammal_ext.preprocessing.
This wrapper maintains backward compatibility with existing code.

Migration:
    # Old (deprecated)
    from preprocessing_utils.keypoint_estimation import estimate_mammal_keypoints

    # New (recommended)
    from mammal_ext.preprocessing.keypoint_estimation import estimate_mammal_keypoints
"""

import warnings
warnings.warn(
    "preprocessing_utils module has moved to mammal_ext.preprocessing. "
    "Please update imports: from mammal_ext.preprocessing import ...",
    DeprecationWarning,
    stacklevel=2
)
