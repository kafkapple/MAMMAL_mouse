"""
MAMMAL Fitting Module Extensions

This module contains extensions and utilities for the MAMMAL fitting pipeline
that are separated from the original code for maintainability.

Modules:
    - debug: Debug image grid collection and compression
"""

from .debug import (
    DebugGridCollector,
    compress_existing_debug_folder,
    create_iteration_grid_from_folder,
)

__all__ = [
    'DebugGridCollector',
    'compress_existing_debug_folder',
    'create_iteration_grid_from_folder',
]
