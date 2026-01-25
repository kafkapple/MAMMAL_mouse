"""
Backward compatibility wrapper for mesh_visualizer.

Usage:
    python -m visualization.mesh_visualizer --result_dir results/fitting/xxx --save_video
"""

# Re-export main entry point
from mammal_ext.visualization.mesh_visualizer import main, MeshVisualizer

if __name__ == "__main__":
    main()
