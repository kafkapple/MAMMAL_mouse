"""
Direct launcher for SAM Annotator GUI
Avoids Hydra conflicts by directly launching Gradio
"""
import sys
from pathlib import Path
import argparse

# Add SAM annotator to path
sam_annotator_path = Path.home() / 'dev/mouse-super-resolution'
sys.path.insert(0, str(sam_annotator_path))

# Import after adding path
from omegaconf import OmegaConf


def create_config(frames_dir, annotations_dir, checkpoint, port):
    """Create configuration for SAM annotator"""
    config = {
        'model': {
            'name': 'sam2.1_hiera_large',
            'checkpoint': str(checkpoint),
            'config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
            'device': 'cuda'
        },
        'data': {
            'input_dir': str(frames_dir),
            'pattern': '*.png',
            'output_dir': str(annotations_dir)
        },
        'ui': {
            'server_name': '0.0.0.0',
            'server_port': int(port),
            'share': False,
            'theme': 'default'
        },
        'annotation': {
            'multimask_output': True,
            'points_per_side': None,
            'pred_iou_thresh': 0.88,
            'stability_score_thresh': 0.95,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 1
        },
        'visualization': {
            'point_size': 8,
            'point_border': 2,
            'foreground_color': [0, 255, 0],
            'background_color': [255, 0, 0],
            'mask_alpha': 0.3
        },
        'advanced': {
            'auto_save': False,
            'save_mask_image': True,
            'save_visualization': True,
            'max_frames': None
        }
    }

    return OmegaConf.create(config)


def main():
    parser = argparse.ArgumentParser(description="Launch SAM Annotator GUI")
    parser.add_argument('--frames-dir', type=str, required=True,
                       help='Directory containing frames')
    parser.add_argument('--annotations-dir', type=str, default=None,
                       help='Output directory for annotations (default: frames_dir/annotations)')
    parser.add_argument('--checkpoint', type=str,
                       default=str(Path.home() / 'dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt'),
                       help='Path to SAM checkpoint')
    parser.add_argument('--port', type=int, default=7860,
                       help='Server port')

    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else frames_dir / 'annotations'
    checkpoint = Path(args.checkpoint)

    # Verify paths
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return

    if not checkpoint.exists():
        print(f"Error: SAM checkpoint not found: {checkpoint}")
        print("Please download SAM checkpoints first:")
        print("  cd ~/dev/segment-anything-2/checkpoints")
        print("  ./download_ckpts.sh")
        return

    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Print info
    print("="*80)
    print("SAM Annotator Launcher")
    print("="*80)
    print(f"Frames: {frames_dir}")
    print(f"Annotations: {annotations_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Port: {args.port}")
    print("="*80)
    print()
    print("Access the web interface:")
    print(f"  Local: http://localhost:{args.port}")
    print(f"  SSH tunnel: ssh -L {args.port}:localhost:{args.port} joon@server")
    print("="*80)
    print()

    # Create config
    cfg = create_config(frames_dir, annotations_dir, checkpoint, args.port)

    # Import and launch annotator
    try:
        from sam_annotator.app import launch_annotator
        launch_annotator(cfg)
    except Exception as e:
        print(f"Error launching annotator: {e}")
        print("\nTrying alternative method...")

        # Alternative: direct import
        from sam_annotator.annotator import WebAnnotator
        from sam_annotator.app import create_ui

        annotator = WebAnnotator(cfg)
        demo = create_ui(annotator, cfg)

        demo.launch(
            server_name=cfg.ui.server_name,
            server_port=cfg.ui.server_port,
            share=cfg.ui.share
        )


if __name__ == "__main__":
    main()
