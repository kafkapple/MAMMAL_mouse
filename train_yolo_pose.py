"""
Train YOLOv8-Pose model on MAMMAL mouse keypoints

Usage:
    python train_yolo_pose.py --epochs 10 --batch 8 --imgsz 256
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8-Pose for mouse keypoints')
    parser.add_argument('--data', type=str, default='data/yolo_mouse_pose/data.yaml',
                        help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt',
                        help='Pretrained model (yolov8n-pose, yolov8s-pose, yolov8m-pose)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=256,
                        help='Image size')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (0, 1, ...) or cpu')
    parser.add_argument('--project', type=str, default='runs/pose',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='mammal_mouse_yolo',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')

    args = parser.parse_args()

    # Check CUDA
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
            args.device = 'cpu'
        else:
            print(f"✅ Using CUDA device: {args.device}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\n📦 Loading model: {args.model}")
    model = YOLO(args.model)

    # Display model info
    print(f"   Model type: Pose estimation")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M")

    # Training configuration
    print(f"\n🎯 Training configuration:")
    print(f"   Dataset: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Device: {args.device}")

    # Train
    print(f"\n🚀 Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Additional settings
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        # Optimization
        optimizer='Adam',
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentation (light for small dataset)
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.2,
        degrees=10,
        translate=0.1,
        scale=0.2,
        shear=5,
        flipud=0.0,  # No vertical flip for mice
        fliplr=0.5,  # Horizontal flip with keypoint swapping
        mosaic=0.5,
    )

    # Validation
    print(f"\n📊 Running validation...")
    metrics = model.val()

    # Print results
    print(f"\n✅ Training complete!")
    print(f"   Best model: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print(f"   Last model: {Path(args.project) / args.name / 'weights' / 'last.pt'}")
    print(f"\n   Validation metrics:")
    if hasattr(metrics, 'box'):
        print(f"      Box mAP50: {metrics.box.map50:.4f}")
        print(f"      Box mAP50-95: {metrics.box.map:.4f}")
    if hasattr(metrics, 'pose'):
        print(f"      Pose mAP50: {metrics.pose.map50:.4f}")
        print(f"      Pose mAP50-95: {metrics.pose.map:.4f}")

    print(f"\n🎉 Done! Model ready for inference.")
    print(f"\nTo use the trained model:")
    print(f"   from ultralytics import YOLO")
    print(f"   model = YOLO('{Path(args.project) / args.name / 'weights' / 'best.pt'}')")
    print(f"   results = model.predict('image.png')")


if __name__ == '__main__':
    main()
