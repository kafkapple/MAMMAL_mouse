#!/usr/bin/env python3
"""
Combine per-view videos into a grid video + snapshot image.

Reads per-view MP4 files from a comparison directory and produces:
  - <output>/grid_NxM.mp4  — all views tiled into a single video
  - <output>/grid_frame<N>.png — snapshot at a specific frame

Layout (default 3×2 for 6 views):
  v0 | v1 | v2
  v3 | v4 | v5

Usage:
    # GT overlay 6-view grid (default)
    python scripts/render_grid_video.py \
        --input results/comparison/production_3600_slerp_gt/ \
        --output results/comparison/production_3600_slerp_gt/ \
        --prefix interpolated_v --views 0 1 2 3 4 5 --cols 3

    # Mesh-only grid
    python scripts/render_grid_video.py \
        --input results/comparison/production_3600_slerp/ \
        --output results/comparison/production_3600_slerp/ \
        --prefix view_ --views 0 1 2 3 4 5 --cols 3
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing per-view videos")
    parser.add_argument("--output", required=True, help="Output directory for grid files")
    parser.add_argument("--prefix", default="interpolated_v", help="Video filename prefix (before view id)")
    parser.add_argument("--suffix", default=".mp4", help="Video filename suffix")
    parser.add_argument("--views", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--cols", type=int, default=3, help="Grid columns")
    parser.add_argument("--scale-w", type=int, default=576, help="Per-tile width after scaling")
    parser.add_argument("--scale-h", type=int, default=256, help="Per-tile height after scaling")
    parser.add_argument("--fps", type=int, default=None, help="Output FPS (default: same as input)")
    parser.add_argument("--snapshot-frame", type=int, default=None,
                        help="Frame index for PNG snapshot (default: middle frame)")
    parser.add_argument("--crf", type=int, default=23)
    return parser.parse_args()


def make_grid_video(input_paths, output_path, cols, tile_w, tile_h, fps_arg, crf):
    n = len(input_paths)
    rows = (n + cols - 1) // cols

    # Build filter_complex
    scale_parts = []
    labels = []
    for i in range(n):
        label = chr(ord('a') + i)
        scale_parts.append(f"[{i}]scale={tile_w}:{tile_h}[{label}]")
        labels.append(f"[{label}]")

    # xstack layout: col*tile_w _ row*tile_h
    layout = "|".join(
        f"{(i % cols) * tile_w}_{(i // cols) * tile_h}"
        for i in range(n)
    )
    xstack = f"{''.join(labels)}xstack=inputs={n}:layout={layout}[out]"
    filter_complex = ";".join(scale_parts) + ";" + xstack

    cmd = []
    for p in input_paths:
        cmd += ["-i", p]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", "fast",
    ]
    if fps_arg:
        cmd += ["-r", str(fps_arg)]
    cmd += ["-y", output_path]

    print(f"Building grid video: {output_path}")
    print(f"  Layout: {cols}×{rows}, tile {tile_w}×{tile_h}, total {cols * tile_w}×{rows * tile_h}")
    result = subprocess.run(["ffmpeg"] + cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr[-2000:])
        sys.exit(1)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Done: {size_mb:.1f}MB")


def make_snapshot(input_paths, output_path, cols, tile_w, tile_h, frame_idx):
    """Extract one frame per video, tile into a PNG."""
    n = len(input_paths)

    # Extract individual frames as PNG via ffmpeg
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    try:
        frames = []
        for i, path in enumerate(input_paths):
            frame_path = os.path.join(tmpdir, f"frame_{i}.png")
            cmd = [
                "ffmpeg", "-y",
                "-i", path,
                "-vf", f"select=eq(n\\,{frame_idx}),scale={tile_w}:{tile_h}",
                "-frames:v", "1",
                "-f", "image2", frame_path
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0 or not os.path.exists(frame_path):
                print(f"  Warning: could not extract frame from {path}")
                frame_path = None
            frames.append(frame_path)

        # Tile with ffmpeg hstack/vstack
        rows = (n + cols - 1) // cols
        row_outputs = []
        for r in range(rows):
            row_frames = frames[r * cols: r * cols + cols]
            row_frames = [f for f in row_frames if f]
            if not row_frames:
                continue
            row_path = os.path.join(tmpdir, f"row_{r}.png")
            if len(row_frames) == 1:
                import shutil
                shutil.copy(row_frames[0], row_path)
            else:
                row_inputs = []
                for p in row_frames:
                    row_inputs += ["-i", p]
                cmd = ["ffmpeg", "-y"] + row_inputs + [
                    "-filter_complex", f"hstack=inputs={len(row_frames)}",
                    row_path
                ]
                subprocess.run(cmd, capture_output=True, check=True)
            row_outputs.append(row_path)

        if len(row_outputs) == 1:
            import shutil
            shutil.copy(row_outputs[0], output_path)
        else:
            stack_inputs = []
            for p in row_outputs:
                stack_inputs += ["-i", p]
            cmd = ["ffmpeg", "-y"] + stack_inputs + [
                "-filter_complex", f"vstack=inputs={len(row_outputs)}",
                output_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        size_kb = os.path.getsize(output_path) / 1e3
        print(f"  Snapshot: {output_path} ({size_kb:.0f}KB, frame {frame_idx})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Resolve input video paths
    input_paths = []
    for v in args.views:
        p = os.path.join(args.input, f"{args.prefix}{v}{args.suffix}")
        if not os.path.exists(p):
            print(f"ERROR: video not found: {p}")
            sys.exit(1)
        input_paths.append(p)
    print(f"Found {len(input_paths)} input videos")

    # Grid video
    rows = (len(args.views) + args.cols - 1) // args.cols
    grid_name = f"grid_{args.cols}x{rows}.mp4"
    grid_path = os.path.join(args.output, grid_name)
    make_grid_video(input_paths, grid_path, args.cols, args.scale_w, args.scale_h,
                    args.fps, args.crf)

    # Snapshot PNG
    snap_frame = args.snapshot_frame
    if snap_frame is None:
        # Detect total frames from first video
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", input_paths[0]],
            capture_output=True, text=True
        )
        import json
        streams = json.loads(result.stdout)["streams"]
        video_stream = next(s for s in streams if s["codec_type"] == "video")
        total = int(video_stream.get("nb_frames", 3600))
        snap_frame = total // 2
    snap_name = f"grid_frame{snap_frame:05d}.png"
    snap_path = os.path.join(args.output, snap_name)
    make_snapshot(input_paths, snap_path, args.cols, args.scale_w, args.scale_h, snap_frame)


if __name__ == "__main__":
    main()
