#!/usr/bin/env python3
"""
UV transplant: copy UV coords from template to refit OBJ files.

Refit OBJ has only vertices + faces. This script adds UV coordinates
from the body model template to create textured OBJ files compatible
with the FaceLift Neural Texture pipeline.

Usage:
    # Refit 23 bad frames (default)
    python scripts/uv_transplant_refit.py

    # Production 3600 slerp OBJs (auto-discover all frames)
    python scripts/uv_transplant_refit.py \
        --src results/fitting/production_3600_slerp/obj/ \
        --dst results/fitting/production_3600_slerp/obj_textured/

    # Explicit source/destination
    python scripts/uv_transplant_refit.py \
        --src results/fitting/refit_accurate_23/obj/ \
        --dst /home/joon/data/synthetic/textured_obj/
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Legacy: 23 bad frames from E2 refit. Used when --src has no OBJ files
# or --frames is explicitly passed.
BAD_FRAMES = [
    720, 1320, 1920, 2040, 2160, 2760, 3600,
    5160, 5520, 5880, 6000, 6120, 6960, 7200,
    8280, 8400, 9360, 9480, 9840, 10080,
    10680, 10800, 11880,
]


def load_vertices(obj_path):
    """Load vertex positions from bare OBJ."""
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                verts.append(f"v {p[1]} {p[2]} {p[3]}\n")
    return verts


def write_textured_obj(verts, uv_lines, face_lines, output_path, source_name):
    """Write OBJ with vertices + UV + textured faces."""
    with open(output_path, "w") as f:
        f.write(f"# UV-transplanted from {source_name}\n")
        f.write(f"# Vertices: {len(verts)}, UV: {len(uv_lines)}, Faces: {len(face_lines)}\n")
        for v in verts:
            f.write(v)
        for vt in uv_lines:
            f.write(vt)
        for face in face_lines:
            f.write(face)


def discover_frames(src_dir):
    """Scan src_dir and return sorted list of frame IDs from OBJ filenames."""
    obj_files = sorted([f for f in os.listdir(src_dir)
                        if f.endswith(".obj") and not f.endswith(".bak")])
    return [int(f.split("frame_")[1].split(".")[0]) for f in obj_files]


def main():
    parser = argparse.ArgumentParser(description="UV transplant for refit OBJ files")
    parser.add_argument("--src", default="results/fitting/refit_accurate_23/obj/",
                        help="Source directory with bare refit OBJs")
    parser.add_argument("--dst", default="/home/joon/data/synthetic/textured_obj/",
                        help="Destination directory for textured OBJs")
    parser.add_argument("--template", default="mouse_model/mouse_txt",
                        help="Body model template directory")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Specific frame IDs to process. If omitted, auto-discovers from --src.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")
    args = parser.parse_args()

    # Resolve frame list: explicit > auto-discover > BAD_FRAMES fallback
    if args.frames:
        frames = args.frames
        print(f"Using {len(frames)} explicitly specified frames")
    else:
        discovered = discover_frames(args.src)
        if discovered:
            frames = discovered
            print(f"Auto-discovered {len(frames)} frames from {args.src}")
        else:
            frames = BAD_FRAMES
            print(f"No OBJ files found in {args.src}, falling back to BAD_FRAMES ({len(frames)} frames)")

    # Load UV template (same for all frames — mesh topology is constant)
    uv_coords = np.loadtxt(os.path.join(args.template, "textures.txt"))
    faces_tex = np.loadtxt(os.path.join(args.template, "faces_tex.txt"), dtype=np.int64)
    faces_vert = np.loadtxt(os.path.join(args.template, "faces_vert.txt"), dtype=np.int64)

    # Pre-build UV and face lines (shared across all output OBJs)
    uv_lines = [f"vt {uv[0]:.6f} {uv[1]:.6f}\n" for uv in uv_coords]
    face_lines = [f"f {fv[0]+1}/{ft[0]+1} {fv[1]+1}/{ft[1]+1} {fv[2]+1}/{ft[2]+1}\n"
                  for fv, ft in zip(faces_vert, faces_tex)]

    os.makedirs(args.dst, exist_ok=True)

    print(f"UV template: {len(uv_lines)} UV coords, {len(face_lines)} faces")
    print(f"Source: {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Frames to process: {len(frames)}")
    print()

    success, skip, fail = 0, 0, 0
    for i, fid in enumerate(frames):
        obj_name = f"step_2_frame_{fid:06d}.obj"
        src_path = os.path.join(args.src, obj_name)
        dst_path = os.path.join(args.dst, obj_name)

        if not os.path.exists(src_path):
            print(f"  SKIP {obj_name}: source not found")
            skip += 1
            continue

        verts = load_vertices(src_path)
        if len(verts) != 14522:
            print(f"  FAIL {obj_name}: unexpected vertex count {len(verts)} (expected 14522)")
            fail += 1
            continue

        if args.dry_run:
            print(f"  DRY-RUN {obj_name}: {len(verts)}v + {len(uv_lines)}vt + {len(face_lines)}f → {dst_path}")
        else:
            # Backup existing file if present
            if os.path.exists(dst_path):
                backup = dst_path + ".bak"
                if not os.path.exists(backup):
                    os.rename(dst_path, backup)

            write_textured_obj(verts, uv_lines, face_lines, dst_path, obj_name)
            size_kb = os.path.getsize(dst_path) / 1024
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(frames)}] {obj_name}: {size_kb:.0f}KB")

        success += 1

    print(f"\nDone: {success} success, {skip} skip, {fail} fail")


if __name__ == "__main__":
    main()
