#!/usr/bin/env python3
"""
Render continuous mesh sequence video from textured OBJ files.

Generates smooth sequential video showing:
- 6-view grid: GT RGB + textured mesh side-by-side per view
- Single-view: GT vs mesh comparison

Usage:
    # All 100 frames, 6-view grid
    CUDA_VISIBLE_DEVICES=5 python scripts/render_sequence_video.py

    # Single view, faster
    CUDA_VISIBLE_DEVICES=5 python scripts/render_sequence_video.py --views 3
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Render continuous mesh sequence video")
    parser.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    parser.add_argument("--obj-dir", default="/home/joon/data/synthetic/textured_obj/")
    parser.add_argument("--output", default="results/comparison/sequence/")
    parser.add_argument("--views", nargs="+", type=int, default=[3])
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find all OBJ files sorted by frame ID
    obj_files = sorted([f for f in os.listdir(args.obj_dir) if f.endswith(".obj")])
    frame_ids = [int(f.split("frame_")[1].split(".")[0]) for f in obj_files]
    print(f"Found {len(frame_ids)} OBJ files, frames {frame_ids[0]}..{frame_ids[-1]}")

    # Load cameras and body model
    import pickle
    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams_raw = pickle.load(f)
    cams = {}
    for i, c in enumerate(cams_raw):
        cams[i] = {"K": c["K"], "R": c["R"], "T": c["T"]}

    from articulation_th import ArticulationTorch
    model = ArticulationTorch()
    faces = model.faces_vert_np

    # Load UV template for vertex colors
    uv_coords = np.loadtxt("mouse_model/mouse_txt/textures.txt")
    faces_tex = np.loadtxt("mouse_model/mouse_txt/faces_tex.txt", dtype=np.int64)
    tex_img = cv2.imread("exports/texture_final.png")
    tex_rgb = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Pre-compute vertex UV mapping
    n_verts = 14522
    vuv = np.zeros((n_verts, 2), dtype=np.float32)
    vcnt = np.zeros(n_verts, dtype=np.int32)
    for fi in range(len(faces)):
        for i in range(3):
            vuv[faces[fi, i]] += uv_coords[faces_tex[fi, i]]
            vcnt[faces[fi, i]] += 1
    valid = vcnt > 0
    vuv[valid] /= vcnt[valid, None]
    H_t, W_t = tex_rgb.shape[:2]
    px = np.clip((vuv[:, 0] * W_t).astype(int), 0, W_t - 1)
    py = np.clip(((1 - vuv[:, 1]) * H_t).astype(int), 0, H_t - 1)
    vert_colors = (tex_rgb[py, px] * 255).astype(np.uint8)

    def load_verts(path):
        v = []
        with open(path) as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split()
                    v.append([float(p[1]), float(p[2]), float(p[3])])
        return np.array(v, dtype=np.float32)

    def render_mesh(verts, view_id):
        K, R, T = cams[view_id]["K"], cams[view_id]["R"], cams[view_id]["T"]
        pc = (R @ verts.T).T + T
        p2h = (K @ pc.T).T
        d = p2h[:, 2]
        p2 = p2h[:, :2] / p2h[:, 2:]
        img = np.zeros((1024, 1152, 3), dtype=np.uint8)
        for fi in np.argsort(-d[faces].mean(axis=1)):
            f = faces[fi]
            if (d[f] < 0).any():
                continue
            fc = vert_colors[f].mean(axis=0).astype(np.uint8)
            cv2.fillPoly(img, [p2[f].astype(np.int32).reshape(-1, 1, 2)],
                         (int(fc[2]), int(fc[1]), int(fc[0])))
        return img

    def load_gt(frame_id, view_id):
        cap = cv2.VideoCapture(os.path.join(args.data_dir, f"videos_undist/{view_id}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else np.zeros((1024, 1152, 3), dtype=np.uint8)

    # Render all frames
    if len(args.views) == 1:
        # Single-view: [GT | Mesh] side-by-side
        vid = args.views[0]
        W, H = 1152 * 2, 1024
        H = H if H % 2 == 0 else H - 1
        W = W if W % 2 == 0 else W - 1
        vid_path = os.path.join(args.output, f"sequence_v{vid}.mp4")

        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", str(args.fps),
               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "23", "-preset", "fast", vid_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for i, fid in enumerate(frame_ids):
            verts = load_verts(os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj"))
            gt = load_gt(fid, vid)
            mesh = render_mesh(verts, vid)
            # Add labels
            cv2.putText(gt, f"GT frame {fid}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(mesh, "Mesh (accurate)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frame = cv2.resize(np.concatenate([gt, mesh], axis=1), (W, H))
            proc.stdin.write(frame.tobytes())
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(frame_ids)}]")

        proc.stdin.close()
        proc.wait()
        print(f"Video: {vid_path} ({len(frame_ids)} frames, {args.fps}fps)")

    else:
        # 6-view grid: 2 rows of 3 views, GT on top, Mesh on bottom
        scale = 0.35
        vW = int(1152 * scale) // 2 * 2  # Force even
        vH = int(1024 * scale) // 2 * 2
        gridW = vW * 3
        gridH = (vH * 2 + 30) * 2  # (GT row + Mesh row + header) × 2 halves
        gridH = gridH // 2 * 2
        vid_path = os.path.join(args.output, "sequence_6view.mp4")

        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-s", f"{gridW}x{gridH}", "-pix_fmt", "bgr24", "-r", str(args.fps),
               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "23", "-preset", "fast", vid_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for i, fid in enumerate(frame_ids):
            verts = load_verts(os.path.join(args.obj_dir, f"step_2_frame_{fid:06d}.obj"))
            gt_views, mesh_views = [], []
            for vid in range(6):
                gt = cv2.resize(load_gt(fid, vid), (vW, vH))
                mesh = cv2.resize(render_mesh(verts, vid), (vW, vH))
                gt_views.append(gt)
                mesh_views.append(mesh)

            # Build grid
            header_gt = np.full((30, gridW, 3), (60, 60, 60), dtype=np.uint8)
            cv2.putText(header_gt, f"GT  |  Frame {fid}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            header_mesh = np.full((30, gridW, 3), (30, 80, 30), dtype=np.uint8)
            cv2.putText(header_mesh, "Mesh (accurate)", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            grid = np.concatenate([
                header_gt,
                np.concatenate(gt_views[:3], axis=1),
                np.concatenate(gt_views[3:], axis=1),
                header_mesh,
                np.concatenate(mesh_views[:3], axis=1),
                np.concatenate(mesh_views[3:], axis=1),
            ], axis=0)
            grid = cv2.resize(grid, (gridW, gridH))
            proc.stdin.write(grid.tobytes())
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(frame_ids)}]")

        proc.stdin.close()
        proc.wait()
        print(f"Video: {vid_path} ({len(frame_ids)} frames, {args.fps}fps)")


if __name__ == "__main__":
    main()
