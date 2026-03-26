#!/usr/bin/env python3
"""
Render smooth video by interpolating vertices between sparse OBJ keyframes.

Takes N OBJ files at sparse intervals and generates smooth video by
linearly interpolating vertex positions between adjacent keyframes.

Usage:
    # Interpolate 100 PoC frames (step=120) → smooth 400-frame video
    CUDA_VISIBLE_DEVICES=5 python scripts/render_interpolated_video.py \
        --obj-dir /home/joon/data/synthetic/textured_obj/ \
        --interp-factor 4 --views 3 --fps 10
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_verts(path):
    v = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                v.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(v, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/markerless_mouse_1_nerf/")
    parser.add_argument("--obj-dir", default="/home/joon/data/synthetic/textured_obj/")
    parser.add_argument("--output", default="results/comparison/sequence_interpolated/")
    parser.add_argument("--views", nargs="+", type=int, default=[3])
    parser.add_argument("--interp-factor", type=int, default=4,
                        help="Interpolation steps between keyframes")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load OBJs
    obj_files = sorted([f for f in os.listdir(args.obj_dir) if f.endswith(".obj") and not f.endswith(".bak")])
    frame_ids = [int(f.split("frame_")[1].split(".")[0]) for f in obj_files]
    print(f"Loading {len(frame_ids)} keyframes...")
    all_verts = {}
    for fid, fname in zip(frame_ids, obj_files):
        all_verts[fid] = load_verts(os.path.join(args.obj_dir, fname))

    # Setup rendering
    import pickle
    with open(os.path.join(args.data_dir, "new_cam.pkl"), "rb") as f:
        cams_raw = pickle.load(f)
    cams = {i: c for i, c in enumerate(cams_raw)}

    from articulation_th import ArticulationTorch
    faces = ArticulationTorch().faces_vert_np

    uv_coords = np.loadtxt("mouse_model/mouse_txt/textures.txt")
    faces_tex = np.loadtxt("mouse_model/mouse_txt/faces_tex.txt", dtype=np.int64)
    tex_img = cv2.imread("exports/texture_final.png")
    tex_rgb = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    n_v = 14522
    vuv = np.zeros((n_v, 2), dtype=np.float32)
    vcnt = np.zeros(n_v, dtype=np.int32)
    for fi in range(len(faces)):
        for i in range(3):
            vuv[faces[fi, i]] += uv_coords[faces_tex[fi, i]]
            vcnt[faces[fi, i]] += 1
    valid = vcnt > 0
    vuv[valid] /= vcnt[valid, None]
    px = np.clip((vuv[:, 0] * tex_rgb.shape[1]).astype(int), 0, tex_rgb.shape[1] - 1)
    py = np.clip(((1 - vuv[:, 1]) * tex_rgb.shape[0]).astype(int), 0, tex_rgb.shape[0] - 1)
    vert_colors = (tex_rgb[py, px] * 255).astype(np.uint8)

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

    # Generate interpolated frames
    N = args.interp_factor
    total_frames = (len(frame_ids) - 1) * N + 1
    print(f"Generating {total_frames} frames ({len(frame_ids)} keyframes × {N} interp)")

    for vid in args.views:
        W, H = 1152 * 2, 1024
        W = W // 2 * 2
        H = H // 2 * 2
        vid_path = os.path.join(args.output, f"interpolated_v{vid}.mp4")

        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", str(args.fps),
               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "23", "-preset", "fast", vid_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        count = 0
        for k in range(len(frame_ids) - 1):
            fid_a, fid_b = frame_ids[k], frame_ids[k + 1]
            va, vb = all_verts[fid_a], all_verts[fid_b]

            for s in range(N):
                alpha = s / N
                verts = va * (1 - alpha) + vb * alpha
                # Interpolate GT frame ID for approximate GT
                gt_fid = int(fid_a + (fid_b - fid_a) * alpha)

                gt = load_gt(gt_fid, vid)
                mesh = render_mesh(verts, vid)
                cv2.putText(gt, f"GT f{gt_fid}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(mesh, f"Mesh (interp {alpha:.1f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                frame = cv2.resize(np.concatenate([gt, mesh], axis=1), (W, H))
                proc.stdin.write(frame.tobytes())
                count += 1

            if (k + 1) % 10 == 0:
                print(f"  [{k+1}/{len(frame_ids)-1}] keyframes, {count} total frames")

        # Last keyframe
        verts = all_verts[frame_ids[-1]]
        gt = load_gt(frame_ids[-1], vid)
        mesh = render_mesh(verts, vid)
        cv2.putText(gt, f"GT f{frame_ids[-1]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(mesh, "Mesh (keyframe)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frame = cv2.resize(np.concatenate([gt, mesh], axis=1), (W, H))
        proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait()
        print(f"Video: {vid_path} ({count+1} frames, {args.fps}fps)")


if __name__ == "__main__":
    main()
