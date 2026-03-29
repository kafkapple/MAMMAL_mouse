#!/usr/bin/env python3
"""Compare IoU: baseline vs warm-start for frame 10080."""
import cv2, numpy as np, pickle, torch, sys, os
sys.path.insert(0, '.')

with open('data/raw/markerless_mouse_1_nerf/new_cam.pkl', 'rb') as f:
    cams = pickle.load(f)

from articulation_th import ArticulationTorch
art = ArticulationTorch()

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def render_silhouette(params, view_id):
    dev = art.tpose_joints_th.device
    thetas = torch.FloatTensor(to_np(params['thetas'])).to(dev)
    bl = torch.FloatTensor(to_np(params['bone_lengths'])).to(dev)
    rot = torch.FloatTensor(to_np(params['rotation'])).to(dev)
    trans = torch.FloatTensor(to_np(params['trans'])).to(dev)
    scale = torch.FloatTensor(to_np(params['scale'])).to(dev)
    chest = torch.FloatTensor(to_np(params['chest_deformer'])).to(dev)
    V, J = art.forward(thetas, bl, rot, trans, scale, chest)
    verts = V[0].detach().cpu().numpy()
    faces = art.faces_vert_np
    K, R, T = cams[view_id]['K'], cams[view_id]['R'], cams[view_id]['T']
    pc = (R @ verts.T).T + T
    p2h = (K @ pc.T).T
    d = p2h[:, 2]
    p2 = p2h[:, :2] / p2h[:, 2:]
    mask = np.zeros((1024, 1152), dtype=np.uint8)
    for fi in range(len(faces)):
        f = faces[fi]
        if (d[f] < 0).any():
            continue
        cv2.fillPoly(mask, [p2[f].astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask

def compute_iou(pred_mask, gt_mask):
    pred = pred_mask > 128
    gt = gt_mask > 128
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + 1e-6)

def load_gt_mask(frame_id, view_id):
    cap = cv2.VideoCapture(f'data/raw/markerless_mouse_1_nerf/simpleclick_undist/{view_id}.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    return frame[:,:,0] if ret else np.zeros((1024, 1152), dtype=np.uint8)

fid = 10080
with open('results/fitting/production_keyframes_part3/params/step_2_frame_010080.pkl', 'rb') as f:
    baseline_params = pickle.load(f)
with open('results/fitting/rearing_test_exp_a/params/step_2_frame_010080.pkl', 'rb') as f:
    exp_a_params = pickle.load(f)
with open('results/fitting/rearing_test_exp_b/params/step_2_frame_010080.pkl', 'rb') as f:
    exp_b_params = pickle.load(f)

print(f'=== Frame {fid} IoU Comparison ===')
print(f'{"View":<6} {"Baseline":>10} {"ExpA(warm)":>12} {"ExpB(rear)":>12} {"dA":>7} {"dB":>7}')
print('-' * 58)
base_ious, a_ious, b_ious = [], [], []
for vid in range(6):
    gt = load_gt_mask(fid, vid)
    bi = compute_iou(render_silhouette(baseline_params, vid), gt)
    ai = compute_iou(render_silhouette(exp_a_params, vid), gt)
    bii = compute_iou(render_silhouette(exp_b_params, vid), gt)
    base_ious.append(bi)
    a_ious.append(ai)
    b_ious.append(bii)
    da = ai - bi
    db = bii - bi
    print(f'  v{vid:<4} {bi:>10.4f} {ai:>12.4f} {bii:>12.4f} {da:>+7.3f} {db:>+7.3f}')
bm = np.mean(base_ious)
am = np.mean(a_ious)
bbm = np.mean(b_ious)
print('-' * 58)
print(f'  Mean  {bm:>10.4f} {am:>12.4f} {bbm:>12.4f} {am-bm:>+7.3f} {bbm-bm:>+7.3f}')
print(f'\nExp A (warm-start): {"IMPROVED" if am-bm > 0.02 else "NO IMPROVEMENT"} ({am-bm:+.4f})')
print(f'Exp B (rearing init): {"IMPROVED" if bbm-bm > 0.02 else "NO IMPROVEMENT"} ({bbm-bm:+.4f})')
