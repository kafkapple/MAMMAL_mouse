"""
Compare old OpenCV preprocessing vs new SAM preprocessing
"""
import pickle
import numpy as np
import cv2

print("="*60)
print("PREPROCESSING COMPARISON")
print("="*60)

# Load old preprocessing keypoints
old_kpts_path = "data/preprocessed_shank3/keypoints2d_undist/result_view_0.pkl"
with open(old_kpts_path, 'rb') as f:
    old_kpts = pickle.load(f)

print(f"\n[OLD] OpenCV Preprocessing:")
print(f"  Shape: {old_kpts.shape}")
print(f"  First frame keypoints (first 5):")
print(old_kpts[0, :5])
print(f"  Mean confidence: {np.mean(old_kpts[:, :, 2]):.3f}")
print(f"  Keypoints at zero: {np.sum(np.all(old_kpts[:, :, :2] == 0, axis=2))} / {old_kpts.shape[0] * old_kpts.shape[1]}")

# Load new SAM preprocessing keypoints
new_kpts_path = "data/preprocessed_shank3_sam/keypoints2d_undist/result_view_0.pkl"
with open(new_kpts_path, 'rb') as f:
    new_kpts = pickle.load(f)

print(f"\n[NEW] SAM Preprocessing:")
print(f"  Shape: {new_kpts.shape}")
print(f"  First frame keypoints (first 5):")
print(new_kpts[0, :5])
print(f"  Mean confidence: {np.mean(new_kpts[:, :, 2]):.3f}")
print(f"  Keypoints at zero: {np.sum(np.all(new_kpts[:, :, :2] == 0, axis=2))} / {new_kpts.shape[0] * new_kpts.shape[1]}")

# Check masks
print(f"\n[MASKS] Comparison:")
old_mask_cap = cv2.VideoCapture("data/preprocessed_shank3/simpleclick_undist/0.mp4")
ret, old_mask = old_mask_cap.read()
if ret:
    old_mask_gray = cv2.cvtColor(old_mask, cv2.COLOR_BGR2GRAY) if len(old_mask.shape) == 3 else old_mask
    print(f"  Old mask coverage: {np.sum(old_mask_gray > 0) / old_mask_gray.size * 100:.1f}%")
old_mask_cap.release()

new_mask_cap = cv2.VideoCapture("data/preprocessed_shank3_sam/simpleclick_undist/0.mp4")
ret, new_mask = new_mask_cap.read()
if ret:
    new_mask_gray = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY) if len(new_mask.shape) == 3 else new_mask
    print(f"  New SAM mask coverage: {np.sum(new_mask_gray > 0) / new_mask_gray.size * 100:.1f}%")
new_mask_cap.release()

print(f"\n[IMPROVEMENT]:")
conf_improvement = (np.mean(new_kpts[:, :, 2]) - np.mean(old_kpts[:, :, 2])) / np.mean(old_kpts[:, :, 2]) * 100
print(f"  Confidence: {conf_improvement:+.1f}%")

old_zeros = np.sum(np.all(old_kpts[:, :, :2] == 0, axis=2))
new_zeros = np.sum(np.all(new_kpts[:, :, :2] == 0, axis=2))
print(f"  Zero keypoints: {old_zeros} → {new_zeros} ({(new_zeros - old_zeros):+d})")
print("="*60)
