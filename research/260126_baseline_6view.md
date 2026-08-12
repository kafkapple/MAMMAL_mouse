# 260126 연구노트 — 6-View 22-Keypoint Baseline Fitting (3600 Frames)

## 목표
- MAMMAL 원본 논문 설정 (6-view, 22 keypoints) 재현, 전체 3600 프레임 fitting
- Downstream task (FaceLift, Pose-Splatter, Temporal deformation) 위한 baseline 데이터 확보

## 진행 내용

### 1. Input Data

**Source**: `/home/joon/data/raw/markerless_mouse_1_nerf/`

| 항목 | 스펙 |
|------|------|
| RGB | 6 views, 18000 frames, 1152x1024, 100fps (`videos_undist/0-5.mp4`) |
| Mask | 6 views, SimpleClick segmentation (`simpleclick_undist/0-5.mp4`) |
| 2D KP | 6 views, 22 keypoints, (x, y, confidence), float64 (`keypoints2d_undist/result_view_*.pkl`) |
| Camera | 6-view intrinsics + extrinsics (`camera_params.h5`) |
| Model | 14,522 vertices, 28,800 faces, 20 bones, 140 joints (`mouse_model/mouse.pkl`) |

**Frame sampling**: interval=5, 18000/5 = **3600 fitting frames** (0, 5, 10, ..., 17995)

> **Warning**: Raw data에 sudden jumps 존재 — frames 5900, 11800, 17700 (DANNCE dataset issue)

### 2. 3-Stage Optimization Pipeline

```
Step 0: Global Initialization (60 iters)
  - PCA init, scale/rotation/translation

Step 1: Coarse Fitting (5 iters)
  - Joint angles + translation + bone lengths

Step 2: Fine Fitting (3 iters, iter_multiplier=2.0)
  - + Mask loss (w=3000) + Temporal smoothness + Chest deformer
```

### 3. Loss Weights

| Loss | Weight | Role |
|------|--------|------|
| theta | 3.0 | Joint angle regularization |
| 3d | 2.5 | 3D keypoint loss |
| 2d | 0.2 | 2D reprojection loss |
| bone | 0.5 | Bone length consistency |
| scale | 0.5 | Scale regularization |
| chest_deformer | 0.1 | Chest deformation reg |
| stretch | 1.0 | Stretch penalty |
| temp | 0.25 | Temporal smoothness (theta) |
| temp_d | 0.2 | Temporal smoothness (deformer) |
| mask (Step 2) | 3000.0 | Silhouette loss |

### 4. Keypoint-specific Weights

| Keypoint | Weight | Note |
|----------|--------|------|
| idx 4 (tail base) | 0.4 | Low confidence |
| idx 5 (spine mid) | 2.0 | High importance |
| idx 6, 7 (spine) | 1.5 | - |
| idx 11, 15 (limbs) | 0.9 | - |
| tail (Step 2) | 10.0 | Precise tail fit |

### 5. Output

**Result path**: `/home/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/`
**Symlink**: `results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249`

| Output | Count | Size | Description |
|--------|-------|------|-------------|
| OBJ meshes | 3600 | 3.3GB total | Per-frame 3D mesh |
| PKL params Step 1 | 3600 | 13MB total | Coarse fitting result |
| PKL params Step 2 | 3600 | 13MB total | Fine fitting result |
| config.yaml | 1 | - | Full experiment config |
| loss_history.json | 1 | - | 7201 entries (frame x step) |
| loss_history.png | 1 | 338KB | Loss visualization |

### 6. PKL Parameter Structure (per frame)

| Key | Shape | Description |
|-----|-------|-------------|
| thetas | 1 x 140 x 3 | 140 joints, axis-angle |
| trans | 1 x 3 | Global translation (x, y, z) |
| scale | 1 x 1 | Body scale factor |
| rotation | 1 x 3 | Global rotation (axis-angle) |
| bone_lengths | 1 x 20 | 20 bone segment lengths |
| chest_deformer | 1 x 1 | Chest deformation |

### 7. OBJ Mesh Spec

| Property | Value |
|----------|-------|
| Vertices | 14,522 |
| Faces | 28,800 (triangles) |
| Unit | mm |
| Coordinate | MAMMAL default (Y-up) |

### 8. Mouse Skeleton (20 Bones)

```
pelvis
  +-- femur - tibia - hind_paw - hind_paw_palm      (x2 L/R)
  +-- scapula - humerus - ulna - fore_paw - front_paw_palm (x2 L/R)
  +-- vertebrae - neck_stretch - head - snout
  |                                    +-- ear
  +-- belly_stretch
  +-- tail - tail_end
```

### 9. Loss Convergence

| Metric | First frame (Step 0) | Last frame (Step 2) |
|--------|---------------------|---------------------|
| total_loss | 508.9 | 366.8 |
| 2d | 611.8 | 227.8 |
| trans_norm | 85.9 | 47.4 |
| theta_norm | 13.6 | 17.5 |
| mask | 0.0 | 0.08 |

## 핵심 발견
- 3600 프레임 전체 fitting 완료 — OBJ 3.3GB, params 26MB
- Temporal propagation 효과: 후반 프레임 loss 안정적 수렴
- Raw data의 sudden jumps (5900, 11800, 17700 frames)는 DANNCE dataset issue
- mask loss (w=3000) Step 2에서만 활성화 → 미세 silhouette 조정

## Downstream Use Cases

1. **FaceLift GS-LRM 비교**: MAMMAL mesh 렌더링 vs GS-LRM geometry 비교
2. **Temporal deformation GT**: 프레임 간 mesh 변형을 temporal loss 설계 참조로 활용
3. **Pose-Splatter 연동**: 3D joint positions → keypoint supervision
4. **Visualization**: OBJ sequence → Blender/Rerun animation rendering

## Reproduction

```bash
cd /home/joon/dev/MAMMAL_mouse
source ~/anaconda3/etc/profile.d/conda.sh && conda activate mammal_stable

# Run fitting (gpu03: GPU 4-7 only)
CUDA_VISIBLE_DEVICES=4 python fit.py \
    fitter.start_frame=0 fitter.end_frame=18000 fitter.interval=5 \
    data.views_to_use=[0,1,2,3,4,5] fitter.keypoint_num=22

# Resume from checkpoint
./run_experiment.sh baseline_6view_keypoint \
    --resume_from results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249
```

## 미해결 / 다음 단계
- Sudden jump 프레임의 fitting 품질 확인
- UV texture mapping 적용 (251210 연구 기반)
- Temporal consistency 정량 평가

---
*Sources: 260126_fitting_baseline_6view_kp22.md*
