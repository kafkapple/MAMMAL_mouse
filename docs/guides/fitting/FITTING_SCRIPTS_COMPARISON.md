# Mesh Fitting Scripts 비교 가이드

**Date**: 2025-11-25

---

## 1. 두 피팅 스크립트 비교

| 항목 | `fitter_articulation.py` | `fit_cropped_frames.py` |
|-----|--------------------------|-------------------------|
| **용도** | Multi-view 3D fitting | Single-view silhouette fitting |
| **입력** | 6카메라 비디오 + 키포인트 | SAM 크롭 이미지 + 마스크 |
| **설정** | Hydra config | CLI arguments |
| **출력** | `mouse_fitting_result/` | `results/` (지정 가능) |

---

## 2. fitter_articulation.py (Multi-View)

### 2.1 활용 가능 데이터셋

| 데이터셋 | Config 이름 | 위치 |
|---------|------------|------|
| **Markerless Mouse** | `default_markerless` | `data/examples/markerless_mouse_1_nerf/` |
| **Shank3** | `shank3` | (별도 다운로드 필요) |
| **Custom** | `custom` | 사용자 지정 |

### 2.2 데이터 구조 요구사항

```
data/examples/markerless_mouse_1_nerf/
├── videos_undist/           # 6개 카메라 비디오
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
├── keypoints2d_undist/      # 2D 키포인트 (pkl)
│   ├── result_view_0.pkl
│   └── ...
├── simpleclick_undist/      # 마스크 비디오
│   └── *.mp4
├── new_cam.pkl              # 6개 카메라 캘리브레이션
└── add_labels_3d_8keypoints.pkl  # 3D GT (optional)
```

### 2.3 실행 방법

```bash
# 기본 실행 (3 프레임, 렌더링 포함)
python fitter_articulation.py \
    dataset=default_markerless \
    fitter.start_frame=0 \
    fitter.end_frame=3 \
    fitter.with_render=true

# Shell 스크립트
./run_mesh_fitting_default.sh 0 10 1 true
# 인자: start_frame, end_frame, interval, with_render
```

### 2.4 출력 구조

```
mouse_fitting_result/results_markerless_mouse_1_nerf_YYYYMMDD_HHMMSS/
├── obj/
│   ├── mesh_000000.obj      # 3D mesh (Blender/MeshLab)
│   ├── mesh_000001.obj
│   └── ...
├── params/
│   ├── param0.pkl           # 최적화 파라미터
│   └── param0_sil.pkl       # Silhouette 단계 파라미터
└── render/
    ├── debug/               # 최적화 과정 시각화
    │   └── fitting_0_global_iter_*.png
    └── fitting_keypoints_*.png  # 키포인트 비교
```

---

## 3. fit_cropped_frames.py (Single-View)

### 3.1 활용 가능 데이터셋

| 데이터셋 | 위치 | 생성 방법 |
|---------|------|----------|
| **SAM 크롭** | `data/100-KO-male-56-20200615_cropped/` | `unified_annotator.py` |
| **Custom** | 사용자 지정 | SAM + crop 처리 |

### 3.2 데이터 구조 요구사항

```
data/my_cropped_frames/
├── frame_000000_cropped.png     # 크롭된 마우스 이미지
├── frame_000000_mask.png        # Binary 마스크 (흰색=마우스)
├── frame_000000_crop_info.json  # 크롭 메타데이터
├── frame_000001_cropped.png
├── frame_000001_mask.png
├── frame_000001_crop_info.json
└── ...
```

### 3.3 실행 방법

```bash
# 기본 실행
python fit_cropped_frames.py \
    data/100-KO-male-56-20200615_cropped \
    --output-dir results/my_fitting \
    --max-frames 10

# Shell 스크립트
./run_mesh_fitting_cropped.sh \
    data/100-KO-male-56-20200615_cropped \
    results/my_fitting \
    10
# 인자: data_dir, output_dir, max_frames
```

### 3.4 출력 구조

```
results/my_fitting/
├── frame_000000/
│   ├── mesh.obj             # 3D mesh
│   ├── params.pkl           # 파라미터
│   ├── silhouette_comparison.png  # 마스크 비교
│   └── keypoints.png        # 키포인트 시각화
├── frame_000001/
│   └── ...
└── summary.json             # 전체 결과 요약
```

---

## 4. 결과 확인 및 시각화

### 4.1 3D Mesh 시각화

```bash
# MeshLab (CLI)
meshlab mouse_fitting_result/*/obj/mesh_000000.obj

# Blender (CLI)
blender --python-expr "import bpy; bpy.ops.import_scene.obj(filepath='mesh.obj')"

# Python (trimesh)
python -c "
import trimesh
mesh = trimesh.load('mouse_fitting_result/*/obj/mesh_000000.obj')
mesh.show()
"
```

### 4.2 2D 결과 이미지 확인

```bash
# 렌더링 이미지 열기
eog mouse_fitting_result/*/render/fitting_keypoints_*.png

# 또는 모든 이미지 확인
feh mouse_fitting_result/*/render/
```

### 4.3 파라미터 확인

```python
import pickle
import numpy as np

with open('mouse_fitting_result/*/params/param0.pkl', 'rb') as f:
    params = pickle.load(f)

print("Keys:", params.keys())
# thetas, bone_lengths, rotation, trans, scale, chest_deformer

print("Scale:", params['scale'])
print("Translation:", params['trans'])
```

---

## 5. Quick Reference

### 빠른 테스트

```bash
# Multi-view (3 프레임)
./run_quick_test.sh default_markerless

# Single-view (3 프레임)
./run_quick_test.sh cropped
```

### 전체 데이터 처리

```bash
# Multi-view (전체)
python fitter_articulation.py \
    dataset=default_markerless \
    fitter.end_frame=-1 \
    fitter.with_render=true

# Single-view (전체)
python fit_cropped_frames.py data/my_frames --output-dir results/full
```

---

## 6. 문제 해결

### ModuleNotFoundError: data_seaker_video_new

```bash
cp scripts/analysis/data_seaker_video_new.py .
```

### FileNotFoundError: colormaps/anliang_paper.txt

```bash
ln -s assets/colormaps colormaps
```

### GPU Out of Memory

```bash
# 프레임 수 줄이기
python fitter_articulation.py fitter.end_frame=5

# 렌더링 비활성화
python fitter_articulation.py fitter.with_render=false
```
