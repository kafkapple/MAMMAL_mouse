# MAMMAL Mouse 22 Keypoint Reference

**Last Updated**: 2025-12-03

---

## GT vs Model Definition 불일치

**중요**: GT annotation (`mouse_22_defs.py`)과 Model definition (`keypoint22_mapper.json`)의 Head keypoint 순서가 다릅니다.

| Index | GT Annotation (mouse_22_defs.py) | Model Definition (keypoint22_mapper.json) |
|-------|----------------------------------|-------------------------------------------|
| **0** | **left_ear_tip** | nose vertex |
| **1** | **right_ear_tip** | left_ear vertex |
| **2** | **nose** | right_ear vertex |
| 3-21 | 동일 | 동일 |

**실제 데이터 사용 시**: GT annotation 기준 (mouse_22_defs.py)을 따릅니다.

---

## GT Keypoint Definition (mouse_22_defs.py 기준)

이것이 **실제 GT annotation 데이터**의 정의입니다.

### Head (idx 0-2)
| Index | GT Label | Description |
|-------|----------|-------------|
| 0 | **L_ear** | 왼쪽 귀 끝 (left_ear_tip) |
| 1 | **R_ear** | 오른쪽 귀 끝 (right_ear_tip) |
| 2 | **nose** | 코 (nose) |

### Body (idx 3-4)
| Index | GT Label | Description |
|-------|----------|-------------|
| 3 | neck | 목 |
| 4 | body_middle | 몸통 중앙 |

### Tail (idx 5-7)
| Index | GT Label | Description |
|-------|----------|-------------|
| 5 | tail_root | 꼬리 기저부 |
| 6 | tail_middle | 꼬리 중간 |
| 7 | tail_end | 꼬리 끝 |

### Left Front Limb (idx 8-11)
| Index | GT Label | Description |
|-------|----------|-------------|
| 8 | left_paw | 왼쪽 앞발 |
| 9 | left_paw_end | 왼쪽 앞발 끝 |
| 10 | left_elbow | 왼쪽 팔꿈치 |
| 11 | left_shoulder | 왼쪽 어깨 |

### Right Front Limb (idx 12-15)
| Index | GT Label | Description |
|-------|----------|-------------|
| 12 | right_paw | 오른쪽 앞발 |
| 13 | right_paw_end | 오른쪽 앞발 끝 |
| 14 | right_elbow | 오른쪽 팔꿈치 |
| 15 | right_shoulder | 오른쪽 어깨 |

### Left Hind Limb (idx 16-18)
| Index | GT Label | Description |
|-------|----------|-------------|
| 16 | left_foot | 왼쪽 뒷발 |
| 17 | left_knee | 왼쪽 무릎 |
| 18 | left_hip | 왼쪽 엉덩이 |

### Right Hind Limb (idx 19-21)
| Index | GT Label | Description |
|-------|----------|-------------|
| 19 | right_foot | 오른쪽 뒷발 |
| 20 | right_knee | 오른쪽 무릎 |
| 21 | right_hip | 오른쪽 엉덩이 |

---

## Part-Aware Color Coding

시각화에서 신체 부위별 색상 구분 (index 범위 기반):

| Body Part | Index Range | Color |
|-----------|-------------|-------|
| Head | 0, 1, 2 | Yellow |
| Body | 3, 4 | Magenta |
| Tail | 5, 6, 7 | Orange |
| Left Front Limb | 8, 9, 10, 11 | Blue |
| Right Front Limb | 12, 13, 14, 15 | Green |
| Left Hind Limb | 16, 17, 18 | Cyan |
| Right Hind Limb | 19, 20, 21 | Red |

---

## Skeleton Topology (Bone Connections)

```
# mouse_22_defs.py - mouse_22_bones
Head:       [0,2], [1,2]           # ears → nose
Spine:      [2,3], [3,4], [4,5]    # nose → neck → body → tail_root
Tail:       [5,6], [6,7]           # tail_root → tail_mid → tail_end
L Front:    [8,9], [9,10], [10,11], [11,3]   # paw → elbow → shoulder → neck
R Front:    [12,13], [13,14], [14,15], [15,3]
L Hind:     [16,17], [17,18], [18,5]         # foot → knee → hip → tail_root
R Hind:     [19,20], [20,21], [21,5]
```

---

## Sparse Keypoint Configurations

### Dataset References

| Dataset | Points | Reference |
|---------|--------|-----------|
| [MARS/CalMS21](https://neuroethology.github.io/MARS/) | 7 | Caltech, top-view behavior |
| [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) | 12 | General mouse pose |
| [DANNCE](https://github.com/spoonsso/dannce) | 18-22 | 3D markerless tracking |

---

### 3 Keypoints (Minimal) - `sixview_sparse_keypoint`
```yaml
sparse_keypoint_indices: [2, 3, 5]  # nose, neck, tail_root
```
- 최소한의 annotation으로 기본 pose 추정
- 머리 방향 정보 부족

### 5 Keypoints (Minimal+) - `sparse_5kp_minimal`
```yaml
sparse_keypoint_indices: [0, 1, 2, 3, 5]  # L_ear, R_ear, nose, neck, tail_root
```
- **양쪽 귀 추가** → 머리 방향/회전 추정 가능
- MARS dataset의 핵심 subset

### 7 Keypoints (MARS-style) - `sparse_7kp_mars`
```yaml
sparse_keypoint_indices: [0, 1, 2, 3, 5, 18, 21]  # ears, nose, neck, tail, hips
```
- [MARS/CalMS21](https://neuroethology.github.io/MARS/) 표준과 호환
- Social behavior 분석에 최적화
- Hip 포인트로 body orientation 보강

### 9 Keypoints (DeepLabCut-style) - `sparse_9kp_dlc`
```yaml
sparse_keypoint_indices: [0, 1, 2, 3, 4, 5, 6, 8, 12]
# L_ear, R_ear, nose, neck, body, tail_root, tail_mid, L_paw, R_paw
```
- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) 표준과 유사
- 균형잡힌 정확도 vs annotation 비용
- 앞발 포함으로 locomotion 분석 가능

### 22 Keypoints (Full) - `baseline_6view_keypoint`
```yaml
# All keypoints with default weights
use_keypoints: true
# No sparse_keypoint_indices (use all)
```
- 전체 skeleton reconstruction
- 최고 정확도, 높은 annotation 비용

---

### Keypoint 수 vs 예상 정확도

```
Keypoints:   3      5      7       9      22
            ┌──────┬──────┬───────┬──────┬──────┐
Accuracy:   │ Low  │ Med- │ Med   │ Med+ │ High │
            └──────┴──────┴───────┴──────┴──────┘
Annotation: │ 1x   │ 1.7x │ 2.3x  │ 3x   │ 7x   │
            └──────┴──────┴───────┴──────┴──────┘
```

---

## 파일 참조

| 파일 | 설명 |
|------|------|
| `mouse_22_defs.py` | GT annotation 정의 (keypoint_names, bones) |
| `mouse_model/keypoint22_mapper.json` | Model vertex/joint 매핑 |
| `data/*/keypoints2d_undist/result_view_*.pkl` | GT 2D keypoints 데이터 |
| `scripts/visualize_gt_keypoints_hires.py` | GT 시각화 스크립트 |

---

## 시각화 결과

고해상도 GT keypoint 시각화:
```
results/keypoint_visualization_hires/
├── view_0_gt_hires.png ~ view_5_gt_hires.png  # 개별 뷰
├── all_views_grid_frame0.png                   # 6개 뷰 그리드
└── keypoint_legend_hires.png                   # 색상 범례
```
