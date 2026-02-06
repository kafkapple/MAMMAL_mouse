# MAMMAL Mouse 22 Keypoint Reference

> 22개 키포인트 정의, skeleton topology, sparse 설정, 시각화 색상

**Last Updated**: 2026-02-06

---

## GT vs Model Definition 불일치

**중요**: GT annotation (`_archive/mouse_22_defs.py`, archived)과 Model definition (`keypoint22_mapper.json`)의 Head keypoint 순서가 다르다.

| Index | GT Annotation (mouse_22_defs.py, archived in `_archive/`) | Model Definition (keypoint22_mapper.json) |
|-------|----------------------------------|-------------------------------------------|
| **0** | **left_ear_tip** | nose vertex |
| **1** | **right_ear_tip** | left_ear vertex |
| **2** | **nose** | right_ear vertex |
| 3-21 | 동일 | 동일 |

**실제 데이터 사용 시**: GT annotation 기준 (`_archive/mouse_22_defs.py`, archived)을 따른다.

---

## GT Keypoint Definition (mouse_22_defs.py 기준, archived in `_archive/`)

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
| 4 | body_middle | 몸통 중앙 (실제로는 neck-tail 62.9% 지점) |

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

## DANNCE vs MAMMAL 키포인트 비교

| Index | DANNCE (Original) | MAMMAL (mouse_22_defs.py, archived in `_archive/`) | 비고 |
|-------|-------------------|---------------------------|------|
| 0 | Left Ear | left_ear_tip | 동일 |
| 1 | Right Ear | right_ear_tip | 동일 |
| 2 | Snout | nose | 동의어 |
| 3 | Anterior Spine | neck | 전방 척추 -> 목 |
| 4 | **Medial Spine** | **body_middle** | 중간 척추 (실제 몸 중심 아님) |
| 5 | Posterior Spine | tail_root | 후방 척추 -> 꼬리 시작 |
| 8 | Left Hand | left_paw | 손 -> 앞발 |
| 9 | - | left_paw_end | MAMMAL 추가 |
| 12 | Right Hand | right_paw | 손 -> 앞발 |
| 13 | - | right_paw_end | MAMMAL 추가 |

**body_middle (idx 4) 주의**: neck(3)에서 tail_root(5) 방향으로 **62.9%** 위치. 실제 몸 중심이 아니라 DANNCE의 "Medial Spine" 정의를 따른 것이다.

```
neck (3)             body_middle (4)        tail_root (5)
  |--------------------------|----------------------|
  0%                        62.9%                 100%
```

---

## Skeleton Topology (Bone Connections)

```
# _archive/mouse_22_defs.py - mouse_22_bones (archived)
Head:       [0,2], [1,2]           # ears -> nose
Spine:      [2,3], [3,4], [4,5]    # nose -> neck -> body -> tail_root
Tail:       [5,6], [6,7]           # tail_root -> tail_mid -> tail_end
L Front:    [8,9], [9,10], [10,11], [11,3]   # paw -> elbow -> shoulder -> neck
R Front:    [12,13], [13,14], [14,15], [15,3]
L Hind:     [16,17], [17,18], [18,5]         # foot -> knee -> hip -> tail_root
R Hind:     [19,20], [20,21], [21,5]
```

---

## Part-Aware Color Coding

시각화에서 신체 부위별 색상 구분:

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

## Sparse Keypoint Configurations

### 데이터셋 참조

| Dataset | Points | Reference |
|---------|--------|-----------|
| MARS/CalMS21 | 7 | Caltech, top-view behavior |
| DeepLabCut | 12 | General mouse pose |
| DANNCE | 18-22 | 3D markerless tracking |

### 프리셋

| 이름 | Config | Indices | 설명 |
|------|--------|---------|------|
| 3kp Minimal | `sixview_sparse_keypoint` | `[2, 3, 5]` | nose, neck, tail_root |
| 5kp Minimal+ | `sparse_5kp_minimal` | `[0, 1, 2, 3, 5]` | + 양쪽 귀 (머리 방향 추정) |
| 7kp MARS | `sparse_7kp_mars` | `[0, 1, 2, 3, 5, 18, 21]` | MARS 표준 호환 |
| 9kp DLC | `sparse_9kp_dlc` | `[0, 1, 2, 3, 4, 5, 6, 8, 12]` | DeepLabCut 유사 |
| 22kp Full | `baseline_6view_keypoint` | (전체) | 최고 정확도 |

### 구현 방식 비교

| 항목 | Weight 방식 | Filtering 방식 |
|------|-------------|----------------|
| `keypoint_num` | 22 | N_sparse |
| 데이터 로드 | 22개 전부 | N개만 |
| `idx_N` 의미 | 원본 인덱스 | sparse 배열 내 위치 |
| 결과 | 동등 | 동등 (더 효율적) |

---

## 파일 참조

| 파일 | 설명 |
|------|------|
| `_archive/mouse_22_defs.py` | GT annotation 정의 (archived) |
| `mouse_model/keypoint22_mapper.json` | Model vertex/joint 매핑 |
| `data/*/keypoints2d_undist/result_view_*.pkl` | GT 2D keypoints 데이터 |
| `scripts/visualize_gt_keypoints_hires.py` | GT 시각화 스크립트 |

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [FITTING_GUIDE](../guides/FITTING_GUIDE.md) | 키포인트 기반 메쉬 피팅 |
| [ANNOTATION_GUIDE](../guides/ANNOTATION_GUIDE.md) | 키포인트 어노테이션 방법 |
| [EXPERIMENTS](EXPERIMENTS.md) | 키포인트 ablation 실험 |

---

*Last updated: 2026-02-06*
