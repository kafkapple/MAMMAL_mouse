# Dataset Specification

> markerless_mouse_1_nerf 데이터셋 상세 스펙

---

## Basic Info

| 항목 | 값 |
|------|-----|
| 데이터셋 이름 | markerless_mouse_1_nerf |
| 원본 FPS | 100 fps |
| 총 길이 | 180초 |
| 총 프레임 | 18,000 |
| 해상도 | 1024 × 1152 (H × W) |
| 카메라 수 | 6 |

---

## Camera Setup

### View IDs

| View | ID | 위치 설명 |
|------|-----|----------|
| 0 | cam0 | 기준 카메라 |
| 1 | cam1 | ~60° 회전 |
| 2 | cam2 | ~120° 회전 |
| 3 | cam3 | 반대편 (180°) |
| 4 | cam4 | ~240° 회전 |
| 5 | cam5 | ~300° 회전 |

### Camera Parameters

| 항목 | 형식 |
|------|------|
| 파일 | `new_cam.pkl` |
| 내용 | K (intrinsic), R (rotation), T (translation) |
| 좌표계 | OpenCV (RDF: Right-Down-Forward) |

---

## Frame Configurations

### pose-splatter 연동 (interval=5)

| Config | 프레임 수 | 용도 |
|--------|-----------|------|
| aligned_posesplatter | 3,600 | 전체 (pose-splatter 정렬) |
| aligned_test_100 | 100 | 표준 테스트 |
| quick_test_30 | 30 | 빠른 검증 |

**Frame alignment**:
```
MAMMAL frame 0 → raw frame 0 → pose-splatter sample 0
MAMMAL frame 1 → raw frame 5 → pose-splatter sample 1
...
MAMMAL frame 3599 → raw frame 17995 → pose-splatter sample 3599
```

---

## Data Structure

```
data/examples/markerless_mouse_1_nerf/
├── videos_undist/           # 왜곡 보정된 비디오
│   ├── 0.mp4               # View 0
│   ├── 1.mp4               # View 1
│   └── ...
├── new_cam.pkl              # 카메라 파라미터
├── keypoints2d_undist/      # 2D 키포인트 (GT)
│   └── *.npz
└── masks/                   # SAM 마스크
    └── *.png
```

---

## Keypoint Data

### 2D Keypoints

| 파일 | Shape | 설명 |
|------|-------|------|
| `keypoints_2d.npy` | (3600, 6, 22, 2) | [T, V, K, 2] |
| `keypoints_2d_confidence.npy` | (3600, 6, 22) | [T, V, K] |

### 3D Keypoints (피팅 결과)

| 파일 | Shape | 설명 |
|------|-------|------|
| `params/*.pkl` | (22, 3) per frame | 3D 키포인트 좌표 |

---

## Related Documents

- [KEYPOINTS.md](KEYPOINTS.md) - 22 키포인트 정의
- [CONFIG.md](CONFIG.md) - Config 파라미터
- [../OPTIMIZATION_GUIDE.md](../OPTIMIZATION_GUIDE.md) - 피팅 설정

---

*Last updated: 2026-01-25*
