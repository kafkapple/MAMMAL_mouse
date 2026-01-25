# Output Format Reference

> 피팅 결과 파일 형식 및 사용법

---

## Output Structure

```
results/fitting/{experiment_name}/
├── config.yaml              # 실험 설정
├── loss_history.json        # Loss 기록
├── loss_history.png         # Loss 그래프
├── loss_per_frame.png       # 프레임별 Loss
│
├── obj/                     # 3D 메쉬 (OBJ)
│   ├── step_2_frame_000000.obj
│   ├── step_2_frame_000005.obj
│   └── ...
│
├── params/                  # 모델 파라미터 (Pickle)
│   ├── step_2_frame_000000.pkl
│   └── ...
│
└── render/                  # 렌더링 이미지 (옵션)
    ├── step_0/              # 초기화 단계
    ├── step_1/              # 트래킹 단계
    └── step_2/              # 최종 단계
```

---

## OBJ Files

### 형식

```
# Wavefront OBJ
v x y z          # 정점 좌표
vt u v           # UV 좌표
vn nx ny nz      # 법선 벡터
f v1/vt1/vn1 ... # 면 정의
```

### 사용법

```python
import trimesh

# 메쉬 로드
mesh = trimesh.load("obj/step_2_frame_000000.obj")

# 정점 접근
vertices = mesh.vertices  # (N, 3)
faces = mesh.faces        # (F, 3)
```

---

## Params Files (PKL)

### 내용

| 키 | Shape | 설명 |
|-----|-------|------|
| `theta` | (72,) | Body pose (24 joints × 3 axis-angle) |
| `trans` | (3,) | Global translation |
| `scale` | (1,) | Scale factor |
| `keypoints_3d` | (22, 3) | 3D 키포인트 좌표 |

### 사용법

```python
import pickle

with open("params/step_2_frame_000000.pkl", "rb") as f:
    params = pickle.load(f)

keypoints_3d = params["keypoints_3d"]  # (22, 3)
```

---

## Downstream Usage

### pose-splatter 연동

```python
# 메쉬를 3D prior로 사용
mesh = trimesh.load(f"obj/step_2_frame_{idx:06d}.obj")
vertices = mesh.vertices  # 3D Gaussian 초기화용
```

### Blender Export

```bash
python scripts/export_to_blender.py \
    --mesh results/fitting/exp/obj/step_2_frame_000000.obj \
    --texture results/uvmap/texture_final.png \
    --output exports/mouse_textured.obj
```

### Rerun 시각화

```bash
python scripts/export_to_rerun.py \
    --result_dir results/fitting/exp \
    --output exports/sequence.rrd
```

---

## Related Documents

- [DATASET.md](DATASET.md) - 입력 데이터 형식
- [../guides/output.md](../guides/output.md) - 상세 출력 가이드

---

*Last updated: 2026-01-25*
