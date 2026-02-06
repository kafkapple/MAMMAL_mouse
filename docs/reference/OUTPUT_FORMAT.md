# Output Format Reference

> 피팅 결과 파일 형식, Loss 해석, downstream 활용법

---

## Output Structure

```
results/fitting/{experiment_name}/
├── config.yaml              # 실험 설정 (재현용)
├── loss_history.json        # Loss 기록
├── loss_history.png         # Loss 그래프
├── loss_per_frame.png       # 프레임별 Loss
│
├── obj/                     # 3D 메쉬 (OBJ)
│   ├── step_2_frame_000000.obj
│   └── ...
│
├── params/                  # 모델 파라미터 (Pickle)
│   ├── step_1_frame_000000.pkl
│   ├── step_2_frame_000000.pkl
│   └── ...
│
└── render/                  # 렌더링 이미지
    ├── step_1_frame_000000.png          # Step1 결과
    ├── step_2_frame_000000.png          # Step2 최종 결과
    ├── step_summary_frame_000000.png    # 3단계 비교 (첫 프레임)
    ├── debug/                           # 중간 iteration grid
    │   ├── step0_frame_000000_grid.jpg
    │   └── step1_frame_000000_grid.jpg
    └── keypoints/                       # GT vs Predicted
        ├── step_1_frame_*_keypoints.png
        └── step_1_frame_*_keypoints_compare.png
```

### Debug Grid Images

Iteration별 렌더링을 **압축 grid JPEG**로 저장한다:
- 저장 용량 ~95% 감소
- 5열 grid, 320x240 썸네일, JPEG 85% 품질
- 최적화 과정을 단일 이미지로 확인 가능

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

mesh = trimesh.load("obj/step_2_frame_000000.obj")
vertices = mesh.vertices  # (N, 3)
faces = mesh.faces        # (F, 3)
```

---

## Params Files (PKL)

### 내용

| 키 | Shape | 설명 |
|-----|-------|------|
| `thetas` | (1, 20, 3) | 관절 회전 (axis-angle) |
| `bone_lengths` | (1, 20) | 뼈 길이 오프셋 |
| `trans` | (1, 3) | 3D 위치 (mm) |
| `rotation` | (1, 3) | 전역 회전 |
| `scale` | (1, 1) | 스케일 팩터 |
| `chest_deformer` | (1, 1) | 가슴 변형 |

### 사용법

```python
import pickle
from bodymodel_th import BodyModelTorch

with open("params/step_2_frame_000000.pkl", "rb") as f:
    params = pickle.load(f)

# 메쉬 재생성
bodymodel = BodyModelTorch(device='cuda')
V, J = bodymodel.forward(
    params["thetas"], params["bone_lengths"],
    params["rotation"], params["trans"], params["scale"],
    params["chest_deformer"]
)
vertices = V[0].cpu().numpy()  # (N_verts, 3)
```

---

## Loss Values 해석

### 출력 값의 의미

출력되는 `theta`, `2d`, `bone` 등은 **weight 적용 전 raw loss**이다.

```
[Step1] iter 0: total=60.91 | theta:1.61 | 2d:129.63 | bone:1.61 | ...
                ^            ^ raw loss (weight 적용 전)
```

### 계산 과정

```python
# 1. 개별 loss 계산 (출력되는 raw 값)
loss_theta = 1.61      # theta:1.61
loss_2d = 129.63       # 2d:129.63
loss_bone = 1.61       # bone:1.61

# 2. Weight 곱해서 total 계산
total = loss_theta * 3.0 +    # = 4.83
        loss_2d * 0.2 +       # = 25.93
        loss_bone * 0.5 + ... # = 0.81
      = 60.91                 # total=60.91
```

### Typical Raw Values

| Loss | Step0 | Step1 | Step2 |
|------|-------|-------|-------|
| theta | 0 (frozen) | 5 -> 0.5 | 0.5 -> 0.3 |
| 2d | 700 -> 100 | 130 -> 20 | 20 -> 15 |
| bone | 0 (frozen) | 5 -> 0.5 | 0.5 -> 0.3 |
| mask | 0 (disabled) | 0 (disabled) | 0.15 -> 0.05 |
| stretch | ~2 | 20 -> 5 | 5 -> 3 |

---

## Downstream 활용

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

### 비디오 생성

```bash
# PKL 기반 (BodyModel 필요)
python scripts/visualize_mesh_sequence.py results/fitting/xxx --output mesh.mp4

# OBJ 직접 사용 (BodyModel 불필요)
python scripts/visualize_mesh_sequence.py results/fitting/xxx --use-obj -o mesh.mp4

# 360도 회전 뷰
python scripts/visualize_mesh_sequence.py results/fitting/xxx --rotating -o rotating.mp4
```

---

## Related Documents

- [DATASET.md](DATASET.md) - 입력 데이터 형식
- [ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처

---

*Last updated: 2026-02-06*
