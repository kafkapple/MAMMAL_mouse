# UV Texture to Blender: 실행 가이드

> MAMMAL 메쉬에 학습된 UV 텍스처를 적용하여 블렌더에서 활용하는 전체 파이프라인

---

## Quick Reference (즉시 사용)

### 단일 프레임

```bash
python scripts/export_to_blender.py \
    --mesh results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254/obj/step_2_frame_000000.obj \
    --texture results/sweep/run_wild-sweep-9/texture_final.png \
    --output exports/mouse_textured.obj
```

### 일괄 실행 (OBJ 전체 프레임 + 6-view 그리드 영상)

```bash
# 100프레임 피팅 결과 대상, 한번에 OBJ 내보내기 + 6-view 그리드 영상 생성
python -m mammal_ext.blender_export.run_all \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --texture results/sweep/run_wild-sweep-9/texture_final.png \
    --output_dir exports/v012345_kp22_20251206/
```

출력 구조:
```
exports/v012345_kp22_20251206/
├── obj/                              # 100 OBJ + MTL + texture (Blender용)
│   ├── step_2_frame_000000.obj
│   ├── step_2_frame_000000.mtl
│   ├── texture_final.png
│   └── ...
└── renders/
    ├── *_6view_grid.mp4              # 6-view 그리드 영상
    └── *_6view_sample.png            # 첫 프레임 샘플 이미지
```

---

## 0. Best Known Configuration (WandB Sweep 결과)

### 전체 Sweep 최고 성능 Top 5

> WandB project: `uvmap-optimization` | 5 sweeps, 204 runs (2025-12-10 ~ 12-12)

| Rank | Sweep | Run Name | Score | Coverage | Seam | Config |
|------|-------|----------|-------|----------|------|--------|
| **1** | aabflhvs | **wild-sweep-9** | **0.836** | 100% | 0.241 | vis=0.35, avg, opt=**F** |
| **2** | n7feev3i | elated-sweep-5 | 0.836 | 100% | 0.241 | vis=0.58, avg, opt=F |
| **3** | gm4e24p6 | upbeat-sweep-76 | 0.617 | 97.8% | 0.924 | vis=0.52, avg, opt=T |
| **4** | kpfwmwhj | grateful-sweep-37 | 0.399 | 100% | 0.173 | vis=0.14, max, opt=T |
| **5** | 0heflsc1 | eager-sweep-3 | 0.396 | 100% | 0.192 | vis=0.33, max, opt=T |

### Best Config 상세 (wild-sweep-9)

```yaml
visibility_threshold: 0.346
fusion_method: average
do_optimization: false     # 수치적 PSNR 우선 (PSNR 역설 참고)
opt_iters: 100             # (미사용, opt=false)
uv_size: 512
w_tv: 0.0049               # (미사용, opt=false)
```

**텍스처 파일**: `results/sweep/run_wild-sweep-9/texture_final.png`

### 블렌더 활용 시 권장 설정 (시각적 품질 우선)

| 용도 | 권장 설정 | 이유 |
|------|----------|------|
| **수치적 최적** (Score 최고) | `wild-sweep-9` (opt=False) | PSNR 역설: 원본 색상 유지 → 높은 score |
| **시각적 최적** (블렌더용) | opt=True, w_tv=0.005~0.01 | TV smoothing → seam 완화, 노이즈 제거 |
| **seam 최소** (3D 프린팅 등) | `grateful-sweep-37` (seam=0.173) | 최저 seam discontinuity |

> **PSNR 역설**: TV regularization이 텍스처를 부드럽게 하여 시각적으로는 개선되지만,
> PSNR은 오히려 낮아질 수 있음. Bayesian optimizer가 opt=False로 수렴하는 경향.

---

## 1. 전체 파이프라인 개요

```
Mesh Fitting (.pkl)
    │
    ▼
UV Map Pipeline ─── 6-view RGB 투영 + visibility weighting
    │
    ▼
texture_final.png (512x512 UV 텍스처맵)
    │
    ├─ (선택) WandB Sweep → 최적 파라미터 탐색
    │
    ▼
export_to_blender.py ─── OBJ + MTL + texture PNG
    │
    ▼
Blender Import (.obj)
```

### 필요 입력물

| 입력 | 경로 | 설명 |
|------|------|------|
| Fitting 결과 | `results/fitting/<exp>/params/*.pkl` | 메쉬 파라미터 |
| OBJ 메쉬 | `results/fitting/<exp>/obj/*.obj` | 3D 메쉬 |
| Multi-view RGB | `data/examples/.../videos_undist/{0-5}.mp4` | 6개 뷰 영상 |
| Segmentation mask | `data/examples/.../simpleclick_undist/{0-5}.mp4` | 전경 마스크 |
| Camera calibration | `data/examples/.../new_cam.pkl` | K, R, T 행렬 |
| Body model UV | `mouse_model/mouse_txt/textures.txt` | UV 좌표 정의 |

---

## 2. Step 1: UV 텍스처 생성

### 기본 실행

```bash
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356 \
    --uv_size 512
```

### Best known config로 실행 (수치적 최적)

```bash
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356 \
    --uv_size 512 \
    --visibility_threshold 0.346 \
    --fusion_method average
```

### 블렌더용 실행 (시각적 최적)

```bash
python -m uvmap.uv_pipeline \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356 \
    --uv_size 512 \
    --visibility_threshold 0.35 \
    --fusion_method average \
    --do_optimization \
    --opt_iters 100 \
    --w_tv 0.005
```

### 출력물

```
results/fitting/<experiment>/uvmap/
├── texture_final.png      # RGB UV 텍스처 (512x512)
├── texture.pt             # PyTorch tensor (프로그래밍용)
├── confidence.png         # 신뢰도 히트맵 (TURBO colormap)
├── confidence_gray.png    # 신뢰도 (grayscale)
└── uv_mask.png            # 유효 UV 영역 마스크
```

### 파이프라인 내부 동작

```
Stage 1: Texture Sampling
  └─ 6개 뷰에서 메쉬 정점을 이미지로 투영 → RGB 샘플링
  └─ Backface culling으로 visibility 계산

Stage 2: Texture Accumulation
  └─ 다중 프레임 가중 평균
  └─ Confidence map 생성

Stage 3: UV Rendering
  └─ Vertex color → UV 좌표 공간으로 rasterize
  └─ 512x512 텍스처맵 생성

Stage 4: (Optional) Photometric Optimization
  └─ Differentiable rendering + L1/SSIM loss
  └─ TV regularization으로 seam 완화
```

---

## 3. 좌표계 변환 (MAMMAL → Blender)

### 3.1 왜 변환이 필요한가?

MAMMAL과 Blender는 **up 축**이 다르기 때문에, 변환 없이 임포트하면 메쉬가 **90° 옆으로 누워** 보입니다.

| 좌표계 | Up 축 | Forward 축 | Right 축 |
|--------|--------|-----------|----------|
| **MAMMAL** | **-Y** | +X (head→tail) | +Z |
| **Blender World** | **+Z** | +Y | +X |

### 3.2 변환 공식

X축 기준 +90° 회전 (Rx(+90°)):

```
(x, y, z)_MAMMAL  →  (x, z, -y)_Blender
```

행렬 표현:
```
┌ x' ┐   ┌ 1  0  0 ┐ ┌ x ┐
│ y' │ = │ 0  0  1 │ │ y │
└ z' ┘   └ 0 -1  0 ┘ └ z ┘
```

### 3.3 추가 처리 (export_to_blender.py 기본값)

| 처리 | 기본값 | 설명 | 비활성화 |
|------|--------|------|----------|
| **좌표 변환** | ON | MAMMAL → Blender World | `--no_transform` |
| **센터링** | ON | 원점 중심으로 이동 | `--no_center` |
| **mm→m 스케일** | ON | MAMMAL은 mm 단위, Blender는 m | `--no_scale` |

### 3.4 검증 방법

Blender에서 정상적으로 보이는 기준:
- **등(back)이 위** (+Z), **배(belly)가 아래** (-Z)
- **머리(head)가 앞** (+Y 또는 +X 방향)
- 크기가 ~0.1m (실제 마우스 체장 ~10cm)

> 상세 좌표계 분석: [`docs/coordinates/coordinate_systems_reference.md`](../coordinates/coordinate_systems_reference.md)

---

## 4. Step 2: 블렌더용 OBJ 내보내기

### 방법 A: 기존 sweep 최고 성능 텍스처 사용 (즉시)

```bash
python scripts/export_to_blender.py \
    --mesh results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260125_174356/obj/step_2_frame_000000.obj \
    --texture results/sweep/run_wild-sweep-9/texture_final.png \
    --output exports/mouse_textured.obj
```

### 방법 B: 새로 생성한 텍스처 사용

```bash
python scripts/export_to_blender.py \
    --mesh results/fitting/<experiment>/obj/step_2_frame_000000.obj \
    --texture results/fitting/<experiment>/uvmap/texture_final.png \
    --output exports/mouse_textured.obj
```

### 방법 C: 좌표 변환 없이 (MAMMAL 원본 좌표)

```bash
python scripts/export_to_blender.py \
    --mesh ... --texture ... --output ... \
    --no_transform --no_center --no_scale
```

### 출력물

```
exports/
├── mouse_textured.obj     # 메쉬 + UV 좌표 (v/vt/f)
├── mouse_textured.mtl     # 머티리얼 파일 (텍스처 참조)
└── texture_final.png      # UV 텍스처 이미지
```

### 방법 D: 일괄 실행 (OBJ 전체 프레임 + 6-view 그리드 영상)

```bash
# 100프레임 (v012345_kp22, 가장 많은 프레임 보유 실험) 일괄 처리
python -m mammal_ext.blender_export.run_all \
    --result_dir results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254 \
    --texture results/sweep/run_wild-sweep-9/texture_final.png \
    --output_dir exports/v012345_kp22_20251206/

# OBJ만 (영상 생략)
python -m mammal_ext.blender_export.run_all \
    --result_dir ... --texture ... --output_dir ... --skip_video

# 영상만 (OBJ 생략)
python -m mammal_ext.blender_export.run_all \
    --result_dir ... --texture ... --output_dir ... --skip_obj

# 첫 10프레임만 테스트
python -m mammal_ext.blender_export.run_all \
    --result_dir ... --texture ... --output_dir ... --max_frames 10
```

### 개별 모듈 실행

```bash
# OBJ만 일괄 내보내기
python -m mammal_ext.blender_export.batch_export \
    --result_dir results/fitting/<experiment> \
    --texture texture_final.png \
    --output_dir exports/obj/

# 6-view 그리드 영상만
python -m mammal_ext.blender_export.sequence_renderer \
    --result_dir results/fitting/<experiment> \
    --texture texture_final.png \
    --output_dir exports/renders/
```
```

> UV 텍스처는 body model 고유의 UV layout이므로, **동일 텍스처를 모든 프레임에 재사용** 가능.

---

## 5. 블렌더에서 임포트

### 기본 임포트

1. **File > Import > Wavefront (.obj)**
2. `mouse_textured.obj` 선택
3. MTL이 자동으로 텍스처 연결

### 텍스처 수동 연결 (MTL 자동 로드 실패 시)

1. Material Properties 탭
2. Base Color 옆 노란 점 클릭
3. **Image Texture** 선택
4. `texture_final.png` 열기

### 뷰포트에서 텍스처 확인

- **Z 키** → Material Preview 또는 Rendered 선택
- Solid 모드에서는 텍스처가 보이지 않음

### 애니메이션 (다중 프레임)

1. 각 프레임 OBJ를 Stop Motion OBJ 애드온으로 임포트
2. 또는 `scripts/blender_mesh_animation.py` 활용:
   ```bash
   # Blender 내 Python console에서
   exec(open("scripts/blender_mesh_animation.py").read())
   ```

### OBJ 파일 구조 (참고)

```obj
# Vertices (메쉬 정점)
v  x y z

# Texture coordinates (UV 좌표, mouse_model/mouse_txt/textures.txt)
vt u v

# Faces (1-indexed, vertex/texture 쌍)
f v1/vt1 v2/vt2 v3/vt3

# Material reference
mtllib mouse_textured.mtl
usemtl mouse_material
```

---

## 6. (선택) WandB Sweep으로 새 파라미터 탐색

### 기존 Sweep 실험 통계

| Sweep ID | Runs | Best Score | Best Run |
|----------|------|-----------|----------|
| **aabflhvs** | 17 | **0.836** | wild-sweep-9 |
| n7feev3i | 12 | 0.836 | elated-sweep-5 |
| gm4e24p6 | 81 | 0.617 | upbeat-sweep-76 |
| kpfwmwhj | 86 | 0.399 | grateful-sweep-37 |
| 0heflsc1 | 8 | 0.396 | eager-sweep-3 |

> **Score 차이 이유**: sweep별로 Score 함수 버전(v1/v2/v3)이 다름.
> kpfwmwhj(86 runs)는 v3(PSNR+SSIM+Coverage+Seam), aabflhvs(17 runs)는 초기 버전.

### 탐색 파라미터 (6개)

| 파라미터 | 범위 | 분포 | 설명 |
|---------|------|------|------|
| `visibility_threshold` | 0.1 ~ 0.7 | uniform | visibility 임계값 |
| `uv_size` | 512 (고정) | - | UV 해상도 |
| `fusion_method` | average / visibility_weighted / max_visibility | categorical | 다중 뷰 합성 방법 |
| `do_optimization` | True / False | categorical | photometric 최적화 |
| `opt_iters` | 30 / 50 / 100 | categorical | 최적화 반복 수 |
| `w_tv` | 1e-5 ~ 1e-2 | log-uniform | TV regularization 강도 |

### Score 함수 (v3 - 최신)

```
Score = 0.50 * PSNR + 0.15 * SSIM + 0.20 * Coverage + 0.15 * Seam
```

- **Coverage Gating**: Coverage < 80% → Score x 0.1 (패널티)
- **Seam score**: `exp(-15 * seam_discontinuity)`

### 새 sweep 실행

```bash
# Full sweep (Bayesian)
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/<experiment> \
    --count 30

# 2-stage 분리 (권장)
# Stage A: 구조 파라미터 탐색 (do_optimization=False 고정)
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/<experiment> \
    --stage stage_a --count 20

# Stage B: Stage A 최적 설정 기반 미세 조정
python -m uvmap.wandb_sweep \
    --result_dir results/fitting/<experiment> \
    --stage stage_b --stage_a_config best_config.json --count 20
```

### Sweep 결과 파일

```bash
# 각 run의 출력물
results/sweep/run_xxx-sweep-NN/
├── texture_final.png          # UV 텍스처
├── texture.pt                 # PyTorch tensor
├── projection_6view_grid.png  # 6-view 비교 (원본 vs 렌더링)
├── confidence.png             # 신뢰도맵
└── uv_mask.png                # UV 마스크
```

---

## 7. 현재 사용 가능한 리소스

### Fitting 실험 (obj+params 보유)

| 실험명 | 날짜 | obj | params |
|--------|------|-----|--------|
| `v012345_kp22_20251213_200852` | 2025-12-13 | O | O |
| `v012345_kp22_20251213_201317` | 2025-12-13 | O | O |
| `v012345_kp22_20260125_174356` | 2026-01-25 | O | O |
| `v012345_kp22_20260125_230540` | 2026-01-25 | O | O |
| `v012345_kp22_20260125_230806` | 2026-01-25 | O | O |
| `v012345_kp22_20260125_231350` | 2026-01-25 | O | O |

> `v012345` = 6-view, `kp22` = 22 keypoints

### Sweep 텍스처 (바로 사용 가능)

| Run | Score | 텍스처 경로 | 추가 파일 |
|-----|-------|-----------|----------|
| **wild-sweep-9** | 0.836 | `results/sweep/run_wild-sweep-9/texture_final.png` | render_front/side/diagonal |
| upbeat-sweep-76 | 0.617 | `results/sweep/run_upbeat-sweep-76/texture_final.png` | render_front/side/diagonal |
| grateful-sweep-37 | 0.399 | `results/sweep/run_grateful-sweep-37/texture_final.png` | 6view grid |
| eager-sweep-3 | 0.396 | `results/sweep/run_eager-sweep-3/texture_final.png` | 6view grid |

---

## 8. Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| Coverage 0~50% | visibility_threshold 과도 | 0.2~0.3으로 낮춤 |
| Seam 아티팩트 | TV regularization 부족 | `w_tv` 0.001~0.01 |
| 색상 반전 (R/B 뒤바뀜) | BGR/RGB 불일치 | `cv2.cvtColor` 확인 |
| 메쉬가 1픽셀로 투영 | 카메라 T 단위 불일치 (mm vs m) | `cam['T'] / 1000` |
| 메쉬가 90° 옆으로 누움 | MAMMAL(-Y up) vs Blender(Z up) | `--no_transform` 제거 (기본값이 변환 ON) |
| 블렌더에서 텍스처 안 보임 | Viewport Shading 모드 | Material Preview(Z) 또는 Rendered로 전환 |
| platformdirs ImportError | wandb 의존성 누락 | `pip install platformdirs` |

---

## 9. 관련 문서

| 문서 | 내용 |
|------|------|
| [HPO Score Design](../reports/251212_uvmap_hpo_score_design.md) | Score v3 설계, do_optimization 수렴 분석 |
| [UV Texture Experiment](../reports/251210_uvmap_texture_experiment.md) | Loss 설계, Grid 시각화, WandB Sweep |
| [UV Score Optimization](../reports/251210_uvmap_score_optimization_strategy.md) | Score 최적화 전략 |
| [UV vs Gaussian Avatar](../reports/251210_uvmap_vs_gaussian_avatar_comparison.md) | UV Map vs 3DGS 비교 |
| [UV Map System](../guides/uvmap_system.md) | UV 파이프라인 시스템 가이드 |
| [Coordinate Systems Reference](../coordinates/coordinate_systems_reference.md) | MAMMAL/Blender/OpenCV 좌표계 정의 및 변환 |

---

*Last updated: 2026-01-27*
*WandB data: kafkapple-joon-kaist/uvmap-optimization*
