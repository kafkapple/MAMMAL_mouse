---
date: 2025-12-10
context_name: "2_Research"
tags: [ai-assisted, uvmap, gaussian-avatar, moremouse, mammal, texture]
project: MAMMAL_mouse
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# UV Map vs Gaussian Avatar: Texture Representation 비교 분석

## 1. 개요

본 문서는 MAMMAL_mouse 프로젝트의 UV Map 기반 텍스처링과 MoReMouse의 Gaussian Avatar 기반 외형 재구성 방식을 비교 분석합니다.

### 연구 배경
- **MAMMAL_mouse**: Multi-view 영상에서 마우스 3D 메시에 텍스처를 입히는 UV mapping 파이프라인
- **MoReMouse**: Single-image 3D 마우스 복원을 위한 Gaussian Avatar 기반 합성 데이터 생성

### 공통 데이터셋
두 프로젝트 모두 **동일한 데이터셋** 사용:
- **markerless_mouse_1** (Dunn et al., 2021)
- 6-view 멀티뷰 카메라 시스템
- 18,000 frames @ 1152×1024, 100FPS
- 단일 C57BL/6 마우스 (유사한 텍스처 특성)

---

## 2. 기술적 비교

### 2.1 Input/Output 비교

| 항목 | MAMMAL UV Map | MoReMouse Gaussian Avatar |
|------|---------------|---------------------------|
| **Input** | 6-view 멀티뷰 영상 | 6-view 멀티뷰 영상 (800 frames) |
| **Output** | 2D UV 텍스처 맵 (256²~1024²) | Per-vertex Gaussian parameters (~250K) |
| **용도** | 직접 텍스처 렌더링 | 합성 학습 데이터 생성 → Single-image 3D 복원 |

### 2.2 텍스처 표현 방식

#### MAMMAL UV Map (Projection-based)
```
Multi-view Images → Camera Projection → UV Coordinate Mapping → 2D Texture Atlas
```

**저장 형식:**
- 2D 텍스처 이미지 (RGB)
- UV 좌표 매핑 테이블
- Visibility weight per texel

**파라미터 수:** O(UV_size²) ≈ 256K~1M pixels

#### MoReMouse Gaussian Avatar (Learning-based)
```
Multi-view Images → Differentiable Optimization → Per-vertex 3D Gaussians
```

**저장 형식 (per Gaussian):**
- Position offset (Δx, Δy, Δz): 3 params
- Color (RGB): 3 params
- Opacity (α): 1 param
- Scale (sx, sy, sz): 3 params (log-space)
- Rotation (quaternion): 4 params

**파라미터 수:** ~19 params × 13,059 vertices ≈ 250K params

### 2.3 변형 처리 (Deformation)

| 측면 | MAMMAL UV Map | MoReMouse Gaussian |
|------|---------------|-------------------|
| **변형 방식** | UV unwrapping (고정) | LBS로 Gaussian 함께 변형 |
| **Seam 처리** | UV seam artifacts 가능 | Seam-free (3D 공간) |
| **Pose 의존성** | 단일 텍스처 (pose-independent) | Pose-dependent deformation |

### 2.4 렌더링 파이프라인

**MAMMAL UV Map:**
```python
# 1. Load mesh vertices (posed via LBS)
V = body_model.forward(pose)

# 2. Project to 2D
proj_2d = camera.project(V)

# 3. Sample UV texture at vertex UV coords
colors = texture.sample(uv_coords[V])

# 4. Rasterize triangles
render_triangles(proj_2d, colors)
```

**MoReMouse Gaussian Avatar:**
```python
# 1. Deform Gaussians with LBS
means = base_positions + offsets
means_posed = LBS(means, pose)

# 2. Transform Gaussian attributes
rotations_posed = transform_rotation(rotations, pose)

# 3. Gaussian splatting
rgb, alpha = gsplat.rasterize(means_posed, colors, scales, rotations_posed, opacities)
```

---

## 3. 학습/최적화 비교

### 3.1 MAMMAL UV Map

| 항목 | 값 |
|------|-----|
| **방법** | Multi-view projection fusion |
| **최적화** | TV regularization, seam continuity |
| **학습 시간** | 수 분 (프레임당 ~1초) |
| **하이퍼파라미터** | visibility_threshold, fusion_method, w_tv |

### 3.2 MoReMouse Gaussian Avatar

| 항목 | 값 |
|------|-----|
| **방법** | End-to-end differentiable optimization |
| **손실 함수** | L1 + SSIM + LPIPS |
| **학습 데이터** | 800 frames (from first 8000 of markerless_mouse_1) |
| **학습 시간** | 400K steps |
| **목적** | Photorealistic synthetic data generation |

---

## 4. 장단점 분석

### 4.1 MAMMAL UV Map

**장점:**
- 구현 간단 (projection-based)
- 빠른 처리 속도 (실시간 가능)
- 메모리 효율적 (2D 텍스처만 저장)
- 표준 렌더링 파이프라인 호환

**단점:**
- UV seam artifacts 발생 가능
- View-dependent effects 표현 불가
- 복잡한 pose 변화 시 텍스처 왜곡
- Multi-view visibility 충돌 해결 필요

### 4.2 MoReMouse Gaussian Avatar

**장점:**
- Seam-free (3D 공간에 저장)
- Photorealistic 품질 (perceptual loss)
- Pose-dependent deformation 자연스러움
- View-dependent effects 가능

**단점:**
- 높은 학습 비용 (400K iterations)
- 메모리 사용량 증가 (~250K params)
- 실시간 렌더링 복잡도 증가
- Differentiable renderer 필요

---

## 5. 파이프라인 연결 관계

```
┌─────────────────────────────────────────────────────────────────┐
│                     markerless_mouse_1                           │
│                 (6-view, 18K frames, 100FPS)                     │
└─────────────────────┬───────────────────────┬───────────────────┘
                      │                       │
                      ▼                       ▼
            ┌─────────────────┐     ┌─────────────────────┐
            │  MAMMAL UV Map  │     │  MoReMouse Gaussian │
            │   (projection)  │     │   Avatar (400K opt) │
            └────────┬────────┘     └──────────┬──────────┘
                     │                         │
                     ▼                         ▼
            ┌─────────────────┐     ┌─────────────────────┐
            │  2D UV Texture  │     │  Gaussian Params    │
            │   (512×512)     │     │  (~250K params)     │
            └────────┬────────┘     └──────────┬──────────┘
                     │                         │
                     ▼                         ▼
            ┌─────────────────┐     ┌─────────────────────┐
            │ Direct Texture  │     │ Synthetic Data Gen  │
            │   Rendering     │     │ (12K scenes, 64-view)│
            └─────────────────┘     └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │ MoReMouse Network   │
                                    │ (DINOv2→Triplane    │
                                    │  →NeRF/DMTet)       │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │ Single-Image → 3D   │
                                    │   Reconstruction    │
                                    └─────────────────────┘
```

---

## 6. 핵심 통찰

### 6.1 목적의 차이
- **MAMMAL UV Map**: 기존 메시에 텍스처를 입혀 **즉시 렌더링** 가능하게 함
- **MoReMouse Gaussian Avatar**: **학습 데이터 생성**을 위한 고품질 avatar → 최종 목표는 single-image 3D reconstruction

### 6.2 동일 데이터, 다른 활용
두 프로젝트 모두 `markerless_mouse_1` 6-view 데이터를 사용하지만:
- MAMMAL: 각 프레임별로 texture extraction
- MoReMouse: 800 프레임으로 single avatar 학습 → 무한 포즈 합성 가능

### 6.3 상호 보완 가능성
- MAMMAL UV Map으로 빠른 프로토타이핑
- MoReMouse Gaussian Avatar로 고품질 학습 데이터 생성
- 두 방식을 결합하여 효율성과 품질 모두 확보 가능

---

## 7. 참고 문헌

1. **MoReMouse**: arXiv:2507.04258v2 - "MoReMouse: Monocular 3D Reconstruction of Mice"
2. **MAMMAL**: An et al. (2023) - "Three-dimensional surface motion capture of multiple freely moving pigs"
3. **markerless_mouse_1**: Dunn et al. (2021) - "Geometric deep learning enables 3D kinematic profiling"

---

*Generated: 2025-12-10*
*Tool: Claude Code (claude-opus-4-5-20250514)*
