---
date: 2025-12-10
context_name: "2_Research"
tags: [ai-assisted, wandb-sweep, loss-function, photometric, ssim, psnr]
project: MAMMAL_mouse
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# WandB Sweep Score 함수 개선: 학술 문헌 기반 분석

## 0. 구현 완료 요약

**구현 일자**: 2025-12-10

### 주요 변경사항

1. **Score v3 구현**: PSNR + SSIM 기반 photometric scoring
2. **Config 가중치 업데이트**: `w_psnr` → `w_photo`, `w_ssim` 추가
3. **Helper Functions 추가**: `compute_psnr_masked()`, `compute_ssim_masked()`, `create_mesh_mask()`
4. **6-view Rendering 개선**: Photometric 메트릭 계산 및 반환

### 최종 Score 공식 (v3)

```python
# uvmap/wandb_sweep.py:794-799
score = (
    w_photo * photo_score +      # 0.50: PSNR-based photometric quality
    w_ssim * ssim_score_val +    # 0.15: Structural similarity
    w_coverage * coverage_score + # 0.20: UV space utilization
    w_seam * seam_score          # 0.15: Texture continuity
)
```

### 변경된 파일

| 파일 | 변경 내용 |
|------|----------|
| `uvmap/wandb_sweep.py` | Score v3 구현, helper functions, config weights |

---

## 1. 이전 문제점 분석

### 1.1 현재 Score 함수 (v2)

```python
# uvmap/wandb_sweep.py:649-653
score = (
    w_coverage * coverage_score +     # UV 공간 커버리지
    w_psnr * confidence_score +       # ⚠️ 실제 PSNR 아님! mean_confidence
    w_seam * seam_score               # Seam discontinuity
)
```

**핵심 문제점:**
1. **`w_psnr`가 실제 PSNR이 아님**: 변수명이 `w_psnr`이지만 실제로는 `mean_confidence` 사용
2. **Photometric Loss 누락**: Rendered vs Original 이미지 비교 없음
3. **Perceptual Loss 누락**: SSIM, LPIPS 등 human perception 기반 메트릭 없음
4. **`do_optimization` 미포함**: Stage 3 최적화 효과가 score에 반영되지 않음

### 1.2 Mesh Coarse 현상 원인

```python
# 현재 렌더링 코드 (wandb_sweep.py)
for face in faces_np[::face_sampling]:  # face_sampling=5 → 5개 중 1개만 렌더링
```

**결과**: 렌더링 속도 최적화를 위해 face를 skip하여 빈 공간(흰색) 발생

---

## 2. 학술 문헌 조사

### 2.1 3D Gaussian Splatting (SIGGRAPH 2023)

**Reference**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"

**Loss Function:**
```
L = (1 - λ) * L1 + λ * D-SSIM
```
- **λ = 0.2** (default)
- L1: Pixel-wise absolute difference
- D-SSIM: Structural similarity (1 - SSIM)

**Evaluation Metrics:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

### 2.2 Extended 3DGS with LPIPS

**Reference**: Water-Adapted 3D Gaussian Splatting (Frontiers in Marine Science, 2025)

**Loss Function:**
```
L = (1 - λ - α) * L1 + λ * SSIM + α * LPIPS
```
- **λ = 0.2**, **α = 0.05**
- LPIPS: AlexNet feature extractor

### 2.3 DiffTex (Texture Optimization)

**Reference**: "DiffTex: Differentiable Texturing for Architectural Proxy Models" (arXiv 2024)

**Loss Function:**
```
L = α * L_Render(L2) + β * L_Persp(L1) + ω * L_Para(L1)
```
- **α=1, β=2, ω=10**
- L_Render: Photometric reconstruction (L2)
- L_Persp: Perspective consistency
- L_Para: Parameter smoothness

### 2.4 IQA Metrics for Optimization

**Reference**: "Comparison of Full-Reference Image Quality Models" (PMC 2021)

**Key Findings:**
| Metric | Best For | Trade-off |
|--------|----------|-----------|
| **LPIPS/DISTS** | Texture-rich tasks | High computational cost |
| **MS-SSIM** | Balanced quality | Tends to over-smooth |
| **MAE (L1)** | Denoising | Produces blurred results |

> "LPIPS is designed to match human perception and yield better scores for images with a higher level of coherence"

### 2.5 Multi-view Consistency

**Reference**: "Improving Neural Radiance Fields with Depth-aware Optimization" (arXiv 2023)

**Key Insight:**
- Photometric consistency + Depth loss → **PSNR +3dB improvement**
- Patch-based consistency outperforms pixel-wise approaches

---

## 3. 제안: 개선된 Score 함수 (v3)

### 3.1 새로운 Score 구조

```python
# Score v3: Photometric-Aware Scoring
score = (
    w_coverage * coverage_score +      # [0, 1] UV coverage
    w_photo * photometric_score +      # [0, 1] NEW: PSNR-based
    w_ssim * ssim_score +              # [0, 1] NEW: Structural similarity
    w_seam * seam_score                # [0, 1] Seam continuity
)
```

### 3.2 Photometric Score 계산

```python
def compute_photometric_score(rendered, target, mask):
    """
    Compute PSNR between rendered and target images.

    Args:
        rendered: [H, W, 3] Rendered RGB (0-255)
        target: [H, W, 3] Original RGB (0-255)
        mask: [H, W] Mesh coverage mask (boolean)

    Returns:
        psnr_score: [0, 1] normalized PSNR score
    """
    # Masked region only
    rendered_masked = rendered[mask]
    target_masked = target[mask]

    # MSE in masked region
    mse = np.mean((rendered_masked - target_masked) ** 2)

    # PSNR calculation
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)

    # Normalize to [0, 1] (PSNR typically 20-40 dB for reasonable results)
    psnr_min, psnr_max = 15.0, 40.0
    psnr_score = np.clip((psnr - psnr_min) / (psnr_max - psnr_min), 0, 1)

    return psnr_score, psnr
```

### 3.3 SSIM Score 계산

```python
from skimage.metrics import structural_similarity as ssim

def compute_ssim_score(rendered, target, mask):
    """
    Compute SSIM between rendered and target images.

    Returns:
        ssim_score: [0, 1] structural similarity
    """
    # Crop to bounding box of mask for efficiency
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return 0.0, 0.0

    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    rendered_crop = rendered[y1:y2, x1:x2]
    target_crop = target[y1:y2, x1:x2]

    # SSIM calculation (channel_axis for RGB)
    ssim_val = ssim(target_crop, rendered_crop, channel_axis=2, data_range=255)

    return ssim_val, ssim_val
```

### 3.4 권장 가중치 (문헌 기반)

```python
# 3DGS 표준 (Kerbl et al., 2023)
w_photo = 0.8 * (1 - 0.2)  # 0.64 (L1 weight from 3DGS)
w_ssim = 0.8 * 0.2         # 0.16 (SSIM weight from 3DGS)
w_coverage = 0.1           # UV completeness
w_seam = 0.1               # Texture continuity

# 총합 = 1.0
```

### 3.5 Alternative: LPIPS 추가 (Optional)

```python
# Extended version with perceptual loss
w_photo = 0.60    # L1/PSNR
w_ssim = 0.15     # Structural
w_lpips = 0.05    # Perceptual (requires torch + lpips library)
w_coverage = 0.10
w_seam = 0.10
```

---

## 4. 구현 완료 내역

### 4.1 Helper Functions (uvmap/wandb_sweep.py:43-158)

```python
def compute_psnr_masked(rendered: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Compute PSNR in masked region, normalized to [0,1] (15-40 dB range)"""

def compute_ssim_masked(rendered: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Compute SSIM using bounding box crop for efficiency"""

def create_mesh_mask(rendered: np.ndarray, background_value: int = 255) -> np.ndarray:
    """Create boolean mask for mesh region (non-white pixels)"""
```

### 4.2 Config 가중치 (uvmap/wandb_sweep.py:63-68)

```python
@dataclass
class WandBSweepConfig:
    # Objective weights (for composite score v3)
    w_photo: float = 0.50   # Photometric (PSNR-based)
    w_ssim: float = 0.15    # Structural similarity
    w_coverage: float = 0.20  # UV space coverage
    w_seam: float = 0.15    # Seam discontinuity
```

### 4.3 6-View Projection 메트릭 반환

`_render_6view_projection_grid()`가 이제 Dict를 반환:

```python
return {
    'grid_path': grid_path,
    'mean_psnr_score': mean_psnr_score,  # [0, 1] normalized
    'mean_ssim_score': mean_ssim_score,  # [0, 1]
    'mean_psnr_db': mean_psnr_db,        # dB value
    'per_view_psnr': psnr_scores,
    'per_view_ssim': ssim_scores,
}
```

### 4.4 face_sampling 관련 결정

**결론**: `face_sampling`은 sweep 대상에서 제외

- face_sampling=1 (full quality)로 고정
- 이유: sampling 시 mesh가 coarse해져 photometric 비교 정확도 저하
- projection_face_sampling은 이미 1로 설정됨

---

## 5. 예상 효과

### 5.1 Score 함수 개선

| 항목 | 현재 (v2) | 개선 후 (v3) |
|------|----------|-------------|
| **Photometric** | 없음 | PSNR 기반 score |
| **Perceptual** | 없음 | SSIM (+ optional LPIPS) |
| **Coverage** | 유지 | 유지 |
| **Seam** | 유지 | 유지 |

### 5.2 Rendering 품질

| 항목 | 현재 | 개선 후 |
|------|------|---------|
| **Face Sampling** | 5 (20% faces) | 1~2 (50~100% faces) |
| **빈 공간** | 많음 | 최소화 |
| **속도** | ~2초 | ~5~10초 (face_sampling=1) |

---

## 6. 참고 문헌

1. **3D Gaussian Splatting**: Kerbl et al., SIGGRAPH 2023
   - [GitHub](https://github.com/graphdeco-inria/gaussian-splatting)

2. **LPIPS**: Zhang et al., CVPR 2018
   - [GitHub](https://github.com/richzhang/PerceptualSimilarity)

3. **DiffTex**: arXiv:2509.23336
   - [Paper](https://arxiv.org/html/2509.23336)

4. **IQA Comparison**: PMC 2021
   - [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7817470/)

5. **Depth-aware NeRF**: arXiv:2304.05218
   - [Paper](https://arxiv.org/html/2304.05218)

---

*Generated: 2025-12-10*
*Tool: Claude Code (claude-opus-4-5-20250514)*
