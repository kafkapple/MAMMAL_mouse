---
date: 2025-12-10
context_name: "2_Research"
tags: [ai-assisted, wandb-sweep, loss-function, photometric, ssim, psnr, uv-mapping]
project: MAMMAL_mouse
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# UV Map Score 최적화 전략 수립: 논리 체계 정리

## 1. 문제 인식

### 1.1 기존 Score 함수 (v2)의 한계

```python
score = w_coverage * coverage + w_psnr * confidence + w_seam * seam
```

**발견된 문제점:**

| 문제 | 설명 | 영향 |
|------|------|------|
| **변수명 오류** | `w_psnr`가 실제 PSNR이 아닌 `mean_confidence` 사용 | Misleading metric |
| **Photometric Loss 부재** | Rendered vs Original 이미지 비교 없음 | 텍스처 품질 미반영 |
| **Perceptual Loss 부재** | Human perception 기반 메트릭 없음 | 시각적 품질 미반영 |

### 1.2 핵심 질문

> "UV 텍스처의 품질을 어떻게 정량적으로 평가할 것인가?"

**답**: 3D 렌더링 후 원본 이미지와 비교 (Photometric Consistency)

---

## 2. 학술 문헌 조사

### 2.1 3D Gaussian Splatting (SIGGRAPH 2023)

**Reference**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"

```
Loss = (1 - λ) × L1 + λ × D-SSIM,  λ = 0.2
```

**핵심 인사이트:**
- L1 (pixel-wise) + SSIM (structural) 조합이 표준
- PSNR/SSIM/LPIPS로 평가

### 2.2 IQA 문헌 조사 결과

| Metric | 장점 | 단점 |
|--------|------|------|
| **PSNR** | 계산 빠름, 직관적 | Perceptual quality 미반영 |
| **SSIM** | Human perception 유사 | Over-smoothing 경향 |
| **LPIPS** | 가장 정확한 perceptual | 계산 비용 높음, GPU 필요 |

**결론**: PSNR + SSIM 조합이 속도/정확도 균형 최적

---

## 3. 설계 결정

### 3.1 Score v3 공식 도출

**3DGS 기반 가중치 설계:**

```
Original 3DGS: L = 0.8 × L1 + 0.2 × D-SSIM  (총 1.0)
```

**UV Map 적용:**
- Photometric (PSNR 기반): 0.50 (주요 품질 지표)
- SSIM: 0.15 (perceptual quality)
- Coverage: 0.20 (UV 공간 활용도)
- Seam: 0.15 (텍스처 연속성)

```python
# Score v3 (uvmap/wandb_sweep.py:794-799)
score = (
    w_photo * photo_score +      # 0.50
    w_ssim * ssim_score +        # 0.15
    w_coverage * coverage_score + # 0.20
    w_seam * seam_score          # 0.15
)
```

### 3.2 PSNR Normalization

**문제**: PSNR은 dB 단위 (일반적으로 20-40 dB)
**해결**: [0, 1] 범위로 정규화

```python
psnr_min, psnr_max = 15.0, 40.0
psnr_score = clip((psnr - psnr_min) / (psnr_max - psnr_min), 0, 1)
```

### 3.3 Masked Region 계산

**문제**: 전체 이미지 비교 시 배경이 결과 왜곡
**해결**: Mesh 영역만 마스킹하여 비교

```python
def create_mesh_mask(rendered, background_value=255):
    """흰색 배경 제외한 mesh 영역만 추출"""
    return ~np.all(rendered == background_value, axis=2)
```

---

## 4. 구현 상세

### 4.1 Helper Functions

| 함수 | 위치 | 역할 |
|------|------|------|
| `compute_psnr_masked()` | :43-90 | PSNR 계산 (masked) |
| `compute_ssim_masked()` | :93-138 | SSIM 계산 (bbox crop) |
| `create_mesh_mask()` | :141-158 | Mesh 영역 마스크 생성 |

### 4.2 데이터 흐름

```
UV Pipeline → vertex_colors
     ↓
6-View Projection Rendering (face_sampling=1)
     ↓
PSNR/SSIM 계산 (rendered vs original, masked)
     ↓
Score v3 계산
     ↓
WandB 로깅
```

### 4.3 face_sampling 결정

**검토 결과**: Sweep 파라미터에서 제외

| face_sampling | 품질 | 속도 |
|---------------|------|------|
| 1 (100%) | 최상 | ~10초 |
| 2 (50%) | 양호 | ~5초 |
| 5 (20%) | Coarse | ~2초 |

**결론**: Photometric 비교 정확도를 위해 `face_sampling=1` 고정

---

## 5. 검증

### 5.1 Syntax Check
```bash
conda run -n mammal_stable python -c "import uvmap.wandb_sweep; print('OK')"
# Output: OK
```

### 5.2 Helper Function Test
```python
from uvmap.wandb_sweep import compute_psnr_masked, compute_ssim_masked, create_mesh_mask
# All functions work correctly
```

---

## 6. 예상 효과

### 6.1 Sweep 최적화 개선

| 항목 | v2 | v3 |
|------|----|----|
| **Objective** | confidence 기반 | Photometric 기반 |
| **품질 신호** | 간접적 | 직접적 (rendered vs original) |
| **WandB 메트릭** | 3개 | 6개 (+PSNR, SSIM) |

### 6.2 새로운 WandB 메트릭

- `mean_psnr_score`: [0-1] 정규화된 PSNR
- `mean_ssim_score`: [0-1] structural similarity
- `mean_psnr_db`: 실제 PSNR (dB)

---

## 7. Action Items

- [x] Score v3 구현
- [x] Helper functions 추가
- [x] Config 가중치 업데이트
- [x] 문서화 완료
- [ ] Sweep 실행 및 결과 분석
- [ ] Best config로 최종 UV Map 생성

---

## 8. 핵심 교훈

1. **변수명 = 실제 의미**: `w_psnr`가 PSNR이 아니었던 것처럼, 변수명과 실제 값의 일치 중요
2. **학술 문헌 기반 설계**: 3DGS의 L1+SSIM 조합은 검증된 표준
3. **Masked Comparison**: 배경 포함 시 메트릭 왜곡 발생
4. **Trade-off 명시**: face_sampling과 품질의 관계 명확히 문서화

---

*Generated: 2025-12-10*
*Tool: Claude Code (claude-opus-4-5-20251101)*
