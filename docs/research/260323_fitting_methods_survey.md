# Mouse 3D Mesh Fitting: Methods Survey & Analysis

> Navigation: [← EXPERIMENTS](../EXPERIMENTS.md)
> Created: 2026-03-23 | Purpose: MAMMAL fitting 개선을 위한 관련 연구 조사

## 1. Why MAMMAL Paper Config (step1=5) Performs Poorly

### Paper Design Intent: Speed Over Quality

MAMMAL (An et al., Nature Communications 2023) 논문의 기본 설정은 **속도 최적화**:

| 설계 선택 | 값 | 이유 |
|----------|---|------|
| step1 iterations | 5 | Temporal init로 미세 조정만 필요 |
| mask loss (wsil) | **0** (OFF) | 속도 우선 |
| Evaluation | Keypoint error only | Mesh surface quality 미평가 |
| Target speed | ~1.2s/frame | "Not real-time" but practical |

논문 인용: *"5 iterations per frame yielded **fairly good** results for T > 0"*

### What the Paper Didn't Do

- **Iteration ablation study 없음**: 5 vs 50 vs 200 비교 실험 미수행
- **Mesh surface quality 미측정**: Silhouette IoU는 pig에서만 보고 (0.80), mouse 미측정
- **Mouse = adaptation**: 논문 주 대상은 돼지. Mouse는 bone length + volume preserving 추가만

### Our Findings (This Project)

| Config | step1 | mask | Mouse IoU (cam_003) | 개선 |
|--------|-------|------|--------------------|----- |
| paper | 5 | 0 | ~0.60 (worst frames) | baseline |
| fast | 50 | 3000 | 0.795 mean | +0.19 |
| **accurate** | **200** | **3000** | **0.888** (frame 720) | **+0.29** |

→ **논문이 다루지 않은 영역에서 significant improvement 확인**

## 2. Mouse/Rodent 3D Reconstruction Methods

### Optimization-based (Per-frame)

| Method | Year | Mesh | Joints | Optimizer | Speed | Code |
|--------|------|------|--------|-----------|-------|------|
| **MAMMAL** | 2023 | 14,522v | 140 | LBFGS | 1.2s-14min | [GitHub](https://github.com/anl13/MAMMAL_mouse) |
| **ArMo** | 2023 | 1,803v | 30 | Adam (batch 300) | slow | Not released |

### Feed-forward (Single Pass)

| Method | Year | Approach | Speed | Mouse Result | Code |
|--------|------|----------|-------|-------------|------|
| **Pose-Splatter** | 2025 NeurIPS | SAM2 → shape carving → 3DGS | **~30ms** | IoU 0.76, PSNR 29.0 | Pending |
| **MoReMouse** | 2025 | Triplane + Gaussian avatar | ~ms | PSNR 22.0 (synthetic) | Not released |
| **RatBodyFormer** | 2024 | Keypoint → body surface | ~ms | Rat only | Not released |
| **AniMer** | 2025 CVPR | ViT + SMAL family-aware | ~ms | No rodent | [GitHub](https://github.com/luoxue-star/AniMer) |

### Pose Estimation (Non-mesh, but Relevant)

| Method | Type | Relevance |
|--------|------|-----------|
| **DANNCE** | 3D CNN | Our data source (markerless_mouse_1) |
| **DeepLabCut + SuperAnimal** | 2D→3D | 26-keypoint mouse foundation model |
| **LocoMouse** | Feature tracking | 400fps 3D paw/nose/tail |

## 3. Available Resources

### Datasets

| Dataset | Species | Views | Frames | GT | Status |
|---------|---------|-------|--------|-----|--------|
| **markerless_mouse_1** | Mouse | 6 | 18K | DANNCE 3D kp | ✅ 보유 |
| Rat7M | Rat | 6 | 7M | MoCap | 공개 |
| PAIR-R24M | Rat pair | Multi | 24.3M | MoCap | 공개 |
| Animal3D | 40 species | Single | 3,379 | SMAL params | 공개 |

### Models

| Model | Type | Species | Status |
|-------|------|---------|--------|
| MAMMAL mouse mesh | LBS + bone length + chest deformer | Mouse (C57BL/6) | ✅ 보유 |
| Virtual Mouse (SAM) | CT scan animated | Mouse | Nature Methods 2021 |
| SMAL | Statistical parametric | Quadrupeds (not rodent) | 공개 |

## 4. Optimization Strategies

### Current (MAMMAL)
- 3-step coarse-to-fine: Global pose → Articulated → Silhouette refinement
- LBFGS optimizer (2nd order, efficient convergence)
- Temporal initialization (sequential mode)

### Potential Improvements

| Strategy | Impact | Effort | Description |
|----------|--------|--------|-------------|
| **Silhouette loss ON** | High | ✅ Done | `accurate` config already enables mask=3000 |
| **Iteration ablation** | High | ✅ In progress | E3 sweep: step1=[100,200,400] × mask=[1000,3000,5000] |
| **nvdiffrast rendering** | Medium | Medium | CUDA-optimized differentiable rendering (faster than PyTorch3D) |
| **Hybrid (feed-forward + refine)** | Very High | High | Network predicts initial pose → LBFGS 2-3 iter refine |

### Hybrid Approach (Most Promising Long-term)

SPIN pattern (Kolotouros, ICCV 2019) — validated in human body fitting:

```
Feed-forward network → Initial pose prediction (~30ms)
         ↓
MAMMAL LBFGS → 2-3 iteration refinement (~5s)
         ↓
High-quality mesh with 100x speedup
```

This could be implemented by combining Pose-Splatter's pose prediction with MAMMAL's mesh refinement.

## 5. Key Insight: MAMMAL vs Parametric Models

MAMMAL is **NOT** a full parametric model like SMAL/SMPL:
- No statistical shape parameters (β)
- Single template mesh + LBS + hand-crafted adaptations
- Mouse-specific: bone length params + volume preserving constraints

This means:
- **Pro**: Works without large 3D scan dataset
- **Con**: Limited shape variation, per-species manual adaptation needed

---

## References

- An et al. "MAMMAL" Nature Communications 2023 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10673844/))
- Goffinet et al. "Pose Splatter" NeurIPS 2025 ([OpenReview](https://openreview.net/forum?id=KuXnKedjAj))
- Bolaños et al. "Virtual Mouse" Nature Methods 2021
- Ye et al. "SuperAnimal" Nature Communications 2024
- Luoxue et al. "AniMer" CVPR 2025

---

*Created: 2026-03-23 | MAMMAL Fitting Experiments*
