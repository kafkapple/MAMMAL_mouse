# Texture Deep Investigation — Full Audit

**Date**: 2026-04-18 (late evening)
**Trigger**: User observation "예전 HPO 당시 대비 GT RGB와 크게 차이" (current texture worse than historical).
**Verdict**: 🔴 **sweep-9 WandB top-score texture is structurally WHITE** (UV space), producing olive-gray render instead of GT RGB's dark-brown mouse. `do_optimization=true` runs **all failed (score=0)** — photometric optimization never completed. Current texture is **not HPO-optimized**, it's just raw `average` projection.

---

## 1. All Textures Inventoried

### On gpu03

| Path | Status | Characteristics |
|------|:------:|----------------|
| `results/sweep/run_wild-sweep-9/texture_final.png` | ✅ current default | White-dominant UV + dark smudges. 160KB, 512×512 |
| `exports/sequence/texture_final.png` | ❌ broken | Similar white+smudges, **additional artifacts** (93KB) |
| `wandb/run-*/files/media/images/uv_texture_*.png` | 15 runs | Only 4 top runs (score 0.6164) produced valid texture; all similar white-dominant pattern |

### Historical reference (Google Drive)

| Path | Source | Appearance |
|------|:------:|-----------|
| `AMILab_my/_Results/MAMMAL_Mesh_UV_texture_media_images_projection_6view_0_818f0d8f0cfb298933fc.png` | WandB artifact (Dec 10 2025) | GT RGB left + **rendered mesh RIGHT = dark brown mouse** |

### Local fetched

| File | Purpose |
|------|---------|
| `/tmp/texture_top_score.png` | sweep-9 texture raw (top WandB score) |
| `/tmp/texture_exports.png` | exports/sequence broken |
| `/tmp/texture_sweep9.png` | same as top_score (directory renamed) |
| `~/results/MAMMAL/260417_novel_view_mvp/gt_rgb_frame1800_view0.png` | **GT RGB reference** (black mouse, pink tail) |
| `~/results/MAMMAL/260417_novel_view_mvp/grid_2x3_sweep9_verify.png` | sweep-9 render output (olive-gray) |

---

## 2. WandB Sweep Audit (15 runs, 2025-12-09~10)

### Top scoring (complete) — all `do_optimization=false`

| Run | Score | do_opt | iters | fusion |
|-----|:---:|:---:|:---:|:---:|
| `run-20251210_010950-7fye4ykb` | 0.6164 | **false** | 100 | average |
| `run-20251209_202940-afb44lzf` | 0.6164 | **false** | 30 | average |
| `run-20251209_223520-yj733z3y` | 0.6164 | **false** | 30 | average |
| `run-20251210_052059-uasp3r43` | 0.6164 | **false** | 30 | average |

All `fusion=average`, coverage=97.8%, seam=0.954, mean_conf=0.751.

### `do_optimization=true` runs — ALL FAILED

| Run | do_opt | Score | media dir |
|-----|:---:|:---:|:---:|
| `run-20251209_205926-gdzbc1wu` | true | **0 (failed)** | missing |
| `run-20251210_030606-oi39ids2` | true | **0** | missing |
| `run-20251210_045819-ki85cpga` | true | **0** | missing |
| `run-20251210_041135-1z2c3nnf` | true | **0** | missing |

🔴 **Photometric optimization path never succeeded** — reason unclear (crash? convergence failure?). All current "optimized" textures are actually **raw projection average** (no optimization).

---

## 3. Color Comparison (Quantitative + Qualitative)

### GT RGB frame 1800 view 0 (source of truth)

**Observation**:
- Body: **dark brown/black** (melanistic lab mouse)
- Tail: **light pink** (un-furred, ~C57BL/6 strain)
- Ears: dark
- Fur texture: slight gradient, matte

### Current sweep-9 render

**Observation**:
- Body: **olive-gray / khaki** — wrong hue
- Tail: dark gray — wrong (should be pink)
- Details: ear/nose dark features present but muted

### Historical (Dec 2025) WandB log

**Observation** (from `MAMMAL_Mesh_UV_texture_media_images_projection_6view_0_818f0d*.png`):
- Rendered mesh right-side: **dark brown mouse** appearance — close to GT
- But this was early experimental code — may have used different render path

### Hypothesis for color mismatch

| Hypothesis | Support |
|-----------|:---:|
| H1: Raw `average` fusion dilutes dark features (averaging-to-mean) | ✅ strong — average fusion mixes all views, black mouse pixels averaged with white background reflections produces gray |
| H2: `do_optimization=true` would have fixed this but never succeeded | 🟡 plausible — photometric L1/SSIM loss would refine toward GT color |
| H3: Lighting/material in pyrender adds highlights | 🟡 partial — pyrender diffuse material + ambient light does NOT significantly change mesh base color |
| H4: Historical render used different rendering engine (not pyrender) | 🟡 possible — Dec 2025 might have used `trimesh` base color or OpenCV projection |

---

## 4. UV Map + Texture Pipeline Config

From `docs/guides/UVMAP_GUIDE.md` + WandB sweep analysis:

### 6 hyperparameters (251212)

| Parameter | Current best (sweep-9) | Range |
|-----------|:---:|:---|
| `visibility_threshold` | 0.193 | 0.1-0.7 |
| `uv_size` | 512 | 256/512/1024 |
| `fusion_method` | **average** | average / visibility_weighted / max_visibility |
| `do_optimization` | **false** ⚠️ | True/False |
| `opt_iters` | 100 (unused) | 30/50/100 |
| `w_tv` | 0.000196 (unused) | 1e-5~1e-2 |

### Score v3 Weights

- 0.5 · PSNR (15-40dB → [0, 1])
- 0.15 · SSIM
- 0.2 · Coverage (UV fill rate)
- 0.15 · Seam smoothness

**Important**: Score 0.616 = 0.5·0.45 + 0.15·SSIM + 0.2·0.978 + 0.15·0.954 = PSNR ≈ 22-26 dB from `render_front/side/diagonal` vs GT. But **render_* images use pyrender** in a single GT camera pose — so PSNR 22-26 dB is moderate (not high).

---

## 5. Recommended Fixes (Priority-ranked)

### P0: Quick post-hoc color correction (1h, no re-HPO)

**A) Gamma/darkening correction of current texture**:
```python
tex_rgb = cv2.imread(texture_path).astype(np.float32) / 255
# Darken (matches lab mouse)
tex_darkened = np.power(tex_rgb, 2.2)  # gamma 2.2 → darker
# OR: apply histogram matching against GT RGB
```

**B) Hue/saturation shift**: olive-gray → dark-brown via color space rotation in Lab.

**Pros**: immediate, no re-optimization
**Cons**: heuristic, doesn't fix underlying fusion weakness

### P1: Re-run `do_optimization=true` WandB sweep (3-5h GPU)

Debug why earlier runs scored 0 (likely crash or NaN loss). Expected outcome: photometric-optimized texture with proper dark coloring.

**Key config changes**:
- Fix whatever caused crash
- Initial iters=50 (not 100, may have hit time limit)
- w_tv=1e-4 (lower regularization, let color vary)

### P2: Direct vertex color from multi-view projection (bypass UV map)

`render_interpolated_video.py` already does this: sample RGB at vertex UV → per-vertex color. Median instead of mean (robust to outliers). Apply to mesh directly.

**Pros**: simple, often produces more realistic appearance
**Cons**: no UV texture artifact (acceptable for novel view)

### P3: Neural texture learning (Phase D, 1-2 weeks)

Train small MLP: (pos, view, time) → RGB using multi-view supervision. Higher quality but significant implementation cost.

---

## 6. Quantitative Comparison (future action — not yet computed)

TODO: Compute per-pixel PSNR/SSIM on same-view render for each texture:

| Texture | Pred_view0 PSNR vs GT | SSIM |
|---------|:---:|:---:|
| sweep-9 (current) | **TBD** | TBD |
| exports/sequence (broken) | **TBD** | TBD |
| gamma-corrected sweep-9 | **TBD** | TBD |
| P2 direct projection | **TBD** | TBD |

Will run after fix P0/P1/P2 implemented.

---

## 7. --audit --fact --devil --explain

### --fact (Verified claims)

- ✅ WandB logs scanned: 15 runs, 4 complete (score 0.6164), 4 with do_opt=true failed
- ✅ sweep-9 = top-score run but with `do_optimization=false` (raw average)
- ✅ GT RGB frame 1800 view 0 shows dark mouse (opposite of current render olive)
- ✅ Historical Dec 2025 image shows darker appearance — but this was early code, rendering pipeline unclear
- ❌ "Current sweep-9 is HPO-optimized" — **False, it's raw projection**

### --devil (Critical concerns)

**D1**: Historical "dark brown" image (Google Drive) may have been GS-LRM rendering, not MAMMAL mesh — cross-modality confusion. Verify before assuming historical MAMMAL-side was better.

**D2**: Fixing texture for MAMMAL rendering is separate from the ICML workshop claim (mesh as GS-LRM bottom-view supplement). Spending hours on pure aesthetic color may divert from F6d belly investigation.

**D3**: `do_optimization=true` failing in all 4 runs suggests **systematic bug**, not bad hyperparameters. Re-running sweep without debugging may reproduce failures.

**D4**: Post-hoc gamma correction (P0) is a band-aid — fundamental fix is photometric optimization OR direct vertex color.

### --explain (Why/Pattern/Delta)

**Why**: UV texture color quality depends on **fusion method + optimization step**. Raw `average` fusion mixes all views including background reflections → dilutes dark foreground → olive-gray result. Without photometric optimization to refine against GT RGB, texture remains approximate.

**Pattern**: HPO score ≠ visual quality. Score v3 weights (PSNR 0.5, SSIM 0.15, coverage 0.2, seam 0.15) reward coverage + smoothness > color fidelity. Top score 0.616 achieves via `do_opt=false` because `do_opt=true` runs crashed. Score convergence masks missing optimization.

**Worked example**:
```
GT RGB: mouse body = RGB(40, 30, 25) dark brown
Current sweep-9: tex_color = average(view_0_sample, view_1_sample, ...)
                ≈ RGB(140, 130, 90) olive (averaged with background reflections)
Rendered: RGB(140, 130, 90) + diffuse lighting → olive mesh
```

**Delta**:
- Before this investigation: "sweep-9 is HPO-optimized" (overclaim)
- After: sweep-9 is raw average projection; HPO's optimization path never executed
- **Don't confuse with**: GS-LRM dark-brown output (Gaussian splat colors, different modality)

---

## 8. 현재 상태 요약 (User Q2)

### 해결된 것 ✅
1. **Pop**: F1/F2 canon slerp patch empirically fixed (+0.54 IoU, 48/48 on pop frames)
2. **Coord system**: 4개 분산 doc 통합 + MVP 검증된 transform pipeline (BRIDGE.md)
3. **Novel view MVP**: v3 6-view rendering working + batch video (180s, 3600 frames)
4. **Obsidian 1-note policy**: 260418 daily note with hierarchical sections
5. **Doc audit cleanup** (/review loop iter 2 converged, 0 critical)

### 미해결 🔴
1. **Belly-dent**: F6a-F6d 중 어느 것도 empirically 확증 안 됨. N=100 kinematic 가설 all no-evidence. F6d/F6a 여전히 untested
2. **Texture quality**: 위 조사 — **olive-gray vs GT dark brown mismatch**. HPO가 실제로는 raw average (do_opt=false). 개선 경로 4개 제시 (P0-P3)
3. **Production switchover**: canon 3600 아직 downstream (FaceLift/PS) 미배포
4. **Git commit**: 15+ uncommitted files across 3 repos

### 다음 P0 후보
- 📌 Texture P0 fix (gamma/색 보정, 1h)
- 📌 F6d (SAM GT mask) PoC — belly 진짜 원인 탐색
- 📌 do_optimization=true sweep debug + re-run

---

*Texture Deep Investigation v1.0 | 2026-04-18 | Historical HPO failure + color mismatch root cause*
