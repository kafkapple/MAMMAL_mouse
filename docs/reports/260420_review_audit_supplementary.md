# Review + Deliberate Audit — Supplementary Findings (2026-04-20)

> **Status**: SUPPLEMENTARY — does not supersede `260419_icml_abstract_draft.md`.
> **Purpose**: Record `/review --audit` (Lv.3) findings for ICML D-2 abstract refinement.
> **Scope**: MAMMAL mesh fitting + UV texture pipeline only. Parallel Session B
> (BehaviorSplatter PS/FL 88d COV F1-F6 audit) is independent — see
> `~/.agent/logging/handoffs/260420_1700.md`.

---

## v3 Code Review (Lv.3 Phase 1, sonnet)

6 scripts audited: `scan_global_iou`, `scan_belly_outliers`, `eval_psnr_ssim_lpips`,
`post_refit_comparison`, `refit_outlier_batch.sh`, `refit_visual_grid`.

**Result**: 3 Critical + 4 Important. 2/3 Critical fixed in commit `b56fc56`.

| # | Finding | Severity | Status |
|:-:|---|:-:|:-:|
| C1 | `eval_psnr_ssim_lpips.py` returns `(0.0, 0.0, 0)` on <100px overlap → mean pollution | 🔴 | ✅ Fixed (None return, caller skip) |
| C2 | `refit_outlier_batch.sh` `ls -td` pattern `v012345_*` race condition | 🔴 | 🟡 Works in practice (114/152 verified), deferred |
| C3 | `post_refit_comparison.py` recovery denominator ambiguity | 🔴 | ✅ Fixed (3 denominators reported) |
| I1 | `scan_global_iou.py` missing-frame count not reported | 🟡 | Abstract caveat needed |
| I2 | Convex hull IoU overestimates non-convex mouse silhouette | 🟡 | Abstract must qualify |
| I3 | SSIM crop includes background pixels → inflation bias | 🟡 | Abstract must qualify |
| I4 | `rm -rf $LATEST_DIR` lacks path whitelist guard | 🟡 | Same as C2, deferred |

---

## Deliberate Audit (Lv.3 Phase 2, 3 external models)

**Verdict**: 🔴 **REFRAME** — abstract "training-free within -2.4dB of SOTA" framing
structurally invalid. Must-fix before submission.

### Consensus Critical (2-3/3 agreement)

**M1. SSIM gap dissimulation** (Haiku + Gemini Critical)
- PSNR gap: 16.00/18.42 = 87% of SOTA (-2.4 dB)
- SSIM gap: 0.475/0.948 = **50% of SOTA** (-0.47)
- SSIM is more perceptually aligned than PSNR → real gap is **wider** than abstract implies
- **Fix**: Abstract must report both metrics explicitly. e.g., "PSNR 16.00 / SSIM 0.475 (vs MoReMouse 18.42 / 0.948)". Add 1-line hypothesis for divergence (temporal flicker or texture granularity).

**M2. Convex-hull IoU un-interpretable** (Haiku + Gemini Critical, o3-mini Major)
- Mouse is non-convex (tail, legs, bottom concavity) → `cv2.convexHull` overestimates silhouette
- 0.642 value incomparable to "pig paper 0.85+ standard"
- **Fix**: Remove IoU=0.642 from abstract OR qualify explicitly as "convex-hull IoU approximation". Remove comparison with pig-paper threshold.

**M3. Non-controlled comparison with MoReMouse** (3/3 consensus)
- Different test sets, masking protocols, N
- Direct "-2.4 dB" parity claim invalid
- **Fix**: Use `cf.` framing. e.g., "16.00 on our markerless_mouse_1 test split (cf. MoReMouse 18.42 on their reported data)". Avoid equal-sign comparison.

### Major (partial consensus)

**M4. Sample size N=30 vs 2775 vs 3580 inconsistency** (o3-mini Critical, Haiku Major)
- PSNR/SSIM: N=30 frame×view pairs
- Belly-dent rate: N=2775 frames
- Global IoU: N=3580 frames
- Abstract currently implies single N — reader confusion
- **Fix**: Report per-metric N explicitly in method table or caption.

**M5. "Training-free" framing narrative flaw** (Gemini Major)
- Analytical vs neural paradigm race is unfair; metrics favor neural
- **Fix**: Reposition as "first high-fidelity analytical baseline, establishing training-free reference point with predictable perceptual delta" — not performance parity race.

**M6. Prior art differentiation missing** (o3-mini Major)
- Analytical mesh texturing has prior art (ARAP, SMPL-X texture, per-vertex color baselines)
- **Fix**: 1-line related-work citation distinguishing P0 γ+hist matching approach.

### Minor
- Reproducibility details (hyperparameters, data splits, hardware) — workshop abstract
  scope constraint acceptable; move to paper/supplementary.

### Acknowledgment (o3-mini Minimum)
> 그나마 MoReMouse의 qualitative 'severe self-penetration' 평가와 belly-dent 수치에
> 기반한 제한 정량화는 타당하다.

---

## Revised Abstract Framing (Proposed — NOT applied)

> "We present a **training-free analytical pipeline** for multi-view 3D mouse mesh
> fitting and UV texture reconstruction. On our markerless_mouse_1 test split, P0
> γ=2.2 + Lab-space histogram matching yields PSNR 16.00 / SSIM 0.475 (N=30
> frame×view), cf. concurrent neural SOTA MoReMouse (AAAI 2026, 400k training
> steps) 18.42 / 0.948. The perceptual (SSIM) gap is wider than PSNR suggests,
> reflecting inherent limits of the training-free regime. A canonical axis-angle
> slerp patch eliminates temporal interpolation pop (+0.54 IoU on 48 verified pop
> frames). Convex-hull silhouette IoU averages 0.642 (N=3580, upper-bound
> approximation). We quantify LBS belly-dent at 12.5% severe rate (N=2775),
> backing concurrent MoReMouse's qualitative 'severe self-penetration' observation;
> accurate-config refit on 152 severe outliers reduces rate to [TBD]. The pipeline
> is **complementary to neural approaches**: interpretable, GPU-training-free,
> multi-view, with quantified limits."

**Key changes from v0.3**:
- "within -2.4dB of SOTA" → removed (M1, M3)
- Explicit both PSNR + SSIM (M1)
- "cf. MoReMouse" (not equal-sign) (M3)
- "convex-hull IoU approximation" (M2)
- Per-metric N inline (M4)
- "complementary to neural" not "parity race" (M5)

---

## Action Items (Next Session)

- [ ] Refit completion check (target 152/152, currently 114+)
- [ ] Run `post_refit_comparison.py` → measure severe→non-severe recovery
- [ ] Rewrite abstract v0.4 incorporating M1-M6 fixes (do NOT modify v0.3 file;
      create v0.4 as separate file)
- [ ] Update Obsidian `260418_MAMMAL_Mesh_Pipeline_for_ICML_Method.md` §19-20
      with post-refit numbers + audit caveats
- [ ] C2 (refit batch pattern) — long-term only, not ICML blocking

---

*Generated 2026-04-20 during /review --audit (Lv.3) on 12 session commits cdbdf33..b56fc56.*
*Supplementary to `260419_icml_abstract_draft.md` v0.3 — does not replace.*
