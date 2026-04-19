# 260419 Accurate Refit on Outlier Frames — Interim Report

**Status**: ⏳ **In progress** (launched 2026-04-19 21:10 KST, ETA ~Mon 11AM, 14-18h)
**Scope**: 152 severe belly-dent outlier frames (Δ < -0.2 from v3 metric)
**Config**: `optim=accurate` (step1=200, step2=50) vs baseline `paper_fast` (step1=5, step2=3)
**GPU**: gpu03 CUDA:5
**Output**: `results/fitting/refit_outliers_152/obj/step_2_frame_XXXXXX.obj`
**Log**: `/node_data/joon/logs/refit_outliers_152.log`

---

## Background + Hypothesis

### Why refit
Global scan (`260419_belly_outlier_scan.csv`, N=2775 frames) identified:
- **152 frames** with Δ (belly_v2 - global) < -0.2 (severe belly-dent)
- **195 frames** with -0.2 ≤ Δ < -0.05 (moderate)
- Pattern: **rearing postures dominate** (frame 2700 6σ outlier, visual-verified)

### Why accurate config
Current production `paper_fast`:
- step1=5 (joint optimization iters)
- step2=3 (body optimization iters)  
- mask_loss=0 (silhouette not supervised)

Accurate config:
- step1=200 (40x more)
- step2=50 (17x more)
- mask_loss=3000 (silhouette-supervised)

### Hypothesis (3-model deliberation consensus)

Targeted accurate refit on outliers → **belly-dent rate 5.5% → 2-3%** (expected)

| Source | Expected ΔIoU/belly-dent |
|--------|:---:|
| Gemini | Severe+moderate: 12.5% → <4% (Δ>8pt), IoU +0.03 |
| Sonnet | Belly-dent: -4~-6pt |
| GPT-4o | IoU ~0.70, belly-dent ~9% |
| **Synthesized** | **Belly-dent 5.5% → 2-3%, IoU +0.03-0.05** |

---

## Pre-Refit Baseline (paper_fast config, N=2775)

| Metric | Value |
|--------|:---:|
| Global silhouette IoU (mean) | **0.642** |
| ≥0.80 frames | 0.2% (8 frames) |
| ≥0.70 frames | 21.8% (781 frames) |
| <0.50 frames | 1.5% (55 frames) |
| Belly-dent severe (Δ<-0.2) | **5.5%** (152 frames) |
| Belly-dent moderate | 7.0% (195 frames) |
| Belly IoU (v3, mean) | 0.838 |

### MoReMouse comparison (reference)

MoReMouse (AAAI 2026, same lab) uses **same MAMMAL mesh** — does not innovate on fitting. Their paper defers to An et al. 2023 for mesh fitting details. So we cannot directly extract their config; we use accurate from existing `conf/optim/accurate.yaml`.

---

## Existing Evidence (J2 POC, frame 2700)

Frame 2700 was single-frame accurate-refit in session J2 (completed 2026-04-19 17:32). However:
- Pre: v3 belly_verts = 9 (near-zero due to rearing pose)
- Post: v3 belly_verts = 0 (worse)
- **v3 metric is not reliable for rearing frames** (measurement breakdown)

Measurement for rearing frames needs alternative:
- Option A: global silhouette IoU (works for all poses)
- Option B: pose-aware belly definition (T-pose vertex group, not canon-z based)

**This interim report uses global silhouette IoU as primary metric** + qualitative visual inspection.

---

## Refit Progress Tracking

```bash
# Check progress on gpu03:
ssh gpu03 "tail -20 /node_data/joon/logs/refit_outliers_152.log"
ssh gpu03 "ls /home/joon/dev/MAMMAL_mouse/results/fitting/refit_outliers_152/obj/ | wc -l"
```

Expected rate: ~7 min/frame × 152 frames = **~18 hours**. Progress:

| Checkpoint | Frames done | ETA |
|-----------|:---:|:---:|
| Launch | 0 | 2026-04-19 21:10 |
| +1h | 8 | — |
| +6h | 48 | 2026-04-20 03:00 |
| +12h | 96 | 2026-04-20 09:00 |
| +18h (complete) | 152 | 2026-04-20 15:00 |

---

## Next Steps (after completion)

1. **Run `scripts/post_refit_comparison.py`**:
   ```bash
   python scripts/post_refit_comparison.py \
     --pre-dir results/fitting/production_3600_canon/obj/ \
     --post-dir results/fitting/refit_outliers_152/obj/ \
     --frame-list conf/frames/outlier_severe_152.txt \
     --output docs/reports/260419_refit_comparison.csv
   ```

2. **Generate visual comparison grid**:
   - 5-10 representative frames: [GT | paper_fast render | accurate render]
   - Include rearing cases (2700, 5230, 17670, 13315)
   - Save `~/results/MAMMAL/260419_refit_comparison_grid.png`

3. **Write final report**:
   - Quantitative: per-frame IoU pre/post, histogram of improvements
   - Qualitative: worst→best transformation examples
   - ICML impact: new belly-dent rate number, updated abstract

4. **Abstract v0.4 update**:
   - Replace "12.5% belly-dent" with post-refit rate
   - Add sentence: "Targeted outlier remediation via accurate config reduces severe belly-dent from 5.5% to X%"

---

## Pre-computed Baseline Artifacts (already available)

- `docs/reports/260419_global_iou_scan.csv` (3580 frames × global IoU)
- `docs/reports/260419_belly_outlier_scan.csv` (2775 frames × v3 belly Δ)
- `docs/reports/260419_psnr_ssim_lpips.csv` (30 samples × PSNR/SSIM)
- `~/results/MAMMAL/canon_3600_p0_video.mp4` (11MB, 180s video)
- `~/results/MAMMAL/260419_belly_dent_investigation/frame_002700_6view_p0/` (GT vs paper_fast render)

---

## Devil's Advocate — Risks

🔴 **R1 — Refit may not converge for all 152 frames**
Some frames (rearing, extreme poses) may hit local minima. Accurate config more iters ≠ guaranteed convergence. Expected success rate: 85-95%.

🔴 **R2 — v3 belly metric fails for rearing frames**
Frame 2700 J2 POC showed v3 returns 0 belly verts in rearing pose. Need global IoU as primary metric.

🟡 **R3 — disk space**
152 frames × 6 views render + OBJ files = ~500 MB. Script deletes intermediate fitting dirs to mitigate.

🟡 **R4 — processing overhead**
Per-frame run_experiment.sh startup = 10-30s. 152 × 25s = ~1h pure overhead. Actual refit time ~6.5 min/frame.

🟢 **R5 — correctness**
Accurate config (step1=200, step2=50) is a known-good config (J2 POC frame 2700 completed cleanly). Risk low.

---

*Interim report | 2026-04-19 21:15 KST | ETA completion ~2026-04-20 15:00 KST*
