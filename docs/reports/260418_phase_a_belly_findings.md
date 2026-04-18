# Phase A — Belly Empirical Probes (Correlation Analysis)

> ⚠️ **Superseded by**: [260418_phase_a_extension_report.md](260418_phase_a_extension_report.md) — N=100 Pearson+Spearman + per-view baseline analysis. N=23 "borderline signal" shown to be statistical noise.

**Date**: 2026-04-18
**Sample**: N=23 keyframes (belly_iou_paperfast_23same.csv × view-mean)
**Verdict**: 🔴 **F6j/F6a (θ-magnitude based) decisively weakened**. F6b rearing shows borderline signal. F6e bone_length drift dismissed.

---

## 1. Correlations (Pearson r)

| Pair | r | Interpretation |
|------|:---:|---|
| belly_iou × −\|θ[49]\| (joint 49 belly_stretch) | **+0.002** | **무관** — dent ↔ belly joint rotation magnitude 인과 없음 |
| belly_iou × −bone_length[13] extreme | +0.188 | 거의 무관 |
| belly_iou × −spine_angle (rearing) | **−0.371** | 약한 부정 — rearing→dent 방향 맞으나 강도 약함 |

Stats:
- Belly IoU mean 0.845, std 0.036, range [0.77, 0.91]
- θ49 mag mean 3.14 (≈π), range [1.76, 4.06]
- Spine angle mean 94.6°, range [38.5°, 160.9°] (wide — includes rearing candidates)

---

## 2. Hypothesis Update Table

| Hypothesis | Pre-Phase-A | Post-Phase-A | Change |
|-----------|:-----------:|:------------:|:------:|
| F6a (LBS no blend shapes, general) | 🟡 Untested | 🟡 Untested | no change (broad, not specifically r(θ)) |
| **F6j (belly_stretch_deformer missing specifically)** | 🟡 Weakened | 🟡 **No support (N=23 underpowered)** | r(θ49)=0.002 — no evidence, but N=23 insufficient for "rejection"; requires N≥50 for α=0.05 |
| F6b (rearing OOD) | 🟡 Untested | 🟡 **Borderline** | Direction correct, strength weak (\|r\|=0.37) |
| F6c (mask_loss=0) | 🟡 Inconclusive | 🟡 Inconclusive | Not re-tested |
| F6d (no GT mask) | 🟡 Untested | 🟡 **Elevated priority** | Other hypotheses weakened → F6d more likely by exclusion |
| F6e (bone drift) | 🟡 Weak link | ❌ **Dismissed** | r=0.19, uncorrelated |

---

## 3. Decision Gate

Originally proposed:
- r(belly_iou, rearing) > 0.6 → Phase B (rearing init)
- r(belly_iou, θ49) > 0.5 → Phase C (deformer)
- All r < 0.3 → F6d (SAM mask)

**Actual results**:
- r(rearing) = **-0.37** (weak, direction-correct)
- r(θ49) = **0.002** (zero)
- r(bone13) = 0.19 (zero)

**Revised strategy**:
- ❌ **Phase C (deformer impl) cancelled** — no empirical support
- 🟡 **Phase B (rearing init pilot) conditional-delay** — weak signal, would need N=200+ sample to confirm
- 🟢 **Phase E (SAM GT mask) elevated** — by elimination, most promising remaining hypothesis
- 🟡 **Phase A extension** — increase sample to 200+ frames before committing to any path

---

## 4. Caveats

1. **Sample size**: N=23 keyframes is small. |r|=0.37 at N=23 has p≈0.08 — not significant at α=0.05. Requires N>50 for |r|=0.3 at p<0.05.
2. **View averaging**: Per-frame = mean of 6 view belly_iou. Could mask view-specific effects (e.g., dent visible only in bottom/side views).
3. **Keyframe-only sample**: All 23 frames are keyframes (%20=0). Non-keyframe interpolated frames may show different correlations.
4. **Spine angle heuristic from MAMMAL kp22**: Uses kp[3]=neck, kp[5]=lumbar — may not capture "rearing" exactly as intended in 260327.

---

## 5. Recommended Next Actions

### Immediate (P0)
- **Texture batch re-render** (sweep9) — complete, regenerate canon_3600 novel view video

### P1 (next session)
- **Phase A extension**: `belly_iou_diagnostic.py --all-100` → 100 frames × 6 views = 600 samples. Re-run correlation at better statistical power
- **F6d (SAM GT mask) feasibility check**: SAM2 install + 1-frame mask quality test

### P2 (conditional)
- **Phase B rearing init pilot** (3 frames × accurate, ~45min) — execute ONLY if Phase A extension confirms r(rearing) > 0.5 at larger N

### P3 (de-prioritized, conditional revival)
- 🟡 Phase C `belly_stretch_deformer` impl — no empirical support at N=23; revive only if N≥50 Phase A ext confirms r(θ49)|>0.3
- 🟡 Phase E bone-length regularization — uncorrelated at N=23; same threshold for revival

**Statistical note**: r=0.188 (bone13) and r=0.002 (θ49) at N=23 have p>0.39, i.e., consistent with no effect OR small effect. Formal rejection requires N≥50 at observed effect sizes.

---

## 6. Method Section Update

ICML method note ([[260418_MAMMAL_Mesh_Pipeline_for_ICML_Method]]) §6 "Belly-Dent" should be updated:

**Old claim**: "F6j (belly_stretch_deformer) weakened but still candidate"
**New claim**: "F6j rejected via Phase A correlation (r(belly_iou, θ49)=0.002, N=23). F6b borderline (r=-0.37). Remediation path favors F6d (GT mask pre-compute) by elimination."

Honest disclosure in limitations:
- "Belly-dent remains unsolved at submission. Phase A eliminates architectural hypothesis (joint-level deformer) but does not conclusively identify cause."

---

*Phase A Report v1.0 | 2026-04-18 | Empirical belly-dent hypothesis pruning*
