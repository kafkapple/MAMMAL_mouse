# Belly Deformer Architecture — Empirical & Literature Investigation

**Date**: 2026-04-17 (evening session)

> 🔗 **Parent**: `../../results/reports/260417_phase0_belly_findings.md` — initial F6j hypothesis
> 🔗 **Taxonomy SSOT**: [260417_mesh_quality_failure_modes.md](260417_mesh_quality_failure_modes.md) — F6j weakened annotation added

**Purpose**: Validate/falsify F6j hypothesis (MAMMAL missing belly_stretch_deformer as belly-dent cause) via two low-cost, high-info tests:
1. Bone_length[belly_stretch] distribution analysis (saturation → fitter wants more DOF)
2. Literature check (upstream repo, paper) for intentional design vs planned-but-unimplemented

**Verdict**: 🟡 **F6j WEAKENED** — bone_length mechanism already provides belly scaling (not saturated), upstream never had the 3×3 deformer. Adding it is still possible but weaker justification than initial claim.

---

## 1. Background

Earlier Phase 0 investigation (`260417_phase0_belly_findings.md`) surfaced hypothesis **F6j**:
> MAMMAL has asymmetric deformer architecture — chest joint has 3×3 matrix scale deformer, belly_stretch joint has only rotation. Missing `belly_stretch_deformer` causes belly-dent.

User request (post /deliberate --devil): Before architectural investment, verify two cheaper tests:
- (A) Does bone_length[belly_stretch] hit saturation? If yes → current 1D scale insufficient → deformer addition justified.
- (B) Was belly_stretch_deformer originally planned or is it intentional architectural choice? Paper/upstream check.

---

## 2. Test A — Bone_length[belly_stretch] Distribution (Empirical)

### Measurement

Source: `tool_use:Bash` on gpu03, `results/fitting/production_900_merged/params/step_1_frame_*.pkl` ×900

Bone_length transformation applied: `bone_length_core = sigmoid(x) + 0.5` → range **[0.5, 1.5]**

### Results (actual post-sigmoid values, 900 keyframes)

| Bone_length index | Name | Min | Mean | Max | Std | @0.5 sat | @1.5 sat |
|-------------------|------|:---:|:----:|:---:|:---:|:-------:|:-------:|
| 0 | perlvis | 0.958 | 1.037 | 1.169 | 0.027 | 0 | 0 |
| 1 | femur | 0.816 | 1.051 | 1.318 | 0.056 | 0 | 0 |
| 2 | tibia | 0.881 | 1.163 | 1.382 | 0.078 | 0 | 0 |
| 10 | vertebrae | 0.575 | 1.273 | 1.466 | 0.144 | 0 | 0 |
| 11 | tail | 0.594 | 0.870 | 1.283 | 0.092 | 0 | 0 |
| **13** | **belly_stretch** | **0.551** | **0.952** | **1.460** | **0.223** | **0** | **0** |
| 19 | tail_end | 0.582 | 0.666 | 0.946 | 0.049 | 0 | 0 |

### Belly_stretch detail

- Median 0.938, 5th pct 0.594, 95th pct 1.328
- **Highest std (0.223)** of all 20 bone_lengths — fitter actively uses wide range for belly
- 58.0% values in [0.8, 1.2] "normal" range, 42% outside
- Raw (pre-sigmoid) values: mean -0.251, std 1.118, |x|>5 saturation: **0/900**

### Interpretation (Test A)

**Signal**: Bone_length[13] belly_stretch is the **most variable bone_length** (std 0.223, vs 0.027-0.144 for other bones). Fitter actively stretches/compresses belly by 1D translation scaling, range 0.55-1.46.

**Anti-signal**: **0/900 saturated** at either bound. If fitter needed more scale range, we'd expect boundary-clustering; we see symmetric distribution centered at 0.95.

**Conclusion A**: Current 1D scale mechanism (bone_length) provides meaningful belly deformation, **not limited by [0.5, 1.5] range**. Adding 3×3 matrix deformer would grant extra DOF (shear/non-uniform scale) but **this specific "fitter is starved" argument is not supported by data**.

---

## 3. Test B — Literature / Upstream Check (Fact Check)

Source: `tool_use:Agent[Fact Checker]` — WebFetch against:
- Nature Communications paper (An et al 2023, DOI 10.1038/s41467-023-43483-w)
- Upstream `anl13/MAMMAL_core` (C++, pigs, original publication)
- Upstream `anl13/MAMMAL_mouse` (Python, mouse adaptation)
- `anl13/PIG_model` README

### Findings

| Target | `belly_stretch_deformer` | `chest_deformer` | Note |
|--------|:------------------------:|:----------------:|------|
| Paper body | Not mentioned | Not mentioned (access limited) | Supplementary PDF not fully scanned (>10MB) |
| `anl13/MAMMAL_core` (C++, upstream) | ❌ Absent | ❌ Absent | Original publication has neither |
| `anl13/MAMMAL_mouse` (Python, upstream) | ❌ Absent | ✅ Present | Chest_deformer exists upstream, same as our fork |
| `kafkapple/MAMMAL_mouse` (our fork) | docstring only (L385) | ✅ Implemented | — |
| `anl13/PIG_model` README | — | — | Explicitly: "without shape-blend-shape or pose-blend-shape" |

### Interpretation (Test B)

1. **chest_deformer was added in the Python mouse adaptation** (upstream `anl13/MAMMAL_mouse`), not in original C++ pig model — so it's a **mouse-specific extension by the authors**
2. **belly_stretch_deformer was NEVER implemented in upstream** either — our fork's docstring reference is inherited vestigial text, not a local oversight
3. **PIG_model author explicit design decision**: "without blend shapes" — means the author made **deliberate architectural choice** to avoid blend-shape mechanisms in the base model
4. Whether `belly_stretch_deformer` was "planned but abandoned" (scenario A) vs "tested and removed" (scenario B) vs "intentional minimal design" (scenario C) **remains undetermined**

**Conclusion B**: F6j is not a "we found a forgotten feature" story. Upstream also lacks this. Adding it would be **our architectural extension**, not "completing" original work.

---

## 4. Revised F6j Assessment

### What is STILL TRUE
- Architectural asymmetry: chest has 3×3 matrix deformer, belly has only 1D bone_length scale. **Fact, verified in code.** (`tool_use:Read articulation_th.py:400-411`)
- Chest deformer is volume-preserving anisotropic scale (y-axis only), with active range [0.305, 2.187] used on 900 KFs (`tool_use:Bash` prior). **Fact.**
- Belly has no equivalent 3×3 matrix mechanism. **Fact.**

### What is WEAKENED
- ❌ "Fitter is hitting wall because no deformer" — **refuted by bone_length distribution** (0/900 saturated, std 0.22 usage within [0.5, 1.5] range)
- ❌ "Missing feature oversight" — **refuted by upstream check** (upstream also doesn't have it)
- 🟡 "Belly has no shape adaptation" — **partial correction**: belly HAS 1D bone_length scale, just not 3×3 matrix like chest

### Devil's Advocate Summary

**The stronger path now**:

| Claim | Status |
|-------|:-----:|
| MAMMAL lacks pose-dependent corrective blend shapes (F6a generic) | ✅ Verified (paper explicit, author stated) |
| This limits rearing/extreme-pose fitting | 🟡 Logical, partial empirical (260327 rearing +0.04 only) |
| Adding chest-style deformer to belly **would fix belly-dent** | ❌ **Not supported** — bone_length not saturated, no prior evidence |
| Belly-dent is caused by specific F6j asymmetry | 🔴 **Overstated** — F6a-generic (no blend shapes anywhere) is consistent with observations |

### Biological / Physical Reality Check

- Mouse belly DOES change shape (breathing, body curling, rearing) — biologically blend shapes would help
- But MAMMAL paper evaluated **pose accuracy**, not belly shape fidelity — original authors may not have prioritized belly shape
- Downstream users (we) observing belly-dent = scope beyond original paper's evaluation

---

## 5. Recommended Next Steps (Updated)

### De-Prioritized
- **R1** (add belly_stretch_deformer) — original priority **downgraded**. Not supported by bone_length data. Could still try as architectural experiment but shouldn't be top priority.

### Elevated
- **F6d (GT silhouette mask)** — addresses **gradient signal** problem. If silhouette is rendered-self-loop, deformer addition has no clean signal to optimize against. Fix F6d FIRST.
- **F6b (rearing init template)** — smaller investment, direct test of "extreme pose OOD" hypothesis. Falsifiable in 2-3 days.

### Still Worth Doing (low cost)
- **F6g (skinning weight visualization)** — 1 hour gpu03 task. Could reveal if belly vertices have pathological weight distribution (e.g., one vertex split across 5+ joints).
- **Belly_iou × bone_length[13] correlation** — do frames with extreme bone_length[13] (< 0.6 or > 1.4) have worse belly IoU? If yes → current mechanism already strained. If no → adding deformer probably won't help.

### Deferred (no change)
- **F6a generic blend shapes** (Phase 3) — still long-term, unchanged
- **DQS / neural skinning** — long-term, unchanged

---

## 6. Measurement Gate Compliance

Per `~/.claude/rules/measurement-gate.md`:

| Claim | Source |
|-------|--------|
| Bone_length[13] mean 0.952, std 0.223 | `tool_use:Bash` pkl analysis on gpu03 (§2 table) |
| 0/900 saturation at [0.5, 1.5] bounds | `tool_use:Bash` same |
| chest_deformer actual range [0.305, 2.187] mean 1.887 | `tool_use:Bash` prior (§260417_phase0_belly_findings.md §2) |
| Upstream anl13 lacks belly_stretch_deformer | `tool_use:Agent[Fact Checker]` WebFetch |
| PIG_model "without blend shapes" | `tool_use:Agent[Fact Checker]` README fetch |
| Joint 49 belly_stretch mean \|θ\|=3.141, 439/900 > π | `tool_use:Bash` prior (phase 0 report §2) |

### Remaining HYPOTHETICAL claims (not verified this session)

- HYPOTHETICAL: belly-dent frames have systematically extreme bone_length[13]
- HYPOTHETICAL: F6d GT mask fix would help F6j deformer addition
- HYPOTHETICAL: blend shapes would outperform current 1D bone_length

---

## 7. Decision for Next Session

Based on this investigation, replace R1 priority ranking:

```
OLD (Phase 0):
R1 (highest) Add belly_stretch_deformer
R2 Regularize theta[49]
R3 H-B2 falsification pilot

NEW (post-investigation):
R_new1 F6b rearing init pilot (2-3 days, cheap, falsifiable)
R_new2 Belly_iou × bone_length[13] correlation (1 hour, informative)
R_new3 F6g skinning weight visualization (1 hour, could reveal F6g/F6i)
R_new4 Archive: Consider F6d (SAM GT mask) as prerequisite for any deformer work
R_deferred belly_stretch_deformer addition (defer until other signals exhausted)
```

---

*Research Note v1.0 | 2026-04-17 PM | F6j weakened via empirical + literature*
