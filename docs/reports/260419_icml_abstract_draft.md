# ICML 2026 AI4Science Workshop — Mesh Method Draft

**Target**: 150-200 word method section for MAMMAL mesh pipeline as bottom-view supplement for GS-LRM novel-view rendering.
**Deadline**: April 21 (D-3, abstract), April 24 (D-6, full paper)
**Date**: 2026-04-19 (early morning)

---

## Method Section Draft (~180 words)

We fit a parametric 3D mouse mesh (MAMMAL, 14,522 vertices, 140
joints, 21 bones) to each frame of 6-view synchronized video using
the original 3-stage optimization pipeline (mask, joint, body).
To eliminate temporal artifacts from quaternion double-cover and
bone-length extrapolation, we introduce **canonical axis-angle
interpolation** (|θ|≤π canonicalization before quaternion slerp),
which reduces single-frame pop artifacts by +0.54 global IoU
across 48/48 detected pop frames. The mesh provides a proxy
reconstruction from below that complements GS-LRM's novel-view
renderings where bottom-view supervision is absent.

For texture, we gamma-correct the WandB-HPO-optimized UV texture
(γ=2.2 with histogram matching against ground-truth RGB) to
mitigate background-leakage artifacts from the baseline
averaging-fusion pipeline. The deployed texture achieves ΔE
(CIELAB) = 17.7 against ground-truth body color.

## Known Limitations

**Rare belly-fit failure on extreme postures (~1.3%)**: Across
N=78 uniformly-sampled frames (step 180 from 3600), our vertex-group
3D-projected belly IoU averages +0.185 *above* the global silhouette
IoU, indicating the belly region is generally well-fit. However, a
single outlier frame (3 SD below mean) occurred during a rearing
posture, suggesting that LBS skinning without a belly blend-shape
deformer (bone #14 "belly_stretch" defined but no deformation
operator) fails to capture ventral deformation in extreme upright
positions. This is consistent with the absence of per-bone blend
shapes in the released MAMMAL model.

**Post-hoc texture correction**: histogram matching (γ=2.2 + RGB-scale)
is a heuristic band-aid; the underlying photometric optimization
pipeline (do_optimization=true) had a camera-path resolution bug in
the post-refactor codebase (fixed this session). Full photometric
re-optimization remains deferred.

## Contributions (Mesh Side)

1. Canonical slerp patch for temporal pop (empirical +0.54 IoU)
2. Two-axis F1-F5 vs F6 failure taxonomy separating temporal
   from spatial mesh quality issues
3. P0 texture correction pipeline deployed (ΔE 17.7 → production)
4. V3 belly metric (vertex-group 3D-projected hull IoU)
   quantifies pose-dependent fitting failures

---

## TODO for D-3 / D-6

- [ ] Cut to 150 words target
- [ ] Add 1-2 ablation numbers (canon slerp frame-IoU, P0 vs raw-average)
- [ ] Reference comparison grid figure path
- [ ] Integrate into full ICML workshop format
- [ ] Cross-check with GS-LRM-side method draft (separate paper section)

---

*Draft v0.1 | 2026-04-19 early AM | Needs revision after Phase A v3 N=100 final results*
