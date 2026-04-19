# Phase A v2 Re-run — N=78 with corrected belly axis

**Date**: 2026-04-19 (early morning)
**Reason**: Prior Phase A (260418) used v1 metric + mis-labeled belly (head verts).
This re-run uses belly_metric_v2 (3D-projected vertex-group convex hull IoU) with
corrected belly definition (y∈[40,90] torso AND z<z25 ventral).

## Setup

- Frames: N=78 (step 180, range [0, 17640])
- Views: 6 per frame
- Samples: 468 total (frame × view)

## Belly IoU statistics

| Metric | Mean | Std | Min | Max |
|--------|:---:|:---:|:---:|:---:|
| iou_v2 (belly hull only) | 0.838 | 0.201 | 0.000 | 1.000 |
| iou_global | 0.653 | 0.111 | 0.389 | 0.915 |
| Δ = v2 - global | +0.185 | 0.198 | -0.757 | +0.543 |

## Kinematic correlations

Not computed: params pkl files not found in results/fitting/production_3600_canon/obj/ or . Only OBJ files available in production_3600_canon.

## Interpretation

- **Δ > 0** on average: belly region fits BETTER than global silhouette
- This **reinforces v1 paradox** (v1 Δ was +0.019, v2 is +0.185)
- **Implication**: 'belly-dent' visual impression may be **rendering/texture artifact**, not 3D geometric defect
- Prior hypothesis 'F6a/F6h blend shape absence' needs reassessment

## Next steps

- Render comparison: P0-textured canon mesh vs raw GT RGB (same frame same view) to test rendering-artifact hypothesis
- If render ≈ GT in belly region → dent is indeed perceptual. Belly track concludes
- If render visibly darker/dented in belly region → separate issue (e.g., P0 texture darkness over-concentrated, geometry fine)
