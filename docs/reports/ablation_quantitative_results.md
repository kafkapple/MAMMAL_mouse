---
date: 2025-12-11
tags: [ablation-study, quantitative-analysis, mesh-comparison]
---

# Ablation Study: Quantitative Mesh Comparison Results

**Generated:** 2025-12-11 09:48:57
**Baseline:** markerless_mouse_1_nerf_v012345_kp22_20251206_165254

## Summary (Sorted by V2V Distance)

| Rank | Views | Keypoints | V2V Mean (mm) | V2V Max (mm) | Chamfer (mm) | Hausdorff (mm) |
|------|-------|-----------|---------------|--------------|--------------|----------------|
| 1 | 6 (0,1,2,3,4,5) | 9 (sparse_9_dlc) | 1.7563 | 10.1591 | 0.5305 | 5.2191 |
| 2 | 6 (0,1,2,3,4,5) | 7 (sparse_7_mars) | 1.9925 | 8.4448 | 0.6626 | 5.3406 |
| 3 | 5 (0,1,2,3,4) | 3 (sparse_3_core) | 4.1622 | 28.9268 | 1.1883 | 9.4884 |
| 4 | 3 (0,2,4) | 3 (sparse_3_core) | 5.1446 | 28.6161 | 1.4496 | 10.0700 |
| 5 | 4 (0,1,2,3) | 3 (sparse_3_core) | 6.3753 | 28.6789 | 1.5494 | 10.3470 |
| 6 | 2 (0,3) | 3 (sparse_3_core) | 6.7582 | 31.6299 | 2.6476 | 14.0383 |
| 7 | 6 (0,1,2,3,4,5) | 5 (sparse_5_minimal) | 17.5889 | 61.7960 | 10.3297 | 54.4368 |

## Trade-off Analysis

### Best Configurations by Category

**Best View Configuration (with 3 keypoints):** 5 views (V2V: 4.1622 mm)
**Best Keypoint Configuration (with 6 views):** 9 keypoints (V2V: 1.7563 mm)

### Efficiency Score (Lower V2V / Lower Resources = Better)

| Configuration | Resources (V×KP) | V2V Mean | Efficiency Score |
|---------------|------------------|----------|------------------|
| 6V 9KP | 54 | 1.7563 | 7.0383 |
| 6V 7KP | 42 | 1.9925 | 7.4943 |
| 5V 3KP | 15 | 4.1622 | 11.5401 |
| 3V 3KP | 9 | 5.1446 | 11.8460 |
| 4V 3KP | 12 | 6.3753 | 16.3524 |
| 2V 3KP | 6 | 6.7582 | 13.1508 |
| 6V 5KP | 30 | 17.5889 | 60.4002 |

## Detailed Results per Configuration

### 6 Views, 9 Keypoints (sparse_9_dlc)

- **Directory:** `markerless_mouse_1_nerf_v012345_sparse9_20251207_081918`
- **View IDs:** [0, 1, 2, 3, 4, 5]
- **Frames Compared:** 20

| Metric | Value |
|--------|-------|
| V2V Mean | 1.7563 mm |
| V2V Max | 10.1591 mm |
| V2V Std | 2.1668 mm |
| V2V Median | 1.0051 mm |
| Chamfer Distance | 0.5305 mm |
| Hausdorff Distance | 5.2191 mm |

### 6 Views, 7 Keypoints (sparse_7_mars)

- **Directory:** `markerless_mouse_1_nerf_v012345_sparse7_20251207_172028`
- **View IDs:** [0, 1, 2, 3, 4, 5]
- **Frames Compared:** 20

| Metric | Value |
|--------|-------|
| V2V Mean | 1.9925 mm |
| V2V Max | 8.4448 mm |
| V2V Std | 1.9531 mm |
| V2V Median | 1.1813 mm |
| Chamfer Distance | 0.6626 mm |
| Hausdorff Distance | 5.3406 mm |

### 5 Views, 3 Keypoints (sparse_3_core)

- **Directory:** `markerless_mouse_1_nerf_v01234_sparse3_20251203_235123`
- **View IDs:** [0, 1, 2, 3, 4]
- **Frames Compared:** 20

| Metric | Value |
|--------|-------|
| V2V Mean | 4.1622 mm |
| V2V Max | 28.9268 mm |
| V2V Std | 4.2948 mm |
| V2V Median | 2.7870 mm |
| Chamfer Distance | 1.1883 mm |
| Hausdorff Distance | 9.4884 mm |

### 3 Views, 3 Keypoints (sparse_3_core)

- **Directory:** `markerless_mouse_1_nerf_v024_sparse3_20251204_153916`
- **View IDs:** [0, 2, 4]
- **Frames Compared:** 20

| Metric | Value |
|--------|-------|
| V2V Mean | 5.1446 mm |
| V2V Max | 28.6161 mm |
| V2V Std | 4.5141 mm |
| V2V Median | 3.9621 mm |
| Chamfer Distance | 1.4496 mm |
| Hausdorff Distance | 10.0700 mm |

### 4 Views, 3 Keypoints (sparse_3_core)

- **Directory:** `markerless_mouse_1_nerf_v0123_sparse3_20251204_074430`
- **View IDs:** [0, 1, 2, 3]
- **Frames Compared:** 20

| Metric | Value |
|--------|-------|
| V2V Mean | 6.3753 mm |
| V2V Max | 28.6789 mm |
| V2V Std | 5.1986 mm |
| V2V Median | 4.6280 mm |
| Chamfer Distance | 1.5494 mm |
| Hausdorff Distance | 10.3470 mm |

### 2 Views, 3 Keypoints (sparse_3_core)

- **Directory:** `markerless_mouse_1_nerf_v03_sparse3_20251205_014945`
- **View IDs:** [0, 3]
- **Frames Compared:** 20

| Metric | Value |
|--------|-------|
| V2V Mean | 6.7582 mm |
| V2V Max | 31.6299 mm |
| V2V Std | 5.3727 mm |
| V2V Median | 4.7776 mm |
| Chamfer Distance | 2.6476 mm |
| Hausdorff Distance | 14.0383 mm |

### 6 Views, 5 Keypoints (sparse_5_minimal)

- **Directory:** `markerless_mouse_1_nerf_v012345_sparse5_20251208_134918`
- **View IDs:** [0, 1, 2, 3, 4, 5]
- **Frames Compared:** 5

| Metric | Value |
|--------|-------|
| V2V Mean | 17.5889 mm |
| V2V Max | 61.7960 mm |
| V2V Std | 12.2363 mm |
| V2V Median | 15.1071 mm |
| Chamfer Distance | 10.3297 mm |
| Hausdorff Distance | 54.4368 mm |
