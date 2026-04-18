# Texture Multipath Comparison — P0 vs P2 Quantitative + Qualitative

**Date**: 2026-04-18 (late evening, 23:00)
**Trigger**: `/audit --fact --devil --explain` finding — sweep-9 WandB default = raw `average` fusion (olive-gray vs GT dark-brown). Four improvement paths proposed; this report measures P0 and P2.

---

## TL;DR

| Method | Approach | ΔE (CIELAB) vs GT | Verdict |
|--------|---------|:--:|:--:|
| **P0** (gamma 2.2 + histogram match) | Post-hoc UV texture correction | **17.7 mean (12.7-22.8)** | ✅ **Production-ready** |
| **P2** (direct 6-view vertex median) | UV bypass, no occlusion check | **105.5 mean (101-113)** | ❌ Fails (bg bleed) |

**Decision**: Deploy P0 as new canonical production texture. P2 path clarifies the root cause of sweep-9 olive-gray and motivates a **P2.5 proposal (occlusion-aware projection)** as future work.

---

## 1. Context

- GT RGB frame 1800 view 0: dark-brown/black mouse body (L* = 26.9), pink tail
- sweep-9 render (do_opt=false, fusion=average): olive-gray body
- 4 paths evaluated; P0 and P2 executed in this session. P1 (do_opt=true debug) deferred due to missing crash logs. P3 (neural MLP) out of scope.

## 2. Quantitative Results

### CIELAB body color measurement

Method: mouse body pixels (non-white, non-pure-black) in each render → mean Lab → ΔE to GT.

| Method | View | L* | a* | b* | ΔE | n_pixels |
|--------|:---:|:--:|:--:|:--:|:--:|:--:|
| GT body | frame 1800 v0 | 26.9 | -4.9 | +9.8 | 0 | 69k |
| **P0 gamma+hist** | Right | 43.6 | -0.7 | +7.4 | **17.4** | 4091 |
| **P0 gamma+hist** | Top | 44.2 | -0.6 | +7.7 | **17.9** | 2935 |
| **P0 gamma+hist** | Bottom | 38.7 | -0.4 | +7.9 | **12.7** | 8712 |
| **P0 gamma+hist** | Front-high | — | — | — | **22.8** | — |
| P2 direct | Right | 130.9 | -0.6 | +7.7 | **104.1** | 4928 |
| P2 direct | Top | 127.9 | -0.4 | +8.2 | **101.1** | 3569 |
| P2 direct | Bottom | 139.9 | +0.3 | +5.1 | **113.2** | 9958 |

**Interpretation**: P0 ΔE ~12-23 falls in the "visible but acceptable" range (typical: <2 imperceptible, 2-10 slight, 10-30 clearly visible, >30 large). P2 ΔE 100+ is extreme — near-complementary color.

The primary failure axis is **L\*** (lightness):
- GT body L = 27 (very dark, almost black)
- P0 body L = 39-44 (dark brown, acceptable)
- P2 body L = 128-140 (light gray, complete miss)

a* and b* (chromaticity) are approximately correct in both methods; the color *hue* is roughly preserved, but P2's lightness is wrong by ~100.

### Raw JSON

See `~/results/MAMMAL/260418_texture_experiment_v1/delta_e_report.json` (gpu03: `results/texture_experiment_v1/delta_e_report.json`).

## 3. Qualitative Comparison Grid

3 × 2 side-by-side: `~/results/MAMMAL/260418_texture_experiment_v1/comparison_grid_3x2.png`

Row 1: [GT Right-like | P0 Right | P2 Right]
Row 2: [GT (same) | P0 Bottom | P2 Bottom]

**Visual observations**:
- P0 Right: dark-brown body + pink tail — GT에 근접
- P0 Bottom: belly 영역도 어둡게 유지됨 (dark brown)
- P2 Right: olive-gray body + pink tail — body 색 실패
- **P2 Bottom: belly 흰-회색 (배경 섞임 명확)** — occlusion 없는 projection이 background pixel을 샘플링한 직접 증거

## 4. Root Cause Analysis — P2 Failure

### Mechanism

`p2_direct_vertex_color` (`scripts/texture_multipath_experiment.py:57`) projects each vertex to each view using:

```python
Pc = (R @ verts.T).T + T
uv = (K @ Pc.T).T[:, :2] / Pc.z
u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (d > 0)
# Sample img[v, u] for all in_img vertices  ← NO occlusion check
```

Every vertex whose projection falls within image bounds samples that pixel — but there is no **depth test** or **silhouette mask check**. Consequences:
- A belly vertex whose projection point lies on the white backdrop (because the front of the body occludes the back → backdrop is what's "at" that 2D pixel from the camera's frustum but a different vertex) gets white color
- Actually more precisely: vertex's own ray from camera passes through body first, but projection math ignores this — vertex is assigned whatever 2D pixel its projection coordinate lands on, not "what the camera actually sees at the vertex"
- Over 6 views with median fusion, this produces consistent L\* > 100 bias

### Same mechanism likely underlies sweep-9 olive-gray

WandB sweep-9 uses `fusion=average` over UV texel → view projections. The underlying projection code (`uvmap/`) likely has the same occlusion gap. Dark mouse pixels get averaged with white backdrop samples → RGB ~(140, 130, 90) olive.

### Proposed P2.5 fix (future work)

```python
# For each vertex × view:
#  1. Project as before → (u, v)
#  2. Check simpleclick_undist/{view}.mp4 mask at (u, v) — must be foreground
#  3. Optional: rasterize mesh once, check if vertex is at depth near rendered depth
#  4. If pass → sample; else → skip
#  5. Median over valid (view, vertex) pairs
```

Expected: ΔE < P0 (12-22) since direct multi-view is higher-fidelity than post-hoc gamma. Time estimate: 2-3h.

Not executed this session due to ICML D-3 deadline — P0 is sufficient.

## 5. P0 Deployment Decision

**Accept**: P0 (gamma 2.2 + histogram match to GT dark pixels) as new production texture.

**Location**: copy `results/texture_experiment_v1/texture_p0_gamma.png` → canonical production path (D3 task).

**SSOT update**: `docs/guides/UVMAP_GUIDE.md` — mark P0 texture as current default, sweep-9 as "reference raw-average".

**Batch re-render**: `results/novel_view_batch/canon_3600_p0/` — 3600 frames × 6 views → 180s video (1-2h GPU overnight).

## 6. Failure Path Audit (for honest reporting)

| Path | Status | Reason |
|------|:------:|--------|
| P0 | ✅ Deployed | ΔE 12-23, acceptable for workshop supplement |
| P1 (do_opt=true debug) | ❌ Deferred | Crash logs unavailable; 3-5h time-sink with uncertain payoff; P0 is sufficient |
| P2 (direct vertex median) | ❌ Rejected | ΔE 100+ due to missing occlusion check; motivates P2.5 |
| P2.5 (occlusion-aware) | 📌 Future work | Expected to beat P0; not executed due to deadline |
| P3 (neural MLP) | ❌ Out of scope | 1-2 weeks |

## 7. Residual Doubt

- **R1**: GT body L=27 was computed from pixels with gray<100 — this threshold may include some background shadow regions. Sensitivity check: at gray<60, L_gt = ~20 (even darker), which would make P0 ΔE larger but P2 ΔE also larger proportionally. Relative ordering unchanged.
- **R2**: ΔE measured on single frame (1800). Temporal variance unassessed — full-batch re-render will provide mean/stddev.
- **R3**: Hue (a*, b*) differences are small (~5-7) — subsequent histogram matching in a\*/b\* could shave another ~5-10 ΔE from P0. Low-priority polish.

## 8. Action Items

1. ✅ D1: ΔE JSON + grid saved
2. ✅ D2: This report
3. 🔄 D3: Deploy P0 as canonical + UVMAP_GUIDE update
4. 🔄 D4: Batch re-render overnight
5. 🔄 D5 (parallel): F6g skinning weight viz for belly
6. 🔄 D6: Git commit + Obsidian sync

---

*260418_texture_multipath_comparison.md v1.0 | 2026-04-18 23:30 | P0 deployed, P2.5 future work*
