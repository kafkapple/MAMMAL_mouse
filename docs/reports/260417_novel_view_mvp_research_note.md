# Research Note: Novel View MVP + Coord System Investigation

**Date**: 2026-04-17 (evening)
**Purpose**: Document the chain of investigations that led to successful MAMMAL mesh rendering from FaceLift v3 novel 6-view, including /deliberate findings and /audit corrections.

**Verdict**: ✅ **MVP PASSED** — `mammal_to_gslrm()` + OpenCV→OpenGL camera flip is the correct coord integration. No axis swap needed. Bottom view shows clean mesh geometry vs GS-LRM's blurry tail artifact — ICML motivation empirically validated.

---

## 1. Task Chain (this session)

1. Belly deformer investigation (`260417_belly_deformer_investigation.md`) — concluded F6j weakened, bone_length mechanism actively used.
2. User request: render MAMMAL mesh from FaceLift novel 6-view for ICML workshop feature analysis
3. `/deliberate --moa --audit --devil --fact --explain` — architecture + use case discovery
4. User confirmations: use case = FaceLift/GS-LRM comparison + bottom-view mesh complement; reference = 260306 v3 image; MVP first
5. Coord system documentation survey (4 docs: 2 in MAMMAL, 2 in FaceLift/Obsidian)
6. `/audit --fact --devil` — scope reduction, symlink/SSOT deferral, empirical test first
7. **Phase 1 MVP executed** — 1 frame × 6 views successfully rendered

---

## 2. The v3 6-View Config (Authoritative)

→ **SSOT**: `docs/coordinates/MAMMAL_FACELIFT_BRIDGE.md §4` (값 정의)
→ Implementation: `scripts/novel_view_render.py::V3_NOVEL_6VIEWS`

Source: `GoogleDrive:AMILab_my/_Results/FaceLift/260306_2nd_phase/_novel_view_rendering/novel_6view_temporal_v3/novel_6view_grid.png` image labels.

---

## 3. Coord System Integration — Empirical Truth

### Transform pipeline (verified MVP-level)

```
MAMMAL mesh (-Y up, +X head, +Z right, mm)
   │
   │ mammal_to_gslrm(verts) = (verts - M5_SCENE_CENTER) * M5_DISTANCE_SCALE
   │   constants (from FaceLift coordinate_utils.py):
   │   M5_SCENE_CENTER = [59.672, 51.517, 107.099] mm
   │   M5_DISTANCE_SCALE = 2.7 / 307.785 ≈ 0.008781
   ▼
Mesh in GS-LRM normalized space (shares OpenCV world: X-right, Y-down, Z-forward)
   │
   │ (no axis swap applied)
   ▼
Camera: spherical_c2w_opencv(elev, azim, radius=2.7)
   returns OpenCV c2w [4×4]
   │
   │ pyrender needs OpenGL: c2w_gl = c2w_cv @ diag(1, -1, -1, 1)
   │ flip is applied to CAMERA only, not mesh
   ▼
pyrender offscreen render → 512×512 PNG
```

### Why no axis swap is needed (confirmed empirically)

Prior doc confusion suggested MAMMAL axes might differ from FaceLift:
- MAMMAL project doc: -Y up, +X head, +Z right
- FaceLift world (OpenCV): X-right, Y-down, Z-forward
- Apparent mismatch: +X head vs +Z forward (semantic ≠ axis)

**Key insight**: "head direction" in MAMMAL world is not mandated to align with "camera forward" in FaceLift world. Both systems use Y-down (MAMMAL -Y up = OpenCV Y-down in magnitude). Mouse "faces" wherever it does in world — X/Z swap semantically, but geometrically both are horizontal axes in a right-handed frame.

**Camera orbit is around world origin**, so mouse orientation relative to camera is determined by where mouse is in world (after `mammal_to_gslrm` centering), not by which axis the mouse "faces."

MVP output confirms: rendered views match v3 reference expectations (top shows back, bottom shows belly, etc.) — empirically validates the transform.

### Existing doc reconciliation (NOT required)

Per Devil's Advocate S1 analysis:
- MAMMAL project `docs/reference/COORDINATES.md`: "-Y up" (2026-02-06, experimentally verified via vertex analysis)
- `mammal_ext/blender_export/coordinate_transform.py::MAMMAL_TO_BLENDER`: `Rx(+90°)` consistent with -Y up
- Obsidian `FaceLift/docs/theory/COORDINATE_SYSTEMS.md` (2026-03-04): defines FaceLift-only, doesn't contradict
- Obsidian `2603_NeurIPS/coordinate system mismatch.md` (2026-03-21): debug log mentioning "Y-up" was in a different context (general comparison, not MAMMAL-specific assertion)
- `coordinate_utils.py` L15 "MAMMAL 축 미확인": **stale** comment, pending since creation (2026-02-24)

**Conclusion**: Code SSOTs (MAMMAL's `coordinate_transform.py` + FaceLift's `coordinate_utils.py`) are internally consistent. Docs lag but don't conflict at a load-bearing level. Wholesale convergence NOT needed. Cross-reference annotations sufficient.

---

## 4. MVP Output (frame 1800)

Location: `results/novel_view_mvp/frame_001800/` (gpu03) + `~/results/MAMMAL/260417_novel_view_mvp/frame_001800/` (local)

Files:
- 6 per-view PNGs (Top.png, Bottom.png, etc.)
- `grid_2x3.png` — 2×3 composite with labels
- `extrinsics.json` — complete 6-camera c2w matrices + intrinsics + transform spec

### Observed render quality

| View | Observation | Comparison to 260306 v3 reference |
|------|-------------|----------------------------------|
| Top | Spine + tail visible, dorsal view ✓ | Matches orientation |
| Front-high | Head + torso 3/4 front ✓ | Matches |
| Right | Full profile, nose-to-right ✓ | Matches |
| Bottom | Belly + legs splayed, **tail clean** | **Better than GS-LRM v3** (no elongated artifact) |
| Back-high | Rear + tail base visible ✓ | Matches |
| Left | Full profile, nose-to-left ✓ | Matches |

Mouse centroid in GS-LRM space: `(-0.24, 0.28, -0.77)`. Within expected range (prior: `[0.33, -0.23, -0.63]` in frame 0; mouse moves across frames).

Mesh bounds in GS-LRM: `[-0.37, -0.43, -0.94]` to `[0.05, 0.64, -0.49]` — max extent ~0.9, well within camera radius 2.7. ✓

---

## 5. ICML Motivation — Empirical Support

User's original hypothesis: "Bottom view에서 GS-LRM artifact 있는데, mesh로 보완 가능성."

**Verified**: MAMMAL mesh Bottom view render shows:
- Clean tail geometry (no elongated white streak)
- Clear belly and limb structure
- Expected mouse silhouette

**vs GS-LRM Bottom view (from 260306 v3 reference image)**:
- Elongated tail artifact
- Blurry/distorted geometry from flat Gaussians seen edge-on

**Implication for ICML workshop**: Sparse (keypoint, 22 pts) → Dense (3DGS, ~100K Gaussians) feature chain can be supplemented by **Mesh (14522 verts, ~28800 faces)** as intermediate representation providing:
- Clean geometry for views unseen by GT cameras
- Structural prior where GS-LRM underfits (bottom/extrapolated views)

---

## 6. Devil's Advocate + Audit Corrections

### What was OVERSTATED in prior session claims

1. **"mammal_to_gslrm() alone"** for mesh rendering — **WRONG**. Needs camera CV→GL flip too. Corrected in MVP script.

2. **"4 coord docs contradict each other"** — **EXAGGERATED**. Actual state:
   - 2 code SSOTs (MAMMAL `coordinate_transform.py`, FaceLift `coordinate_utils.py`) handle different concerns and are consistent
   - 1 older Obsidian note has "Y-up" mention in unrelated context
   - 1 "pending" comment in FaceLift doc is stale but not load-bearing
   - **No active code conflict.**

3. **"All doc/code/memory mass update"** — **REFRAMED**. Code correct, docs lag. Solution: cross-reference annotations where needed, not wholesale rewrite.

4. **"Cross-project symlink SSOT"** — **DEFERRED**. BehaviorSplatter already has independent novel view code (`camera_trajectory.py::get_novel_view_cameras()`). Integration must audit cross-dependencies first. Next session.

### What was CORRECT

1. **Empirical verification priority** — MVP render before wholesale doc revision was the right call.
2. **v3 6-view from user's reference** — preserved exact angles (Top +80/270, etc.)
3. **Camera flip necessity** — `diag(1,-1,-1,1)` correctly applied after initial correction

---

## 7. Next Steps (Conditional)

### Immediate (if user wants)
- **Batch 3600 frames** — scale MVP to full canon sequence × 6 views → video (1-2h estimate)
- **Overlay with GS-LRM renders** — side-by-side: mesh view vs GS-LRM view for direct artifact comparison (requires GS-LRM render at matching cameras)

### Near-term (next session)
- **Doc cross-references** — add 1-line pointers in MAMMAL `COORDINATES.md` and FaceLift `COORDINATE_SYSTEMS.md` linking to this note as bridge validation record
- **Remove stale "MAMMAL 축 미확인" comment** from FaceLift `COORDINATE_SYSTEMS.md` L15 — replaced with "Verified 2026-04-17 in novel_view_mvp" reference

### Deferred (further sessions)
- **BehaviorSplatter novel view integration** — audit their `camera_trajectory.py` API, decide shared interface
- **Cross-project SSOT architecture** — options: pip-installable package, single canonical doc with URL pointers, or intentional versioned copies
- **ICML writeup section** — Sparse→Dense→Mesh feature chain as hypothesis

---

## 8. Files Created This Session (MVP scope)

- `scripts/novel_view_mvp.py` — initial MVP render script (untextured, **deprecated** 2026-04-17 after superset available)
- `scripts/novel_view_render.py` — **production superset**: textured + batch + MP4 video + reuses v3 6-view config
- `results/novel_view_mvp/frame_001800/` — untextured 1-frame validation (gpu03 + local)
- `results/novel_view_mvp/frame_1800_textured/` — textured 1-frame validation (gpu03)
- `results/novel_view_batch/canon_3600/` — full 3600-frame textured batch + MP4 (gpu03)
- `docs/coordinates/MAMMAL_FACELIFT_BRIDGE.md` — cross-project coord SSOT bridge
- This research note

## 9. Measurement Gate Compliance

Per `~/.claude/rules/measurement-gate.md`:

| Claim | Source |
|-------|--------|
| Mouse centroid in GSLRM = (-0.235, 0.279, -0.774) | `tool_use:Bash` script output |
| Mesh bounds in GSLRM max extent ~0.9 | `tool_use:Bash` script output |
| 6 views empirically render correct orientation | `tool_use:Read` grid_2x3.png image |
| `precompute_maps.py` applies camera flip diag(1,-1,-1,1) | `tool_use:Agent[Fact Checker]` — file line reference |
| Bottom view mesh cleaner than GS-LRM reference | `tool_use:Read` local + GoogleDrive reference image comparison |

### HYPOTHETICAL (unverified this session)

- Batch 3600 frames will complete in 1-2h (estimate based on single-frame time ~5s × 3600 ÷ 4 workers)
- Other MAMMAL frames will render identically correctly (only tested frame 1800)

---

*Research Note v1.0 | 2026-04-17 PM | MVP success, coord transform validated*
