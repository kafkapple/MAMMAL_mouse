# MAMMAL ↔ FaceLift Coordinate Bridge (Verified)

**Status**: ✅ Empirically validated 2026-04-17 via Phase 1 MVP render
**Scope**: Cross-project rendering MAMMAL mesh at FaceLift-style novel view cameras
**Supersedes**: Ambiguity in FaceLift `mouse_extensions/docs/COORDINATE_SYSTEMS.md` L15 ("MAMMAL 축 방향 미확인") and stale Obsidian note contradictions

---

## 1. TL;DR

To render a MAMMAL mesh from a FaceLift novel-view camera:

```python
# 1. Load MAMMAL mesh (native: -Y up, +X head, +Z right, mm)
verts_mm, faces = load_obj(path)

# 2. Translate + scale only — NO AXIS SWAP
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])  # mm
M5_DISTANCE_SCALE = 2.7 / 307.785                        # ≈ 0.008781
verts_gslrm = (verts_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE

# 3. Camera: OpenCV c2w from spherical coords
c2w_cv = get_turntable_cameras(elev_deg, azim_deg, radius=2.7)  # or equivalent

# 4. For OpenGL renderers (pyrender etc.): flip Y and Z of the camera pose
c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])
# Apply flip to CAMERA ONLY. Mesh stays in OpenCV/FaceLift world unchanged.

# 5. Render
scene.add(mesh, pose=identity)       # mesh uses world coords
scene.add(camera, pose=c2w_gl)
```

**This pipeline is verified visually correct for all 6 v3 novel views** (Top/Front-high/Right/Bottom/Back-high/Left).

---

## 2. Coord Systems Involved

| System | Axis (right, up, forward) | Unit | Usage |
|--------|---------------------------|------|-------|
| **MAMMAL native** | +Z right, **-Y up** (=Y-down), +X head (forward) | mm | Raw `.obj` output from fitter |
| **FaceLift GS-LRM world (OpenCV)** | +X right, -Y up (=Y-down), +Z forward | normalized (~[-1, 1]) | Camera rig frame, GS-LRM input space |
| **pyrender / OpenGL camera frame** | +X right, +Y up, -Z forward (Z-back) | same as GS-LRM | Camera-internal for pyrender |

**Key observation**:
- MAMMAL and FaceLift both use **Y-down** (both "-Y up" in magnitude). This axis is already aligned.
- Horizontal axes (MAMMAL +X head vs FaceLift +Z forward, MAMMAL +Z right vs FaceLift +X right) are **semantically swapped but geometrically free** — mouse can face any direction in world; what matters is internal consistency.
- As long as (mesh + camera) are in the same world frame, rendering is correct. Mouse orientation relative to camera is set by `mammal_to_gslrm()` centering + camera spherical pose, not by axis labels.

---

## 3. Why "axis swap" Is Not Needed

This was a source of confusion in prior docs (FaceLift doc L15 marked "MAMMAL 축 방향 미확인"). The resolution:

| Hypothesis | Verdict | Evidence |
|-----------|:------:|----------|
| "MAMMAL +X head ≠ FaceLift +Z forward → need swap" | ❌ False | Both are right-handed world frames. Mouse "forward" is determined by body pose, not world axis. Camera sees whatever the mesh looks like in world after centering. |
| "Y-down in both systems means Y-axis is aligned" | ✅ True | Renders correctly in Top/Bottom view — dorsal/ventral correctly identified |
| "Need `(x, z, -y)` MAMMAL → Blender transform for novel views" | ❌ False | That transform is for Blender's Z-up world. FaceLift uses Y-down OpenCV world. Different target. |
| "`mammal_to_gslrm()` single-step insufficient" | 🟡 Partial | Mesh is covered by `mammal_to_gslrm()`. Camera additionally needs OpenCV→OpenGL flip `diag(1,-1,-1,1)`. Two separate concerns, not missing third step. |

---

## 4. v3 Novel 6-View Camera Config (Canonical)

Source: `260306_2nd_phase/_novel_view_rendering/novel_6view_temporal_v3/novel_6view_grid.png` (image labels).

| View | Elevation (°) | Azimuth (°) | Grid |
|------|:-----:|:-----:|:----:|
| Top | +80 | 270 | row 0, col 0 |
| Front-high | +40 | 270 | row 0, col 1 |
| Right | +20 | 0 | row 0, col 2 |
| Bottom | -85 | 270 | row 1, col 0 |
| Back-high | +40 | 90 | row 1, col 1 |
| Left | +20 | 180 | row 1, col 2 |

- Radius: 2.7 (FaceLift canonical)
- Up vector: `(0, 0, 1)` (Z-up in world)
- FOV: 50° yfov
- Resolution: 512×512
- Camera convention: OpenCV

Note: Current FaceLift `cinematic_sequence.py::_seg_grid_novel_6views` uses different elevations [80, 70, 60, -85, -40, -30] (newer iteration). The v3 above is preserved as user-confirmed reference for cross-project bridging.

---

## 5. Transform Pipeline Code Reference

### Canonical module
`mammal_ext/novel_view_render.py` (implementation) — `scripts/novel_view_render.py` (entry).

### Key constants (do not duplicate)

**SSOT**: `FaceLift/mouse_extensions/coordinate_utils.py`

```python
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])  # mm, M5 6-cam rig centroid
M5_DISTANCE_SCALE = 2.7 / 307.785  # ≈ 0.008781
```

For cross-project usage, hardcode these values or import via explicit path. Keep in sync manually — if FaceLift upgrades to M6 scene, update here too.

---

## 6. Validation Record

### 2026-04-17 MVP (frame 1800, canon_slerp)

- Mouse centroid in GSLRM: `(-0.235, 0.279, -0.774)` — within expected range (prior verified `[0.33, -0.23, -0.63]` for frame 0)
- Mesh bounds in GSLRM: `[-0.37, -0.43, -0.94]` to `[0.05, 0.64, -0.49]` — max extent 0.94, well within camera radius 2.7
- Rendered views empirically correct:
  - Top (+80°): dorsal (spine + tail from above)
  - Bottom (-85°): ventral (belly + splayed limbs)
  - Right (a=0°): right profile, nose right
  - Left (a=180°): left profile, nose left
  - Front-high (+40°, 270°): head + torso 3/4 front
  - Back-high (+40°, 90°): rear + tail base

### Downstream comparison
FaceLift GS-LRM Bottom view (from 260306 v3 reference) shows **elongated tail artifact** from edge-on flat Gaussians. MAMMAL mesh Bottom view in our MVP shows **clean tail geometry**. This confirms ICML workshop hypothesis: mesh can supplement GS-LRM in under-observed views.

---

## 7. Resolution of Prior Doc Contradictions

### Docs surveyed (4 sources)

1. `MAMMAL_mouse/docs/reference/COORDINATES.md` — "**-Y up, +X head, +Z right**" (authoritative, vertex analysis verified 2026-02-06)
2. `FaceLift/mouse_extensions/docs/COORDINATE_SYSTEMS.md` L15 — "MAMMAL 축 방향 **미확인 (OBJ 로딩 후 검증 필요)**" (stale, predates 2026-04-17 MVP)
3. `Obsidian/30_Projects/_CODES/FaceLift/docs/theory/COORDINATE_SYSTEMS.md` — math basics, no MAMMAL-specific claims
4. `Obsidian/30_Projects/2603_NeurIPS_3D-Animal-Recon/_Notes/_queue/.../coordinate system mismatch.md` (2026-03-21) — contains "MAMMAL body model | Y-up" in a debug comparison table. **Context**: general animal-model orientation comparison, not MAMMAL-specific ground truth claim.

### Reconciliation

- **MAMMAL project doc is authoritative**: -Y up, +X head, +Z right.
- FaceLift "미확인" is a pending annotation predating verification. Should be updated to "Verified 2026-04-17: -Y up, mammal_to_gslrm() + camera flip sufficient — see `MAMMAL_mouse/docs/coordinates/MAMMAL_FACELIFT_BRIDGE.md`".
- Obsidian "Y-up" note is preserved as historical debug record (different context). No edit needed.

### Actions to propagate (light touch, per Devil audit)

- [ ] Add 1-line pointer in `MAMMAL_mouse/docs/reference/COORDINATES.md` → this bridge doc (MAMMAL ↔ FaceLift section)
- [ ] Update `FaceLift/mouse_extensions/docs/COORDINATE_SYSTEMS.md` L15 to reference this doc instead of "미확인"
- [ ] Cross-reference in `FaceLift/mouse_extensions/docs/CAMERA_CALIBRATION.md` (if mesh-rendering section exists)

**NOT required** (scope deferred):
- Symbolic link cross-project SSOT (per Devil audit: BehaviorSplatter has independent novel-view code)
- Mass memory/Obsidian rewrite (no active code conflict)

---

## 8. Failure Modes to Watch

If someone tries to use this and gets wrong results, suspect (in order):

1. **Forgot `mammal_to_gslrm()`**: mesh appears far from camera, tiny or off-screen
2. **Forgot camera flip `@ diag(1,-1,-1,1)`**: mesh upside-down or mirrored
3. **Applied flip to mesh instead of camera**: mesh visible but inverted handedness (inside-out)
4. **OBJ face winding**: if mesh looks "inside-out" with backface culling on, flip faces or disable culling
5. **Wrong `M5_SCENE_CENTER`**: mesh centroid very far from origin (> 1.0 in GSLRM space)
6. **Wrong radius**: camera too close (cropped) or too far (mesh tiny)

Debug sequence:
```python
# After mammal_to_gslrm
print(f"GSLRM bounds: {verts_g.min(0)}, {verts_g.max(0)}")
print(f"GSLRM centroid: {verts_g.mean(0)}")
# Expected: bounds within ~[-1, 1], centroid magnitude < 1.0
```

---

## 9. File References

| Artifact | Path |
|----------|------|
| This doc | `MAMMAL_mouse/docs/coordinates/MAMMAL_FACELIFT_BRIDGE.md` |
| Implementation | `MAMMAL_mouse/scripts/novel_view_render.py` |
| MVP result — untextured (1 frame) | `MAMMAL_mouse/results/novel_view_mvp/frame_001800/` (gpu03) / `~/results/MAMMAL/260417_novel_view_mvp/frame_001800/` (local) |
| MVP result — textured (1 frame) | `MAMMAL_mouse/results/novel_view_mvp/frame_1800_textured/` (gpu03) / `~/results/MAMMAL/260417_novel_view_mvp/grid_2x3_textured.png` (local) |
| Batch render (3600 frames + video) | `MAMMAL_mouse/results/novel_view_batch/canon_3600/` (gpu03, running) |
| Research note | `MAMMAL_mouse/docs/reports/260417_novel_view_mvp_research_note.md` |
| FaceLift coord SSOT | `FaceLift/mouse_extensions/coordinate_utils.py` |
| FaceLift camera utils | `FaceLift/mouse_extensions/visualization/camera_utils.py::get_turntable_cameras` |
| MAMMAL coord doc | `MAMMAL_mouse/docs/reference/COORDINATES.md` |

---

*Coord Bridge v1.0 | 2026-04-17 | MVP-verified transform*
