# 251117 연구노트 — Mesh Fitting 시스템 구축 및 E2E 파이프라인

## 목표
- Multi-dataset 지원 mesh fitting 시스템 체계적 정리 (default / cropped / upsampled / custom)
- 일반 생쥐 영상 대상 End-to-End 파이프라인 구축 (Video → SAM → Crop → Fitting → 3D Mesh)
- PyTorch3D 호환성 문제 해결

## 진행 내용

### 1. Multi-Dataset Configuration System

4가지 dataset type을 Hydra config로 통합 관리:

| Dataset | 위치 | 특징 | 용도 |
|---------|------|------|------|
| **Default Markerless** | `data/examples/markerless_mouse_1_nerf/` | 6-view, masks, keypoints | Reference/validation |
| **Cropped** | `data/100-KO-male-56-20200615_cropped/` | Single-view, masks, crop metadata | Silhouette fitting |
| **Upsampled** | `data/100-KO-male-56-20200615_upsampled/` | Single-view, high-res, no masks | 전처리 필요 |
| **Custom** | User-defined | Flexible | 사용자 실험 |

Config 파일: `conf/dataset/{default_markerless, cropped, upsampled, custom}.yaml`

### 2. E2E Video Processing Pipeline

```
Video (15 min, 640x480, 30fps)
  → Frame Extraction (20 frames, evenly spaced)
    → SAM 2.1 Annotation (Gradio UI, interactive)
      → Frame Cropping (bbox + 50px padding)
        → 3D Mesh Fitting (silhouette-based, neutral pose)
          → Visualization (3-panel: target / rendered / overlay)
```

**실험 비디오**: `100-KO-male-56-20200615.avi`

### 3. SAM Annotation 구현

- SAM 2.1 Hiera Large 모델, Gradio web UI (port 7860)
- SSH tunnel로 원격 접속 지원
- `conda run` + Hydra 충돌 해결: `run_sam_gui.py` (Hydra 우회, OmegaConf 직접 사용)

### 4. Silhouette 기반 Mesh Fitting

최적화 대상: Translation (XYZ) + Scale만 (pose 고정)

```python
optimizer = torch.optim.Adam([
    {"params": [params["translation"]]},
    {"params": [params["scale"]]},
])
# thetas (joint angles)는 최적화에서 제외 → 고정된 neutral pose
```

### 5. Convenience Scripts

| Script | 용도 |
|--------|------|
| `run_mesh_fitting_default.sh` | Default dataset fitting |
| `run_mesh_fitting_cropped.sh` | Cropped frames fitting |
| `run_mesh_fitting_custom.sh` | Custom configuration |
| `run_quick_test.sh` | 3-frame 빠른 테스트 (~1분) |

### 6. PyTorch3D 호환성 해결

**문제**: PyTorch3D 0.7.8 (precompiled) vs PyTorch 2.0.0+cu118 incompatible
(`ImportError: undefined symbol: _ZNK3c105Error4whatEv`)

**해결**: Source 빌드 (`pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"`)
→ 5-10분 소요, `fix_pytorch3d.sh` 스크립트화

### 7. ArticulationTorch API 수정

4가지 API mismatch 해결:

```python
# Correct usage
vertices, _ = bodymodel.forward(
    thetas=params["thetas"],              # Not "theta"
    bone_lengths_core=params["bone_lengths"],  # Shape [1, 28], not 20
    R=params["rotation"],
    T=params["translation"],
    s=params["scale"],
    chest_deformer=params["chest_deformer"]
)
rendered_mask = renderer.render_from_vertices_faces(vertices, faces, camera)
```

## 핵심 발견
- **Silhouette-only fitting의 IoU**: 46-51% (neutral pose 제약 하에서)
- Silhouette만으로는 정확한 pose 복원 불가 → keypoint supervision 필수
- Pose 최적화 시 learning rate 높으면 mesh가 화면 밖으로 이탈 → translation+scale만 안정적
- PyTorch3D는 반드시 source 빌드 권장 (binary 호환성 문제 빈번)

## 성능 수치

| 단계 | 시간 |
|------|------|
| Frame extraction | <1분 (20 frames) |
| SAM annotation | ~3-5분/frame (수동) |
| Frame cropping | <5초 (20 frames) |
| Mesh fitting | ~2-3분/frame (RTX 3060) |

## 미해결 / 다음 단계
- Keypoint detection 통합 (DLC, SLEAP 등) → 정확한 pose 복원
- Multi-stage optimization (global → coarse pose → fine pose)
- Temporal consistency (optical flow 기반 tracking)
- Batch processing 도구 개발

## 생성된 파일 (11개)

**Config**: `conf/dataset/{default_markerless, cropped, upsampled}.yaml`
**Scripts**: `run_mesh_fitting_{default,cropped,custom}.sh`, `run_quick_test.sh`, `fix_pytorch3d.sh`
**Docs**: `docs/MESH_FITTING_GUIDE.md` (40KB), `MESH_FITTING_CHEATSHEET.md` (6.6KB), `docs/PYTORCH3D_FIX.md`

---
*Sources: 251117_mesh_fitting_system_setup.md, 251117_mouse_video_processing_summary.md*
