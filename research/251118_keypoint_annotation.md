# 251118 연구노트 — Keypoint Annotation 시스템

## 목표
- MAMMAL mesh fitting에 유연한 keypoint annotation workflow 구축
- 1~22개 keypoint 유연 지원 + confidence 기반 자동 필터링 구현
- 기존 fit_cropped_frames.py의 pose 고정 문제 분석 및 해결 방향 제시

## 진행 내용

### 1. 기존 문제 분석

`fit_cropped_frames.py`에서 `thetas` (joint angles)가 optimizer에서 제외되어 neutral pose로 고정:

```python
# fit_cropped_frames.py:58-70
params = {"thetas": torch.zeros(1, 140, 3)}  # 0으로 고정

# fit_cropped_frames.py:97-100
optimizer = torch.optim.Adam([
    {"params": [params["translation"]]},  # translation만 최적화
    {"params": [params["scale"]]},
])
```

### 2. MAMMAL Keypoint 처리 메커니즘 분석

**핵심 발견 — Confidence 기반 자동 필터링**:

```python
# fitter_articulation.py:214
diff = (J2d_projected - target_2d) * confidence
loss = mean(norm(diff * keypoint_weight))
```

- `confidence > 0` → loss에 기여
- `confidence = 0` → loss에서 자동 제외
- **결론: 1개 keypoint만 있어도 동작**

**Data loading threshold**: `data_seaker_video_new.py`에서 confidence < 0.25인 keypoint는 (0, 0, 0)으로 자동 제거

### 3. MAMMAL 3단계 최적화

| Stage | 최적화 대상 | KP Weight | Mask Weight |
|-------|-------------|-----------|-------------|
| **Step 0** | rotation, translation, scale | Normal | 0 |
| **Step 1** | + thetas, bone_lengths | Normal | 0 |
| **Step 2** | + chest_deformer | Foot x10 | 3000 |

**Loss 비중**:
```
total_loss = keypoint_2d * 0.2 + theta_reg * 3 + bone_length * 0.5
           + scale * 0.5 + mask * 3000 (Step 2 only)
```

**Mask는 선택사항** — Step 0-1은 keypoint만, Step 2에서 mask 추가 (미세 조정용)

### 4. Annotation Pipeline 구현

```
Manual Annotation (Gradio UI) → JSON → Converter → PKL → MAMMAL Fitter → 3D Mesh
```

**구성 요소**:

| 모듈 | 파일 | 역할 |
|------|------|------|
| Annotator | `keypoint_annotator_v2.py` | Gradio UI, 7 core keypoints |
| Converter | `convert_keypoints_to_mammal.py` | JSON → NumPy (N, 22, 3) |
| SAM Extractor | `extract_sam_keypoints.py` | SAM clicks에서 point 추출 (비권장) |
| Test | `test_keypoint_conversion.sh` | E2E 테스트 |

### 5. Keypoint Mapping (7 manual → 22 MAMMAL)

```python
KEYPOINT_MAPPING = {
    "nose": 0, "neck": 1, "spine_mid": 2,
    "hip": 3, "tail_base": 4, "left_ear": 5, "right_ear": 6,
    # Indices 7-21: Reserved (conf=0.0, 자동 무시)
}
```

### 6. Data Format

**Manual JSON**:
```json
{"frame_000000": {"nose": {"x": 50.0, "y": 30.0, "visibility": 1.0}}}
```

**MAMMAL PKL**: NumPy array `(num_frames, 22, 3)` — [x, y, confidence]

## 핵심 발견
- **MAMMAL은 매우 유연**: 1~22개 keypoint, confidence 기반 자동 필터링
- **Mask 없이 keypoint만으로 충분한 fitting 가능** (mask는 Step 2 미세 조정용)
- **최소 5-7개 keypoint 권장**: spine landmarks (nose, spine_mid, hip, tail_base) + ears

## 최소 Keypoint 구성별 품질

| Keypoints | 품질 | 용도 |
|-----------|------|------|
| 1-2개 | Poor | 위치 추정만 |
| 3-4개 | Fair | 기본 orientation |
| **5-7개** | **Good** | **Full body pose** |
| 10+개 | Excellent | 세부 사지 정확 |

## 성능 수치
- Manual annotation: ~30초/frame (7 keypoints)
- Format conversion: <1초 (100 frames)
- Mesh fitting: ~10초/frame (iteration 수에 따라 변동)

## 미해결 / 다음 단계
- ML-assisted annotation: SuperAnimal/DLC pretrained → 수동 교정
- Auto-interpolation: key frames만 annotate, 중간 프레임 보간
- Temporal consistency: optical flow propagation
- Extended keypoint set: 발 (7-14), 꼬리 세그먼트 (15-16)

---
*Sources: 251118_keypoint_annotation_system.md*
