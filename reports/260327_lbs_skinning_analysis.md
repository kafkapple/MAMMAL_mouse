# MAMMAL LBS Skinning & Rigging Analysis Report

> Navigation: [← MOC](../MOC.md) | [← Experiments](../EXPERIMENTS.md)
> Created: 2026-03-27
> Method: Multi-model deliberation (Claude Sonnet + Gemini Pro + GPT-4o)

## 1. Executive Summary

MAMMAL은 이미 **140-joint LBS (Linear Blend Skinning)** 체계를 갖추고 있다.
이는 SMPL(인간)/SMAL(동물)과 동일한 근간 기술이며, params slerp 보간, pose transfer,
PCA 기반 속도 개선 등을 즉시 활용할 수 있다. 다만 blend shapes가 없어 극단 포즈에서
candy-wrapper artifact가 발생하며, 이는 중기 개선 과제이다.

## 2. 핵심 개념: Skinning vs Fitting

```
Skinning (Forward):  θ (관절 각도) → LBS → 3D Mesh
Fitting  (Inverse):  Multi-view Images → Optimize θ → LBS → 3D Mesh

Skinning = 도구 (즉시, <1ms)
Fitting  = 도구를 사용하는 역문제 (14min/frame)
```

### 2.1 LBS (Linear Blend Skinning) 원리

```
T-pose 꼭짓점 v_i에 대해:
  v_i' = Σ_j (w_ij × T_j) × v_i

  w_ij = 꼭짓점 i가 관절 j에 받는 가중치 (skinning weight)
  T_j  = 관절 j의 변환행렬 (rotation + translation)

MAMMAL 구체:
  - 14,522 vertices × 140 joints → skinning weight matrix [14522, 140]
  - Parameters: thetas [140, 3] (axis-angle), bone_lengths [28], R, T, s
  - chest_deformer: Y축 흉부 변형 (추가 표현력)
```

### 2.2 왜 Vertex Lerp가 문제이고 Params Slerp가 해결인가

```
Vertex Lerp (OBJ 기반):
  v_interp = v_A × (1-t) + v_B × t
  문제: 관절 회전 무시 → 팔꿈치/어깨 찌그러짐 (volume loss)

Params Slerp (LBS 기반):
  θ_interp = slerp(θ_A, θ_B, t)    ← 회전을 구면 보간
  v_interp = LBS_forward(θ_interp)  ← 자연스러운 중간 포즈

비유:
  Vertex Lerp = "종이접기 작품 A와 B를 녹여서 섞기" → 형태 붕괴
  Params Slerp = "접는 각도를 A에서 B로 서서히 바꾸기" → 자연스러운 변환
```

### 2.3 MAMMAL vs SMPL/SMAL 비교

| 특성 | MAMMAL (Mouse) | SMPL (Human) | SMAL (Animal) |
|------|---------------|-------------|--------------|
| Vertices | 14,522 | 6,890 | 3,889 |
| Joints | 140 | 24 | 33 |
| Skinning | LBS | LBS | LBS |
| Shape space | ❌ 없음 | PCA β (10D) | PCA β (41D) |
| Pose blend shapes | ❌ 없음 | ✅ 207 shapes | ✅ pose correctives |
| Joint limits | ❌ 미구현 | ✅ | ✅ |
| Learned from | 수동 설계 | ~10K body scans | ~40 toy animals |

**핵심 차이**: MAMMAL은 skinning 인프라는 갖추었으나, SMPL/SMAL이 갖춘
shape space와 pose-dependent correctives가 없어 표현력이 제한적.

## 3. Worst Frame 진단

### 3.1 증상

| Frame | Mean IoU (fast) | Mean IoU (accurate) | Delta |
|-------|----------------|-------------------|-------|
| 10080 | 0.737 | 0.774 | +0.037 |
| 9840 | 0.724 | 0.760 | +0.036 |
| (good ref) 1920 | 0.702 | 0.869 | +0.167 |

Accurate가 거의 개선되지 않음 (Δ~0.04 vs 일반 Δ~0.15).

### 3.2 원인 분석

**Rearing (뒷다리 서기) 자세** — 100.8초, 98.4초 시점 (연속 행동 에피소드)

- 쥐가 뒷다리로 서서 몸을 수직으로 세운 상태 (grooming/exploration)
- MAMMAL skeleton = 사족보행 기준 설계 → rearing은 극단적 out-of-distribution
- 일부 뷰에서 extreme foreshortening (몸이 카메라 방향으로 축약)
- Pre-computed mask 없음 → silhouette rendering 기반 mask에 의존
- **Local minimum**: 사족보행 초기값에서 수직 자세로 수렴 불가능

### 3.3 개선 방안

| 방안 | 난이도 | 기대 효과 |
|-----|-------|---------|
| Rearing 초기 포즈 템플릿 | 낮음 | 해당 프레임 fitting 대폭 개선 |
| Spine joint limits 완화 | 낮음 | 극단 굴곡 허용 |
| Multi-init fitting (3개 초기값) | 중간 | Best-of-3로 local min 탈출 |
| Behavior detection → 초기값 분기 | 중간 | 자동화된 포즈 분류 |

## 4. LBS 활용 로드맵

### Phase 1: 즉시 (현재 파이프라인)

**Params Slerp 보간 구현** — `scripts/interpolate_keyframes.py`에 이미 계획됨

```python
# Quaternion-based slerp (axis-angle 직접 slerp은 불안정)
def slerp_thetas(theta_A, theta_B, t):
    q_A = axis_angle_to_quaternion(theta_A)  # [140, 4]
    q_B = axis_angle_to_quaternion(theta_B)  # [140, 4]
    # Sign consistency (q와 -q는 같은 회전)
    dot = (q_A * q_B).sum(dim=-1)
    q_B[dot < 0] *= -1
    q_interp = quaternion_slerp(q_A, q_B, t)
    return quaternion_to_axis_angle(q_interp)
```

⚠️ **Critical**: Quaternion sign-flip 미처리 시 메시가 갑자기 뒤집히는 artifact 발생

### Phase 2: 단기 (1-2주)

**PCA Pose Prior** — 900 keyframe의 theta 데이터로 pose PCA 학습

```
900 × [140, 3] thetas → flatten [900, 420] → PCA
예상: 20-40 PC가 95%+ 분산 설명
→ fitting 시 420 params 대신 40 params 최적화 = ~10× 속도 향상
→ PCA 공간 밖 포즈 자동 제한 (regularization 효과)
```

**Joint Angle Limits** — 관절별 ROM (Range of Motion) 제한

```python
loss_total = loss_silhouette + loss_keypoint + λ * loss_joint_limits
# 해부학적으로 불가능한 포즈 페널티 → 수렴 속도 향상
```

### Phase 3: 중기 (1-2개월)

**Pose-Dependent Correctives (Blend Shapes)**

```
1. 900 keyframe의 fitted mesh와 LBS mesh 간 residual 계산
   delta_v = v_observed - v_lbs(theta)
2. delta_v를 theta의 함수로 회귀 (linear model 또는 MLP)
3. v_corrected = v_lbs(theta) + f(theta)  ← SMPL 방식
```

### Phase 4: 장기

**Feed-forward Pose Estimation**

```
Image features → Neural network → θ prediction (< 100ms)
→ LBS forward → Mesh
→ 현재 14min → 0.1s (8400× speedup)
필요: 900+ keyframe supervision data (이미 생산 중)
```

## 5. 실용적 결론

### 즉시 활용 가능한 것
- **Params slerp**: 900 keyframe → 3600 frame 자연스러운 보간 (vertex lerp artifact 제거)
- **Parametric 저장**: 14,522×3 vertex 대신 ~500 float로 압축
- **Pose transfer**: skeleton 공유로 다른 쥐에 포즈 전이 가능

### 현재 파이프라인에서 달라지는 것
- `interpolate_keyframes.py`가 OBJ vertex 대신 **pkl params**를 slerp → LBS forward
- 보간 품질 대폭 향상 (특히 관절 부근)
- 추가 비용: 없음 (LBS forward < 1ms)

### Multi-view Fitting이 여전히 필요한 이유
- LBS는 **순방향 함수**일 뿐, θ를 알아내는 것은 별개 문제
- 6대 카메라 관측 → θ 역추정 = 본질적으로 비싼 inverse problem
- 장기적으로 feed-forward network가 대체 가능 (Phase 4)

## 6. Worst Frame Audit (Frames 10080, 9840)

### 6.1 Audit 결과 (3-model consensus)

| # | Severity | Finding | Fix |
|---|----------|---------|-----|
| 1 | Critical | T-pose 초기값 → rearing 수렴 불가 | Multi-hypothesis init (best-of-3) |
| 2 | Critical | Slerp으로 rearing 경유 시 invalid 중간 포즈 | Behavior-aware interpolation |
| 3 | Major | Occluded keypoint loss가 역효과 | Per-view visibility masking |
| 4 | Major | mask_step2=3000 일부 뷰 과적합 | Adaptive early stopping |
| 5 | Major | Bone lengths가 프레임마다 변동 | 해부학 상수로 고정 |

### 6.2 Rearing 프레임 개선 실험 계획

```
E1: Baseline         — 현재 config (T-pose init, accurate)     → IoU ~0.76
E2: Rearing init     — manual rearing pose 초기값               → IoU +0.08~0.12 예상
E3: Occlusion mask   — T-pose + 숨겨진 keypoint 비활성화       → IoU +0.03~0.05 예상
E4: Adaptive mask    — T-pose + mask_step2 early stopping      → 과적합 방지
E5: Combined         — rearing init + occlusion + adaptive     → 최대 효과
E6: Multi-hypothesis — 3개 초기값 중 best loss 선택 + E5        → 가장 robust
```

### 6.3 Rearing 감지 (자동)

```python
# Spine vector heuristic (step0 3D keypoints 기반)
V = kp3d['neck'] - kp3d['pelvis']  # 3D spine vector
V_norm = V / np.linalg.norm(V)
is_rearing = V_norm[1] > 0.7  # Y-up threshold (MAMMAL: -Y up → 반전 필요)
```

### 6.4 추가 평가 지표 (IoU 외)

- **P-KPE**: Keypoint reprojection error (visible keypoints only, pixels)
- **Bone Length Stability**: per-bone std across frames (should ≈ 0)
- **Self-Intersection Score**: trimesh face intersection count
- **Temporal Jitter**: MPJPE between consecutive frames (mm)

## 7. Params Slerp 상세 설명

### 7.1 전체 파이프라인

```
900 keyframe PKL files
    ↓
[Load params]  thetas[140,3], bone_lengths[28], R[3], T[3], s[1], chest_deformer[1]
    ↓
[Convert]      axis-angle → quaternion (per joint): q[140,4]
    ↓
[Sign check]   dot(q_prev, q_curr) < 0 → negate q_curr (shortest path)
    ↓
[Slerp]        q_interp = slerp(q_A, q_B, t)  ← 140 joints 각각 독립
               T_interp = lerp(T_A, T_B, t)
               s_interp = lerp(s_A, s_B, t)
               chest_interp = lerp(chest_A, chest_B, t)
               bone_lengths = CONSTANT (세션 평균, lerp 아님!)
    ↓
[Convert back] quaternion → axis-angle
    ↓
[LBS forward]  ArticulationTorch.forward(thetas, bone_lengths, R, T, s, chest)
    ↓
[Output]       14,522 vertices → OBJ file
```

### 7.2 핵심 수식

```
Quaternion Slerp:
  slerp(q₁, q₂, t) = q₁ · sin((1-t)·Ω) / sin(Ω) + q₂ · sin(t·Ω) / sin(Ω)
  where Ω = arccos(q₁ · q₂)

LBS Forward:
  v'ᵢ = Σⱼ wᵢⱼ · Gⱼ(θ) · vᵢ
  where Gⱼ(θ) = ∏ₖ∈ancestors(j) Rₖ(θₖ) · Tₖ

Vertex lerp vs Params slerp:
  Vertex: v = (1-t)·vₐ + t·vᵦ           ← 회전 무시, 관절 찌그러짐
  Params:  θ = slerp(θₐ, θᵦ, t) → LBS(θ) ← 회전 보간, 자연스러운 포즈
```

### 7.3 Behavior Transition 처리

```
일반 구간:       keyframe_A ──slerp──→ keyframe_B  (같은 행동)
전환 구간:       quadruped ──?──→ rearing  (다른 행동)

해결: 전환 구간에 추가 keyframe 삽입
  walk_frame → transition_frame_1 → transition_frame_2 → rearing_frame
  (기존 interval=20 → 전환부 interval=4~5로 밀집)

  또는: per-joint-group 별도 alpha (spine 먼저, 사지 나중)
```

## 8. Multi-Model Deliberation Summary

### Round 1: LBS/Skinning 분석 (MoA)

| 관점 | 핵심 기여 |
|-----|---------|
| 🔵 Claude | Quaternion sign-flip 경고, PCA speedup, 코드 예시 |
| 🟢 Gemini | Phase별 검증-우선 로드맵, skinning weight 품질 리스크 |
| 🟠 GPT | Blend shapes → Neural deformer 장기 대안 |

### Round 2: Worst Frame Audit + 개선 (Audit → MoA)

| 관점 | 핵심 기여 |
|-----|---------|
| 🔵 Claude | Multi-hypothesis fitting, OcclusionAwareLoss, 7-experiment suite |
| 🟢 Gemini | Spine vector classifier, Z-buffer occlusion, IK 기반 보간, 검증 기준 |
| 🟠 GPT | Phased roadmap, adaptive mask weight, behavior-specific sub-pipeline |

**Full consensus (6/6)**: Rearing init가 가장 큰 개선 (>parameter tuning)

---

## References

- MAMMAL: An et al., "Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL", Nature Communications 2023
- SMPL: Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH Asia 2015
- SMAL: Zuffi et al., "3D Menagerie: Modeling the 3D Shape and Pose of Animals", CVPR 2017
- MoA: Wang et al., "Mixture-of-Agents Enhances Large Language Model Capabilities", ICLR 2025

---

*Generated: 2026-03-27 | Method: MoA Deliberation (Sonnet + Gemini + GPT-4o) + Worst Frame Diagnosis*
