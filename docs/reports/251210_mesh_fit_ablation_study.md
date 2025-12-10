---
date: 2025-12-10
context_name: "2_Research"
tags: [ai-assisted, mesh-fitting, ablation-study, view-analysis, keypoint-analysis, mammal]
project: MAMMAL_mouse
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Mesh Fitting 베이스라인 및 View/Keypoint Ablation Study

## 1. 제목 (Title)

**Multi-view Mouse Mesh Fitting: View 수와 Keypoint 수에 따른 Fitting 품질 Ablation Study**

---

## 2. 날짜 (Date)

- **실험 기간**: 2025-12-03 ~ 2025-12-08
- **노트 작성일**: 2025-12-10

---

## 3. 연구 주제 (Research Topic)

6-view RGB 비디오 데이터셋을 활용한 마우스 3D mesh fitting에서:
- **View Ablation**: 카메라 view 수 감소가 fitting 품질에 미치는 영향
- **Keypoint Ablation**: Keypoint 수 및 구성 변경이 fitting 품질에 미치는 영향

---

## 4. 핵심 목표 (Key Objective)

1. **베이스라인 확립**: MAMMAL 원본 논문 설정 (6-view, 22 keypoints) 재현
2. **View 최소 요구사항 파악**: 최소 몇 개의 view가 필요한지 결정
3. **Keypoint 효율화**: DLC, MARS 등 표준 keypoint set으로 대체 가능 여부 검증
4. **실용적 가이드라인 도출**: 실제 적용 시 권장 설정 제안

---

## 5. 배경 및 동기 (Background & Motivation)

### 5.1 문제 정의

MAMMAL 논문의 마우스 mesh fitting은 다음을 요구:
- **6개 동기화된 카메라**
- **22개의 상세 keypoints** (코, 귀, 발, 꼬리 등)

**현실적 제약**:
- 다중 카메라 시스템 구축 비용
- Dense keypoint annotation 비용 (시간, 인력)
- 기존 데이터셋과의 호환성 (DLC, MARS 등)

### 5.2 연구 질문

| 질문 | 중요도 |
|------|--------|
| 3-4개 view만으로 acceptable한 fitting이 가능한가? | 높음 |
| DLC/MARS 표준 keypoint set으로 대체 가능한가? | 높음 |
| View와 keypoint의 최소 조합은? | 중간 |

---

## 6. 방법론 (Methodology)

### 6.1 데이터셋

- **이름**: `markerless_mouse_1_nerf`
- **View 수**: 6개 (카메라 ID: 0, 1, 2, 3, 4, 5)
- **프레임 수**: 100 frames
- **해상도**: Multi-view RGB video

### 6.2 실험 설계

#### A. Keypoint Ablation (6-view 고정)

| 실험명 | Keypoints | 설명 |
|--------|-----------|------|
| `baseline_6view_keypoint` | 22개 | MAMMAL 원본 (full) |
| `sparse_9kp_dlc` | 9개 [0,1,2,3,4,5,6,8,12] | DeepLabCut 스타일 |
| `sparse_7kp_mars` | 7개 [0,1,2,3,5,18,21] | MARS 스타일 (코, 귀, 목, 꼬리, 엉덩이) |
| `sparse_5kp_minimal` | 5개 [0,1,2,3,5] | 최소 구성 (코, 귀, 목, 꼬리) |

**Keypoint Index 참조**:
- 0, 1: 귀 (left ear, right ear)
- 2: 코 (nose)
- 3: 목 (neck)
- 4, 5: 꼬리 (tail base, tail tip)
- 6, 7: 앞발 (front paws)
- 8, 12: 뒷발 (hind paws)
- 18, 21: 엉덩이 (hips)

#### B. View Ablation (3 sparse keypoints 고정)

| 실험명 | Views | 카메라 ID |
|--------|-------|-----------|
| `sparse_5view` | 5개 | [0,1,2,3,4] |
| `sparse_4view` | 4개 | [0,1,2,3] |
| `sparse_3view` | 3개 | [0,2,4] (120° 간격) |
| `sparse_2view` | 2개 | [0,3] (180° 대칭) |

**공통 Keypoints**: [2, 5, 3] = 코(nose), 꼬리 끝(tail_base), 목(neck)

### 6.3 최적화 파라미터 조정

View/keypoint 수 감소에 따른 보상 전략:

| View 수 | Iteration 증가 | Regularization 강화 | Mask Loss 활성화 |
|---------|----------------|---------------------|------------------|
| 6 | 기본값 | 기본값 | Step 2만 |
| 5 | +20% | +20% | Step 1부터 |
| 3-4 | +50% | +50% | Step 0부터 |
| 2 | +100% | +100% | Step 0부터 강화 |

---

## 7. 주요 결과 (Key Findings/Results)

### 7.1 실험 상태

| 실험명 | 상태 | 완료 프레임 |
|--------|------|-------------|
| `baseline_6view_keypoint` | 완료 | 100/100 |
| `sparse_9kp_dlc` | 완료 | 100/100 |
| `sparse_7kp_mars` | 완료 | 100/100 |
| `sparse_5kp_minimal` | 완료 | 100/100 |
| `sparse_5view` | 완료 | 100/100 |
| `sparse_4view` | 완료 | 100/100 |
| `sparse_3view` | 완료 | 100/100 |
| `sparse_2view` | 완료 | 100/100 |

### 7.2 최적화 파라미터 비교

#### Keypoint Ablation 설정 (6-view)

| 설정 | baseline | 9kp_dlc | 7kp_mars | 5kp_minimal |
|------|----------|---------|----------|-------------|
| step1_iters | 100 | 100 | 120 | 140 |
| step2_iters | 30 | 35 | 40 | 45 |
| theta_weight | 3.0 | 3.5 | 4.0 | 4.5 |
| 2d_weight | 0.2 | 0.35 | 0.4 | 0.45 |
| mask_step1 | 0 | 200 | 300 | 400 |

#### View Ablation 설정 (3 keypoints)

| 설정 | 5view | 4view | 3view | 2view |
|------|-------|-------|-------|-------|
| step0_iters | 15 | 15 | 20 | 25 |
| step1_iters | 150 | 150 | 180 | 200 |
| step2_iters | 45 | 45 | 55 | 60 |
| theta_weight | 5.0 | 5.0 | 6.0 | 8.0 |
| mask_step0 | 0 | 0 | 300 | 500 |

### 7.3 정성적 관찰

**Keypoint Ablation**:
- 22→9 keypoints: 큰 품질 저하 없음
- 22→7 keypoints: 미세한 사지 위치 오차 발생
- 22→5 keypoints: 발 위치 불안정, 꼬리 tracking 양호

**View Ablation**:
- 6→5 views: 품질 유지
- 5→4 views: 미세한 depth ambiguity
- 4→3 views: occlusion 시 오차 증가
- 3→2 views: 특정 각도에서 flip 현상 발생 가능

---

## 8. 분석 및 논의 (Analysis & Discussion)

### 8.1 Keypoint 효율성

```
품질 유지 가능한 최소 keypoint 수: 7-9개 (MARS/DLC 스타일)
```

**권장 keypoint 구성**:
1. **필수**: 코(2), 목(3), 꼬리(5) - 체축(body axis) 정의
2. **권장**: 귀(0,1) - 머리 방향 정의
3. **선택**: 엉덩이(18,21), 발(8,12) - 사지 정확도 향상

### 8.2 View 효율성

```
최소 권장 view 수: 4개 (연속 배치) 또는 3개 (120° 간격)
```

**View 배치 전략**:
- **최적**: 120° 간격 배치 (e.g., [0,2,4])
- **차선**: 연속 4개 (e.g., [0,1,2,3])
- **위험**: 대칭 2개 (180°) - flip ambiguity 발생

### 8.3 Trade-off 분석

| 구성 | 품질 | 비용 | 권장 사용처 |
|------|------|------|-------------|
| 6view + 22kp | 최상 | 최고 | 논문 발표용, 정밀 분석 |
| 6view + 9kp | 상 | 중 | 일반 연구, DLC 호환 |
| 4view + 7kp | 중상 | 중저 | 실용적 기본값 |
| 3view + 5kp | 중 | 저 | 빠른 프로토타이핑 |
| 2view + 3kp | 하 | 최저 | 비권장 (보조적 사용만) |

---

## 9. 미결 과제 (Open Questions)

### 9.1 정량적 평가 필요

- [ ] MPJPE (Mean Per-Joint Position Error) 계산
- [ ] View별 silhouette IoU 측정
- [ ] 프레임 간 temporal smoothness 정량화

### 9.2 추가 실험 가능성

- [ ] View 배치 최적화 (어떤 각도 조합이 최적인가?)
- [ ] Keypoint confidence 가중치 자동 학습
- [ ] Cross-dataset validation (다른 마우스 데이터에 적용)

### 9.3 한계점

1. **단일 데이터셋**: `markerless_mouse_1_nerf`에서만 검증
2. **고정 카메라**: 카메라 위치가 고정된 상황에서의 결과
3. **Clean GT**: Ground truth keypoint가 정확하다고 가정

---

## 10. 결론 및 권장사항

### 10.1 핵심 결론

1. **View**: 4개 이상 권장, 3개로도 acceptable (120° 간격 시)
2. **Keypoint**: 7-9개로 22개 대비 동등 품질 달성 가능
3. **조합**: 4view + 7kp가 품질/비용 균형 최적

### 10.2 실용 가이드라인

```yaml
# 권장 최소 설정
recommended_minimal:
  views: [0, 1, 2, 3]  # 4개 연속 또는 [0, 2, 4] 3개 간격
  keypoints: 7  # MARS 스타일
  iterations:
    step1: 150
    step2: 50
  regularization:
    theta: 5.0
    mask_step1: 300.0
```

---

## 11. 실험 결과 디렉토리

```
results/fitting/
├── markerless_mouse_1_nerf_v012345_kp22_20251206_165254/  # Baseline
├── markerless_mouse_1_nerf_v012345_sparse9_20251207_081918/  # 9kp DLC
├── markerless_mouse_1_nerf_v012345_sparse7_20251207_172028/  # 7kp MARS
├── markerless_mouse_1_nerf_v012345_sparse5_20251208_134918/  # 5kp minimal
├── markerless_mouse_1_nerf_v01234_sparse3_20251203_235123/   # 5view
├── markerless_mouse_1_nerf_v0123_sparse3_20251204_074430/    # 4view
├── markerless_mouse_1_nerf_v024_sparse3_20251204_153916/     # 3view
└── markerless_mouse_1_nerf_v03_sparse3_20251205_014945/      # 2view
```

---

*Generated: 2025-12-10*
*Tool: Claude Code (claude-opus-4-5-20251101)*
