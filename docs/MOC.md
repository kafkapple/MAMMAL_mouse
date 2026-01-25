# Map of Contents (MoC)

> MAMMAL_mouse 프로젝트 전체 문서 네비게이션 허브

---

## Quick Start

| 목적 | 바로가기 |
|------|----------|
| 빠른 시작 | [../QUICKSTART.md](../QUICKSTART.md) |
| 실험 실행 | [practical/EXPERIMENTS.md](practical/EXPERIMENTS.md) |
| 최적화 설정 | [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) |
| 트러블슈팅 | [troubleshooting/](troubleshooting/) |

---

## Documentation Hierarchy

```
docs/
├── MOC.md                     # 이 문서 (전체 네비게이션)
├── OPTIMIZATION_GUIDE.md      # 피팅 최적화 가이드 ★
│
├── practical/                 # 실습 가이드 (How-to) ★
│   ├── EXPERIMENTS.md              # 실험 실행 가이드 ✓
│   └── VISUALIZATION.md            # 시각화 가이드 ✓
│
├── reference/                 # 기술 레퍼런스 (What) - 변하지 않는 정보 ★
│   ├── KEYPOINTS.md                # 22 키포인트 정의 ✓
│   ├── DATASET.md                  # 데이터셋 스펙 ✓
│   ├── MAMMAL_PAPER.md             # 논문 핵심 설정 ✓
│   └── OUTPUT_FORMAT.md            # 출력 파일 형식 ✓
│
├── guides/                    # 상세 가이드 (기존 레거시)
│   ├── fitting/                    # 메쉬 피팅 상세
│   ├── annotation/                 # 어노테이션 상세
│   └── preprocessing/              # 전처리 상세
│
├── reports/                   # 날짜별 연구 보고서
│   └── YYMMDD_*.md
│
├── notes/                     # TIL, 임시 메모
│   └── YYMMDD_*.md
│
├── setup/                     # 환경 설정
│   └── PYTORCH3D_*.md
│
├── troubleshooting/           # 문제 해결
│   └── *.md
│
└── _archive/                  # 폐기/중복 문서
    └── *.md
```

> ★ = 핵심 문서 | ✓ = 신규/업데이트

---

## 1. Practical Guides (How-to)

> 실제 작업 수행을 위한 실습 가이드

| 문서 | 설명 |
|------|------|
| [EXPERIMENTS](practical/EXPERIMENTS.md) | 실험 실행 명령어 |
| [PREPROCESSING](practical/PREPROCESSING.md) | 비디오 → 마스크 → 키포인트 |
| [VISUALIZATION](practical/VISUALIZATION.md) | 메쉬 시각화, 비디오 생성 |
| [ANNOTATION](practical/ANNOTATION.md) | 키포인트 어노테이션 워크플로우 |

---

## 2. Reference (기술 레퍼런스)

> 변하지 않는 핵심 정보

### Core Specifications

| 문서 | 설명 |
|------|------|
| [KEYPOINTS](reference/KEYPOINTS.md) | 22 키포인트 정의 + 인덱스 |
| [DATASET](reference/DATASET.md) | 데이터셋 스펙 (FPS, 프레임, 카메라) |
| [CONFIG](reference/CONFIG.md) | Hydra Config 파라미터 |
| [OUTPUT_FORMAT](reference/OUTPUT_FORMAT.md) | 출력 파일 형식 (.obj, .pkl) |

### Literature

| 문서 | 설명 |
|------|------|
| [MAMMAL_PAPER](reference/MAMMAL_PAPER.md) | 논문 핵심 설정 + 인용 |

---

## 3. Reports (연구 보고서)

> 날짜별 연구 진행 기록 - `reports/` 디렉토리

### 2026-01 ★ Recent

| 날짜 | 제목 | 주요 내용 |
|------|------|----------|
| **260125** | **paper_fast 설정** | **논문 기반 전체 피팅 최적화, 문서 리팩토링** |

### 2025-12

| 날짜 | 제목 | 주요 내용 |
|------|------|----------|
| 251210 | [UV Map 최적화](reports/251210_uvmap_score_optimization_strategy.md) | HPO 스코어 설계 |
| 251210 | [Ablation Study](reports/251210_mesh_fit_ablation_study.md) | 메쉬 피팅 정량 평가 |

### 2025-11

| 날짜 | 제목 | 주요 내용 |
|------|------|----------|
| 251125 | [메쉬 피팅 메커니즘](reports/251125_mesh_fitting_mechanism.md) | 피팅 원리 분석 |
| 251117 | [비디오 처리](reports/251117_mouse_video_processing_summary.md) | 전처리 파이프라인 |
| 251103 | [SAM 전처리](reports/251103_sam_preprocessing_validation.md) | 마스크 획득 검증 |

---

## 4. Key Specifications

### Dataset

| 항목 | 값 |
|------|-----|
| 카메라 수 | 6 |
| 원본 FPS | 100 fps |
| 총 프레임 | 18,000 (180초) |
| frame_jump=5 | 3,600 샘플 |
| 해상도 | 1024 × 1152 |

### Paper Settings (An et al., 2023)

| 항목 | 값 | 근거 |
|------|-----|------|
| step0_iters | 60 | 첫 프레임 초기화 |
| step1_iters | 5 | "3-5 iterations per frame" |
| step2_iters | 3 | "3 iterations" |
| mask_loss | 0.0 | "wsil=0 for faster inference" |

### 22 Keypoints

```
0: L_ear       8: L_paw      16: L_foot
1: R_ear       9: L_paw_end  17: L_knee
2: nose       10: L_elbow    18: L_hip
3: neck       11: L_shoulder 19: R_foot
4: body       12: R_paw      20: R_knee
5: tail_root  13: R_paw_end  21: R_hip
6: tail_mid   14: R_elbow
7: tail_end   15: R_shoulder
```

---

## Navigation Tips

1. **처음 시작**: [../QUICKSTART.md](../QUICKSTART.md)
2. **전체 피팅**: [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)
3. **실험 실행**: [practical/EXPERIMENTS.md](practical/EXPERIMENTS.md)
4. **파라미터 확인**: [reference/CONFIG.md](reference/CONFIG.md)
5. **문제 발생**: [troubleshooting/](troubleshooting/)

---

*Last updated: 2026-01-25*
