# Codebase Cleanup Plan - 2025-11-15

## 🎯 목표

프로젝트 구조를 체계적이고 일관성 있게 정리:
1. **폴더 구조 단순화** - 명확한 계층 구조
2. **파일명 일관성** - YYMMDD_ 접두사 표준화
3. **스크립트 모듈화** - 중복 제거, 재사용성 향상
4. **사용하지 않는 파일 제거** - 정리 및 아카이빙
5. **문서 통합** - 일관된 위치 및 명명 규칙

---

## 📋 현재 상태 분석

### 문제점

#### 1. 폴더 구조 혼란
```
outputs/ - 너무 많은 서브폴더 (날짜별 산재)
  ├── 2025-10-30/
  ├── 2025-10-31/
  ├── 2025-11-02/
  ├── 2025-11-03/
  ├── 2025-11-04/
  ├── monocular_poc/
  ├── monocular_poc_batch/
  ├── mouse_fitting_result/
  ├── preprocessing_debug/
  └── sam_test_results/

reports/ vs docs/reports/ - 중복된 보고서 위치
  - reports/ (루트): 6개 오래된 보고서
  - docs/reports/: 5개 최신 보고서

test/ vs test_*_output/ - 테스트 결과 분산
  - test/: 테스트 스크립트
  - test_geometric_output/: 테스트 결과
  - test_superanimal_output/: 테스트 결과
```

#### 2. 문서 명명 불일치
```
docs/reports/:
  ✅ 251114_ml_keypoint_detection_integration.md (YYMMDD_)
  ✅ 251114_monocular_mammal_fitting_poc.md
  ✅ 251114_session_summary.md
  ✅ 251115_session_continuation_summary.md
  ✅ COMPREHENSIVE_SUMMARY_ML_KEYPOINT_DETECTION.md

reports/:
  ❌ preprocessing_improvement_report_20251103.md (YYYYMMDD)
  ❌ SUCCESS_REPORT_20251103.md
  ❌ silhouette_fitting_final_report_20251104.md
  ❌ SAM_preprocessing_validation_report.md (날짜 없음)
  ❌ IMPLEMENTATION_PLAN.md (날짜 없음)
```

#### 3. 루트 디렉토리 혼잡
```
루트에 너무 많은 파일:
  - 22개 .py 파일 (모듈화 필요)
  - 6개 .md 파일 (일부 docs/로 이동 필요)
  - test/, reports/, outputs/ 등 여러 실험 폴더
```

#### 4. 데이터 폴더 구조 불명확
```
data/:
  - manual_labeling/ (진행 중)
  - yolo_mouse_pose/ (생성됨)
  - markerless_mouse_1_nerf/ (예제 데이터?)
  - shank3/ (특정 실험)
  - preprocessed_shank3/ (전처리 결과)
  - preprocessed_shank3_sam/ (SAM 전처리 결과)
```

---

## 🎯 정리 계획

### 1. 폴더 구조 재정의

#### 제안하는 최종 구조:

```
MAMMAL_mouse/
├── README.md                    # 프로젝트 개요
├── requirements.txt             # 의존성
│
├── configs/                     # 모든 설정 파일 (conf/ 통합)
│   ├── config.yaml
│   ├── dataset/
│   ├── optim/
│   └── preprocess/
│
├── src/                         # 모든 소스 코드 (NEW)
│   ├── core/                    # 핵심 모델 및 로직
│   │   ├── articulation_th.py
│   │   ├── bodymodel_np.py
│   │   ├── bodymodel_th.py
│   │   └── mouse_22_defs.py
│   │
│   ├── preprocessing/           # preprocessing_utils/ 이동
│   │   ├── __init__.py
│   │   ├── keypoint_estimation.py
│   │   ├── mask_processing.py
│   │   ├── sam_inference.py
│   │   ├── silhouette_renderer.py
│   │   ├── yolo_keypoint_detector.py
│   │   ├── superanimal_detector.py
│   │   ├── dannce_to_yolo.py
│   │   └── visualize_yolo_labels.py
│   │
│   ├── fitting/                 # 피팅 관련 스크립트
│   │   ├── fit_monocular.py
│   │   ├── fit_silhouette_prototype.py
│   │   └── fitter_articulation.py
│   │
│   ├── training/                # 학습 관련
│   │   └── train_yolo_pose.py
│   │
│   └── utils/                   # 유틸리티
│       ├── utils.py
│       └── visualize_DANNCE.py
│
├── scripts/                     # 실행 스크립트 (NEW)
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── download_superanimal.py
│   ├── sample_images_for_labeling.py
│   └── debug/                   # 디버그 스크립트
│       ├── debug_pickle.py
│       ├── compare_preprocessing.py
│       └── fix_inverted_masks.py
│
├── tests/                       # test/ 이름 변경
│   ├── unit/                    # 단위 테스트
│   ├── integration/             # 통합 테스트
│   └── outputs/                 # 테스트 결과 (test_*_output/ 통합)
│       ├── geometric/
│       └── superanimal/
│
├── data/                        # 데이터셋
│   ├── raw/                     # 원본 데이터
│   │   └── shank3/
│   ├── preprocessed/            # 전처리 결과 (정리 후)
│   │   ├── shank3_opencv/
│   │   └── shank3_sam/
│   ├── training/                # 학습 데이터
│   │   ├── yolo_mouse_pose/     # YOLO 학습 데이터
│   │   └── manual_labeling/     # 수동 라벨링 (진행 중)
│   └── examples/                # 예제 데이터
│       └── markerless_mouse_1_nerf/
│
├── models/                      # 학습된 모델 및 체크포인트
│   ├── checkpoints/             # checkpoints/ 통합
│   ├── pretrained/              # 사전학습 모델
│   │   └── superanimal_topviewmouse/
│   └── trained/                 # 학습된 모델 (runs/ 통합 후)
│       └── yolo/
│           └── mammal_mouse_test/
│
├── outputs/                     # 실험 결과 (아카이빙)
│   └── archives/                # 오래된 실험 결과
│       ├── 2025-10-30/
│       ├── 2025-10-31/
│       ├── 2025-11-02/
│       ├── 2025-11-03/
│       └── 2025-11-04/
│
├── results/                     # 최신 실험 결과 (NEW, outputs/의 현재 버전)
│   ├── monocular/               # Monocular fitting
│   ├── preprocessing/           # 전처리 결과
│   └── training/                # 학습 결과
│
├── docs/                        # 모든 문서
│   ├── guides/                  # 사용 가이드
│   │   ├── MONOCULAR_FITTING_GUIDE.md
│   │   ├── MANUAL_LABELING_GUIDE.md
│   │   ├── ROBOFLOW_LABELING_GUIDE.md
│   │   ├── QUICK_START_LABELING.md
│   │   └── SAM_MASK_ACQUISITION_MANUAL.md
│   │
│   └── reports/                 # 연구 보고서 (reports/ 통합)
│       ├── 251103_preprocessing_improvement.md
│       ├── 251103_success_report.md
│       ├── 251104_silhouette_fitting_final.md
│       ├── 251114_ml_keypoint_detection_integration.md
│       ├── 251114_monocular_mammal_fitting_poc.md
│       ├── 251114_session_summary.md
│       ├── 251115_session_continuation_summary.md
│       └── 251115_comprehensive_ml_keypoint_summary.md
│
├── assets/                      # 정적 자원
│   ├── colormaps/               # colormaps/ 이동
│   ├── mouse_model/             # 마우스 모델 정의
│   └── figs/                    # 그림 파일
│
└── deprecated/                  # 사용하지 않는 파일 (삭제 전 임시)
    ├── scripts/
    └── reports/
```

---

### 2. 파일 이름 표준화

#### 문서 파일 (*.md)

**규칙**: `YYMMDD_카테고리_간단한_설명.md`

**변환 계획**:
```bash
# reports/ → docs/reports/ (이름 변경 포함)
reports/preprocessing_improvement_report_20251103.md
  → docs/reports/251103_preprocessing_improvement.md

reports/SUCCESS_REPORT_20251103.md
  → docs/reports/251103_success_report.md

reports/silhouette_fitting_final_report_20251104.md
  → docs/reports/251104_silhouette_fitting_final.md

reports/SAM_preprocessing_validation_report.md
  → docs/reports/251103_sam_preprocessing_validation.md (날짜 추정)

reports/IMPLEMENTATION_PLAN.md
  → docs/guides/implementation_plan.md (가이드로 분류)

# 루트 → docs/guides/
SAM_MASK_ACQUISITION_MANUAL.md
  → docs/guides/SAM_MASK_ACQUISITION_MANUAL.md

QUICK_START_LABELING.md
  → docs/guides/QUICK_START_LABELING.md

README_MONOCULAR.md
  → docs/guides/MONOCULAR_FITTING_GUIDE.md (이미 존재, 병합 검토)

# docs/reports/ 내 이름 변경
COMPREHENSIVE_SUMMARY_ML_KEYPOINT_DETECTION.md
  → 251115_comprehensive_ml_keypoint_summary.md
```

#### 스크립트 파일 (*.py)

**규칙**: `동사_명사.py` or `명사_처리.py`

**모듈화 및 이동**:
```bash
# src/core/
articulation_th.py → src/core/articulation_th.py
bodymodel_np.py → src/core/bodymodel_np.py
bodymodel_th.py → src/core/bodymodel_th.py
mouse_22_defs.py → src/core/mouse_22_defs.py

# src/preprocessing/ (preprocessing_utils/ 이동)
preprocessing_utils/*.py → src/preprocessing/*.py

# src/fitting/
fit_monocular.py → src/fitting/fit_monocular.py
fit_silhouette_prototype.py → src/fitting/fit_silhouette_prototype.py
fitter_articulation.py → src/fitting/fitter_articulation.py

# src/training/
train_yolo_pose.py → src/training/train_yolo_pose.py

# scripts/
preprocess.py → scripts/preprocess.py
evaluate.py → scripts/evaluate.py
download_superanimal.py → scripts/download_superanimal.py
sample_images_for_labeling.py → scripts/sample_images_for_labeling.py

# scripts/debug/
debug_pickle.py → scripts/debug/debug_pickle.py
compare_preprocessing.py → scripts/debug/compare_preprocessing.py
fix_inverted_masks.py → scripts/debug/fix_inverted_masks.py

# Deprecated (사용하지 않음)
data_seaker_video_new.py → deprecated/scripts/
visualize_DANNCE.py → deprecated/scripts/ (또는 src/utils/)
```

---

### 3. 데이터 및 결과 정리

#### 3.1 `data/` 재구조화

```bash
# 원본 데이터
data/shank3/ → data/raw/shank3/

# 전처리 결과
data/preprocessed_shank3/ → data/preprocessed/shank3_opencv/
data/preprocessed_shank3_sam/ → data/preprocessed/shank3_sam/

# 학습 데이터
data/yolo_mouse_pose/ → data/training/yolo_mouse_pose/
data/manual_labeling/ → data/training/manual_labeling/

# 예제 데이터
data/markerless_mouse_1_nerf/ → data/examples/markerless_mouse_1_nerf/
```

#### 3.2 `outputs/` 정리 및 `results/` 분리

```bash
# 오래된 실험 → archives/
outputs/2025-10-30/ → outputs/archives/2025-10-30/
outputs/2025-10-31/ → outputs/archives/2025-10-31/
outputs/2025-11-02/ → outputs/archives/2025-11-02/
outputs/2025-11-03/ → outputs/archives/2025-11-03/
outputs/2025-11-04/ → outputs/archives/2025-11-04/
outputs/mouse_fitting_result/ → outputs/archives/mouse_fitting_result/

# 최신 실험 → results/
outputs/monocular_poc/ → results/monocular/poc/
outputs/monocular_poc_batch/ → results/monocular/poc_batch/
outputs/preprocessing_debug/ → results/preprocessing/debug/
outputs/sam_test_results/ → results/preprocessing/sam_test/
```

#### 3.3 `models/` 통합

```bash
# 체크포인트
checkpoints/ → models/checkpoints/

# 학습된 모델
runs/pose/ → models/trained/yolo/

# 사전학습 모델 (기존 유지)
models/superanimal_topviewmouse/ → models/pretrained/superanimal_topviewmouse/
```

#### 3.4 `tests/` 정리

```bash
# 테스트 스크립트 정리
test/*.py → tests/integration/*.py (또는 unit/)

# 테스트 결과 통합
test_geometric_output/ → tests/outputs/geometric/
test_superanimal_output/ → tests/outputs/superanimal/
```

---

### 4. 사용하지 않는 파일 식별

#### 4.1 삭제 후보 (deprecated/로 이동 후 확인)

```bash
# 스크립트
data_seaker_video_new.py - 오래된 데이터 탐색 스크립트?
manual.md - 내용 불명, 확인 필요
CODEBASE_CLEANUP_PLAN.md - 이전 정리 계획 (이번 계획으로 대체)
CODEBASE_SUMMARY.md - 오래된 요약 (최신 문서로 대체)

# 보고서 (reports/ 내)
reports/Report.md - 일반 이름, 내용 확인 필요
reports/shank3_workflow_debugging_report.md - 특정 실험, 아카이브
reports/keypoint_optimization_analysis.md - 분석 완료, 아카이브
reports/commit_message.txt - 임시 파일, 삭제

# 테스트 스크립트 (일부)
test/preprocess_sam_improved.py vs test/preprocess_sam.py - 중복? 병합 검토
test/refine_with_silhouette.py - Prototype, 아카이브?
```

#### 4.2 병합 후보

```bash
# README
README.md (루트) + README_MONOCULAR.md
  → README.md 통합 (섹션 추가)

# Monocular Fitting 가이드
docs/MONOCULAR_FITTING_GUIDE.md + README_MONOCULAR.md
  → 중복 확인 후 병합
```

---

### 5. 실행 계획 (단계별)

#### Step 1: 백업 생성 (필수!)
```bash
# 전체 프로젝트 백업
cd /home/joon/dev/
tar -czf MAMMAL_mouse_backup_251115.tar.gz MAMMAL_mouse/

# 또는 Git commit
cd MAMMAL_mouse/
git add .
git commit -m "backup: Before major codebase cleanup (251115)"
```

#### Step 2: 새 폴더 구조 생성
```bash
mkdir -p src/{core,preprocessing,fitting,training,utils}
mkdir -p scripts/debug
mkdir -p tests/{unit,integration,outputs}
mkdir -p data/{raw,preprocessed,training,examples}
mkdir -p models/{checkpoints,pretrained,trained}
mkdir -p outputs/archives
mkdir -p results/{monocular,preprocessing,training}
mkdir -p docs/guides
mkdir -p assets/{colormaps,mouse_model,figs}
mkdir -p configs
mkdir -p deprecated/{scripts,reports}
```

#### Step 3: 파일 이동 (스크립트 작성)
```bash
# cleanup_codebase.py 스크립트 작성
# - 파일 이동
# - 이름 변경
# - import 경로 자동 수정
```

#### Step 4: Import 경로 수정
```python
# Before
from preprocessing_utils.keypoint_estimation import estimate_mammal_keypoints

# After
from src.preprocessing.keypoint_estimation import estimate_mammal_keypoints
```

#### Step 5: 테스트 및 검증
```bash
# 주요 스크립트 실행 테스트
python scripts/preprocess.py --help
python src/fitting/fit_monocular.py --help
python src/training/train_yolo_pose.py --help

# Import 검증
python -c "from src.core.articulation_th import ArticulationTorch"
python -c "from src.preprocessing.keypoint_estimation import estimate_mammal_keypoints"
```

#### Step 6: 문서 업데이트
```bash
# README.md 업데이트 (새 구조 반영)
# docs/ 내 모든 파일 경로 수정
# requirements.txt 검증
```

#### Step 7: 정리 및 삭제
```bash
# deprecated/ 검토 후 삭제
# 빈 폴더 제거
# Git commit
```

---

### 6. 예상 결과

#### Before (현재)
```
MAMMAL_mouse/
├── 22 Python files (루트)
├── 6 Markdown files (루트)
├── conf/ (설정)
├── preprocessing_utils/ (전처리)
├── test/ (테스트)
├── reports/ (구 보고서)
├── docs/reports/ (신 보고서)
├── outputs/ (10+ 서브폴더 혼재)
├── checkpoints/
├── runs/
└── [기타 혼재 폴더]

총 폴더: ~30개
루트 파일: ~30개
```

#### After (목표)
```
MAMMAL_mouse/
├── README.md
├── requirements.txt
├── configs/
├── src/
│   ├── core/
│   ├── preprocessing/
│   ├── fitting/
│   ├── training/
│   └── utils/
├── scripts/
│   └── debug/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── outputs/
├── data/
│   ├── raw/
│   ├── preprocessed/
│   ├── training/
│   └── examples/
├── models/
│   ├── checkpoints/
│   ├── pretrained/
│   └── trained/
├── outputs/archives/
├── results/
│   ├── monocular/
│   ├── preprocessing/
│   └── training/
├── docs/
│   ├── guides/
│   └── reports/
└── assets/
    ├── colormaps/
    ├── mouse_model/
    └── figs/

총 폴더: ~25개 (계층적 정리)
루트 파일: 2개 (README, requirements)
```

#### 개선 효과
- ✅ 루트 디렉토리 깔끔 (2개 파일만)
- ✅ 논리적 계층 구조 (src/, scripts/, tests/, data/, models/, docs/)
- ✅ 일관된 명명 규칙 (YYMMDD_, 동사_명사)
- ✅ 명확한 파일 위치 (찾기 쉬움)
- ✅ 모듈화된 코드 (import 경로 명확)

---

## ✅ 체크리스트

### 실행 전
- [ ] Git commit (현재 상태 백업)
- [ ] 전체 프로젝트 tar.gz 백업
- [ ] 정리 스크립트 작성 및 검토
- [ ] 팀원에게 공지 (있다면)

### 실행 중
- [ ] 새 폴더 구조 생성
- [ ] 파일 이동 (스크립트 실행)
- [ ] Import 경로 수정
- [ ] 문서 업데이트

### 실행 후
- [ ] 주요 스크립트 실행 테스트
- [ ] Import 검증
- [ ] Git commit (정리 완료)
- [ ] deprecated/ 검토 및 삭제
- [ ] README 최종 업데이트

---

## 📝 주의사항

1. **점진적 접근**: 한 번에 모두 이동하지 말고 카테고리별로 진행
2. **테스트 필수**: 각 단계마다 기능 검증
3. **백업 유지**: deprecated/에 최소 1주일 보관
4. **문서 우선**: 코드보다 문서 정리를 먼저 (위험 낮음)
5. **Import 주의**: Python import 경로 변경 시 모든 파일 검색 필수

---

**작성일**: 2025-11-15
**작성자**: Codebase Cleanup Planning
**실행 예정**: 2025-11-15 (즉시 또는 사용자 확인 후)
**예상 소요 시간**: 1-2시간 (스크립트 + 수동 검증)
