# Final Codebase Cleanup Plan - 2025-11-15

## 📊 폴더 분류 및 처리 방안

### ✅ 원본 폴더 (Git에 있던 것들) - 유지 필요

| 폴더 | 용도 | 상태 | 조치 |
|------|------|------|------|
| `conf/` | Hydra 설정 | 필수 | ✅ 유지 |
| `mouse_model/` | 3D 마우스 모델 | 필수 | ✅ 유지 |
| `preprocessing_utils/` | 전처리 모듈 | 필수 | ✅ 유지 |
| `colormaps/` | 시각화 컬러맵 | 필수 | ✅ 유지 (assets/로 이동 고려) |
| `figs/` | README 이미지 | 필수 | ✅ 유지 (assets/로 이동 고려) |
| `test/` | 테스트 스크립트 | 필요 | ✅ 유지 (tests/로 이름 변경 고려) |
| `outputs/` | Hydra 출력 | 자동생성 | ✅ 유지 (.gitignore) |

### 🆕 새로 추가된 폴더 - 정리 필요

| 폴더 | 용도 | 상태 | 조치 |
|------|------|------|------|
| `docs/` | 문서 | 필수 | ✅ 유지 (정리 완료) |
| `data/` | 데이터셋 | 필수 | ✅ 유지 (정리 완료) |
| `models/` | 모델 가중치 | 필수 | ✅ 유지 (정리 완료) |
| `results/` | 최신 실험 결과 | 필요 | ✅ 유지 (정리 완료) |
| `deprecated/` | 참고용 구버전 | 임시 | ⚠️ 검토 후 삭제 |
| `checkpoints/` | 학습 체크포인트 | 중복 | ❌ 삭제 (models/로 통합) |
| `runs/` | YOLO 학습 결과 | 중복 | ❌ 삭제 (models/trained/로 이동 완료) |
| `reports/` | 구버전 보고서 | 중복 | ❌ 삭제 (docs/reports/로 이동 완료) |

### ⚠️ 테스트 출력 폴더 - 정리 필요

| 폴더 | 용도 | 상태 | 조치 |
|------|------|------|------|
| `test_geometric_output/` | 테스트 결과 | 임시 | ❌ 삭제 또는 tests/outputs/로 이동 |
| `test_superanimal_output/` | 테스트 결과 | 임시 | ❌ 삭제 또는 tests/outputs/로 이동 |

---

## 🎯 최종 정리 작업

### 1. 불필요한 폴더 삭제

```bash
# 빈 폴더 및 중복 폴더 삭제
rm -rf checkpoints/  # 비어있거나 models/로 통합됨
rm -rf runs/         # models/trained/yolo/로 이동 완료
rm -rf reports/      # docs/reports/로 이동 완료

# 임시 테스트 출력 삭제
rm -rf test_geometric_output/
rm -rf test_superanimal_output/
```

### 2. Assets 폴더 생성 및 정리

```bash
# 정적 자원 통합
mkdir -p assets
mv colormaps/ assets/
mv figs/ assets/
mv mouse_model/ assets/
```

### 3. Tests 폴더 재구성

```bash
# test/ → tests/로 이름 변경
mv test/ tests/

# 테스트 출력 폴더 생성
mkdir -p tests/outputs/
```

### 4. Deprecated 검토 및 삭제

```bash
# deprecated/ 내용 확인 후 완전 삭제
# (1-2주 후 문제없으면 삭제 예정)
ls -la deprecated/
```

---

## 📁 최종 목표 구조

```
MAMMAL_mouse/
├── README.md
├── requirements.txt
├── setup.sh
├── run_preprocess.sh
├── run_fitting.sh
│
├── conf/                        # ✅ 원본 유지
│   ├── config.yaml
│   ├── dataset/
│   ├── preprocess/
│   └── optim/
│
├── src/                         # 🆕 제안: Python 소스 모듈화
│   ├── core/                    # 핵심 모델
│   │   ├── articulation_th.py
│   │   ├── bodymodel_th.py
│   │   └── bodymodel_np.py
│   ├── preprocessing/           # preprocessing_utils/ 이동
│   ├── fitting/                 # fit_*.py 이동
│   └── utils/
│
├── scripts/                     # 🆕 실행 스크립트
│   ├── preprocess.py
│   ├── train_yolo_pose.py
│   └── evaluate.py
│
├── tests/                       # test/ 이름 변경
│   ├── unit/
│   ├── integration/
│   └── outputs/                 # 테스트 결과
│
├── data/                        # ✅ 정리 완료
│   ├── raw/
│   ├── preprocessed/
│   ├── training/
│   └── examples/
│
├── models/                      # ✅ 정리 완료
│   ├── pretrained/
│   └── trained/
│
├── results/                     # ✅ 정리 완료
│   ├── monocular/
│   └── preprocessing/
│
├── outputs/                     # ✅ Hydra 자동생성
│   └── archives/
│
├── docs/                        # ✅ 정리 완료
│   ├── guides/
│   └── reports/
│
└── assets/                      # 🆕 정적 자원
    ├── colormaps/
    ├── figs/
    └── mouse_model/
```

---

## 🔍 Python 스크립트 모듈화 (선택적)

### 현재 루트의 Python 파일들

**핵심 모델** (src/core/로 이동 고려):
- `articulation_th.py`
- `bodymodel_th.py`
- `bodymodel_np.py`
- `mouse_22_defs.py`

**피팅 스크립트** (src/fitting/로 이동 고려):
- `fitter_articulation.py` (메인)
- `fit_monocular.py` (신규)
- `fit_silhouette_prototype.py`

**유틸리티** (src/utils/로 이동 고려):
- `utils.py`
- `visualize_DANNCE.py`
- `data_seaker_video_new.py`

**실행 스크립트** (scripts/로 이동 고려):
- `preprocess.py`
- `train_yolo_pose.py`
- `evaluate.py`
- `download_superanimal.py`
- `sample_images_for_labeling.py`

**디버그/임시** (scripts/debug/로 이동):
- `debug_pickle.py`
- `compare_preprocessing.py`
- `fix_inverted_masks.py`

### 모듈화 시 고려사항

**장점**:
- 명확한 코드 구조
- Import 경로 체계화
- 전문적인 프로젝트 구조

**단점**:
- Import 경로 수정 필요
- 기존 사용자 스크립트 변경 필요
- 추가 작업 시간 필요

**권장사항**:
- **지금은 스킵**, 프로젝트가 안정화되면 추후 진행
- 현재는 폴더 정리에만 집중

---

## 🚀 즉시 실행 작업

### Step 1: 불필요한 폴더 삭제

```bash
# 중복 및 빈 폴더 삭제
rm -rf checkpoints/
rm -rf runs/
rm -rf reports/
rm -rf test_geometric_output/
rm -rf test_superanimal_output/
```

### Step 2: Assets 폴더 생성

```bash
mkdir -p assets
mv colormaps/ assets/
mv figs/ assets/
mv mouse_model/ assets/
```

### Step 3: Tests 폴더 정리

```bash
mv test/ tests/
mkdir -p tests/outputs/
```

### Step 4: Git 커밋

```bash
git add -A
git commit -m "refactor: Final cleanup - remove duplicates and organize assets

- Remove: checkpoints/, runs/, reports/ (duplicates)
- Remove: test_*_output/ (temporary test outputs)
- Create: assets/ (colormaps, figs, mouse_model)
- Rename: test/ → tests/
- Result: Cleaner root directory, professional structure"
```

---

## 📊 정리 전후 비교

### Before (현재)
```
루트 디렉토리: 20개 폴더
- checkpoints/ (중복)
- runs/ (중복)
- reports/ (중복)
- test_geometric_output/ (임시)
- test_superanimal_output/ (임시)
- colormaps/ (분산)
- figs/ (분산)
- mouse_model/ (분산)
```

### After (목표)
```
루트 디렉토리: 12개 폴더
- assets/ (colormaps, figs, mouse_model 통합)
- tests/ (test 표준화)
- 중복 제거 (5개 폴더)
```

---

## ⚠️ 주의사항

1. **삭제 전 확인**:
   - checkpoints/가 비어있는지 확인
   - runs/의 내용이 models/trained/로 이동되었는지 확인
   - reports/가 비어있는지 확인

2. **테스트**:
   - 삭제 후 주요 스크립트 실행 테스트
   - Import 경로 확인
   - 문서 링크 확인

3. **백업**:
   - Git commit으로 이미 백업됨
   - 필요시 deprecated/에 보관

---

## ✅ 체크리스트

- [ ] checkpoints/ 내용 확인 후 삭제
- [ ] runs/ 내용 확인 후 삭제
- [ ] reports/ 내용 확인 후 삭제
- [ ] test_*_output/ 삭제
- [ ] assets/ 폴더 생성
- [ ] colormaps/, figs/, mouse_model/ → assets/로 이동
- [ ] test/ → tests/로 이름 변경
- [ ] Git commit
- [ ] README 경로 업데이트
- [ ] 주요 스크립트 실행 테스트

---

**작성일**: 2025-11-15
**예상 소요 시간**: 10-15분
**위험도**: 낮음 (Git 백업 완료)
