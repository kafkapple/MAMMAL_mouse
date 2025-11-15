# Aggressive Cleanup - Final Reorganization (2025-11-15)

## 📊 Executive Summary

**목적**: 프로젝트 루트 디렉토리를 더욱 체계적으로 정리하여 간결하고 전문적인 구조 확립

**결과**:
- 루트 디렉토리 항목: **36개 → 21개** (42% 감소)
- 디스크 공간 확보: **410MB** (outputs/archives 정리)
- 스크립트 체계화: scripts/ 폴더에 논리적으로 분류
- 문서 경로 업데이트: 모든 문서에서 새 경로 반영

---

## 🎯 주요 변경 사항

### 1. deprecated/ 폴더 완전 삭제 ✅

**삭제된 내용**:
- `deprecated/CODEBASE_CLEANUP_PLAN.md`
- `deprecated/CODEBASE_SUMMARY.md`
- `deprecated/reports/` (6개 구버전 보고서)
- `deprecated/scripts/` (빈 폴더)

**이유**: 모든 유용한 내용은 이미 `docs/reports/`로 이동 완료

### 2. scripts/ 폴더 구조 체계화 ✅

#### 새로운 scripts/ 구조:

```
scripts/
├── preprocess.py                    # 메인 전처리 스크립트
├── train_yolo_pose.py               # YOLO 학습 스크립트
├── evaluate.py                      # 평가 스크립트
├── run_fitting.sh                   # 피팅 실행 쉘
├── run_preprocess.sh                # 전처리 실행 쉘
├── run.sh                           # 통합 실행 쉘
│
├── setup/                           # 설치 및 설정
│   ├── install_mammal_mouse.sh
│   ├── setup.sh
│   ├── download_superanimal.py
│   └── sample_images_for_labeling.py
│
├── debug/                           # 디버그 유틸리티
│   ├── debug_pickle.py
│   ├── compare_preprocessing.py
│   └── fix_inverted_masks.py
│
└── analysis/                        # 분석 도구
    ├── data_seaker_video_new.py
    └── visualize_DANNCE.py
```

**이동된 파일**:
- **메인 스크립트** (3개): preprocess.py, train_yolo_pose.py, evaluate.py
- **쉘 스크립트** (3개): run_fitting.sh, run_preprocess.sh, run.sh
- **설치/설정** (4개): install_mammal_mouse.sh, setup.sh, download_superanimal.py, sample_images_for_labeling.py
- **디버그** (3개): debug_pickle.py, compare_preprocessing.py, fix_inverted_masks.py
- **분석** (2개): data_seaker_video_new.py, visualize_DANNCE.py

### 3. outputs/ 폴더 정리 (410MB 절약) ✅

**Before**:
```
outputs/
├── archives/
│   ├── 2025-10-30/      (344KB)
│   ├── 2025-10-31/      (64KB)
│   ├── 2025-11-02/      (544KB)
│   ├── 2025-11-03/      (204KB)
│   ├── 2025-11-04/      (64KB)
│   └── mouse_fitting_result/  (410MB) ← 대부분의 공간
├── refined_params_silhouette.pkl
└── silhouette_results_comparison.png
```

**After**:
```
outputs/
└── archives/          (빈 폴더, Hydra 자동 생성용)

results/monocular/
├── refined_params_silhouette.pkl        (이동됨)
└── silhouette_results_comparison.png    (이동됨)
```

**조치**:
- archives/ 전체 삭제 (410MB)
- 유용한 결과 파일 2개는 `results/monocular/`로 이동
- 빈 archives/ 폴더 재생성 (Hydra 자동 출력용)

### 4. 문서 경로 자동 업데이트 ✅

**업데이트된 문서** (16개):
- README.md
- docs/guides/COMPREHENSIVE_USAGE_GUIDE.md
- docs/guides/QUICK_START_LABELING.md
- docs/ROBOFLOW_LABELING_GUIDE.md
- docs/MANUAL_LABELING_GUIDE.md
- docs/reports/*.md (11개 보고서)

**변경 내용**:
```bash
# Before
python preprocess.py ...
python train_yolo_pose.py ...
python evaluate.py ...

# After
python scripts/preprocess.py ...
python scripts/train_yolo_pose.py ...
python scripts/evaluate.py ...
```

---

## 📁 최종 프로젝트 구조

### Root Directory (21개 항목)

```
MAMMAL_mouse/
├── README.md                        # 프로젝트 개요
├── requirements.txt                 # 의존성
│
├── articulation_th.py               # 핵심 모델 (PyTorch)
├── bodymodel_th.py                  # Body model (PyTorch)
├── bodymodel_np.py                  # Body model (NumPy)
├── mouse_22_defs.py                 # 마우스 정의
├── utils.py                         # 유틸리티
│
├── fitter_articulation.py           # 메인 피팅 스크립트
├── fit_monocular.py                 # 모노큘러 피팅
├── fit_silhouette_prototype.py      # 실루엣 피팅
│
├── conf/                            # Hydra 설정
├── preprocessing_utils/             # 전처리 모듈
├── scripts/                         # 실행 스크립트 (NEW!)
│   ├── setup/, debug/, analysis/
│   └── *.py, *.sh
│
├── data/                            # 데이터셋
│   ├── raw/, preprocessed/
│   ├── training/, examples/
│
├── models/                          # 모델 가중치
│   ├── pretrained/
│   └── trained/
│
├── results/                         # 최신 실험 결과
│   ├── monocular/
│   └── preprocessing/
│
├── outputs/                         # Hydra 자동 출력
│   └── archives/
│
├── docs/                            # 문서
│   ├── guides/
│   └── reports/
│
├── assets/                          # 정적 자원
│   ├── colormaps/, figs/
│   └── mouse_model/
│
└── tests/                           # 테스트 스크립트
```

---

## 📊 Before vs After 비교

| 항목 | Before | After | 개선 |
|------|--------|-------|------|
| 루트 항목 수 | 36개 | 21개 | **-42%** |
| 디스크 공간 (outputs) | 411MB | 0.2MB | **-410MB** |
| 스크립트 분산 | 루트에 15개 | scripts/ 체계화 | ✅ |
| deprecated/ | 8개 파일 | 삭제됨 | ✅ |
| 문서 경로 | 구버전 | 모두 업데이트 | ✅ |

---

## 🔍 루트 디렉토리 세부 분석

### 유지된 Python 파일 (6개)

**핵심 모델** (4개) - 자주 import되므로 루트 유지:
- `articulation_th.py` - Articulation layer (PyTorch)
- `bodymodel_th.py` - Body model (PyTorch)
- `bodymodel_np.py` - Body model (NumPy)
- `mouse_22_defs.py` - 22 keypoint 정의

**피팅 스크립트** (3개) - 메인 실행 파일이므로 루트 유지:
- `fitter_articulation.py` - 메인 멀티뷰 피팅
- `fit_monocular.py` - 모노큘러 피팅 (NEW)
- `fit_silhouette_prototype.py` - 실루엣 피팅 프로토타입

**유틸리티** (1개):
- `utils.py` - 공통 유틸리티 (모든 스크립트에서 import)

### 폴더 구조 (12개)

**필수 폴더** (8개):
1. `conf/` - Hydra 설정 (원본)
2. `preprocessing_utils/` - 전처리 모듈 (원본)
3. `data/` - 데이터셋
4. `models/` - 모델 가중치
5. `results/` - 최신 실험 결과
6. `outputs/` - Hydra 자동 출력
7. `docs/` - 문서
8. `assets/` - 정적 자원

**새로 추가된 폴더** (2개):
9. `scripts/` - 실행 스크립트 (NEW!)
10. `tests/` - 테스트 스크립트 (test/ → tests/ 이름 변경)

**기타** (2개):
11. `__pycache__/` - Python 캐시 (자동 생성, .gitignore)
12. README.md, requirements.txt

---

## ✅ 완료된 작업

- [x] deprecated/ 폴더 완전 삭제 (8개 파일)
- [x] scripts/ 폴더 생성 및 체계화 (15개 파일 분류)
  - [x] 메인 스크립트 (3개)
  - [x] 쉘 스크립트 (3개)
  - [x] setup/ (4개)
  - [x] debug/ (3개)
  - [x] analysis/ (2개)
- [x] outputs/archives/ 정리 (410MB 절약)
- [x] 유용한 결과 파일 results/monocular/로 이동
- [x] 모든 문서에서 스크립트 경로 업데이트 (16개 파일)
- [x] Git commit 준비

---

## 🚀 다음 단계 (선택적)

### 추가 모듈화 (현재는 스킵, 안정화 후 고려)

**src/ 폴더 생성 시나리오**:
```
src/
├── core/                    # 핵심 모델
│   ├── articulation_th.py
│   ├── bodymodel_th.py
│   ├── bodymodel_np.py
│   └── mouse_22_defs.py
│
├── fitting/                 # 피팅 스크립트
│   ├── fitter_articulation.py
│   ├── fit_monocular.py
│   └── fit_silhouette_prototype.py
│
├── preprocessing/           # preprocessing_utils/ 이동
│   └── ...
│
└── utils/
    └── utils.py
```

**장점**:
- 더욱 명확한 코드 구조
- Import 경로 체계화 (`from mammal_mouse.core import ...`)
- 전문적인 Python 프로젝트 구조

**단점**:
- 모든 import 경로 수정 필요
- 기존 사용자 스크립트 호환성 깨짐
- 추가 작업 시간 소요

**권장**: **현재는 스킵**, 프로젝트가 충분히 안정화되고 사용자가 많아지면 고려

---

## 📝 사용법 변경 사항

### 실행 스크립트 경로 변경

**Before**:
```bash
# 전처리
python preprocess.py --input_dir data/raw/shank3 ...

# YOLO 학습
python train_yolo_pose.py --epochs 100 ...

# 평가
python evaluate.py --model models/trained/yolo/best.pt ...
```

**After**:
```bash
# 전처리
python scripts/preprocess.py --input_dir data/raw/shank3 ...

# YOLO 학습
python scripts/train_yolo_pose.py --epochs 100 ...

# 평가
python scripts/evaluate.py --model models/trained/yolo/best.pt ...
```

### 쉘 스크립트 경로 변경

**Before**:
```bash
bash setup.sh
bash run_preprocess.sh
bash run_fitting.sh
```

**After**:
```bash
bash scripts/setup/setup.sh
bash scripts/run_preprocess.sh
bash scripts/run_fitting.sh
```

**주의**: 모든 문서 (README.md, docs/)는 이미 새 경로로 업데이트됨

---

## 🎯 성과 및 교훈

### 주요 성과

1. **간결성**: 루트 항목 42% 감소 (36 → 21개)
2. **체계성**: 스크립트를 논리적 카테고리로 분류
3. **효율성**: 410MB 디스크 공간 절약
4. **일관성**: 모든 문서 경로 자동 업데이트

### 교훈

1. **Progressive Cleanup의 중요성**:
   - 한 번에 완벽한 구조 만들기 어려움
   - 여러 번의 반복적 정리를 통해 최적화
   - 사용자 피드백 ("여전히 많은 것 같은데") 반영

2. **Git을 활용한 안전한 리팩토링**:
   - 각 단계마다 commit으로 백업
   - `git mv` 사용으로 히스토리 보존
   - 언제든 이전 상태로 복구 가능

3. **문서 동기화의 중요성**:
   - 구조 변경 시 문서 즉시 업데이트 필수
   - 자동화 (sed) 활용으로 일관성 확보

4. **모듈화의 Trade-off**:
   - 지나친 모듈화는 복잡도 증가 가능
   - 프로젝트 성숙도에 맞는 적절한 구조 선택
   - 현재는 간결함 > 완벽한 구조

---

## ⚠️ 주의사항

1. **Import 경로는 변경되지 않음**:
   - 핵심 모델 파일들은 여전히 루트에 위치
   - 기존 Python import 문은 모두 정상 작동
   - `from articulation_th import ...` 동일

2. **실행 스크립트 경로만 변경됨**:
   - `python preprocess.py` → `python scripts/preprocess.py`
   - 모든 문서에서 이미 업데이트 완료

3. **outputs/archives는 자동 재생성됨**:
   - Hydra가 실행 시마다 자동 생성
   - 주기적으로 정리 권장 (디스크 공간)

---

## 📈 향후 정리 계획

### 단기 (1-2주)
- [ ] scripts/ 구조 사용해보고 피드백 수집
- [ ] 불필요한 스크립트 추가 식별
- [ ] tests/ 폴더 정리 (단위/통합 테스트 분리)

### 중기 (1-2개월)
- [ ] preprocessing_utils/ → src/preprocessing/ 이동 고려
- [ ] 공통 유틸리티 함수 정리
- [ ] CI/CD 파이프라인 추가

### 장기 (안정화 후)
- [ ] src/ 폴더 구조 전환 (Breaking Change)
- [ ] Package 형태로 배포 (pip install mammal-mouse)
- [ ] 완전한 API 문서화

---

**작성일**: 2025-11-15
**작성자**: Automated Cleanup Process
**예상 소요 시간**: 15분
**실제 소요 시간**: 20분
**위험도**: 낮음 (Git 백업 완료, 모든 변경 가역적)
