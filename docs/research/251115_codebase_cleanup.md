# 251115 연구노트 — 코드베이스 정리 및 ML Keypoint 2-Session 요약

## 목표
- ML Keypoint Detection 통합 2-session (251114-251115) 종합 정리
- Manual labeling 워크플로우 완성 및 문서화
- 프로젝트 루트 디렉토리 42% 감소 (aggressive cleanup)
- 장기 리팩토링 계획 수립

## 진행 내용

### 1. ML Keypoint 2-Session 종합 (251114-251115)

**전체 여정**:
1. Phase 1 (YOLOv8 Infrastructure) -- 완료: DANNCE->YOLO 변환, 학습 파이프라인
2. Phase 2 (SuperAnimal) -- 90%: 모델 다운로드, 27->22 매핑 설계, DLC API 제한 발견
3. Phase 3 (Manual Labeling 준비) -- 완료: 20 images 샘플링, Roboflow 가이드 작성

**코드 생산량**: ~1,400 lines
- `dannce_to_yolo.py`: 329 lines
- `yolo_keypoint_detector.py`: 368 lines
- `superanimal_detector.py`: 570+ lines
- `train_yolo_pose.py`: 121 lines
- `download_superanimal.py`: 35 lines

**문서 생산량**: 6개 가이드, 3개 보고서
- `QUICK_START_LABELING.md` (307 lines) -- 전체 워크플로우
- `docs/ROBOFLOW_LABELING_GUIDE.md` (263 lines) -- Roboflow 특화
- `docs/MANUAL_LABELING_GUIDE.md` -- 일반 라벨링 가이드

**22 Keypoint 정의 표준화** (순서 정확히 유지 필수):
- Head (0-5): nose, left/right_ear, left/right_eye, head_center
- Spine (6-13): spine_1~8 (neck -> tail base, 균등 분포)
- Paws (14-17): left/right_front_paw, left/right_rear_paw
- Tail (18-20): tail_base, tail_mid, tail_tip
- Body (21): centroid

**Manual Labeling 예상 ROI**:
- 투자: 2-3시간 (20 images x 5-10분)
- 예상 수익: confidence 2x (0.5->0.85+), loss 10-20x 감소, paw detection 0%->70-80%
- 전체 워크플로우: 라벨링 3시간 + 학습/평가 1.5시간 = ~4.5시간

### 2. Aggressive Cleanup 실행

**루트 디렉토리: 36개 -> 21개 (42% 감소)**

**주요 변경**:

| 작업 | 상세 | 절약 |
|------|------|------|
| deprecated/ 삭제 | 8개 파일 (모두 docs/reports/로 이동 완료) | - |
| outputs/archives/ 정리 | 날짜별 폴더 + mouse_fitting_result | **410MB** |
| scripts/ 체계화 | 15개 파일을 setup/debug/analysis로 분류 | - |
| 문서 경로 업데이트 | 16개 문서에서 scripts/ 접두사 반영 | - |
| checkpoints/ 삭제 | models/로 통합 완료 | - |
| runs/ 삭제 | models/trained/로 이동 완료 | - |
| reports/ 삭제 | docs/reports/로 이동 완료 | - |
| test_*_output/ 삭제 | 임시 테스트 출력 | - |
| assets/ 생성 | colormaps/, figs/, mouse_model/ 통합 | - |
| test/ -> tests/ | 표준 명명 규칙 | - |

**scripts/ 최종 구조**:
```
scripts/
├── preprocess.py, train_yolo_pose.py, evaluate.py
├── run_fitting.sh, run_preprocess.sh, run.sh
├── setup/ (install, setup, download, sample)
├── debug/ (debug_pickle, compare, fix_inverted_masks)
└── analysis/ (data_seaker, visualize_DANNCE)
```

### 3. Codebase 분석 (CODEBASE_ANALYSIS.md 기반)

**원본 vs 현재 비교**:

| Category | Original | Current | 변화 |
|----------|----------|---------|------|
| Python files | ~10 | ~85+ | 확장 |
| Config 시스템 | argparse (하드코딩) | Hydra (YAML) | 대폭 개선 |
| fitter_articulation.py | ~530 lines | ~1,712 lines | +223% |
| 새 모듈 | 0 | 6개 | visualization, preprocessing, uvmap 등 |
| Experiment configs | 0 | 28개 | - |
| 중복 파일 | 0 | 3+ | 정리 필요 |
| Deprecated 코드 | 0 | 14 files | 정리 완료 |

**fitter_articulation.py 주요 확장**: Hydra integration, GPU auto-detection, configurable loss weights, step-specific mask weights, sparse keypoint support, debug grid collector

### 4. 리팩토링 계획 (REFACTORING_PLAN.md)

**Phase 1: Cleanup (Very Low Risk)** -- 부분 완료
- deprecated scripts 삭제, duplicate 제거, backup archive

**Phase 2: Result Consolidation (Low Risk)** -- 계획
- 분산된 결과 폴더 통합: results/fitting/ (21GB), outputs/ (18MB), logs/ (36KB), wandb/ (4.4MB), wandb_sweep_results/ (32MB)
- 목표 구조: results/{fitting, logs, sweep, debug, visualizations, wandb}

**Phase 3: Module Extraction (Medium Risk)** -- 계획
- `mammal_ext/` 패키지 생성
- config/, fitting/, visualization/, preprocessing/ 분리
- 원본 MAMMAL 코드 최소 수정 원칙

**src/ 폴더 전환 (장기)**:
- core/ (articulation, bodymodel, mouse_22_defs)
- fitting/ (fitter, monocular, silhouette)
- preprocessing/ (현재 preprocessing_utils/)
- Import 경로 전체 수정 필요 -> 프로젝트 안정화 후 고려

### 5. 결과 폴더 현황

| Folder | Size | 내용 |
|--------|------|------|
| results/fitting/ | 21GB | 실험 결과 |
| results/fitting/_backup/ | 1.8GB | 구버전 백업 (archive 후 삭제 예정) |
| wandb_sweep_results/ | 32MB | Sweep export |
| wandb/ | 4.4MB | WandB logs |
| outputs/ | 18MB | Debug images |
| logs/ (root) | 36KB | Runtime logs |

## 핵심 발견

- **Progressive cleanup 중요**: 한 번에 완벽한 구조 불가, 반복적 정리를 통해 최적화. 사용자 피드백 반영
- **Git을 활용한 안전한 리팩토링**: 각 단계마다 commit, `git mv`로 히스토리 보존
- **문서 동기화 필수**: 구조 변경 시 16개 문서의 경로를 즉시 업데이트 (sed 자동화)
- **모듈화의 적정 수준**: 지나친 모듈화는 복잡도 증가. 현재는 간결함 > 완벽한 구조
- **Import 경로는 미변경**: 핵심 모델 파일은 루트 유지, 기존 Python import 모두 정상 작동

## 미해결 / 다음 단계

**즉시**:
- Manual labeling 실행 (Roboflow, 20 images, 2-3시간)
- YOLOv8 fine-tuning (100 epochs, ~30분)
- Geometric vs YOLO 벤치마크

**단기 (1-2주)**:
- results/ 폴더 통합 (Phase 2)
- scripts/ 구조 피드백 수집
- tests/ 정리 (unit/integration 분리)

**중기 (1-2개월)**:
- mammal_ext/ 패키지 추출 (Phase 3)
- CI/CD 파이프라인
- 공통 유틸리티 정리

**장기**:
- src/ 폴더 전환 (Breaking Change)
- Package 배포 (pip install)
- 완전한 API 문서화

---
*Sources: 251115_comprehensive_ml_keypoint_summary.md, 251115_aggressive_cleanup_final.md, 251115_final_cleanup_plan.md, 251115_session_continuation_summary.md, CODEBASE_ANALYSIS.md, CODEBASE_CLEANUP_PLAN_251115.md, REFACTORING_PLAN.md*
