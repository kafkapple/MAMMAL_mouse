# MAMMAL Mouse 코드베이스 정리 요약

**작성일**: 2025-11-04  
**목적**: SAM 매뉴얼 작성 및 코드베이스 정리 계획

---

## 1. 완료된 작업

### 1.1 SAM Mask Acquisition Manual 작성

**파일**: \`SAM_MASK_ACQUISITION_MANUAL.md\`

**내용**:
- SAM 기본 개념 및 설치 방법
- 5가지 마스크 획득 방법 상세 설명
  1. Automatic Mask Generation (자동)
  2. Point Prompt (포인트 지정)
  3. Negative Prompt (배경 제외)
  4. Size-Based Filtering (크기 필터링)
  5. Multi-Stage Strategy (다단계)
- 각 방법의 장단점 비교표
- 코드베이스 파일 가이드
- 실전 사용 가이드
- 문제 해결 (Troubleshooting)

**주요 발견사항**:
- 현재 SAM이 아레나 전체를 선택하는 문제 문서화
- 마스크 반전 이슈 문서화
- 5가지 방법론 체계적 정리

### 1.2 코드베이스 분석 및 정리 계획

**파일**: \`CODEBASE_CLEANUP_PLAN.md\`

**발견된 중복 파일**:
1. \`sam_point_prompt.py\` (root) vs \`test/sam_point_prompt.py\`
2. \`sam_mouse_only.py\` (root) vs \`test/sam_mouse_only.py\`

**Deprecated 파일**:
- \`test/preprocess_sam.py\` → \`preprocess_sam_improved.py\`로 대체됨

**제안된 구조 개선**:
- \`models/\` 폴더 신설 (articulation, bodymodel 이동)
- \`scripts/\` 폴더 신설 (메인 스크립트 이동)
- \`tools/\` 폴더 신설 (CLI 도구 통합)
- \`docs/\` 폴더 신설 (문서 정리)
- \`test/\` → \`experiments/\`로 이름 변경

**통합 제안**:
1. SAM 관련 5개 스크립트 → \`tools/sam_mask_tool.py\` 단일 CLI
2. Silhouette 관련 스크립트 → \`tools/fit_silhouette.py\` 단일 CLI

---

## 2. 파일 목록 요약

### 2.1 SAM 관련 파일 (7개)

| 파일 | 위치 | 용도 | 상태 |
|------|------|------|------|
| \`sam_inference.py\` | \`preprocessing_utils/\` | SAM 래퍼 클래스 | ✅ 사용 중 |
| \`test_sam.py\` | \`test/\` | SAM 테스트 | ✅ 개발용 |
| \`preprocess_sam_improved.py\` | \`test/\` | SAM 전처리 파이프라인 | ✅ 최신 버전 |
| \`preprocess_sam.py\` | \`test/\` | SAM 전처리 (이전) | ❌ Deprecated |
| \`sam_point_prompt.py\` | root | 포인트 프롬프트 | ⚠️ 중복 확인 필요 |
| \`sam_mouse_only.py\` | root | 네거티브 프롬프트 | ⚠️ 중복 확인 필요 |
| \`visualize_sam_mouse_detection.py\` | \`test/\` | SAM 시각화 | ✅ 사용 중 |

### 2.2 Silhouette 관련 파일 (6개)

| 파일 | 위치 | 용도 | 상태 |
|------|------|------|------|
| \`silhouette_renderer.py\` | \`preprocessing_utils/\` | PyTorch3D 렌더러 | ✅ 사용 중 |
| \`fit_silhouette_prototype.py\` | root | Silhouette 피팅 (프로토타입) | ✅ 개발용 |
| \`refine_with_silhouette.py\` | \`test/\` | Silhouette 피팅 (Refinement) | ✅ 사용 중 |
| \`test_silhouette_renderer.py\` | \`test/\` | 렌더러 테스트 | ✅ 개발용 |
| \`test_silhouette_simple.py\` | \`test/\` | 간단한 테스트 | ✅ 개발용 |
| \`visualize_silhouette_results.py\` | \`test/\` | 결과 시각화 | ✅ 사용 중 |

### 2.3 메인 스크립트 (8개)

| 파일 | 위치 | 용도 | 상태 |
|------|------|------|------|
| \`preprocess.py\` | root | 전처리 메인 | ✅ 사용 중 |
| \`fitter_articulation.py\` | root | 피팅 메인 | ✅ 사용 중 |
| \`articulation_th.py\` | root | 관절 모델 | ✅ 사용 중 |
| \`bodymodel_np.py\` | root | Body 모델 (NumPy) | ✅ 사용 중 |
| \`bodymodel_th.py\` | root | Body 모델 (PyTorch) | ✅ 사용 중 |
| \`evaluate.py\` | root | 평가 | ✅ 사용 중 |
| \`utils.py\` | root | 유틸리티 | ✅ 사용 중 |
| \`visualize_DANNCE.py\` | root | DANNCE 시각화 | ✅ 사용 중 |

### 2.4 문서 파일 (17개)

위치: \`./\`, \`./reports/\`

- \`README.md\` - 프로젝트 개요
- \`manual.md\` - MAMMAL 모델 아키텍처 매뉴얼
- \`SAM_MASK_ACQUISITION_MANUAL.md\` - SAM 매뉴얼 (신규)
- \`CODEBASE_CLEANUP_PLAN.md\` - 정리 계획 (신규)
- \`reports/*.md\` - 15개의 분석 보고서

---

## 3. 우선순위별 액션 아이템

### Priority 1: 즉시 실행 가능 (중복 제거)

\`\`\`bash
# 1. 중복 파일 비교
diff sam_point_prompt.py test/sam_point_prompt.py
diff sam_mouse_only.py test/sam_mouse_only.py

# 2. Deprecated 파일 백업 및 삭제
mkdir -p archive/deprecated_20251104
mv test/preprocess_sam.py archive/deprecated_20251104/

# 3. Git commit
git add -A
git commit -m "docs: Add SAM manual and cleanup plan

- Created SAM_MASK_ACQUISITION_MANUAL.md
- Created CODEBASE_CLEANUP_PLAN.md
- Archived deprecated preprocess_sam.py"
\`\`\`

### Priority 2: 단기 (폴더 구조 개선)

\`\`\`bash
# 디렉토리 생성
mkdir -p models scripts tools docs experiments/archived

# 파일 이동
mv articulation_th.py bodymodel_*.py models/
mv preprocess.py fitter_articulation.py evaluate.py scripts/
mv *.md docs/ (선택적)

# Import 경로 업데이트
python update_imports.py
\`\`\`

### Priority 3: 중기 (코드 통합)

개발 작업:
- \`tools/sam_mask_tool.py\` - 통합 SAM CLI
- \`tools/fit_silhouette.py\` - 통합 Silhouette CLI
- \`preprocessing_utils/sam/\` 모듈 정리
- \`preprocessing_utils/silhouette/\` 모듈 정리

---

## 4. 기대 효과

### 4.1 정량적

- Root 파일 수: 14개 → 4개 (-71%)
- 중복 파일: 4개 → 0개 (-100%)
- SAM 스크립트: 5개 → 1개 (-80%)

### 4.2 정성적

- ✅ 코드 가독성 향상
- ✅ 유지보수 용이
- ✅ 신규 개발자 온보딩 시간 단축
- ✅ 모듈 재사용성 증가

---

## 5. 다음 단계

1. **사용자 확인**: 중복 파일 비교 결과 검토
2. **백업 생성**: 마이그레이션 전 전체 백업
3. **단계별 실행**: Priority 1 → 2 → 3 순서
4. **회귀 테스트**: 각 단계 후 기능 검증

---

## 6. 참고 문서

- **SAM 매뉴얼**: \`SAM_MASK_ACQUISITION_MANUAL.md\`
- **정리 계획**: \`CODEBASE_CLEANUP_PLAN.md\`
- **MAMMAL 매뉴얼**: \`manual.md\`
- **프로젝트 분석**: \`reports/PROJECT_ANALYSIS.md\`

---

**작성자**: Claude Code  
**작성일**: 2025-11-04  
**프로젝트**: MAMMAL Mouse
