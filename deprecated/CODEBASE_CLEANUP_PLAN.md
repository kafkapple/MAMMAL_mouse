# MAMMAL Mouse 코드베이스 정리 계획

**작성일**: 2025-11-04  
**목적**: 중복 파일 제거, 코드 통합, 프로젝트 구조 개선

---

## 1. 현황 분석

### 1.1 전체 파일 구조

\`\`\`
MAMMAL_mouse/
├── *.py                      # 메인 스크립트들 (14개)
├── preprocessing_utils/      # 전처리 유틸리티 모듈
├── test/                     # 실험 및 테스트 스크립트 (12개)
├── conf/                     # Hydra 설정 파일
├── data/                     # 데이터셋
├── mouse_fitting_result/     # 피팅 결과물
├── checkpoints/              # SAM, SuperAnimal 체크포인트
└── reports/                  # 분석 보고서
\`\`\`

### 1.2 중복 파일 발견

#### SAM 관련 중복

| Root 위치 | Test 위치 | 상태 | 비고 |
|-----------|-----------|------|------|
| \`sam_point_prompt.py\` | \`test/sam_point_prompt.py\` | ⚠️ 중복 가능성 | 최신 버전 확인 필요 |
| \`sam_mouse_only.py\` | \`test/sam_mouse_only.py\` | ⚠️ 중복 가능성 | 최신 버전 확인 필요 |

#### Preprocessing 관련 중복

| 파일 | 상태 | 비고 |
|------|------|------|
| \`test/preprocess_sam.py\` | ❌ Deprecated | \`preprocess_sam_improved.py\`로 대체됨 |
| \`test/preprocess_sam_improved.py\` | ✅ 최신 버전 | 사용 중 |

---

## 2. 정리 계획

### 2.1 중복 파일 처리

#### Phase 1: 중복 확인 및 최신 버전 결정

\`\`\`bash
# 1. sam_point_prompt.py 비교
diff sam_point_prompt.py test/sam_point_prompt.py

# 2. sam_mouse_only.py 비교
diff sam_mouse_only.py test/sam_mouse_only.py
\`\`\`

**결정 규칙**:
- Root 버전이 최신이면 → test/ 버전 삭제
- Test 버전이 최신이면 → Root 버전 삭제
- 기능이 다르면 → 파일명 변경하여 구분

#### Phase 2: Deprecated 파일 삭제

\`\`\`bash
# Backup 생성
mkdir -p archive/deprecated_20251104

# Deprecated 파일 이동
mv test/preprocess_sam.py archive/deprecated_20251104/
\`\`\`

**삭제 후보 목록**:
- \`test/preprocess_sam.py\` → \`preprocess_sam_improved.py\`로 대체됨

### 2.2 코드 통합 제안

#### 통합 1: SAM 관련 스크립트 → 단일 CLI 도구

**현재 상황**:
- \`test/test_sam.py\` - SAM 테스트
- \`sam_point_prompt.py\` - 포인트 프롬프트
- \`sam_mouse_only.py\` - 네거티브 프롬프트
- \`test/visualize_sam_mouse_detection.py\` - 시각화

**통합 제안**:
\`\`\`python
# tools/sam_mask_tool.py (새로 생성)

\"""
통합 SAM 마스크 생성 도구
\"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='SAM Mask Generation Tool')
    parser.add_argument('--mode', choices=['auto', 'point', 'negative'], 
                       default='point', help='SAM mode')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default='sam_output')
    parser.add_argument('--point', type=int, nargs=2, help='Center point (x, y)')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == 'auto':
        run_automatic_mode(args)
    elif args.mode == 'point':
        run_point_prompt_mode(args)
    elif args.mode == 'negative':
        run_negative_prompt_mode(args)

if __name__ == '__main__':
    main()
\`\`\`

**사용 예**:
\`\`\`bash
# Automatic mode
python tools/sam_mask_tool.py --mode auto --video data/video.mp4

# Point prompt mode
python tools/sam_mask_tool.py --mode point --video data/video.mp4 --point 320 240

# Negative prompt mode
python tools/sam_mask_tool.py --mode negative --video data/video.mp4
\`\`\`

**장점**:
- ✅ 하나의 인터페이스로 모든 SAM 기능 접근
- ✅ 코드 중복 제거
- ✅ 유지보수 용이

#### 통합 2: Silhouette 관련 스크립트 정리

**현재 상황**:
- \`fit_silhouette_prototype.py\` (root)
- \`test/refine_with_silhouette.py\`
- \`test/test_silhouette_renderer.py\`
- \`test/test_silhouette_simple.py\`
- \`test/visualize_silhouette_results.py\`

**정리 제안**:
\`\`\`
preprocessing_utils/
└── silhouette/
    ├── __init__.py
    ├── renderer.py           # silhouette_renderer.py 이동
    ├── fitter.py             # fit_silhouette_prototype.py 통합
    └── visualizer.py         # visualize_silhouette_results.py 통합

tools/
└── fit_silhouette.py         # CLI 진입점
\`\`\`

### 2.3 디렉토리 구조 개선

#### 제안 구조

\`\`\`
MAMMAL_mouse/
├── conf/                     # Hydra 설정
├── preprocessing_utils/      # 핵심 모듈
│   ├── sam/                  # SAM 관련 모듈 (신규)
│   │   ├── inference.py
│   │   ├── prompts.py
│   │   └── visualizer.py
│   ├── silhouette/           # Silhouette 관련 모듈 (신규)
│   │   ├── renderer.py
│   │   ├── fitter.py
│   │   └── loss.py
│   ├── mask_processing.py
│   └── keypoint_estimation.py
├── tools/                    # CLI 도구들 (신규)
│   ├── sam_mask_tool.py
│   ├── fit_silhouette.py
│   └── visualize_results.py
├── scripts/                  # 메인 실행 스크립트
│   ├── preprocess.py         # 현재 root에서 이동
│   ├── fitter_articulation.py
│   └── evaluate.py
├── models/                   # 모델 정의 (신규)
│   ├── articulation_th.py    # 현재 root에서 이동
│   ├── bodymodel_np.py
│   └── bodymodel_th.py
├── experiments/              # test/ 폴더 이름 변경
│   ├── archived/             # 이전 실험 보관
│   └── active/               # 현재 진행 중인 실험
├── docs/                     # 문서화 (신규)
│   ├── manual.md
│   ├── SAM_MASK_ACQUISITION_MANUAL.md
│   └── API_REFERENCE.md
└── reports/                  # 분석 보고서
\`\`\`

### 2.4 파일 분류 및 이동 계획

#### Root 파일 정리

| 현재 위치 | 새 위치 | 이유 |
|-----------|---------|------|
| \`articulation_th.py\` | \`models/articulation_th.py\` | 모델 정의 |
| \`bodymodel_np.py\` | \`models/bodymodel_np.py\` | 모델 정의 |
| \`bodymodel_th.py\` | \`models/bodymodel_th.py\` | 모델 정의 |
| \`preprocess.py\` | \`scripts/preprocess.py\` | 메인 스크립트 |
| \`fitter_articulation.py\` | \`scripts/fitter_articulation.py\` | 메인 스크립트 |
| \`evaluate.py\` | \`scripts/evaluate.py\` | 메인 스크립트 |
| \`utils.py\` | \`preprocessing_utils/utils.py\` | 유틸리티 |
| \`sam_point_prompt.py\` | \`tools/sam_mask_tool.py\` | CLI 도구로 통합 |
| \`sam_mouse_only.py\` | \`tools/sam_mask_tool.py\` | CLI 도구로 통합 |
| \`fit_silhouette_prototype.py\` | \`tools/fit_silhouette.py\` | CLI 도구로 정리 |

#### Test 폴더 정리

| 현재 위치 | 새 위치 / 처리 | 이유 |
|-----------|---------------|------|
| \`test/preprocess_sam.py\` | \`experiments/archived/\` | Deprecated |
| \`test/test_sam.py\` | \`experiments/active/\` | 개발 중 |
| \`test/sam_point_prompt.py\` | 삭제 | Root 버전이 최신 |
| \`test/sam_mouse_only.py\` | 삭제 | Root 버전이 최신 |
| \`test/refine_with_silhouette.py\` | \`tools/fit_silhouette.py\` | 통합 |
| \`test/visualize_*.py\` | \`tools/\` | CLI 도구로 정리 |

---

## 3. 실행 계획

### 3.1 우선순위

#### 우선순위 1 (즉시): 중복 파일 제거
- [ ] \`sam_point_prompt.py\` vs \`test/sam_point_prompt.py\` 비교 및 통합
- [ ] \`sam_mouse_only.py\` vs \`test/sam_mouse_only.py\` 비교 및 통합
- [ ] \`test/preprocess_sam.py\` 삭제 (deprecated)

#### 우선순위 2 (단기): 폴더 구조 개선
- [ ] \`models/\` 폴더 생성 및 모델 파일 이동
- [ ] \`scripts/\` 폴더 생성 및 메인 스크립트 이동
- [ ] \`docs/\` 폴더 생성 및 문서 이동
- [ ] \`test/\` → \`experiments/\` 이름 변경

#### 우선순위 3 (중기): 코드 통합
- [ ] \`tools/sam_mask_tool.py\` 통합 CLI 개발
- [ ] \`tools/fit_silhouette.py\` 통합 CLI 개발
- [ ] Silhouette 모듈 정리 (\`preprocessing_utils/silhouette/\`)
- [ ] SAM 모듈 정리 (\`preprocessing_utils/sam/\`)

#### 우선순위 4 (장기): 문서화 및 테스트
- [ ] API 문서 작성 (\`docs/API_REFERENCE.md\`)
- [ ] 단위 테스트 추가
- [ ] CI/CD 설정

### 3.2 마이그레이션 스크립트

\`\`\`bash
#!/bin/bash
# migrate_codebase.sh

echo "=== MAMMAL Mouse Codebase Migration ==="

# 1. Backup
echo "[1/5] Creating backup..."
mkdir -p backup_20251104
cp -r . backup_20251104/ --exclude=backup_20251104

# 2. Create new directories
echo "[2/5] Creating new directory structure..."
mkdir -p models scripts tools docs experiments/archived experiments/active
mkdir -p preprocessing_utils/sam preprocessing_utils/silhouette

# 3. Move model files
echo "[3/5] Moving model files..."
mv articulation_th.py models/
mv bodymodel_np.py models/
mv bodymodel_th.py models/

# 4. Move main scripts
echo "[4/5] Moving main scripts..."
mv preprocess.py scripts/
mv fitter_articulation.py scripts/
mv evaluate.py scripts/

# 5. Move documentation
echo "[5/5] Moving documentation..."
mv manual.md docs/
mv SAM_MASK_ACQUISITION_MANUAL.md docs/
mv *.md docs/ 2>/dev/null

echo "✅ Migration complete!"
echo "⚠️  Please update import statements in affected files"
\`\`\`

---

## 4. 임포트 업데이트 가이드

### 4.1 필요한 수정 사항

파일 이동 후 import 경로를 업데이트해야 합니다.

#### 예: articulation_th.py 이동 후

**Before**:
\`\`\`python
from articulation_th import ArticulationTorch
\`\`\`

**After**:
\`\`\`python
from models.articulation_th import ArticulationTorch
\`\`\`

### 4.2 자동 업데이트 스크립트

\`\`\`python
# update_imports.py

import os
import re

replacements = {
    'from articulation_th': 'from models.articulation_th',
    'from bodymodel_np': 'from models.bodymodel_np',
    'from bodymodel_th': 'from models.bodymodel_th',
    'import articulation_th': 'import models.articulation_th',
}

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    modified = False
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            modified = True
    
    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Updated: {filepath}")

# Scan all Python files
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            update_file(filepath)
\`\`\`

---

## 5. 검증 체크리스트

### 5.1 마이그레이션 후 검증

- [ ] 모든 import 경로 정상 작동
- [ ] \`scripts/preprocess.py\` 정상 실행
- [ ] \`scripts/fitter_articulation.py\` 정상 실행
- [ ] SAM 도구 정상 작동
- [ ] Silhouette 피팅 정상 작동
- [ ] 기존 결과 재현 가능

### 5.2 회귀 테스트

\`\`\`bash
# 1. 전처리 테스트
python scripts/preprocess.py mode=single_view_preprocess

# 2. 피팅 테스트
python scripts/fitter_articulation.py fitter.end_frame=1

# 3. SAM 테스트
python tools/sam_mask_tool.py --mode point --video test_video.mp4

# 4. Silhouette 테스트
python tools/fit_silhouette.py --input results/param0.pkl
\`\`\`

---

## 6. 롤백 계획

문제 발생 시 원래 상태로 복구:

\`\`\`bash
# 백업에서 복구
rm -rf models scripts tools docs experiments
cp -r backup_20251104/* .

# Git을 사용하는 경우
git reset --hard HEAD
\`\`\`

---

## 7. 예상 효과

### 7.1 정량적 개선

| 항목 | Before | After | 개선 |
|------|--------|-------|------|
| Root 파일 수 | 14개 | 4개 | -71% |
| 중복 파일 | 4개 | 0개 | -100% |
| SAM 스크립트 | 5개 | 1개 | -80% |
| 폴더 깊이 | 2 level | 3 level | 구조화 |

### 7.2 정성적 개선

- ✅ 코드 가독성 향상
- ✅ 유지보수 용이성 증가
- ✅ 신규 개발자 온보딩 시간 단축
- ✅ 모듈 재사용성 증가
- ✅ 테스트 작성 용이

---

## 8. 다음 단계

1. **즉시 실행**: 중복 파일 제거 (우선순위 1)
2. **주간 계획**: 폴더 구조 개선 (우선순위 2)
3. **월간 계획**: 코드 통합 및 문서화 (우선순위 3-4)

---

**작성자**: Claude Code  
**검토 필요**: 프로젝트 관리자  
**승인 후 실행**: 백업 완료 후
