# MAMMAL_mouse Documentation

> 문서 허브 - 모든 문서의 위치와 역할

---

## 문서 구조

```
docs/
├── INDEX.md                 <- 이 파일 (허브)
├── QUICK_REFERENCE.md       <- 자주 쓰는 명령어 모음
│
├── guides/                  # 주제별 실행 가이드
│   ├── FITTING_GUIDE.md
│   ├── ANNOTATION_GUIDE.md
│   ├── PREPROCESSING_GUIDE.md
│   ├── UVMAP_GUIDE.md
│   └── VISUALIZATION_GUIDE.md
│
├── reference/               # 시스템 레퍼런스
│   ├── ARCHITECTURE.md
│   ├── KEYPOINTS.md
│   ├── DATASET.md
│   ├── OUTPUT_FORMAT.md
│   ├── COORDINATES.md
│   ├── PAPER.md
│   └── EXPERIMENTS.md
│
├── research/                # 날짜별 연구노트
│   └── YYMMDD_*.md
│
└── setup/                   # 환경 설정
    └── PYTORCH3D.md
```

---

## 가이드 (guides/)

주제별 실행 방법을 다룬다.

| 문서 | 내용 |
|------|------|
| **[FITTING_GUIDE](guides/FITTING_GUIDE.md)** | 메쉬 피팅 (multi-view + monocular + silhouette-only) |
| **[ANNOTATION_GUIDE](guides/ANNOTATION_GUIDE.md)** | 어노테이션 (도구 비교, 키포인트, SAM, Roboflow 워크플로우) |
| **[PREPROCESSING_GUIDE](guides/PREPROCESSING_GUIDE.md)** | 전처리 (비디오 추출, SAM 마스크, 파이프라인) |
| **[UVMAP_GUIDE](guides/UVMAP_GUIDE.md)** | UV 텍스처 (생성, Blender, HPO) |
| **[VISUALIZATION_GUIDE](guides/VISUALIZATION_GUIDE.md)** | 시각화 (mesh_visualizer, Blender, Rerun) |

---

## 레퍼런스 (reference/)

시스템 설계와 명세를 다룬다.

| 문서 | 내용 |
|------|------|
| **[ARCHITECTURE](reference/ARCHITECTURE.md)** | 시스템 아키텍처, Hydra config, 3단계 최적화, 시나리오별 사용법, Shell scripts |
| **[MAMMAL_EXT](reference/MAMMAL_EXT.md)** | mammal_ext 확장 모듈 아키텍처, 의존성 그래프, CLI 명령어 |
| **[KEYPOINTS](reference/KEYPOINTS.md)** | 22 키포인트 정의, GT vs Model 불일치, Skeleton topology, Sparse configs |
| **[DATASET](reference/DATASET.md)** | markerless_mouse_1_nerf 데이터셋 스펙 (6cam, 18K frames, pose-splatter 연동) |
| **[OUTPUT_FORMAT](reference/OUTPUT_FORMAT.md)** | 출력 파일 형식 (OBJ/PKL/render), Loss 해석, Downstream 활용법 |
| **[COORDINATES](reference/COORDINATES.md)** | 좌표계 (MAMMAL -Y up / OpenGL / OpenCV / Blender), 변환, PoseSplatter 정렬 |
| **[PAPER](reference/PAPER.md)** | 논문 설정 (iterations, wsil=0), Citation, paper_fast config |
| **[EXPERIMENTS](reference/EXPERIMENTS.md)** | 실험 config, Keypoint/View ablation, Batch runs, Background 실행 |

---

## 연구노트 (research/)

날짜순 연구 기록이다.

| 날짜 | 주제 |
|------|------|
| [251103](research/251103_sam_preprocessing.md) | SAM 기반 전처리 전환 |
| [251104](research/251104_silhouette_fitting.md) | Silhouette 기반 Fitting |
| [251114](research/251114_ml_keypoint_monocular.md) | ML Keypoint Detection + Monocular Fitting |
| [251115](research/251115_codebase_cleanup.md) | 코드베이스 정리 + ML Keypoint 요약 |
| [251117](research/251117_mesh_fitting_setup.md) | Mesh Fitting 시스템 구축 |
| [251118](research/251118_keypoint_annotation.md) | Keypoint Annotation 시스템 |
| [251125](research/251125_keypoint_pipeline.md) | Auto Keypoint Pipeline + Fitting 분석 |
| [251127](research/251127_silhouette_only.md) | Silhouette-Only Mesh Fitting |
| [251210](research/251210_ablation_uvmap.md) | View/Keypoint Ablation + UV Texture |
| [251212](research/251212_uvmap_hpo.md) | UV Map HPO Score + TV Regularization |
| [260126](research/260126_baseline_6view.md) | 6-View 22-Keypoint Baseline Fitting |

---

## 환경 설정 (setup/)

| 문서 | 내용 |
|------|------|
| **[PYTORCH3D](setup/PYTORCH3D.md)** | PyTorch3D 설치 (소스 빌드), ABI 호환성 문제 해결, Troubleshooting |

---

## Quick Links

### 처음 시작 시

1. [PYTORCH3D 설치](setup/PYTORCH3D.md)
2. [ARCHITECTURE](reference/ARCHITECTURE.md) - 시스템 이해
3. [QUICK_REFERENCE](QUICK_REFERENCE.md) - 자주 쓰는 명령어

### 실험 실행 시

1. [EXPERIMENTS](reference/EXPERIMENTS.md) - 실험 config 및 batch runs
2. [PAPER](reference/PAPER.md) - 논문 설정 참조
3. [OUTPUT_FORMAT](reference/OUTPUT_FORMAT.md) - 결과 파일 해석

### 데이터 작업 시

1. [DATASET](reference/DATASET.md) - 데이터셋 스펙
2. [KEYPOINTS](reference/KEYPOINTS.md) - 키포인트 정의
3. [COORDINATES](reference/COORDINATES.md) - 좌표계 변환

---

*Last updated: 2026-02-06*
