# MAMMAL_mouse 모델 상세 분석 및 활용 매뉴얼

## 1. 개요

이 문서는 `MAMMAL_mouse` 프로젝트의 상세 아키텍처, 데이터 요구사항, 실행 방법 및 일반화 방안을 설명합니다. 이 프로젝트는 다중 시점 비디오, 2D 키포인트, 실루엣 마스크를 입력받아, 각 프레임에 대한 마우스의 3D 관절형 메쉬(articulated mesh)를 생성하는 최적화 기반의 파이프라인입니다.

## 2. 전체 파이프라인

`run.sh` 스크립트를 실행하면 `fitter_articulation.py`가 호출되며, 다음 과정이 프레임별로 순차적으로 진행됩니다.

```
Input Data --> [Data Loader] --> [Fitter] --> Output Files
```

-   **Input Data**: 특정 프레임에 대한 다중 시점 이미지, 키포인트, 마스크, 카메라 파라미터
-   **Data Loader (`DataSeakerDet`)**: 입력 데이터를 파이프라인에 맞는 형식으로 가공하여 `Fitter`에 전달
-   **Fitter (`MouseFitter`)**: 핵심 로직. 최적화를 통해 3D 모델을 데이터에 피팅
-   **Output Files**: 최적화된 모델 파라미터(`.pkl`), 렌더링 이미지(`.png`), 3D 메쉬(`.obj`)

---

## 3. 입력 데이터 상세 명세 (`data/markerless_mouse_1_nerf/`)

모델 학습을 위해 아래와 같은 구조와 명세를 가진 데이터가 필요합니다.

| 경로/파일명 | 데이터 타입 | 차원/형식 | 설명 |
| :--- | :--- | :--- | :--- |
| `new_cam.pkl` | `pickle` | `dict` | 6개 시점의 카메라 파라미터. 각 카메라(key: `0`~`5`)는 다음을 포함:<ul><li>`'K'`: (3, 3) `np.float64` - 내부 파라미터</li><li>`'R'`: (3, 3) `np.float64` - 회전 행렬</li><li>`'T'`: (3, 1) `np.float64` - 이동 벡터</li></ul> |
| `videos_undist/*.mp4` | 비디오 | (H, W, 3) | 왜곡 보정된 6개 시점의 컬러 비디오. (예: `0.mp4`, `1.mp4`, ...) |
| `simpleclick_undist/*.mp4`| 비디오 | (H, W, 3) | 마우스 영역만 흰색(255), 배경은 검은색(0)인 마스크 비디오. |
| `keypoints2d_undist/result_view_*.pkl` | `pickle` | `np.ndarray` | 각 시점별 2D 키포인트.<ul><li>**Shape**: `(18000, 22, 3)`</li><li>**Dimension 0**: 프레임 인덱스</li><li>**Dimension 1**: 22개 키포인트 인덱스</li><li>**Dimension 2**: `(x, y, confidence)`</li></ul> |

---

## 4. 핵심 로직 상세 (`fitter_articulation.py`)

### `MouseFitter` 클래스

-   **초기화 (`__init__`)**:
    -   PyTorch `device` 설정 ('cuda')
    -   관절 모델 `ArticulationTorch` 로드
    -   미분 가능 렌더러 `MeshRenderer` (from Pytorch3D) 초기화
    -   각종 손실 함수의 가중치(`term_weights`), 키포인트 가중치(`keypoint_weight`) 등 설정

-   **최적화 프로세스**: L-BFGS 옵티마이저를 사용하여 3단계에 걸쳐 파라미터를 최적화합니다.

    1.  **`solve_step0` (Global Alignment)**:
        -   **목표**: 전체적인 위치(`trans`), 회전(`rotation`), 크기(`scale`) 맞춤.
        -   **활성 파라미터**: `trans`, `rotation`, `scale`
        -   **비활성 파라미터**: `thetas`, `bone_lengths`, `chest_deformer`
        -   **주요 손실**: `2d` (키포인트 재투영 오차)

    2.  **`solve_step1` (Skeletal Fitting)**:
        -   **목표**: 관절 각도(`thetas`)와 뼈 길이(`bone_lengths`)를 최적화하여 세부 포즈 맞춤.
        -   **활성 파라미터**: `trans`, `rotation`, `scale`, `thetas`, `bone_lengths`
        -   **비활성 파라미터**: `chest_deformer`
        -   **주요 손실**: `2d`, `theta` (정규화), `bone` (정규화), `temp` (이전 프레임과의 유사도)

    3.  **`solve_step2` (Mask-based Refinement)**:
        -   **목표**: 실루엣 마스크를 사용하여 메쉬 표면을 정교하게 맞춤.
        -   **활성 파라미터**: 모든 파라미터 (`chest_deformer` 포함)
        -   **주요 손실**: `mask` (실루엣 오차), `2d`, `theta`, `bone`, `temp`

### 모델 파라미터 (`body_param`)

최적화 대상이 되는 파라미터들입니다. `torch.Tensor`로 변환되어 `requires_grad_(True)`가 설정됩니다.

| 파라미터 | 데이터 타입 | 차원 | 설명 |
| :--- | :--- | :--- | :--- |
| `thetas` | `np.ndarray` | `(1, 140, 3)` | 140개 관절의 회전 (axis-angle). |
| `trans` | `np.ndarray` | `(1, 3)` | 모델의 전역 이동 (x, y, z). |
| `scale` | `np.ndarray` | `(1, 1)` | 모델의 전역 크기. |
| `rotation`| `np.ndarray` | `(1, 3)` | 모델의 전역 회전 (root joint). |
| `bone_lengths`|`np.ndarray` | `(1, 20)` | 주요 20개 뼈의 길이 변화량. |
| `chest_deformer`|`np.ndarray` | `(1, 1)` | 가슴 부분 메쉬를 변형시키는 파라미터. |

---

## 5. 출력 데이터 상세 명세 (`mouse_fitting_result/results/`)

최적화 완료 후 생성되는 파일들입니다.

| 경로/파일명 | 데이터 타입 | 차원/형식 | 설명 |
| :--- | :--- | :--- | :--- |
| `params/param{id}.pkl` | `pickle` | `dict` | `solve_step1` 완료 후의 모델 파라미터. 값은 `torch.Tensor`. |
| `params/param{id}_sil.pkl`| `pickle` | `dict` | `solve_step2` 완료 후의 최종 모델 파라미터. 값은 `torch.Tensor`. |
| `render/fitting_{id}.png` | 이미지 | (H, W, 3) | `solve_step1` 결과 렌더링. 여러 시점이 하나의 이미지로 합쳐짐. |
| `render/fitting_{id}_sil.png`| 이미지 | (H, W, 3) | `solve_step2` 결과 렌더링. |
| `fitting_keypoints_{id}.png`| 이미지 | (H, W, 3) | 원본 이미지에 2D 키포인트 예측을 시각화한 결과. |
| `obj/mesh_{id:06d}.obj` | 텍스트 | `.obj` | 최종 3D 메쉬.<ul><li>`v x y z`: 정점(vertex) 좌표</li><li>`f v1 v2 v3`: 면(face)을 구성하는 정점 인덱스</li></ul> |

---

## 6. 일반화 방안

### 옵션 1: Hydra를 이용한 모드 전환

-   `conf/config.yaml` 파일을 통해 `mode`를 `multi_view` 또는 `single_view`로 설정하여 기존 파이프라인과 단일 뷰 파이프라인을 선택적으로 실행할 수 있도록 합니다.

### 옵션 2: 단일 비디오 자동 전처리

-   **입력**: 비디오 파일 1개 (e.g., `my_video.mp4`)
-   **프로세스 (`preprocess.py`)**:
    1.  **마스크 생성**: `cv2.createBackgroundSubtractorMOG2`와 같은 배경 제거 기법으로 마스크 영상(`simpleclick_undist/0.mp4`)을 생성합니다.
    2.  **키포인트 생성**: 생성된 마스크의 윤곽선(contour)에서 기하학적 특징(e.g., 최상단, 최하단, 무게중심 등)을 추출하여 `keypoints2d_undist/result_view_0.pkl` 형식에 맞는 "가짜" 키포인트를 생성합니다. (정확도는 낮지만 파이프라인 실행을 위함)
    3.  **카메라 파라미터 생성**: 표준적인 값으로 채워진 가상 카메라(`new_cam.pkl`)를 생성합니다.
-   **결과**: `fitter_articulation.py`를 바로 실행할 수 있는 데이터 폴더 구조를 자동으로 생성합니다.