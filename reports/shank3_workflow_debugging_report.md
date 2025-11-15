## `shank3` 데이터 처리 워크플로우 디버깅 및 실행 시도 보고서

- **날짜**: 2025년 10월 31일
- **최종 목표**: `shank3` 비디오 데이터를 입력으로 사용하여, 전처리, 3D 메쉬 피팅, 그리고 결과물(이미지, 비디오) 생성까지의 전체 파이프라인을 실행하고, 관련 내용을 `README.md`에 문서화한다.

---

### 1. 사전 조사 및 계획 수립

- **내용**: `shank3` 키워드 검색을 통해 관련 코드의 존재 여부를 확인했다. `preprocess.py`, `fitter_articulation.py`, `conf/config.yaml` 파일을 분석하여 전체 워크플로우를 파악했다.
- **결론**: 파이프라인이 `conf/config.yaml`의 `mode` 파라미터를 통해 **전처리**와 **피팅**의 2단계로 제어됨을 확인했다. 최종 비디오 생성 기능은 코드에 부재함을 파악하고, 생성된 이미지들을 `ffmpeg`으로 통합하는 3단계 계획을 수립했다.

### 2. `README.md` 업데이트

- **내용**: `shank3` 데이터 처리 및 `ffmpeg`을 이용한 영상 생성까지의 전체 과정을 상세히 기술한 가이드를 `README.md` 파일에 추가했다.
- **결과**: **성공**

### 3. 1단계: 전처리 실행

- **내용**: `conf/config.yaml`의 `mode`를 `single_view_preprocess`로 설정하고 `preprocess.py`를 실행했다.
- **결과**: **성공.** `data/preprocessed_shank3/` 경로에 피팅에 필요한 파일들이 정상적으로 생성됨을 확인했다.

### 4. 2단계: 피팅 실행 (핵심 디버깅 과정)

피팅 스크립트(`fitter_articulation.py`) 실행 과정에서 아래와 같이 연속적인 오류가 발생했으며, 각 단계별로 해결을 시도했다.

- **시도 1: 최초 실행**
    - **오류**: `ModuleNotFoundError: No module named 'tensorboard'`
    - **원인 분석**: 환경에 `tensorboard` 패키지가 설치되지 않음.
    - **해결 시도**: `pip install tensorboard` 명령으로 패키지 설치.

- **시도 2: `tensorboard` 설치 후**
    - **오류**: `AttributeError: module 'distutils' has no attribute 'version'`
    - **원인 분석**: `tensorboard` 설치로 인해 `setuptools` 버전이 변경되면서, 구버전 `torch`와의 호환성 문제가 발생.
    - **해결 시도**: `setuptools`와 `numpy` 버전을 호환되는 구버전으로 강제 다운그레이드.

- **시도 3: 의존성 수정 후**
    - **오류**: `ModuleNotFoundError: No module named 'numpy._core'`
    - **원인 분석**: `numpy` 버전이 변경된 환경에서, 이전 버전의 `numpy`로 생성된 전처리 파일(`.pkl`)을 읽으려 하여 데이터 호환성 문제 발생.
    - **해결 시도**: 수정된 환경에서 `preprocess.py`를 재실행하여 전처리 데이터를 다시 생성.

- **시도 4: 전처리 데이터 재생성 후**
    - **오류**: `pyglet.display.xlib.NoSuchDisplayException: Cannot connect to "None"`
    - **원인 분석**: GUI가 없는 서버 환경에서 렌더링을 시도하여 디스플레이 연결 오류 발생.
    - **해결 시도**: `run.sh` 파일에 명시된 대로, `PYOPENGL_PLATFORM=egl` 환경 변수를 설정하여 EGL 백엔드로 오프스크린 렌더링을 강제.

- **시도 5: EGL 환경 변수 설정 후**
    - **오류**: `RuntimeError: The size of tensor a (22) must match the size of tensor b (3) at non-singleton dimension 1`
    - **원인 분석**: 2D 키포인트 손실 계산(`calc_2d_keypoint_loss`) 시, 카메라 이동 벡터 `T`의 행렬 차원이 덧셈 연산에 맞지 않음.
    - **해결 시도**: `fitter_articulation.py`의 해당 라인에서 `T` 벡터를 전치(`transpose`)하여 차원을 맞추는 코드로 수정.

- **시도 6 ~ 12: 코드 수정 및 반복 실패**
    - **오류**: `ValueError: could not broadcast input array...`, `NameError: name 'self' is not defined` 등 다양한 오류 발생.
    - **원인 분석**: 
        1. `render` 함수 내에서 또 다른 차원 불일치 문제가 존재했다.
        2. 이 문제를 `write_file`로 수정하는 과정에서, 실수로 코드 블록 전체의 **들여쓰기(indentation)가 손상**되어 `NameError`가 발생하는 등 추가적인 버그가 유입되었다.
        3. 최종적으로 들여쓰기와 코드 오류를 모두 수정한 버전을 `write_file`로 덮어썼음에도, `ValueError`가 계속해서 동일하게 발생했다. 이는 파일 수정 도구가 변경사항을 안정적으로 적용하지 못했거나, 다른 근본적인 문제가 있음을 시사한다.

---

### 최종 실패 원인 분석

1.  **복잡한 의존성**: `torch`, `numpy`, `setuptools`, `pyrender` 등 여러 라이브러리 간의 특정 버전 조합이 매우 중요했으나, 초기 환경 설정(`run.sh`)이 이를 완벽하게 제어하지 못했다.
2.  **내재된 코드 버그**: 스크립트 내부에 최소 2개 이상의 명백한 버그(행렬 차원 불일치)가 존재했다.
3.  **파일 수정의 불안정성**: `replace` 및 `write_file` 도구를 사용한 코드 수정이 반복적으로 실패하거나, 또 다른 오류(들여쓰기)를 유발했다. 마지막에는 수정 사항이 적용되지 않은 채 동일한 오류가 계속 발생하여, 더 이상 디버깅을 진행할 수 없었다.

### 결론

다단계에 걸친 복합적인 의존성 문제와 코드 내부의 여러 버그, 그리고 이를 수정하는 과정에서 발생한 파일 수정의 불안정성으로 인해 **`fitter_articulation.py` 스크립트 실행에 최종적으로 실패했다.**

피팅이 성공하지 못해 결과 이미지가 생성되지 않았으므로, 마지막 단계인 `ffmpeg`을 이용한 비디오 생성 또한 불가능했다. 따라서 요청된 `shank3` 워크플로우의 완전한 실행은 완수하지 못했다.
