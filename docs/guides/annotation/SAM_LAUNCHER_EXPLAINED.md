# SAM Annotator 실행 방법 상세 설명

## 문제의 근본 원인: Hydra + conda run 충돌

### 왜 `bash run_sam_annotator.sh`가 안 되나?

`run_sam_annotator.sh`는 다음과 같이 실행합니다:
```bash
conda run -n mammal_stable python -m sam_annotator \
    data.input_dir=... \
    model.device=cuda
```

**문제점 2가지:**

1. **conda run + Hydra 충돌**
   ```
   ValueError: GlobalHydra is already initialized
   ```
   - `conda run`이 Python 프로세스를 특수한 방식으로 실행
   - Hydra가 이미 초기화된 상태로 인식
   - Hydra 재초기화 시도 → 오류

2. **Relative import 오류**
   ```bash
   # 이렇게 실행하면:
   python sam_annotator/cli.py

   # cli.py 내부:
   from .app import launch_annotator  # ← 오류!
   ```
   - Python이 `sam_annotator`를 패키지로 인식 못함
   - Relative import (`.app`) 사용 불가

### 왜 `run_sam_gui.py`는 되나?

`run_sam_gui.py`는:
```python
# 1. 절대 경로로 패키지 추가
sys.path.insert(0, str(sam_annotator_path))

# 2. 절대 import 사용
from sam_annotator.app import launch_annotator  # ← OK!

# 3. Hydra를 직접 사용하지 않고 OmegaConf만 사용
cfg = OmegaConf.create(config)  # ← Hydra 우회
```

**장점:**
- ✅ conda run/activate 둘 다 OK
- ✅ Hydra 초기화 문제 회피
- ✅ 어디서든 실행 가능

---

## 올바른 실행 방법 정리

### 현재 프로젝트 (MAMMAL_mouse)

#### 방법 1: run_sam_gui.py (권장)
```bash
conda activate mammal_stable
cd /home/joon/dev/MAMMAL_mouse

python run_sam_gui.py \
    --frames-dir data/100-KO-male-56-20200615_frames \
    --port 7860
```

**장점:**
- 간단하고 확실함
- 어떤 conda 환경에서도 작동
- 오류 없음

#### 방법 2: conda 환경 활성화 후 직접 실행
```bash
conda activate mammal_stable
cd /home/joon/dev/mouse-super-resolution

python -m sam_annotator \
    data.input_dir=/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_frames \
    data.output_dir=/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_frames/annotations \
    ui.server_port=7860
```

**주의:**
- ⚠️ `conda activate` 필수 (conda run 사용 금지)
- ⚠️ `cd /home/joon/dev/mouse-super-resolution` 필수
- ⚠️ 절대 경로 사용 필수

---

## 다른 프로젝트에서 사용하는 방법

### 시나리오 1: 다른 프로젝트에서 sam_annotator 사용

예: `/home/joon/dev/my_project`에서 사용

#### Option A: run_sam_gui.py 복사 (가장 쉬움)

```bash
# 1. run_sam_gui.py를 프로젝트에 복사
cp /home/joon/dev/MAMMAL_mouse/run_sam_gui.py /home/joon/dev/my_project/

# 2. 실행
cd /home/joon/dev/my_project
conda activate mammal_stable

python run_sam_gui.py \
    --frames-dir ./my_frames \
    --port 7860
```

**장점:**
- 어디서든 동일하게 작동
- 경로만 수정하면 됨
- 독립적으로 관리 가능

#### Option B: 원본 sam_annotator 직접 사용

```bash
cd /home/joon/dev/mouse-super-resolution
conda activate mammal_stable

python -m sam_annotator \
    data.input_dir=/home/joon/dev/my_project/frames \
    data.output_dir=/home/joon/dev/my_project/annotations \
    ui.server_port=7860
```

**주의:**
- ⚠️ 반드시 `mouse-super-resolution` 디렉토리에서 실행
- ⚠️ 절대 경로 사용

#### Option C: Python 스크립트로 래핑

```python
# /home/joon/dev/my_project/start_sam.py
import sys
from pathlib import Path

# sam_annotator 경로 추가
sys.path.insert(0, str(Path.home() / 'dev/mouse-super-resolution'))

from omegaconf import OmegaConf
from sam_annotator.app import launch_annotator

# 설정
cfg = OmegaConf.create({
    'model': {
        'name': 'sam2.1_hiera_large',
        'checkpoint': str(Path.home() / 'dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt'),
        'config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
        'device': 'cuda'
    },
    'data': {
        'input_dir': './my_frames',
        'pattern': '*.png',
        'output_dir': './annotations'
    },
    'ui': {
        'server_name': '0.0.0.0',
        'server_port': 7860,
        'share': False
    },
    # ... (나머지 설정)
})

launch_annotator(cfg)
```

실행:
```bash
conda activate mammal_stable
python start_sam.py
```

---

## mouse-super-resolution 원본 프로젝트에서 사용

### 원본 프로젝트에서 직접 실행

```bash
cd /home/joon/dev/mouse-super-resolution
conda activate mammal_stable

# 방법 1: Hydra 방식 (권장)
python -m sam_annotator \
    data.input_dir=./data/my_images \
    data.output_dir=./data/annotations

# 방법 2: Simple CLI (대안)
python -c "
from sam_annotator.cli import simple_cli
simple_cli()
" \
    --input ./data/my_images \
    --output ./data/annotations \
    --port 7860
```

**중요:**
- ✅ `conda activate` 사용 (conda run 금지)
- ✅ `cd /home/joon/dev/mouse-super-resolution` 필수

---

## 왜 이렇게 복잡한가?

### Python 패키지 import 시스템 때문

**정상 작동 조건:**
1. Python이 `sam_annotator`를 패키지로 인식
2. `python -m sam_annotator` 실행 시 현재 디렉토리가 부모 디렉토리여야 함

```
mouse-super-resolution/     ← 여기서 실행해야 함
├── sam_annotator/
│   ├── __init__.py
│   ├── __main__.py
│   └── ...
```

**잘못된 실행:**
```bash
cd /some/other/path
python -m sam_annotator  # ← sam_annotator를 못 찾음!
```

### Hydra 초기화 문제

Hydra는 전역 싱글톤 패턴:
- 한 프로세스에서 한 번만 초기화 가능
- `conda run`이 이미 초기화된 것으로 인식
- 재초기화 시도 → 오류

---

## 권장 사용 패턴

### 1. MAMMAL_mouse 프로젝트에서

```bash
# run_sam_gui.py 사용 (가장 간단)
conda activate mammal_stable
python run_sam_gui.py --frames-dir ./data/frames --port 7860
```

### 2. 새 프로젝트에서

```bash
# run_sam_gui.py 복사해서 사용
cp /home/joon/dev/MAMMAL_mouse/run_sam_gui.py ./
conda activate mammal_stable
python run_sam_gui.py --frames-dir ./frames --port 7860
```

### 3. mouse-super-resolution에서

```bash
cd /home/joon/dev/mouse-super-resolution
conda activate mammal_stable
python -m sam_annotator data.input_dir=./data/images
```

---

## run_sam_annotator.sh 수정 방안

`run_sam_annotator.sh`를 작동하게 만들려면:

```bash
#!/bin/bash
# 수정된 run_sam_annotator.sh

FRAMES_DIR="/home/joon/dev/MAMMAL_mouse/data/100-KO-male-56-20200615_frames"
PORT=${1:-7860}

# conda activate 확인
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: Please activate conda environment first:"
    echo "  conda activate mammal_stable"
    exit 1
fi

# run_sam_gui.py 실행
cd /home/joon/dev/MAMMAL_mouse

python run_sam_gui.py \
    --frames-dir "${FRAMES_DIR}" \
    --port ${PORT}
```

사용:
```bash
conda activate mammal_stable
bash run_sam_annotator.sh 7860
```

**하지만** 이것도 결국 `run_sam_gui.py`를 호출하는 것이므로, 직접 실행하는 것이 더 간단합니다.

---

## 요약

| 방법 | 작동 여부 | 난이도 | 권장 |
|------|----------|--------|------|
| `conda run + python -m sam_annotator` | ❌ | - | ❌ |
| `conda activate + python -m sam_annotator` | ✅ | 중 | △ |
| `run_sam_gui.py` | ✅ | 하 | ⭐ |
| `run_sam_gui.py` 복사 | ✅ | 하 | ⭐ |

**결론:**
- **현재 프로젝트**: `run_sam_gui.py` 직접 실행
- **다른 프로젝트**: `run_sam_gui.py` 복사해서 사용
- **원본 프로젝트**: `cd mouse-super-resolution` 후 `python -m sam_annotator`

---

## 실전 예시

### 예시 1: 다른 비디오 처리

```bash
cd /home/joon/dev/MAMMAL_mouse

# 프레임 추출
conda run -n mammal_stable python extract_video_frames.py \
    /path/to/video2.avi \
    --output-dir data/video2_frames \
    --num-frames 20

# SAM 실행
conda activate mammal_stable
python run_sam_gui.py \
    --frames-dir data/video2_frames \
    --port 7860
```

### 예시 2: 다른 프로젝트

```bash
# 1. run_sam_gui.py 복사
cp /home/joon/dev/MAMMAL_mouse/run_sam_gui.py /home/joon/dev/other_project/

# 2. 실행
cd /home/joon/dev/other_project
conda activate mammal_stable

python run_sam_gui.py \
    --frames-dir ./my_frames \
    --port 7860
```

---

**핵심: `run_sam_gui.py`는 재사용 가능한 범용 런처입니다!**
