# PyTorch3D Installation & Troubleshooting

> PyTorch3D 설치, ABI 호환성 문제 해결, 대안

---

## 문제 상황

```
ImportError: .../pytorch3d/_C.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZNK3c105Error4whatEv
```

**원인**: PyTorch3D binary가 현재 설치된 PyTorch 버전과 다른 PyTorch로 컴파일됨 (ABI 불일치).

**환경**:
- PyTorch: 2.0.0+cu118
- PyTorch3D: 0.7.5 권장

---

## Quick Fix (권장)

```bash
cd /home/joon/dev/MAMMAL_mouse
./fix_pytorch3d.sh
```

자동으로: 기존 제거 -> 소스에서 0.7.5 빌드 (5-10분) -> 검증

---

## 수동 설치 옵션

### Option 1: 소스 빌드 (권장)

```bash
conda activate mammal_stable

# 기존 제거
pip uninstall pytorch3d -y

# 빌드 의존성
pip install fvcore iopath

# 소스에서 설치 (5-10분)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### Option 2: Conda (빠름, 호환성 불확실)

```bash
conda activate mammal_stable
pip uninstall pytorch3d -y
conda install -c conda-forge pytorch3d -y
```

### Option 3: CUB 수동 지정 (CUDA 에러 시)

```bash
# NVIDIA CUB 다운로드
cd /tmp
wget https://github.com/NVIDIA/cub/archive/1.17.2.tar.gz -O cub-1.17.2.tar.gz
tar -xzf cub-1.17.2.tar.gz

# 환경변수 설정
export CUB_HOME=/tmp/cub-1.17.2
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 빌드
MAX_JOBS=4 pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### Option 4: 렌더링 비활성화 (Workaround)

PyTorch3D 없이도 핵심 피팅은 동작한다. `fitter.with_render=false`로 실행하고 silhouette refinement (Step2)를 건너뛴다.

---

## 검증

```bash
conda activate mammal_stable
python -c "
import torch
import pytorch3d
from pytorch3d import _C
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch3D: {pytorch3d.__version__}')
print('PyTorch3D C++ extensions loaded successfully!')
"
```

기대 출력:
```
PyTorch: 2.0.0+cu118
PyTorch3D: 0.7.5
PyTorch3D C++ extensions loaded successfully!
```

피팅 테스트:
```bash
./run_mesh_fitting_default.sh quick_test
```

---

## Troubleshooting

### CUDA 컴파일 에러

```
error: command '/usr/local/cuda/bin/nvcc' failed
```

**해결**:
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version  # CUDA 11.8 확인
```

### GCC 에러

```
error: no matching function for call to 'at::Tensor::item()'
```

**해결**:
```bash
conda install gcc_linux-64 gxx_linux-64 -y
```

### 메모리 부족 (컴파일 중)

```
c++: internal compiler error: Killed
```

**해결**:
```bash
export MAX_JOBS=2
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

### 재설치 후에도 Symbol 에러

```bash
pip cache purge
pip uninstall pytorch3d -y
rm -rf ~/anaconda3/envs/mammal_stable/lib/python3.10/site-packages/pytorch3d*
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

---

## 예방 (새 환경 설정 시)

```bash
# PyTorch 먼저 설치
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 그 다음 PyTorch3D를 소스에서 설치
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"

# pip/conda의 pre-built binary 사용 금지 (호환성 문제)
# pip install pytorch3d  # <-- 이렇게 하지 말 것
```

---

## 시스템 요구사항

| 항목 | 요구 |
|------|------|
| GPU | NVIDIA (CUDA 지원) |
| CUDA | 11.8 |
| GCC | 7.x - 11.x |
| RAM | 8GB 이상 |
| 디스크 | ~5GB |

---

## Reference

- PyTorch3D GitHub: https://github.com/facebookresearch/pytorch3d
- Installation Guide: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

---

*Last updated: 2026-02-06*
