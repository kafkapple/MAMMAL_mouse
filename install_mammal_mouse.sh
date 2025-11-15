#!/bin/bash

echo "========================================="
echo "MAMMAL Mouse 환경 설치 시작"
echo "========================================="

# 에러 발생 시 스크립트 중단
set -e

# 환경 재생성
echo ""
echo "📦 [1/9] Conda 환경 재생성..."
conda deactivate 2>/dev/null || true
conda remove -n mouse --all -y 2>/dev/null || true
conda create -n mouse python=3.9 -y

# Conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse

# CUDA 11.8 환경 변수 설정
echo ""
echo "🔧 [2/9] CUDA 11.8 환경 변수 설정..."
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# nvcc 버전 확인
echo ""
echo "✓ [3/9] CUDA 컴파일러 버전 확인..."
nvcc --version

# PyTorch 설치 (CUDA 11.8)
echo ""
echo "🔥 [4/9] PyTorch (CUDA 11.8) 설치..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 인식 확인
echo ""
echo "✓ [5/9] PyTorch CUDA 인식 확인..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# 핵심 의존성 먼저 설치 (버전 충돌 방지)
echo ""
echo "📚 [6/9] 핵심 의존성 설치 (버전 호환성 고려)..."
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78  # numpy 1.24.3과 호환되는 버전
pip install tensorboard==2.14.0
pip install trimesh==3.23.5
pip install scipy==1.10.1
pip install scikit-image==0.21.0
pip install tqdm

# 렌더링 관련 패키지
echo ""
echo "🎨 [7/9] 렌더링 패키지 설치..."
pip install pyrender==0.1.45
pip install pyopengl==3.1.7
pip install pyopengl-accelerate==3.1.7

# PyTorch3D 설치
echo ""
echo "🐍 [8/9] PyTorch3D 설치..."
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

# 추가 requirements.txt가 있다면 설치
if [ -f "requirements.txt" ]; then
    echo ""
    echo "📄 [추가] requirements.txt 확인 중..."
    # requirements.txt에서 numpy와 opencv-python 제외하고 설치
    grep -v "numpy" requirements.txt | grep -v "opencv-python" > requirements_filtered.txt
    if [ -s requirements_filtered.txt ]; then
        pip install -r requirements_filtered.txt
    fi
    rm -f requirements_filtered.txt
fi

# Conda 환경에 환경 변수 영구 설정
echo ""
echo "💾 [9/9] Conda 환경에 환경 변수 영구 설정..."
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'ENVEOF'
#!/bin/sh
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
ENVEOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << 'ENVEOF'
#!/bin/sh
unset CUDA_HOME
unset PYOPENGL_PLATFORM
ENVEOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# 최종 확인
echo ""
echo "========================================="
echo "🔍 설치 확인 중..."
echo "========================================="
python << 'PYEOF'
import sys
import torch
import pytorch3d
import numpy as np
import cv2
import trimesh
import pyrender

print("=" * 50)
print("✅ 설치 완료!")
print("=" * 50)
print(f"✓ Python: {sys.version.split()[0]}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ CUDA version: {torch.version.cuda}")
print(f"✓ GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
print(f"✓ PyTorch3D: {pytorch3d.__version__}")
print(f"✓ NumPy: {np.__version__}")
print(f"✓ OpenCV: {cv2.__version__}")
print(f"✓ Trimesh: {trimesh.__version__}")
print(f"✓ PyRender: {pyrender.__version__}")
print("=" * 50)
PYEOF

echo ""
echo "========================================="
echo "✅ 모든 설치가 완료되었습니다!"
echo "========================================="
echo ""
echo "다음 명령어로 환경을 활성화하세요:"
echo "  conda activate mouse"
echo ""
echo "프로그램 실행:"
echo "  bash run.sh"
echo ""