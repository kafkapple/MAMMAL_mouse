# EGL 렌더링 강제 설정
export PYOPENGL_PLATFORM=egl

# CUDA 경로 설정
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================="
echo "MAMMAL Mouse 실행 시작"
echo "========================================="
echo "렌더링 모드: $PYOPENGL_PLATFORM"
echo "CUDA 버전: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo ""

# 환경 변수 확인
if [ -z "$PYOPENGL_PLATFORM" ]; then
    echo "⚠️  오류: PYOPENGL_PLATFORM이 설정되지 않았습니다!"
    exit 1
fi

# --- Clean and Recreate Conda Environment ---
echo "--- Cleaning conda cache ---"
conda clean --all -y

echo "--- Removing and Recreating 'mouse' conda environment ---"
conda env remove -n mouse -y
conda create -n mouse python=3.9 -y
source activate mouse

# --- Install Dependencies ---
echo "--- Installing pip ---"
pip install --upgrade pip

echo "--- Installing PyTorch ---"
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3 -c pytorch -y

# Downgrade NumPy for compatibility with PyTorch and PyTorch3D
echo "--- Downgrading NumPy to 1.23.5 ---"
pip install numpy==1.23.5 --force-reinstall

echo "--- Installing requirements.txt ---"
pip install -r requirements.txt

echo "--- Installing PyTorch3D dependencies ---"
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install jupyter -y
pip install black usort flake8 flake8-bugbear flake8-comprehensions

echo "--- Installing PyTorch3D ---"
conda install pytorch3d==0.6.2 -c pytorch3d -y

echo "--- Installing Hydra/OmegaConf ---"
pip install omegaconf hydra-core # Ensure omegaconf and hydra-core are installed in the activated environment

# --- Run the main script ---
echo "--- Running fitter_articulation.py ---"
python fitter_articulation.py

echo ""
echo "========================================="
echo "처리 완료! 결과는 mouse_fitting_result/ 에 저장되었습니다."
echo "========================================="