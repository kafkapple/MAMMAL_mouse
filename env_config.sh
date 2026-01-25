#!/bin/bash
# ===== Environment-specific Configuration =====
# Auto-detect server and set appropriate GPU/paths
#
# Usage: source env_config.sh (automatically sourced by run scripts)

HOSTNAME=$(hostname)

case "$HOSTNAME" in
    gpu05|gpu05.*)
        # gpu05 (joon) 서버: GPU 1 사용
        export GPU_ID="${GPU_ID:-1}"
        export CONDA_PATH="$HOME/miniconda3"
        export MAMMAL_ENV="mammal_stable"
        ;;
    gpu03|gpu03.*)
        # gpu03 서버: GPU 0-3 사용 가능 (H100), 기본 GPU 0
        export GPU_ID="${GPU_ID:-0}"
        export CONDA_PATH="$HOME/anaconda3"
        export MAMMAL_ENV="mammal_stable"
        ;;
    bori|bori.*)
        # bori 서버: GPU 0 사용 (GPU 1개만 있음)
        export GPU_ID="${GPU_ID:-0}"
        export CONDA_PATH="$HOME/miniconda3"
        export MAMMAL_ENV="mammal_stable"
        ;;
    *)
        # 기타 서버: 기본값
        export GPU_ID="${GPU_ID:-0}"
        export CONDA_PATH="$HOME/miniconda3"
        export MAMMAL_ENV="mammal_stable"
        ;;
esac

# Common GPU settings
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EGL_DEVICE_ID="$GPU_ID"
export PYOPENGL_PLATFORM=egl
export DISPLAY=""

echo "[env_config] Host: $HOSTNAME → GPU_ID=$GPU_ID, Conda=$CONDA_PATH, Env=$MAMMAL_ENV"
