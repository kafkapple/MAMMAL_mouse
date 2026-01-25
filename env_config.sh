#!/bin/bash
# ===== Environment-specific Configuration =====
# Auto-detect server and set appropriate GPU/paths
#
# Usage: source env_config.sh (automatically sourced by run scripts)
# Note: If CUDA_VISIBLE_DEVICES is already set externally, it takes precedence

HOSTNAME=$(hostname)

# Preserve externally set CUDA_VISIBLE_DEVICES
EXTERNAL_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

case "$HOSTNAME" in
    gpu05|gpu05.*)
        # gpu05 (joon) 서버: GPU 1 사용
        export GPU_ID="${GPU_ID:-1}"
        export CONDA_PATH="$HOME/miniconda3"
        export MAMMAL_ENV="mammal_stable"
        ;;
    gpu03|gpu03.*)
        # gpu03 서버: GPU 4-7 사용 (A6000, sm_86)
        # GPU 0-3은 Blackwell (sm_120) - PyTorch 미지원
        export GPU_ID="${GPU_ID:-4}"
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

# Use external CUDA_VISIBLE_DEVICES if set, otherwise use GPU_ID
if [ -n "$EXTERNAL_CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$EXTERNAL_CUDA_DEVICES"
    export GPU_ID="$EXTERNAL_CUDA_DEVICES"
else
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

export EGL_DEVICE_ID="$GPU_ID"
export PYOPENGL_PLATFORM=egl
export DISPLAY=""

echo "[env_config] Host: $HOSTNAME → GPU_ID=$GPU_ID, Conda=$CONDA_PATH, Env=$MAMMAL_ENV"
