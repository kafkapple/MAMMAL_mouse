#!/bin/bash
# ===== Environment-specific Configuration =====
# Auto-detect server and set appropriate GPU/paths
#
# Usage: source env_config.sh (automatically sourced by run scripts)

HOSTNAME=$(hostname)

case "$HOSTNAME" in
    gpu05|gpu05.*)
        # gpu05 서버: GPU 1 사용 (GPU 0은 다른 용도)
        export GPU_ID="${GPU_ID:-1}"
        ;;
    bori|bori.*)
        # bori 서버: GPU 0 사용 (GPU 1개만 있음)
        export GPU_ID="${GPU_ID:-0}"
        ;;
    *)
        # 기타 서버: 기본 GPU 0
        export GPU_ID="${GPU_ID:-0}"
        ;;
esac

# Common GPU settings
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EGL_DEVICE_ID="$GPU_ID"
export PYOPENGL_PLATFORM=egl
export DISPLAY=""

echo "[env_config] Host: $HOSTNAME → GPU_ID=$GPU_ID"
