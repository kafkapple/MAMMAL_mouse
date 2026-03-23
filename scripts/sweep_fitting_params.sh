#!/bin/bash
# Sweep fitting parameters on worst frames
# Usage: bash scripts/sweep_fitting_params.sh [GPU_ID]

set -e
cd "$(dirname "$0")/.."
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate mouse

GPU_ID=${1:-5}
echo "=== Fitting Parameter Sweep on GPU $GPU_ID ==="
echo "Start: $(date)"
export CUDA_VISIBLE_DEVICES=$GPU_ID

WORST="9480,9360,5520,1320,8400"

run_config() {
    local S1=$1 M=$2
    local NAME="s1_${S1}_m_${M}"
    local OUT="results/fitting/sweep_${NAME}"
    echo "--- ${NAME} (step1=${S1}, mask=${M}) ---"
    python fitter_articulation.py \
        experiment=accurate_6views \
        result_folder="$OUT" \
        "+fitter.frame_list=[$WORST]" \
        fitter.with_render=false \
        fitter.generate_visualizations=false \
        optim.solve_step1_iters=$S1 \
        loss_weights.mask_step2=$M 2>&1 | tail -3
    echo "  OBJs: $(ls "$OUT/obj/"*.obj 2>/dev/null | wc -l)/5"
}

run_config 100 1000
run_config 100 3000
run_config 100 5000
run_config 200 1000
run_config 200 3000
run_config 200 5000
run_config 400 1000
run_config 400 3000
run_config 400 5000

echo ""
echo "=== Sweep Complete: $(date) ==="
