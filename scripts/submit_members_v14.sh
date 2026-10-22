#!/bin/bash
#SBATCH --job-name=members_v14
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --array=0-4

# ==============================================================================
# Members 1-3 for the five v14 runs (diary 2026-10-22, "Equal Sampling"), so each has 4 x 10 years like v19.
#
# Member 0 of each run exists: its calendar series from final_regime_model.pt with plan seed 42
# (scripts/submit_regenerate_final.sh for v14, v14_s1, v14_s2, v14_scratch; scripts/submit_retrain.sh for
# v14_scratch_s1). This repeats that command unchanged with seeds 43, 44, 45, the member seeds of
# oneday/slurm/train.sbatch (member K: seed 42 + K). Outputs <BASE>_m1 .. _m3; an existing file is never
# overwritten.
#
#   sbatch scripts/submit_members_v14.sh                  # all five runs
#   bash scripts/submit_members_v14.sh list               # the table
#   SMOKE=1 bash scripts/submit_members_v14.sh all        # laptop: every entry, CPU, ~1 day of rain, to $SMOKE_DIR
# ==============================================================================

set -euo pipefail

# BASE (member 0's series name) | RUN_DIR (under logs/ImagenFew/Rainfall_Regime)
TABLE=(
    "v14_final_nobridge_cal|v14"
    "v14_s1_final_nobridge_cal|v14_s1"
    "v14_s2_final_nobridge_cal|v14_s2"
    "v14_scratch_final_nobridge_cal|v14_scratch"
    "v14_scratch_s1_nobridge_cal|v14_scratch_s1"
)
CONFIG=regime_training/config_v14.yaml
TRANS=data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl

if [ "${1:-}" = "list" ]; then
    for i in "${!TABLE[@]}"; do echo "${i}  ${TABLE[$i]}"; done
    exit 0
fi

cd "${SLURM_SUBMIT_DIR:-.}"
if [ "${SMOKE:-0}" = "1" ]; then
    PY=.venv/bin/python; DEVICE=cpu; YEARS=0.003; OUT_DIR="${SMOKE_DIR:-/tmp/members_v14_smoke}"
else
    source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate
    PY=python; DEVICE=cuda; YEARS=10; OUT_DIR=results/generated_data
    mkdir -p logs
    echo "Job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
    echo "commit: $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(uncommitted changes!)')"
    python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"
fi
mkdir -p "${OUT_DIR}"

run_entry() {
    IFS='|' read -r BASE RUN <<< "$1"
    local RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN}"
    local WEIGHTS="${RUN_DIR}/final_regime_model.pt"
    for f in "${CONFIG}" "${TRANS}" "${WEIGHTS}" "${RUN_DIR}/scaler.pkl"; do
        [ -f "${f}" ] || { echo "ERROR: ${f} not found"; return 1; }
    done
    # Check every output first: nothing is generated if any member exists
    for K in 1 2 3; do
        local OUT="${OUT_DIR}/rainfall_synthetic_10y_${BASE}_m${K}.csv"
        [ ! -e "${OUT}" ] || { echo "ERROR: ${OUT} exists; not overwritten"; return 1; }
    done
    "${PY}" -c "import torch,hashlib,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); \
print('weights', sys.argv[1], 'epoch', c['epoch'], 'sha256', hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])" "${WEIGHTS}"
    for K in 1 2 3; do
        local NAME="rainfall_synthetic_10y_${BASE}_m${K}.csv"
        echo "── ${BASE} member ${K} (seed $((42 + K))) -> ${OUT_DIR}/${NAME}"
        "${PY}" regime_training/generate_hmm_v1.py \
            --model_ckpt "${WEIGHTS}" \
            --scaler_path "${RUN_DIR}/scaler.pkl" \
            --config "${CONFIG}" \
            --transition_matrix_path "${TRANS}" --assembly_mode calendar --transition_overlap 8 --bridge_blocks 0 \
            --years "${YEARS}" --freq "5min" --overlap 4 \
            --seed $((42 + K)) --device "${DEVICE}" \
            --output_dir "${OUT_DIR}" \
            --output_name "${NAME}"
    done
}

if [ "${1:-}" = "all" ]; then
    for e in "${TABLE[@]}"; do run_entry "${e}"; done
else
    IDX="${SLURM_ARRAY_TASK_ID:-${1:?give an index (see: list), 'all', or run as an sbatch array}}"
    run_entry "${TABLE[$IDX]}"
fi
echo "Done at: $(date)"
