#!/bin/bash
#SBATCH --job-name=gen_v17_ctx
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# E3 (strengthening run 5, code_plan/STRENGTHENING_RUNS.md): chained generation from v17,
# the model trained with known prefixes (config_v17.yaml). Each block after the first is
# given the previous block's last K*8 steps as clean known pixels (--known_pixels clean,
# exactly as v17 saw them in training; no RePaint resampling, no history noise).
#
#   sbatch scripts/submit_generation_v17_ctx.sh          # K = 2 (80 min of context), the E3 run
#   sbatch scripts/submit_generation_v17_ctx.sh 0        # K = 0: same model and code, blocks
#                                                        # independent (the no-context control)
# Outputs (calendar timeline, 4 independent 10-year chains each):
#   K=2 -> results/generated_data/rainfall_synthetic_10y_v17_ctx_cal.csv   (+ _m1.._m3)
#   K=0 -> results/generated_data/rainfall_synthetic_10y_v17_ctx0_cal.csv  (+ _m1.._m3)
#
# Calendar, because that is the timeline the deliverable uses (v14_nobridge_cal).
# About 1.5 s per block on an A100 (E2 needed 6-8 s with 10 resampling passes): ~9 h for
# K = 2 (21,900 blocks), ~7 h for K = 0. Progress is saved every 500 blocks: if the job hits
# the time limit, submit the same command again and it resumes.
# Needs the trained v17 (scripts/submit_strengthening.sh chains this after training).
# ==============================================================================

set -euo pipefail

K="${1:-2}"
MEMBERS="${MEMBERS:-4}"
YEARS="${YEARS:-10}"
NAME="rainfall_synthetic_10y_v17_ctx"
[ "${K}" = "2" ] || NAME="${NAME}${K}"
[ "${YEARS}" = "10" ] || NAME="${NAME}_${YEARS}y"
NAME="${NAME}_cal"
RUN_DIR="logs/ImagenFew/Rainfall_Regime/v17"

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results/generated_data
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
echo "K=${K} members=${MEMBERS} years=${YEARS} -> ${NAME}"
echo "commit: $(git rev-parse HEAD)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

for f in "${RUN_DIR}/best_regime_model.pt" "${RUN_DIR}/scaler.pkl" regime_training/config_v17.yaml; do
    [ -f "${f}" ] || { echo "ERROR: ${f} not found (train v17 first: scripts/submit_strengthening.sh)"; exit 1; }
done
grep -q "known_pixels" regime_training/generate_context.py \
    || { echo "ERROR: generate_context.py lacks --known_pixels: git pull"; exit 1; }

python regime_training/generate_context.py \
    --run v17 --mode calendar --years "${YEARS}" --seed 42 \
    --members "${MEMBERS}" --context_cols "${K}" \
    --known_pixels clean --resample 1 --history_noise 0 \
    --device cuda --output_name "${NAME}"

ls -lh results/generated_data/${NAME}*.csv
echo "Job completed at: $(date)"
