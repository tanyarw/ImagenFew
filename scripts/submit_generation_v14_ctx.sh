#!/bin/bash
#SBATCH --job-name=gen_v14_ctx
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# ==============================================================================
# E2: context-chained generation from the v14 checkpoint (no retraining).
# Each 64-step block is generated with the previous block's last K*8 steps as known
# context (RePaint + noisy history); see regime_training/generate_context.py.
#   markov   -> results/generated_data/rainfall_synthetic_10y_v14_ctx.csv      (+ _m1.._m3)
#   calendar -> results/generated_data/rainfall_synthetic_10y_v14_ctx_cal.csv  (+ _m1.._m3)
#
#   sbatch scripts/submit_generation_v14_ctx.sh markov
#   sbatch scripts/submit_generation_v14_ctx.sh calendar
#
# Optional environment settings (defaults = the pre-registered E2 run):
#   K=2 (context columns, 40 min each)  RESAMPLE=10  MEMBERS=4  YEARS=10
#   K=0 gives independent blocks through the same code (no-context control).
#   Non-default settings get their own file name, e.g. K=1 -> ..._v14_ctx_k1.csv
#
# The log prints s/block and an ETA every 100 blocks. Progress is saved every 500 blocks:
# if the job hits the time limit, submit the same command again and it resumes.
# ==============================================================================

set -euo pipefail

MODE="${1:-markov}"
K="${K:-2}"
RESAMPLE="${RESAMPLE:-10}"
MEMBERS="${MEMBERS:-4}"
YEARS="${YEARS:-10}"
NAME="rainfall_synthetic_10y_v14_ctx"
[ "${K}" = "2" ] || NAME="${NAME}_k${K}"
[ "${RESAMPLE}" = "10" ] || NAME="${NAME}_u${RESAMPLE}"
[ "${YEARS}" = "10" ] || NAME="${NAME}_${YEARS}y"
if [ "${MODE}" = "calendar" ]; then NAME="${NAME}_cal"; fi

mkdir -p logs results/generated_data
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
echo "mode=${MODE} K=${K} resample=${RESAMPLE} members=${MEMBERS} years=${YEARS} -> ${NAME}"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

for f in logs/ImagenFew/Rainfall_Regime/v14/best_regime_model.pt logs/ImagenFew/Rainfall_Regime/v14/scaler.pkl \
         regime_training/generate_context.py; do
    [ -f "${f}" ] || { echo "ERROR: ${f} not found (run git pull?)"; exit 1; }
done

python regime_training/generate_context.py \
    --run v14 --mode "${MODE}" --years "${YEARS}" --seed 42 \
    --members "${MEMBERS}" --context_cols "${K}" --resample "${RESAMPLE}" \
    --device cuda --output_name "${NAME}"

ls -lh results/generated_data/${NAME}*.csv
echo "Job completed at: $(date)"
