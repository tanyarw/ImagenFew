#!/bin/bash
#SBATCH --job-name=gen_v14
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# ==============================================================================
# v14 (asinh transform): generation only, from the trained v14 checkpoint.
# Same flags as run_v10_generation.sh, both assemblies:
#   markov   -> results/generated_data/rainfall_synthetic_10y_v14.csv
#   calendar -> results/generated_data/rainfall_synthetic_10y_v14_cal.csv
#
#   sbatch scripts/submit_generation_v14.sh          # seed 42 (the main run)
#   sbatch scripts/submit_generation_v14.sh 7        # extra seed -> ..._v14_seed7.csv
#
# Optional environment settings (defaults reproduce the main run):
#   MODES="markov calendar"   which assemblies to generate
#   BRIDGE_BLOCKS=1           0 = no bridge blocks (removes the calendar drift) -> ..._v14_nobridge*.csv
#   e.g.  MODES=calendar BRIDGE_BLOCKS=0 sbatch scripts/submit_generation_v14.sh
#         -> results/generated_data/rainfall_synthetic_10y_v14_nobridge_cal.csv
# ==============================================================================

set -euo pipefail

RUN_ID="v14"
SEED="${1:-42}"
CONFIG="regime_training/config_${RUN_ID}.yaml"
RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN_ID}"
TRANS_MATRIX="data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl"
OUTPUT_DIR="results/generated_data"
TAG="${RUN_ID}"; [ "${SEED}" = "42" ] || TAG="${RUN_ID}_seed${SEED}"
MODES="${MODES:-markov calendar}"
BRIDGE_BLOCKS="${BRIDGE_BLOCKS:-1}"
[ "${BRIDGE_BLOCKS}" = "1" ] || TAG="${TAG}_nobridge"

mkdir -p logs "${OUTPUT_DIR}"
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Seed: ${SEED} | Start: $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

for f in "${RUN_DIR}/best_regime_model.pt" "${RUN_DIR}/scaler.pkl" "${TRANS_MATRIX}"; do
    [ -f "${f}" ] || { echo "ERROR: ${f} not found"; exit 1; }
done
# Blocks must be converted to mm before stitching (commit 136f723), or log1p/asinh lose rain
grep -q "Inverse-scaling blocks" regime_training/generate_hmm_v1.py \
    || { echo "ERROR: generate_hmm_v1.py lacks the stitch-in-mm fix, run git pull"; exit 1; }

for MODE in ${MODES}; do
    if [ "${MODE}" = "calendar" ]; then SUFFIX="_cal"; else SUFFIX=""; fi
    python regime_training/generate_hmm_v1.py \
        --model_ckpt "${RUN_DIR}/best_regime_model.pt" \
        --scaler_path "${RUN_DIR}/scaler.pkl" \
        --config "${CONFIG}" \
        --transition_matrix_path "${TRANS_MATRIX}" \
        --assembly_mode "${MODE}" \
        --years 10 --freq "5min" --overlap 4 \
        --transition_overlap 8 --bridge_blocks "${BRIDGE_BLOCKS}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --output_name "rainfall_synthetic_10y_${TAG}${SUFFIX}.csv"
done

ls -lh "${OUTPUT_DIR}"/rainfall_synthetic_10y_${TAG}*.csv
echo "Job completed at: $(date)"
