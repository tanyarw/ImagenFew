#!/bin/bash
#SBATCH --job-name=v13_log1p
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# v13 (log1p transform): train -> generate, in one job.
# Same settings as v10 (run_v10_training.sh + run_v10_generation.sh), but with
# config_v13.yaml and its own run_id / output files, so nothing from v10 is
# overwritten. The ONLY difference from v10 is the rainfall transform.
#
#   sbatch scripts/submit_v13.sh
# ==============================================================================

set -euo pipefail

RUN_ID="v13"
CONFIG="regime_training/config_${RUN_ID}.yaml"
RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN_ID}"
TRANS_MATRIX="data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl"
OUTPUT_DIR="results/generated_data"

mkdir -p logs "${OUTPUT_DIR}"
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s) | Start: $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

# ── 1. Train (as v10: ImagenFew_64.ckpt, 500 epochs, bs 2048, lr 1e-4) ─
python regime_training/train_regime.py \
    --model_ckpt models_ckpt/ImagenFew/ImagenFew_64.ckpt \
    --config "${CONFIG}" \
    --epochs 500 --batch_size 2048 --learning_rate 0.0001 \
    --run_id "${RUN_ID}"

# ── 2. Generate 10 years (same flags as run_v10_generation.sh) ───────
# markov -> rainfall_synthetic_10y_v13.csv, calendar -> ..._v13_cal.csv.
# The pickled scaler does the inverse transform — no generation code changes.
for MODE in markov calendar; do
    if [ "${MODE}" = "calendar" ]; then SUFFIX="_cal"; else SUFFIX=""; fi
    python regime_training/generate_hmm_v1.py \
        --model_ckpt "${RUN_DIR}/best_regime_model.pt" \
        --scaler_path "${RUN_DIR}/scaler.pkl" \
        --config "${CONFIG}" \
        --transition_matrix_path "${TRANS_MATRIX}" \
        --assembly_mode "${MODE}" \
        --years 10 --freq "5min" --overlap 4 \
        --transition_overlap 8 --bridge_blocks 1 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}" \
        --output_name "rainfall_synthetic_10y_${RUN_ID}${SUFFIX}.csv"
done

ls -lh "${OUTPUT_DIR}"/rainfall_synthetic_10y_${RUN_ID}*.csv
echo "Job completed at: $(date)"
