#!/bin/bash
#SBATCH --job-name=v15_attn8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# v15 (E1, attention at 8x8): fine-tune FROM THE v14 CHECKPOINT -> generate, in one job.
# Same data, transform, loss and generation flags as v14. The only differences are
# config_v15.yaml's attn_min_heads: 1 and the starting weights (v14's).
# Compare against v15_ctrl (same fine-tune, no attention change), not against v14.
#
#   sbatch scripts/submit_job_v15.sh
# ==============================================================================

set -euo pipefail

RUN_ID="v15"
EPOCHS=200
INIT_CKPT="logs/ImagenFew/Rainfall_Regime/v14/best_regime_model.pt"
CONFIG="regime_training/config_${RUN_ID}.yaml"
RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN_ID}"
TRANS_MATRIX="data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl"
OUTPUT_DIR="results/generated_data"

mkdir -p logs "${OUTPUT_DIR}"
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

[ -f "${INIT_CKPT}" ] || { echo "ERROR: v14 checkpoint ${INIT_CKPT} not found"; exit 1; }
grep -q "attn_min_heads" models/ImagenFew/networks.py \
    || { echo "ERROR: networks.py has no attn_min_heads option, run git pull"; exit 1; }

# ── 1. Fine-tune from v14 (bs 2048, lr 1e-4 as v14) ──────────────────
python regime_training/train_regime.py \
    --model_ckpt "${INIT_CKPT}" \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" --batch_size 2048 --learning_rate 0.0001 \
    --run_id "${RUN_ID}"

# ── 2. Generate 10 years (same flags as the v14 job) ─────────────────
# markov -> rainfall_synthetic_10y_v15.csv, calendar -> ..._v15_cal.csv
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
