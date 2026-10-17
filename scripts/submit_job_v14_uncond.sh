#!/bin/bash
#SBATCH --job-name=v14_uncond
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# v14_uncond (v14 without the seasonal-state label): train -> generate, in one job.
# Training is submit_job_v14.sh with config_v14_uncond.yaml; generation is
# run_v12_generation.sh (unconditional assembly, 4-step crossfade, no bridges),
# so v14 vs v14_uncond is the same comparison as v10 vs v12.
#
#   sbatch scripts/submit_job_v14_uncond.sh
# ==============================================================================

set -euo pipefail

RUN_ID="v14_uncond"
CONFIG="regime_training/config_${RUN_ID}.yaml"
RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN_ID}"
OUTPUT_DIR="results/generated_data"

mkdir -p logs "${OUTPUT_DIR}"
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s) | Start: $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

# ── 1. Train (as v14: ImagenFew_64.ckpt, 500 epochs, bs 2048, lr 1e-4) ─
python regime_training/train_regime.py \
    --model_ckpt models_ckpt/ImagenFew/ImagenFew_64.ckpt \
    --config "${CONFIG}" \
    --epochs 500 --batch_size 2048 --learning_rate 0.0001 \
    --run_id "${RUN_ID}"

# ── 2. Generate 10 years (same flags as run_v12_generation.sh) ───────
python regime_training/generate_hmm_v1.py \
    --model_ckpt "${RUN_DIR}/best_regime_model.pt" \
    --scaler_path "${RUN_DIR}/scaler.pkl" \
    --config "${CONFIG}" \
    --assembly_mode unconditional \
    --years 10 --freq "5min" --overlap 4 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "rainfall_synthetic_10y_${RUN_ID}.csv"

ls -lh "${OUTPUT_DIR}/rainfall_synthetic_10y_${RUN_ID}.csv"
echo "Job completed at: $(date)"
