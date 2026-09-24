#!/bin/bash
#SBATCH --job-name=v13
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --nodelist=xgpg[0-7]
# ==============================================================================
# v13: train -> generate -> denoising-error diagnostic, in one job.
# Same as submit_job_v12.sh + submit_generation_v12.sh, but with
# config_v13.yaml (asinh transform) and its own run_id / output file,
# so nothing from v12 is overwritten.
#
#   sbatch scripts/submit_v13.sh
# ==============================================================================

set -euo pipefail

RUN_ID="v13"
CONFIG="regime_training/config_v13.yaml"
RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN_ID}"
V12_DIR="logs/ImagenFew/Rainfall_Regime/v12"          # the "before" model

mkdir -p logs results/generated_data results/transform_diagnostics
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s) | Start: $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

# ── 1. Train (500 epochs, same base checkpoint as v12) ───────────────
python regime_training/train_regime.py \
    --model_ckpt models_ckpt/ImagenFew/ImagenFew_64.ckpt \
    --config "${CONFIG}" \
    --epochs 500 --batch_size 2048 --learning_rate 0.0001 \
    --run_id "${RUN_ID}"

# ── 2. Generate 10 years (same settings as run_v12_generation.sh) ───
# The pickled AsinhScaler does the inverse transform — no generation code changes.
python regime_training/generate_hmm_v1.py \
    --model_ckpt "${RUN_DIR}/best_regime_model.pt" \
    --scaler_path "${RUN_DIR}/scaler.pkl" \
    --config "${CONFIG}" \
    --assembly_mode "unconditional" \
    --years 10 --freq "5min" --overlap 4 --seed 42 \
    --output_dir results/generated_data \
    --output_name "rainfall_synthetic_10y_${RUN_ID}.csv"

# ── 3. Before/after denoising error by rain intensity (in mm) ────────
python scripts/diagnose_denoising_error.py \
    --run v12="${V12_DIR}" \
    --run v13="${RUN_DIR}" \
    --config "${CONFIG}" \
    --out_dir results/transform_diagnostics

# Same again on the 2000-2007 split: more storms, so a steadier read on the tail
python scripts/diagnose_denoising_error.py \
    --run v12="${V12_DIR}" \
    --run v13="${RUN_DIR}" \
    --config "${CONFIG}" \
    --split train \
    --out_dir results/transform_diagnostics

echo "Job completed at: $(date)"
