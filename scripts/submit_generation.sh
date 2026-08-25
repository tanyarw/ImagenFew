#!/bin/bash
#SBATCH --job-name=generate_synthetic
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ── Configurable Variables (with defaults or CLI arguments) ────────
RUN_ID_105120="${1:-3559f298}"
RUN_ID_365="${2:-045d9e20}"
YEARS="${3:-10}"
FREQ="${4:-5min}"

mkdir -p logs
mkdir -p results/generated_data
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s)"
echo "105k Run ID:   $RUN_ID_105120"
echo "365  Run ID:   $RUN_ID_365"
echo "Years:         $YEARS | Freq: $FREQ"
echo "Start Time:    $(date)"
echo "=========================================================="

# ───────────────────────────────────────────────────────────────────
# 1. Generate for 105120-fit variant
# ───────────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/2] Generating ${YEARS}-year synthetic data for 105120-fit variant (Run ID: ${RUN_ID_105120})..."
python regime_training/generate_hmm_v1.py \
    --model_ckpt "logs/ImagenFew/Rainfall_Regime/${RUN_ID_105120}/best_regime_model.pt" \
    --scaler_path "logs/ImagenFew/Rainfall_Regime/${RUN_ID_105120}/scaler.pkl" \
    --years "${YEARS}" \
    --freq "${FREQ}" \
    --hmm_variant 105120 \
    --output_dir results/generated_data

# ───────────────────────────────────────────────────────────────────
# 2. Generate for 365-fit variant
# ───────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/2] Generating ${YEARS}-year synthetic data for 365-fit variant (Run ID: ${RUN_ID_365})..."
python regime_training/generate_hmm_v1.py \
    --model_ckpt "logs/ImagenFew/Rainfall_Regime/${RUN_ID_365}/best_regime_model.pt" \
    --scaler_path "logs/ImagenFew/Rainfall_Regime/${RUN_ID_365}/scaler.pkl" \
    --years "${YEARS}" \
    --freq "${FREQ}" \
    --hmm_variant 365 \
    --output_dir results/generated_data

echo ""
echo "=========================================================="
echo "Generation complete! Output files in results/generated_data/:"
ls -lh results/generated_data/
echo "End Time: $(date)"
echo "=========================================================="
