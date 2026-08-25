#!/bin/bash
#SBATCH --job-name=gen_365
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ── Configurable Variables (with defaults or CLI arguments) ────────
RUN_ID="${1:-045d9e20}"
YEARS="${2:-10}"
FREQ="${3:-5min}"

mkdir -p logs
mkdir -p results/generated_data
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s)"
echo "Variant:       365 (Day-of-Year HMM)"
echo "Run ID:        $RUN_ID"
echo "Years:         $YEARS | Freq: $FREQ"
echo "Start Time:    $(date)"
echo "=========================================================="

python regime_training/generate_hmm_v1.py \
    --model_ckpt "logs/ImagenFew/Rainfall_Regime/${RUN_ID}/best_regime_model.pt" \
    --scaler_path "logs/ImagenFew/Rainfall_Regime/${RUN_ID}/scaler.pkl" \
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
