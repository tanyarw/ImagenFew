#!/bin/bash
#SBATCH --job-name=gen_105120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ── Configurable Variables (with defaults or CLI arguments) ────────
RUN_ID="${1:-1bff544a}"
YEARS="${2:-10}"
FREQ="${3:-5min}"

mkdir -p logs
mkdir -p results/generated_data
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s)"
echo "Variant:       105120 (5-min Interval HMM v1, Retrained 500 epochs)"
echo "Run ID:        $RUN_ID"
echo "Years:         $YEARS | Freq: $FREQ"
echo "Start Time:    $(date)"
echo "=========================================================="

python regime_training/generate_hmm_v1.py \
    --model_ckpt "logs/ImagenFew/Rainfall_Regime/${RUN_ID}/best_regime_model.pt" \
    --scaler_path "logs/ImagenFew/Rainfall_Regime/${RUN_ID}/scaler.pkl" \
    --years "${YEARS}" \
    --freq "${FREQ}" \
    --hmm_variant 105120 \
    --output_name "rainfall_synthetic_10y_v5.csv" \
    --output_dir results/generated_data

echo ""
echo "=========================================================="
echo "Generation complete! Output files in results/generated_data/:"
ls -lh results/generated_data/
echo "End Time: $(date)"
echo "=========================================================="
