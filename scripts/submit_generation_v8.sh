#!/bin/bash
#SBATCH --job-name=gen_v8_holdout
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

RUN_ID="${1:-ed17d299}"
YEARS="${2:-10}"
SEED="${3:-42}"
MODE="${4:-both}"

mkdir -p logs
mkdir -p results/generated_data
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s)"
echo "Clean v8 Synthetic Generation (Holdout Split 2000–2007)"
echo "Run ID:        $RUN_ID"
echo "Years:         $YEARS | Freq: 5min | Seed: $SEED | Mode: $MODE"
echo "Start Time:    $(date)"
echo "=========================================================="

bash scripts/run_v8_generation.sh "$RUN_ID" "$YEARS" "$SEED" "$MODE"

echo ""
echo "=========================================================="
echo "Generation complete! Output files in results/generated_data/:"
ls -lh results/generated_data/rainfall_synthetic_10y_v8*.csv
echo "End Time: $(date)"
echo "=========================================================="
