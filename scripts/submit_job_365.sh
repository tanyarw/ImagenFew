#!/bin/bash
#SBATCH --job-name=hmm_365_finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

mkdir -p logs
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s) | Start: $(date)"

# Run full pipeline for the 365-day HMM variant
bash scripts/run_hmm_training.sh 365

echo "Job completed at: $(date)"
