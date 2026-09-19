#!/bin/bash
#SBATCH --job-name=v8_holdout_finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

mkdir -p logs
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s) | Start: $(date)"

# Train clean v8 for 500 epochs (~12.5h on A100)
bash scripts/run_v8_training.sh 500

echo "Job completed at: $(date)"
