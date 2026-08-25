#!/bin/bash
#SBATCH --job-name=hmm_105120_v2_finetune
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

# Train for 500 epochs (~12.5h on A100, fully converged by epoch 300)
# (Pass 1001 if you increase --time to 28:00:00)
bash scripts/run_hmm_training.sh 105120_v2 500

echo "Job completed at: $(date)"
