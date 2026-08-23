#!/bin/bash
#SBATCH --job-name=imagen_train        # Job name in squeue
#SBATCH --output=logs/%x_%j.out        # Standard output and error log (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x_%j.err         # Separate error log (optional)
#SBATCH --time=12:00:00                # Max runtime (HH:MM:SS) — request realistic time for faster queueing
#SBATCH --cpus-per-task=8              # CPU cores for data loaders
#SBATCH --mem=32G                      # RAM required
#SBATCH --partition=gpu                # Partition/queue name
#SBATCH --gres=gpu:1                   # Request 1 GPU (or specify type, see suggestions below)

# ----------------------------------------------------
# 1. Prepare Environment
# ----------------------------------------------------
mkdir -p logs                          # Ensure logs folder exists

# Load modules (if required by your cluster, e.g., module load cuda/12.1)
# Activate virtual environment
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

# Print job details for debugging
echo "========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Running on:    $(hostname -s)"
echo "Assigned GPU:  $CUDA_VISIBLE_DEVICES"
echo "Start Time:    $(date)"
echo "========================================="

# ----------------------------------------------------
# 2. Run Your Script
# ----------------------------------------------------
# Example A: Pretraining
# python run.py --no_test_model --seq_len 24 --ddp --config configs/pretrain/pretrain.yaml

# Example B: Fine-tuning
python run.py --subset_p 100 --model_ckpt models_ckpt/pretrain_24.pth --config configs/finetune/Weather.yaml

echo "Job completed at: $(date)"
