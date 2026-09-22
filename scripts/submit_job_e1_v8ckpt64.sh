#!/bin/bash
#SBATCH --job-name=e1_v8ckpt64_finetune
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

# E1 Control Run B: seq_len=24 (v8's config), initialised from ImagenFew_64.ckpt
# instead of ImagenFew_24.ckpt. See code_plan/AUDIT_2026-09-22.md §3.1 / §6 E1.
bash scripts/run_e1_v8ckpt64_training.sh 500

echo "Job completed at: $(date)"
