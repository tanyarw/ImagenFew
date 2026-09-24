#!/bin/bash
#SBATCH --job-name=v11_len64_uncond_finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --nodelist=xgpg[0-7]

mkdir -p logs
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s) | Start: $(date)"

# Fast CUDA sanity check to fail immediately if driver is unhealthy
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

# Train clean v11 unconditional ablation (seq_len=64) for 500 epochs with run_id 'v11'
bash scripts/run_v11_training.sh 500 v11

echo "Job completed at: $(date)"
