#!/bin/bash
#SBATCH --job-name=denoise_diag
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# ==============================================================================
# Denoising error by true rain band, v10 vs v13 vs v14 (scripts/diagnose_denoising_error.py).
# The held-out split has only 7 extreme steps, so the training split (40 extreme steps,
# in-sample) is the steadier read on the tail. It did not finish on a laptop CPU.
#   -> results/transform_diagnostics/denoise_<split>/
#
#   sbatch scripts/submit_denoise_diagnostic.sh              # train and test splits
#   sbatch scripts/submit_denoise_diagnostic.sh train        # one split
# ==============================================================================

set -euo pipefail

SPLITS="${*:-train test}"

mkdir -p logs
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

for SPLIT in ${SPLITS}; do
    python scripts/diagnose_denoising_error.py \
        --run v10=logs/ImagenFew/Rainfall_Regime/aebe363f \
        --run v13=logs/ImagenFew/Rainfall_Regime/v13 \
        --run v14=logs/ImagenFew/Rainfall_Regime/v14 \
        --config regime_training/config_v14.yaml \
        --split "${SPLIT}" \
        --out_dir results/transform_diagnostics
done

echo "Job completed at: $(date)"
