#!/bin/bash
#SBATCH --job-name=base_metrics
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# ==============================================================================
# The ImagenFew paper's own metrics (discriminative score, predictive score, context-FID) on
# every generated version, for the metric-validity analysis (scripts/metric_validity.py).
# Reads results/evaluation/base_paper_windows.npz (made on the laptop by
# `python scripts/base_paper_metrics.py pack`, committed), so the 10-year CSVs are not needed.
# Saves after every series and skips finished ones, so resubmitting continues a stopped job.
#   -> results/evaluation/base_paper_metrics.json  (copy back to the laptop)
#      logs/TS2VEC/Rainfall_3000_64_1_ref2000-2007.ckpt  (the context-FID encoder)
#
#   sbatch scripts/submit_base_paper_metrics.sh              # all series, 10 repeats
#   sbatch scripts/submit_base_paper_metrics.sh 3            # 3 repeats (quicker)
# ==============================================================================

set -euo pipefail

ITERS="${1:-10}"

mkdir -p logs
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"
test -f results/evaluation/base_paper_windows.npz || { echo 'ERROR: base_paper_windows.npz missing (git pull?)'; exit 1; }

python scripts/base_paper_metrics.py score --device cuda --iters "${ITERS}"

echo "Job completed at: $(date)"
