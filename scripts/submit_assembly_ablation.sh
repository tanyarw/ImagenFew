#!/bin/bash
#SBATCH --job-name=assembly_ablation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
# ==============================================================================
# Block-assembly ablation (bridging x crossfade), same blocks joined four ways.
# The 2-year CPU run is in the diary (Sept 30); this is the full 10-year version for
# the thesis table. -> results/assembly_ablation/assembly_<run>_<years>y_seed42.csv
#
#   sbatch scripts/submit_assembly_ablation.sh            # v14, 10 years
#   sbatch scripts/submit_assembly_ablation.sh v13 10
# ==============================================================================

set -euo pipefail

RUN="${1:-v14}"
YEARS="${2:-10}"

mkdir -p logs
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

python scripts/ablate_block_assembly.py --run "${RUN}" --years "${YEARS}" --device cuda

echo "Job completed at: $(date)"
