#!/bin/bash
#SBATCH --job-name=gd_ctx2h
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100-40:1

# The chosen pilot setting rerun with 2 hours of context (3 columns) and compared with 80 minutes
# -> results/guidance/context2h.json. Exits quietly if no setting passed the pilot.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
python -m guidance pilot-context2h --device cuda
