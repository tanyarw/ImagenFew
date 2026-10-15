#!/bin/bash
#SBATCH --job-name=v19_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# v19 before training (code_plan/ONE_DAY_WINDOW_PLAN.md, section 3): speed and memory of the 24x24 network at
# batch 2048 and 1024, sampling speed, and this partition's time limit -> results/oneday/smoke.json.
# About 10 minutes. The epoch count and batch size are then set by the plan's rule.
# ==============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs
echo "job ${SLURM_JOB_ID:-} | $(hostname -s) | GPU ${CUDA_VISIBLE_DEVICES:-} | $(date)"
echo "commit $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(UNCOMMITTED CHANGES)')"
python -m unittest discover -s oneday/tests -t .
python -m oneday check
python -m oneday smoke
echo "done $(date)"
