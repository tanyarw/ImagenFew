#!/bin/bash
#SBATCH --job-name=gd_pilot
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100-40:1
#SBATCH --array=0-19

# One guidance pilot setting per array task (guidance/pilot.py TASKS; `python -m guidance tasks` lists them).
# Task 0 runs the unit tests first (they need torch, which login nodes cannot load) and stops the whole
# pilot if one fails. Sent by guidance/slurm/submit_pilot.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
echo "job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} | $(hostname -s) | $(date) | commit $(git rev-parse HEAD)"
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    python -m unittest discover -s guidance/tests -t .
fi
python -m guidance pilot-task --index "${SLURM_ARRAY_TASK_ID}" --device cuda
echo "done $(date)"
