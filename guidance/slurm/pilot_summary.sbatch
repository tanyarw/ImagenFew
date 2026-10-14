#!/bin/bash
#SBATCH --job-name=gd_summary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Collect the 20 pilot tasks, apply the pass rules, choose a setting -> results/guidance/pilot.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
python -m guidance pilot-summary
