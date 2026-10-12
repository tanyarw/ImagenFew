#!/bin/bash
#SBATCH --job-name=st_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=normal

# ==============================================================================
# Transfer site: score every series (now including the trained model's) and write the verdict.
# Runs after train_generate.sbatch (submit.sh chains it). Argument: SITE.
# The pilot is regenerated here because its .npy arrays are not in git (same seeds, same result).
#   results/transfer/<SITE>/evaluation.json, results/transfer/verdict.md, verdict.json
# ==============================================================================
set -euo pipefail
SITE="${1:?usage: evaluate.sbatch SITE}"
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
echo "job ${SLURM_JOB_ID:-} | $(hostname -s) | $(date) | commit $(git rev-parse HEAD)"

python -m sitetransfer pilot --site "${SITE}"
python -m sitetransfer evaluate --site "${SITE}"
python -m sitetransfer pilot --site astlingen
python -m sitetransfer evaluate --site astlingen
python -m sitetransfer verdict --site "${SITE}"
cat results/transfer/verdict.md
echo "done $(date)"
