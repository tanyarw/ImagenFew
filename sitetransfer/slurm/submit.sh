#!/bin/bash
# ==============================================================================
# Send the transfer run to SLURM from the repository root on the login node:
#   sitetransfer/slurm/submit.sh                 # site singapore, run sg_v14
#   sitetransfer/slurm/submit.sh --dry-run       # print the sbatch commands only
#   SITE=singapore RUN_ID=sg_v14 sitetransfer/slurm/submit.sh
#
# 1. refuses uncommitted tracked files (every job records the commit it ran)
# 2. checks the synced data is there and runs the unit tests and the invariants
# 3. GPU job: train + generate (train_generate.sbatch); CPU job afterwards: evaluate + verdict
# ==============================================================================
set -euo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
SITE="${SITE:-singapore}"
RUN_ID="${RUN_ID:-sg_v14}"

[ -f sitetransfer/__main__.py ] || { echo "run from the ImagenFew repository root"; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "tracked files have uncommitted changes: commit and pull first (ALLOW_DIRTY=1 to override)"; exit 1
fi
for f in "data/${SITE}/processed/cluster4.csv" "data/${SITE}/splits/train_years_labelled.csv" \
         "data/${SITE}/splits/seasonal_transition_matrix_train_len64.pkl" "results/transfer/${SITE}/derived.json" \
         "results/transfer/${SITE}/bands.json" "regime_training/config_${RUN_ID}.yaml"; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing (sync data/${SITE}/ from the laptop; see sitetransfer/README.md)"; exit 1; }
done

if [ "${DRY}" = "0" ]; then
    source .venv/bin/activate
    python -m unittest discover -s sitetransfer/tests -t . 2>&1 | tail -3
    python -m sitetransfer check --site "${SITE}" --run-id "${RUN_ID}" --dataset
    python -m sitetransfer check --site astlingen
fi
mkdir -p logs

if [ "${DRY}" = "1" ]; then
    echo "sbatch --parsable -J st_${RUN_ID} sitetransfer/slurm/train_generate.sbatch ${RUN_ID} ${SITE}"
    echo "sbatch --dependency=afterok:<that job> sitetransfer/slurm/evaluate.sbatch ${SITE}"
    exit 0
fi
TRAIN=$(sbatch --parsable -J "st_${RUN_ID}" sitetransfer/slurm/train_generate.sbatch "${RUN_ID}" "${SITE}")
EVAL=$(sbatch --parsable --dependency=afterok:"${TRAIN}" sitetransfer/slurm/evaluate.sbatch "${SITE}")
echo "train + generate: ${TRAIN}   evaluate + verdict: ${EVAL} (after ${TRAIN})"
echo "watch: squeue -u \$USER ; logs/st_${RUN_ID}_${TRAIN}.out ; logs/st_eval_${EVAL}.out"
