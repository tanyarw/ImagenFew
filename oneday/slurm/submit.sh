#!/bin/bash
# ==============================================================================
# Send v19 to SLURM from the repository root on the login node:
#   oneday/slurm/submit.sh smoke                        # ~10 min GPU: speed, memory, time limit
#   TRAIN_TIME=40:00:00 oneday/slurm/submit.sh train    # after the config is fixed from the smoke result
#   RUN_ID=v19_scratch INIT=none EPOCHS=325 SAVE_EVERY=50 TRAIN_TIME=48:00:00 oneday/slurm/submit.sh train
#                                                       # another run of the recipe (train.sbatch header)
#   ... --dry-run                                       # print the sbatch command only
#   GPU_PARTITION=gpu-long (default) | gpu
#
# Refuses uncommitted tracked files, checks the checkpoint (it is not in git: rsync it from the laptop to
# models_ckpt/ImagenFew/) and the config; the unit tests run at the start of each GPU job.
# ==============================================================================
set -euo pipefail
WHAT="${1:?usage: submit.sh smoke|train [--dry-run]}"
DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
GPU_PARTITION="${GPU_PARTITION:-gpu-long}"

[ -f oneday/__main__.py ] || { echo "run from the ImagenFew repository root"; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "tracked files have uncommitted changes: commit and pull first (ALLOW_DIRTY=1 to override)"; exit 1
fi
for f in data/rainfall/splits/train_years_labelled.csv data/rainfall/splits/val_years_labelled.csv \
         data/rainfall/splits/test_years_labelled.csv models_ckpt/ImagenFew/ImagenFew_288_1200epochs.ckpt \
         regime_training/config_v19.yaml; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing (see oneday/README.md)"; exit 1; }
done
mkdir -p logs
if [ "${DRY}" = "0" ]; then
    source .venv/bin/activate
    python -m oneday check
fi

case "${WHAT}" in
    smoke)
        CMD=(sbatch --parsable -p "${GPU_PARTITION}" oneday/slurm/smoke.sbatch)
        ;;
    train)
        T="${TRAIN_TIME:?set TRAIN_TIME (e.g. 40:00:00) from the smoke result and the rule in the plan}"
        R="${RUN_ID:-v19}"
        [ "${GENERATE_ONLY:-0}" = "1" ] || [ ! -e "logs/ImagenFew/Rainfall_Regime/${R}/best_regime_model.pt" ] || \
            { echo "ERROR: logs/ImagenFew/Rainfall_Regime/${R} already holds a model"; exit 1; }
        CMD=(sbatch --parsable -p "${GPU_PARTITION}" -t "${T}" -J "${R}" oneday/slurm/train.sbatch)
        ;;
    *)
        echo "unknown: ${WHAT} (smoke | train)"; exit 1
        ;;
esac
if [ "${DRY}" = "1" ]; then echo "${CMD[*]}"; exit 0; fi
J=$("${CMD[@]}")
echo "job ${J}: logs/$([ "${WHAT}" = smoke ] && echo v19_smoke || echo "${RUN_ID:-v19}")_${J}.out"
