#!/bin/bash
# ==============================================================================
# Send the two-resolution runs to SLURM from the repository root on the login node:
#   tworesolution/slurm/submit.sh train            # both models in parallel; stage 1 after the coarse model,
#                                                  # stage 2 after the fine model
#   tworesolution/slurm/submit.sh retrain          # follow-up (plan section 11): the coarse model for v14's
#                                                  # step count, then its stage 1 (outputs tagged _long)
#   tworesolution/slurm/submit.sh stage3           # after reading stage1.json and stage2.json
#   COARSE_RUN=twores_coarse30_long tworesolution/slurm/submit.sh stage3   # stage 3 from the follow-up
#   tworesolution/slurm/submit.sh train --dry-run  # print the sbatch commands only
#   GPU_PARTITION=gpu-long (default) | gpu
#
# 1. refuses uncommitted tracked files (every job records the commit it ran)
# 2. checks the data and models are there, rebuilds the 30-minute files (they must hash as recorded in the
#    committed derived.json) and runs the checks that need no torch (login nodes cannot load torch); the unit
#    tests and the torch checks run at the start of each training job
# ==============================================================================
set -euo pipefail
WHAT="${1:?usage: submit.sh train|retrain|stage3 [--dry-run]}"
COARSE_RUN="${COARSE_RUN:-twores_coarse30}"
DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
GPU_PARTITION="${GPU_PARTITION:-gpu-long}"
J=tworesolution/slurm

[ -f tworesolution/__main__.py ] || { echo "run from the ImagenFew repository root"; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "tracked files have uncommitted changes: commit and pull first (ALLOW_DIRTY=1 to override)"; exit 1
fi
for f in data/rainfall/splits/train_years_labelled.csv data/rainfall/splits/val_years_labelled.csv \
         data/rainfall/splits/test_years_labelled.csv models_ckpt/ImagenFew/ImagenFew_64.ckpt \
         logs/ImagenFew/Rainfall_Regime/v14/best_regime_model.pt logs/ImagenFew/Rainfall_Regime/v14/scaler.pkl \
         results/tworesolution/derived.json tworesolution/configs/coarse30.yaml \
         tworesolution/configs/coarse30_long.yaml tworesolution/configs/fine.yaml; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing (see tworesolution/README.md)"; exit 1; }
done
mkdir -p logs

if [ "${DRY}" = "0" ]; then
    source .venv/bin/activate
    python -m tworesolution build-data > /dev/null
    python -m tworesolution check
fi

case "${WHAT}" in
    train)
        if [ "${DRY}" = "1" ]; then
            echo "sbatch --parsable -p ${GPU_PARTITION} -J tr_coarse ${J}/train.sbatch coarse"
            echo "sbatch --parsable -p ${GPU_PARTITION} -J tr_fine ${J}/train.sbatch fine"
            echo "sbatch -p ${GPU_PARTITION} -J tr_stage1 --dependency=afterok:<coarse job> ${J}/stage.sbatch 1"
            echo "sbatch -p ${GPU_PARTITION} -J tr_stage2 --dependency=afterok:<fine job> ${J}/stage.sbatch 2"
            exit 0
        fi
        C=$(sbatch --parsable -p "${GPU_PARTITION}" -J tr_coarse "${J}/train.sbatch" coarse)
        F=$(sbatch --parsable -p "${GPU_PARTITION}" -J tr_fine "${J}/train.sbatch" fine)
        S1=$(sbatch --parsable -p "${GPU_PARTITION}" -J tr_stage1 --dependency=afterok:"${C}" "${J}/stage.sbatch" 1)
        S2=$(sbatch --parsable -p "${GPU_PARTITION}" -J tr_stage2 --dependency=afterok:"${F}" "${J}/stage.sbatch" 2)
        echo "coarse ${C} -> stage 1 ${S1} | fine ${F} -> stage 2 ${S2}"
        echo "watch: squeue -u \$USER ; logs/tr_coarse_${C}.out ; logs/tr_fine_${F}.out"
        ;;
    retrain)
        if [ "${DRY}" = "1" ]; then
            echo "sbatch --parsable -p ${GPU_PARTITION} -t 16:00:00 -J tr_coarse_long ${J}/train.sbatch coarse_long"
            echo "sbatch -p ${GPU_PARTITION} -J tr_stage1_long --dependency=afterok:<that job> ${J}/stage.sbatch 1 twores_coarse30_long"
            exit 0
        fi
        C=$(sbatch --parsable -p "${GPU_PARTITION}" -t 16:00:00 -J tr_coarse_long "${J}/train.sbatch" coarse_long)
        S1=$(sbatch --parsable -p "${GPU_PARTITION}" -J tr_stage1_long --dependency=afterok:"${C}" \
             "${J}/stage.sbatch" 1 twores_coarse30_long)
        echo "coarse (v14's step count) ${C} -> stage 1 ${S1}"
        echo "watch: squeue -u \$USER ; logs/tr_coarse_long_${C}.out ; logs/tr_stage1_long_${S1}.out"
        ;;
    stage3)
        if [ "${DRY}" = "0" ]; then
            python -m tworesolution stage3-methods --coarse-run "${COARSE_RUN}" > /dev/null   # stops if stage 1 or 2 did
        fi
        if [ "${DRY}" = "1" ]; then
            echo "sbatch -p ${GPU_PARTITION} -J tr_stage3 ${J}/stage.sbatch 3 ${COARSE_RUN}"; exit 0
        fi
        S3=$(sbatch --parsable -p "${GPU_PARTITION}" -J tr_stage3 "${J}/stage.sbatch" 3 "${COARSE_RUN}")
        echo "stage 3 ${S3}: logs/tr_stage3_${S3}.out"
        ;;
    *)
        echo "unknown: ${WHAT} (train | retrain | stage3)"; exit 1
        ;;
esac
