#!/bin/bash
# ==============================================================================
# Send the guidance pilot from the repository root on the login node:
#   GPU_PARTITION=gpu-long guidance/slurm/submit_pilot.sh            (--dry-run to print only)
# 1. 20 GPU array tasks, one per setting (pilot_task.sbatch)
# 2. CPU: collect, apply the pass rules, choose (pilot_summary.sbatch), after all tasks succeed
# 3. GPU: the chosen setting with 2 h of context (pilot_context2h.sbatch), after the summary
# Results: results/guidance/pilot.json, results/guidance/context2h.json
# ==============================================================================
set -euo pipefail
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
GPU_PARTITION="${GPU_PARTITION:-gpu}"
CPU_PARTITION="${CPU_PARTITION:-normal}"
[ -f guidance/__main__.py ] || { echo "run from the ImagenFew repository root"; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "tracked files have uncommitted changes: commit and pull first (ALLOW_DIRTY=1 to override)"; exit 1
fi
for f in logs/ImagenFew/Rainfall_Regime/v14/best_regime_model.pt logs/ImagenFew/Rainfall_Regime/v14/scaler.pkl \
         data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing"; exit 1; }
done
[ -e results/guidance/pilot.json ] && { echo "ERROR: results/guidance/pilot.json exists; move it away first"; exit 1; }
mkdir -p logs
if [ "${DRY}" = "1" ]; then
    echo "sbatch --parsable -p ${GPU_PARTITION} guidance/slurm/pilot_task.sbatch"
    echo "sbatch --parsable -p ${CPU_PARTITION} --dependency=afterok:<array> guidance/slurm/pilot_summary.sbatch"
    echo "sbatch -p ${GPU_PARTITION} --dependency=afterok:<summary> guidance/slurm/pilot_context2h.sbatch"
    exit 0
fi
ARR=$(sbatch --parsable -p "${GPU_PARTITION}" guidance/slurm/pilot_task.sbatch)
SUM=$(sbatch --parsable -p "${CPU_PARTITION}" --dependency=afterok:"${ARR}" guidance/slurm/pilot_summary.sbatch)
CTX=$(sbatch --parsable -p "${GPU_PARTITION}" --dependency=afterok:"${SUM}" guidance/slurm/pilot_context2h.sbatch)
echo "pilot array: ${ARR} (20 tasks)   summary: ${SUM}   2-hour context: ${CTX}"
echo "watch: squeue -u \$USER ; logs/gd_pilot_${ARR}_<task>.out ; logs/gd_summary_${SUM}.out ; logs/gd_ctx2h_${CTX}.out"
