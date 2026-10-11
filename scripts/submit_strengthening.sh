#!/bin/bash
# ==============================================================================
# Sends every ImagenFew job of the strengthening runs (code_plan/STRENGTHENING_RUNS.md) to
# SLURM, from the repository root on the login node. Run 1 (the agent test) is sent from
# flood-control; see the plan.
#
#   scripts/submit_strengthening.sh --dry-run      # print the sbatch commands, send nothing
#   FLOOD_CONTROL=~/flood-control scripts/submit_strengthening.sh
#
#   run 2  training seeds   v14_s1, v14_s2, v10_s1          scripts/submit_retrain.sh
#   run 3  no pretraining   v14_scratch                     scripts/submit_retrain.sh
#   run 4  Gate B KPIs      (CPU)                           scripts/submit_gate_b_kpis.sh
#   run 5  E3               v17 training, then chained      scripts/submit_retrain.sh,
#                           generation K = 2 and K = 0      scripts/submit_generation_v17_ctx.sh
#   run 7  denoising check  training years                  scripts/submit_denoise_diagnostic.sh
#
# Refuses to send uncommitted code (every job records the commit it ran; ALLOW_DIRTY=1 to
# override) and checks the code first with scripts/check_strengthening_code.py.
# ==============================================================================

set -euo pipefail
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

[ -f regime_training/train_regime.py ] || { echo "run from the ImagenFew repository root"; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "tracked files have uncommitted changes: commit and pull them first (ALLOW_DIRTY=1 to override)"
    exit 1
fi
for f in models_ckpt/ImagenFew/ImagenFew_64.ckpt logs/ImagenFew/Rainfall_Regime/v14/best_regime_model.pt \
         logs/ImagenFew/Rainfall_Regime/aebe363f/best_regime_model.pt logs/ImagenFew/Rainfall_Regime/v13/best_regime_model.pt; do
    [ -f "${f}" ] || { echo "ERROR: ${f} not found"; exit 1; }
done
for r in v14_s1 v14_s2 v10_s1 v14_scratch v17; do
    [ ! -e "logs/ImagenFew/Rainfall_Regime/${r}/best_regime_model.pt" ] \
        || { echo "ERROR: logs/ImagenFew/Rainfall_Regime/${r} already holds a model: nothing sent"; exit 1; }
done

if [ "${DRY}" = "0" ]; then
    source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate
    python scripts/check_strengthening_code.py
fi
mkdir -p logs

send() {   # send <args...>: sbatch --parsable, prints and returns the job id
    if [ "${DRY}" = "1" ]; then
        echo "sbatch $*" >&2
        echo "DRY"
    else
        local id
        id=$(sbatch --parsable "$@")
        echo "sbatch $*  ->  ${id}" >&2
        echo "${id}"
    fi
}

echo "== run 2: training seeds" >&2
send -J v14_s1 scripts/submit_retrain.sh v14_s1 config_v14.yaml ImagenFew_64.ckpt 1 >/dev/null
send -J v14_s2 scripts/submit_retrain.sh v14_s2 config_v14.yaml ImagenFew_64.ckpt 2 >/dev/null
send -J v10_s1 scripts/submit_retrain.sh v10_s1 config_v10.yaml ImagenFew_64.ckpt 1 >/dev/null
echo "== run 3: no pretraining" >&2
send -J v14_scratch scripts/submit_retrain.sh v14_scratch config_v14.yaml none 0 >/dev/null
echo "== run 5: E3 (v17), training then chained generation" >&2
V17=$(send -J v17 scripts/submit_retrain.sh v17 config_v17.yaml ImagenFew_64.ckpt 0)
send --dependency=afterok:${V17} scripts/submit_generation_v17_ctx.sh 2 >/dev/null
send --dependency=afterok:${V17} scripts/submit_generation_v17_ctx.sh 0 >/dev/null
echo "== run 7: denoising check on the training years" >&2
send scripts/submit_denoise_diagnostic.sh train >/dev/null
echo "== run 4: Gate B KPIs (CPU, flood-control's .venv)" >&2
if [ -n "${FLOOD_CONTROL:-}" ] || [ "${DRY}" = "1" ]; then
    FLOOD_CONTROL="${FLOOD_CONTROL:-${HOME}/flood-control}" send --export=ALL scripts/submit_gate_b_kpis.sh >/dev/null
else
    echo "   skipped: set FLOOD_CONTROL=<path to flood-control> to send it" >&2
fi
echo "done. Watch with: squeue -u \$USER ; results land in results/generated_data/ and results/reference/" >&2
