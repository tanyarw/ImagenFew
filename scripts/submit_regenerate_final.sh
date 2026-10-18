#!/bin/bash
#SBATCH --job-name=regen_final
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --array=0-14

# ==============================================================================
# Regenerate every scored version from its FINAL weights (diary 2026-10-18, "Best-Checkpoint Bug").
#
# From commit c9784fc (2026-08-25) until 2026-10-18, train_regime.py saved best_regime_model.pt
# only on logging epochs (every 25), while the lowest loss so far was updated every epoch. One low
# loss on a non-logging epoch blocked every later save, so "best" froze early: v11 and v15 at
# epoch 1, v9 at 1, v17 / v8 / v14_s1 / e1_v10ckpt24 at 50, v10 / v12 at 75, sg_v14 at 25 of 698.
# Every scored series was generated from that file.
#
# Each task repeats the version's ORIGINAL generation command (same config, transition table,
# assembly, overlaps, plan seed 42); the only change is final_regime_model.pt in place of
# best_regime_model.pt. Outputs get a _final tag, so no existing series is overwritten:
#   markov / unconditional  -> results/generated_data/rainfall_synthetic_10y_<NAME>_final.csv
#   calendar, no bridges     -> results/generated_data/rainfall_synthetic_10y_<NAME>_final_nobridge_cal.csv
#   sg_v14                   -> results/generated_data/singapore/rainfall_synthetic_10y_sg_v14_final_nobridge_cal.csv
#
#   sbatch scripts/submit_regenerate_final.sh                 # all 15 tasks
#   sbatch --array=10-12 scripts/submit_regenerate_final.sh   # some of them (indices: ... list)
#   bash scripts/submit_regenerate_final.sh list              # print the table
#   SMOKE=1 bash scripts/submit_regenerate_final.sh all       # laptop: every entry, CPU, ~1 day of rain,
#                                                             # written to $SMOKE_DIR (default /tmp)
#
# v17's chained series (9 h + 7 h): CKPT_NAME=final_regime_model.pt SUFFIX=_final, see
# scripts/submit_generation_v17_ctx.sh. v19: GENERATE_ONLY=1 with oneday/slurm/train.sbatch.
# ==============================================================================

set -euo pipefail

# NAME | RUN_DIR (under logs/ImagenFew/Rainfall_Regime) | CONFIG (under regime_training) | TRANSITION TABLE | ASSEMBLIES
S=data/rainfall/splits/seasonal_transition_matrix_train_len
TABLE=(
    "v8|ed17d299|config_v8.yaml|${S}24.pkl|markov"
    "v9|160156d4|config_v9.yaml|${S}36.pkl|markov"
    "v10|aebe363f|config_v10.yaml|${S}64.pkl|markov"
    "e1_v10ckpt24|e1_v10ckpt24|config_v10.yaml|${S}64.pkl|markov"
    "v11|v11|config_v11.yaml|-|unconditional"
    "v12|v12|config_v12.yaml|-|unconditional"
    "v13|v13|config_v13.yaml|${S}64.pkl|markov"
    "v14|v14|config_v14.yaml|${S}64.pkl|markov nobridge_cal"
    "v15|v15|config_v15.yaml|${S}64.pkl|markov"
    "v15_ctrl|v15_ctrl|config_v15_ctrl.yaml|${S}64.pkl|markov"
    "v14_s1|v14_s1|config_v14.yaml|${S}64.pkl|markov nobridge_cal"
    "v14_s2|v14_s2|config_v14.yaml|${S}64.pkl|markov nobridge_cal"
    "v14_scratch|v14_scratch|config_v14.yaml|${S}64.pkl|markov nobridge_cal"
    "v17|v17|config_v17.yaml|${S}64.pkl|markov nobridge_cal"
    "sg_v14|sg_v14|config_sg_v14.yaml|data/singapore/splits/seasonal_transition_matrix_train_len64.pkl|nobridge_cal"
    # 15: not in the default array. Only if v14_uncond was sent before this fix (its own job generated from
    # best_regime_model.pt then): sbatch --array=15 scripts/submit_regenerate_final.sh
    "v14_uncond|v14_uncond|config_v14_uncond.yaml|-|unconditional"
)

if [ "${1:-}" = "list" ]; then
    for i in "${!TABLE[@]}"; do echo "${i}  ${TABLE[$i]}"; done
    exit 0
fi

cd "${SLURM_SUBMIT_DIR:-.}"
if [ "${SMOKE:-0}" = "1" ]; then
    PY=.venv/bin/python; DEVICE=cpu; YEARS=0.003; OUT_ROOT="${SMOKE_DIR:-/tmp/regen_final_smoke}"
else
    source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate
    PY=python; DEVICE=cuda; YEARS=10; OUT_ROOT=results/generated_data
    mkdir -p logs
    echo "Job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
    echo "commit: $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(uncommitted changes!)')"
    python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"
fi

run_entry() {
    IFS='|' read -r NAME RUN CONFIG TRANS ASSEMBLIES <<< "$1"
    local RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN}"
    local WEIGHTS="${RUN_DIR}/final_regime_model.pt"
    local OUT_DIR="${OUT_ROOT}"
    [ "${NAME}" != "sg_v14" ] || OUT_DIR="${OUT_ROOT}/singapore"
    mkdir -p "${OUT_DIR}"

    for f in "regime_training/${CONFIG}" "${WEIGHTS}" "${RUN_DIR}/scaler.pkl"; do
        [ -f "${f}" ] || { echo "ERROR: ${f} not found"; return 1; }
    done
    [ "${TRANS}" = "-" ] || [ -f "${TRANS}" ] || { echo "ERROR: ${TRANS} not found"; return 1; }

    # Provenance: which epoch the weights are from, and their hash
    "${PY}" -c "import torch,hashlib,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); \
print('weights', sys.argv[1], 'epoch', c['epoch'], 'sha256', hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])" "${WEIGHTS}"

    for A in ${ASSEMBLIES}; do
        local ARGS=() SUFFIX=""
        case "${A}" in
            markov)        ARGS=(--transition_matrix_path "${TRANS}" --assembly_mode markov --transition_overlap 8 --bridge_blocks 1) ;;
            nobridge_cal)  ARGS=(--transition_matrix_path "${TRANS}" --assembly_mode calendar --transition_overlap 8 --bridge_blocks 0)
                           SUFFIX="_nobridge_cal" ;;
            unconditional) ARGS=(--assembly_mode unconditional) ;;
            *) echo "ERROR: unknown assembly ${A}"; return 1 ;;
        esac
        local OUT_NAME="rainfall_synthetic_10y_${NAME}_final${SUFFIX}.csv"
        echo "── ${NAME}: ${A} -> ${OUT_DIR}/${OUT_NAME}"
        "${PY}" regime_training/generate_hmm_v1.py \
            --model_ckpt "${WEIGHTS}" \
            --scaler_path "${RUN_DIR}/scaler.pkl" \
            --config "regime_training/${CONFIG}" \
            "${ARGS[@]}" \
            --years "${YEARS}" --freq "5min" --overlap 4 \
            --seed 42 --device "${DEVICE}" \
            --output_dir "${OUT_DIR}" \
            --output_name "${OUT_NAME}"
    done
}

if [ "${1:-}" = "all" ]; then
    for e in "${TABLE[@]}"; do run_entry "${e}"; done
else
    IDX="${SLURM_ARRAY_TASK_ID:-${1:?give an index (see: list), 'all', or run as an sbatch array}}"
    run_entry "${TABLE[$IDX]}"
fi
echo "Done at: $(date)"
