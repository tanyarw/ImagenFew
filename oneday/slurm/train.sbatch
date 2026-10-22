#!/bin/bash
#SBATCH --job-name=v19
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# v19: train the v14 recipe with a one-day window, generate 4 ten-year members, evaluate
# (code_plan/ONE_DAY_WINDOW_PLAN.md). Sent by oneday/slurm/submit.sh train, with --time from the smoke result.
#
#   model    logs/ImagenFew/Rainfall_Regime/v19/      from models_ckpt/ImagenFew/ImagenFew_288_1200epochs.ckpt
#   series   results/generated_data/rainfall_synthetic_10y_v19_nobridge_cal[_m1.._m3].csv
#            (as v14_nobridge_cal: calendar timeline, no bridge blocks, 4-step overlap; seeds 42-45)
#   verdict  results/oneday/evaluation.json (statistics; the sewer check runs on the laptop)
# GENERATE_ONLY=1 skips training (to regenerate from a finished model).
#
# Other runs of the same recipe (defaults reproduce v19 exactly):
#   RUN_ID      run name: model in logs/ImagenFew/Rainfall_Regime/RUN_ID/, series rainfall_synthetic_10y_RUN_ID_*,
#               verdict results/oneday/evaluation_RUN_ID.json (v19 keeps evaluation.json)     default v19
#   INIT        pretrained (the checkpoint CKPT below) | none (random weights; RUN_ID must then differ from v19)
#   EPOCHS      overrides the config's epochs on the command line (recorded in run_args.json)  default: config's
#   SAVE_EVERY  keep the weights every N epochs as epoch_NNNN_regime_model.pt                  default 0 (none)
#   TRAIN_SEED  training seed                                                                  default 0
# v19 from scratch (diary 2026-10-19), 325 epochs to match v19's evaluated snapshot:
#   RUN_ID=v19_scratch INIT=none EPOCHS=325 SAVE_EVERY=50 TRAIN_TIME=48:00:00 oneday/slurm/submit.sh train
# CKPT_NAME = the weights file in RUN_DIR to generate from (default final_regime_model.pt, the last epoch;
# best_regime_model.pt could freeze early under the train_regime.py save rule used before 2026-10-18, diary
# "Best-Checkpoint Bug"). Not to be confused with CKPT below, the pretrained checkpoint training starts from.
# The first v19 training job stopped at its time limit near epoch 330 of 500, so no final_regime_model.pt
# exists; its last snapshot (epoch 325) is generated with (a recorded deviation, diary 2026-10-19):
#   GENERATE_ONLY=1 CKPT_NAME=best_regime_model.pt TRAIN_TIME=02:00:00 oneday/slurm/submit.sh train
# SERIES_ID   names the series and the verdict instead of RUN_ID (default RUN_ID), so a snapshot of a run gets its
#             own files. A series that exists is never overwritten (ALLOW_OVERWRITE=1 to regenerate on purpose).
# HELDOUT     the real years evaluate compares with: 2008-09 (default, as before) | 2008 (keeps 2009 closed for
#             the test year, code_plan/TEST_YEAR_PLAN.md)
# v19 scratch at epoch 50, the generator the test year is for (diary 2026-10-22):
#   RUN_ID=v19_scratch SERIES_ID=v19_scratch_ep50 CKPT_NAME=epoch_0050_regime_model.pt HELDOUT=2008 GENERATE_ONLY=1 \
#       TRAIN_TIME=02:00:00 oneday/slurm/submit.sh train
# ==============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs
RUN_ID="${RUN_ID:-v19}"
INIT="${INIT:-pretrained}"
EPOCHS="${EPOCHS:-}"
SAVE_EVERY="${SAVE_EVERY:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN_ID}"
case "${INIT}" in
    pretrained) CKPT="models_ckpt/ImagenFew/ImagenFew_288_1200epochs.ckpt" ;;
    none)       CKPT="none"
                [ "${RUN_ID}" != "v19" ] || { echo "ERROR: INIT=none needs its own RUN_ID (v19 is the pretrained run)"; exit 1; } ;;
    *)          echo "ERROR: INIT=${INIT} (pretrained | none)"; exit 1 ;;
esac
CAL="data/rainfall/splits/seasonal_transition_matrix_train_len288.pkl"
CKPT_NAME="${CKPT_NAME:-final_regime_model.pt}"
WEIGHTS="${RUN_DIR}/${CKPT_NAME}"
SERIES_ID="${SERIES_ID:-${RUN_ID}}"
HELDOUT="${HELDOUT:-2008-09}"
FIRST="results/generated_data/rainfall_synthetic_10y_${SERIES_ID}_nobridge_cal.csv"
if [ -e "${FIRST}" ] && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
    echo "ERROR: ${FIRST} exists; set SERIES_ID to a new name (or ALLOW_OVERWRITE=1 to regenerate on purpose)"
    exit 1
fi
# Generating from an existing model: stop now, before tests and setup, if its weights file is missing
if [ "${GENERATE_ONLY:-0}" = "1" ] && [ ! -f "${WEIGHTS}" ]; then
    echo "ERROR: ${WEIGHTS} not found. Weights files in ${RUN_DIR}: $(ls "${RUN_DIR}" 2>/dev/null | grep '\.pt$' | tr '\n' ' ')"
    echo "       Choose one with CKPT_NAME=... (e.g. CKPT_NAME=best_regime_model.pt)"
    exit 1
fi
echo "job ${SLURM_JOB_ID:-} | $(hostname -s) | GPU ${CUDA_VISIBLE_DEVICES:-} | $(date)"
echo "run ${RUN_ID} | init ${INIT} (${CKPT}) | epochs ${EPOCHS:-config} | save every ${SAVE_EVERY} | training seed ${TRAIN_SEED}"
echo "commit $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(UNCOMMITTED CHANGES)')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Checks that need torch (login nodes cannot load it): stop here, before any training, if one fails
python -m unittest discover -s oneday/tests -t .
python -m oneday check

if [ "${GENERATE_ONLY:-0}" != "1" ]; then
    [ ! -e "${RUN_DIR}/best_regime_model.pt" ] || { echo "ERROR: ${RUN_DIR} already holds a model"; exit 1; }
    python regime_training/train_regime.py --model_ckpt "${CKPT}" --config regime_training/config_v19.yaml \
        --seed "${TRAIN_SEED}" --run_id "${RUN_ID}" --save_every "${SAVE_EVERY}" ${EPOCHS:+--epochs "${EPOCHS}"}
fi

[ -f "${WEIGHTS}" ] || { echo "ERROR: ${WEIGHTS} not found after training"; exit 1; }
# Provenance: which weights file, the epoch stored in it, and its hash
python -c "import torch,hashlib,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); \
print('generating from', sys.argv[1], 'epoch', c['epoch'], 'sha256', hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])" "${WEIGHTS}"

for K in 0 1 2 3; do
    NAME="rainfall_synthetic_10y_${SERIES_ID}_nobridge_cal$([ "${K}" = 0 ] || echo "_m${K}").csv"
    python regime_training/generate_hmm_v1.py \
        --model_ckpt "${WEIGHTS}" --scaler_path "${RUN_DIR}/scaler.pkl" \
        --config regime_training/config_v19.yaml --transition_matrix_path "${CAL}" \
        --assembly_mode calendar --bridge_blocks 0 --years 10 --freq 5min --overlap 4 --transition_overlap 8 \
        --seed $((42 + K)) --output_dir results/generated_data --output_name "${NAME}"
done

python -m oneday evaluate --run-id "${SERIES_ID}" --heldout "${HELDOUT}"
echo "done $(date)"
