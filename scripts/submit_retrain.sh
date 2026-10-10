#!/bin/bash
#SBATCH --job-name=retrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# Train one model with a production config, then generate its 10-year series the
# way the scored versions were generated. Used by strengthening runs 2, 3 and 6
# (code_plan/STRENGTHENING_RUNS.md):
#
#   sbatch -J v14_s1      scripts/submit_retrain.sh v14_s1      config_v14.yaml ImagenFew_64.ckpt 1   # run 2
#   sbatch -J v14_s2      scripts/submit_retrain.sh v14_s2      config_v14.yaml ImagenFew_64.ckpt 2   # run 2
#   sbatch -J v10_s1      scripts/submit_retrain.sh v10_s1      config_v10.yaml ImagenFew_64.ckpt 1   # run 2
#   sbatch -J v14_scratch scripts/submit_retrain.sh v14_scratch config_v14.yaml none              0   # run 3
#   sbatch -J v18         scripts/submit_retrain.sh v18         config_v18.yaml ImagenFew_64.ckpt 0   # run 6
#
# (scripts/submit_strengthening.sh sends all of them.)
#
# Arguments: RUN_ID CONFIG (in regime_training/) CHECKPOINT (in models_ckpt/ImagenFew/, or
# "none" = random weights) SEED (training seed).
#
# Training: 500 epochs, batch 2048, lr 1e-4: the settings of every scored version since v8
# (taken from the config; nothing is overridden on the command line).
# Generation, both with plan seed 42 so that only the trained weights differ from the
# version being compared against:
#   markov, 1 bridge block per change -> rainfall_synthetic_10y_<RUN_ID>.csv
#       (how the scored rows v10, v13, v14 were made: scripts/submit_job_v14.sh)
#   calendar, no bridge blocks        -> rainfall_synthetic_10y_<RUN_ID>_nobridge_cal.csv
#       (how v14_nobridge_cal, v14's seasonal version, was made)
#
# Refuses to start if the run directory already holds a trained model, so no existing
# run (v10, v14, ...) can be overwritten by a mistyped RUN_ID. If generation fails after
# training, rerun with GENERATE_ONLY=1 to skip training.
# ==============================================================================

set -euo pipefail

RUN_ID="${1:?usage: submit_retrain.sh RUN_ID CONFIG CHECKPOINT|none SEED}"
CONFIG="regime_training/${2:?config missing}"
CKPT_NAME="${3:?checkpoint missing (or none)}"
SEED="${4:?training seed missing}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"

RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN_ID}"
TRANS_MATRIX="data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl"
OUTPUT_DIR="results/generated_data"
if [ "${CKPT_NAME}" = "none" ]; then CKPT="none"; else CKPT="models_ckpt/ImagenFew/${CKPT_NAME}"; fi

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs "${OUTPUT_DIR}"
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-} | GPU: ${CUDA_VISIBLE_DEVICES:-} | Host: $(hostname -s) | Start: $(date)"
echo "run=${RUN_ID} config=${CONFIG} checkpoint=${CKPT} seed=${SEED} generate_only=${GENERATE_ONLY}"
echo "commit: $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(uncommitted changes!)')"
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

for f in "${CONFIG}" "${TRANS_MATRIX}"; do
    [ -f "${f}" ] || { echo "ERROR: ${f} not found"; exit 1; }
done
[ "${CKPT}" = "none" ] || [ -f "${CKPT}" ] || { echo "ERROR: ${CKPT} not found"; exit 1; }
# Training code must have the strengthening options (seed, random init) - fail now, not in 16 h
grep -q "def known_prefix_mask" regime_training/train_regime.py \
    || { echo "ERROR: train_regime.py is older than the strengthening runs: git pull"; exit 1; }

# ── 1. Train ──────────────────────────────────────────────────────────
if [ "${GENERATE_ONLY}" = "1" ]; then
    [ -f "${RUN_DIR}/best_regime_model.pt" ] || { echo "ERROR: GENERATE_ONLY=1 but ${RUN_DIR} has no model"; exit 1; }
else
    if [ -e "${RUN_DIR}/best_regime_model.pt" ] || [ -e "${RUN_DIR}/final_regime_model.pt" ]; then
        echo "ERROR: ${RUN_DIR} already holds a trained model; refusing to overwrite it."
        echo "       Choose another RUN_ID, or GENERATE_ONLY=1 to only generate from it."
        exit 1
    fi
    python regime_training/train_regime.py \
        --model_ckpt "${CKPT}" \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --run_id "${RUN_ID}"
fi

# ── 2. Generate 10 years, the two ways the scored versions were made ──
for f in "${RUN_DIR}/best_regime_model.pt" "${RUN_DIR}/scaler.pkl"; do
    [ -f "${f}" ] || { echo "ERROR: ${f} missing after training"; exit 1; }
done
for MODE in markov calendar; do
    if [ "${MODE}" = "calendar" ]; then BRIDGE=0; SUFFIX="_nobridge_cal"; else BRIDGE=1; SUFFIX=""; fi
    python regime_training/generate_hmm_v1.py \
        --model_ckpt "${RUN_DIR}/best_regime_model.pt" \
        --scaler_path "${RUN_DIR}/scaler.pkl" \
        --config "${CONFIG}" \
        --transition_matrix_path "${TRANS_MATRIX}" \
        --assembly_mode "${MODE}" \
        --years 10 --freq "5min" --overlap 4 \
        --transition_overlap 8 --bridge_blocks "${BRIDGE}" \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}" \
        --output_name "rainfall_synthetic_10y_${RUN_ID}${SUFFIX}.csv"
done

ls -lh "${OUTPUT_DIR}"/rainfall_synthetic_10y_${RUN_ID}.csv "${OUTPUT_DIR}"/rainfall_synthetic_10y_${RUN_ID}_nobridge_cal.csv
echo "Job completed at: $(date)"
