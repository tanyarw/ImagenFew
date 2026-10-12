#!/bin/bash
#SBATCH --job-name=st_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# Transfer site: train the v14 recipe on the site's data, then generate 10 years exactly as
# v14_nobridge_cal was generated (calendar timeline, no bridge blocks, plan seed 42).
# Sent by sitetransfer/slurm/submit.sh; arguments: RUN_ID SITE.
#
#   config   regime_training/config_<RUN_ID>.yaml (written by `python -m sitetransfer train-config`)
#   model    logs/ImagenFew/Rainfall_Regime/<RUN_ID>/   (run_args.json records commit, seed, config)
#   series   results/generated_data/<SITE>/rainfall_synthetic_10y_<RUN_ID>_nobridge_cal.csv
#
# The site's wet threshold is applied when the series is scored (sitetransfer.site.load_generated);
# generate_hmm_v1.py's own 0.005 mm cut-off runs after stitching, so the result is the same.
# ==============================================================================
set -euo pipefail

RUN_ID="${1:?usage: train_generate.sbatch RUN_ID SITE}"
SITE="${2:?usage: train_generate.sbatch RUN_ID SITE}"
CONFIG="regime_training/config_${RUN_ID}.yaml"
RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN_ID}"
BUNDLE="data/${SITE}/splits/seasonal_transition_matrix_train_len64.pkl"
OUT_DIR="results/generated_data/${SITE}"
CKPT="models_ckpt/ImagenFew/ImagenFew_64.ckpt"

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs "${OUT_DIR}"
echo "job ${SLURM_JOB_ID:-} | $(hostname -s) | GPU ${CUDA_VISIBLE_DEVICES:-} | $(date)"
echo "commit $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(UNCOMMITTED CHANGES)')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

for f in "${CONFIG}" "${BUNDLE}" "${CKPT}"; do
    [ -f "${f}" ] || { echo "ERROR: ${f} not found"; exit 1; }
done
if [ -e "${RUN_DIR}/best_regime_model.pt" ] && [ "${GENERATE_ONLY:-0}" != "1" ]; then
    echo "ERROR: ${RUN_DIR} already holds a model (GENERATE_ONLY=1 to only generate)"; exit 1
fi

if [ "${GENERATE_ONLY:-0}" != "1" ]; then
    python regime_training/train_regime.py --model_ckpt "${CKPT}" --config "${CONFIG}" --seed 0 --run_id "${RUN_ID}"
fi

python regime_training/generate_hmm_v1.py \
    --model_ckpt "${RUN_DIR}/best_regime_model.pt" \
    --scaler_path "${RUN_DIR}/scaler.pkl" \
    --config "${CONFIG}" \
    --transition_matrix_path "${BUNDLE}" \
    --assembly_mode calendar --bridge_blocks 0 \
    --years 10 --freq 5min --overlap 4 --transition_overlap 8 --seed 42 \
    --output_dir "${OUT_DIR}" \
    --output_name "rainfall_synthetic_10y_${RUN_ID}_nobridge_cal.csv"

ls -lh "${OUT_DIR}/rainfall_synthetic_10y_${RUN_ID}_nobridge_cal.csv"
echo "done $(date)"
