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
#   model    logs/ImagenFew/Rainfall_Regime/v19/      from models_ckpt/ImagenFew/ImagenFew_288_807epochs.ckpt
#   series   results/generated_data/rainfall_synthetic_10y_v19_nobridge_cal[_m1.._m3].csv
#            (as v14_nobridge_cal: calendar timeline, no bridge blocks, 4-step overlap; seeds 42-45)
#   verdict  results/oneday/evaluation.json (statistics; the sewer check runs on the laptop)
# GENERATE_ONLY=1 skips training (to regenerate from a finished model).
# ==============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs
RUN_DIR="logs/ImagenFew/Rainfall_Regime/v19"
CKPT="models_ckpt/ImagenFew/ImagenFew_288_807epochs.ckpt"
CAL="data/rainfall/splits/seasonal_transition_matrix_train_len288.pkl"
echo "job ${SLURM_JOB_ID:-} | $(hostname -s) | GPU ${CUDA_VISIBLE_DEVICES:-} | $(date)"
echo "commit $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(UNCOMMITTED CHANGES)')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Checks that need torch (login nodes cannot load it): stop here, before any training, if one fails
python -m unittest discover -s oneday/tests -t .
python -m oneday check

if [ "${GENERATE_ONLY:-0}" != "1" ]; then
    [ ! -e "${RUN_DIR}/best_regime_model.pt" ] || { echo "ERROR: ${RUN_DIR} already holds a model"; exit 1; }
    python regime_training/train_regime.py --model_ckpt "${CKPT}" --config regime_training/config_v19.yaml \
        --seed 0 --run_id v19
fi

for K in 0 1 2 3; do
    NAME="rainfall_synthetic_10y_v19_nobridge_cal$([ "${K}" = 0 ] || echo "_m${K}").csv"
    python regime_training/generate_hmm_v1.py \
        --model_ckpt "${RUN_DIR}/best_regime_model.pt" --scaler_path "${RUN_DIR}/scaler.pkl" \
        --config regime_training/config_v19.yaml --transition_matrix_path "${CAL}" \
        --assembly_mode calendar --bridge_blocks 0 --years 10 --freq 5min --overlap 4 --transition_overlap 8 \
        --seed $((42 + K)) --output_dir results/generated_data --output_name "${NAME}"
done

python -m oneday evaluate
echo "done $(date)"
