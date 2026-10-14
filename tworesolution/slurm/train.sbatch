#!/bin/bash
#SBATCH --job-name=tr_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# Two-resolution generator: train one model (code_plan/TWO_RESOLUTION_PLAN.md, stages 1 and 2).
# Sent by tworesolution/slurm/submit.sh; argument: coarse | fine.
#
#   coarse  regime_training/train_regime.py, v14's recipe on 30-minute totals, from ImagenFew_64.ckpt
#           -> logs/ImagenFew/Rainfall_Regime/twores_coarse30/
#   fine    python -m tworesolution train-fine, v14 + the half-hour totals as an extra input, from v14
#           -> logs/ImagenFew/Rainfall_Regime/twores_fine/
# Both: 500 epochs x 68 steps (about 34,000 optimiser steps), seed 0.
# ==============================================================================
set -euo pipefail
MODEL="${1:?usage: train.sbatch coarse|fine}"
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs
echo "job ${SLURM_JOB_ID:-} | $(hostname -s) | GPU ${CUDA_VISIBLE_DEVICES:-} | $(date)"
echo "commit $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(UNCOMMITTED CHANGES)')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Checks that need torch (login nodes cannot load it): stop here, before any training, if one fails
python -m tworesolution build-data > /dev/null
python -m unittest discover -s tworesolution/tests -t .
python -m tworesolution check --torch --device cpu      # CPU: bit-exact comparison

case "${MODEL}" in
    coarse)
        RUN_DIR="logs/ImagenFew/Rainfall_Regime/twores_coarse30"
        [ ! -e "${RUN_DIR}/best_regime_model.pt" ] || { echo "ERROR: ${RUN_DIR} already holds a model"; exit 1; }
        python regime_training/train_regime.py --model_ckpt models_ckpt/ImagenFew/ImagenFew_64.ckpt \
            --config tworesolution/configs/coarse30.yaml --seed 0 --run_id twores_coarse30
        ;;
    fine)
        python -m tworesolution train-fine --seed 0
        ;;
    *)
        echo "unknown model ${MODEL} (coarse | fine)"; exit 1
        ;;
esac
echo "done $(date)"
