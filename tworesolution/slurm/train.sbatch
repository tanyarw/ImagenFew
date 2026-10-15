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
# Sent by tworesolution/slurm/submit.sh; argument: coarse | coarse_long | fine.
#
#   coarse  regime_training/train_regime.py, v14's recipe on 30-minute totals, from ImagenFew_64.ckpt
#           -> logs/ImagenFew/Rainfall_Regime/twores_coarse30/
#   coarse_long  the same for v14's 205,000 optimiser steps (3,015 epochs; plan section 11; about 11 h,
#           sent with --time 16:00:00) -> logs/ImagenFew/Rainfall_Regime/twores_coarse30_long/
#   fine    python -m tworesolution train-fine, v14 + the half-hour totals as an extra input, from v14
#           -> logs/ImagenFew/Rainfall_Regime/twores_fine/
# coarse and fine: 500 epochs x 68 steps (about 34,000 optimiser steps); all seed 0.
# ==============================================================================
set -euo pipefail
MODEL="${1:?usage: train.sbatch coarse|coarse_long|fine}"
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
    coarse|coarse_long)
        if [ "${MODEL}" = "coarse" ]; then RUN=twores_coarse30; CFG=coarse30; else RUN=twores_coarse30_long; CFG=coarse30_long; fi
        RUN_DIR="logs/ImagenFew/Rainfall_Regime/${RUN}"
        [ ! -e "${RUN_DIR}/best_regime_model.pt" ] || { echo "ERROR: ${RUN_DIR} already holds a model"; exit 1; }
        python regime_training/train_regime.py --model_ckpt models_ckpt/ImagenFew/ImagenFew_64.ckpt \
            --config "tworesolution/configs/${CFG}.yaml" --seed 0 --run_id "${RUN}"
        ;;
    fine)
        python -m tworesolution train-fine --seed 0
        ;;
    *)
        echo "unknown model ${MODEL} (coarse | coarse_long | fine)"; exit 1
        ;;
esac
echo "done $(date)"
