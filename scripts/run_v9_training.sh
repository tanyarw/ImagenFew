#!/usr/bin/env bash
# ==============================================================================
# Clean v9 Training Pipeline (Strict Holdout 2000–2007, seq_len = 36)
# ==============================================================================
# Fine-tunes ImagenFew (v9) on the clean 2000–2007 seasonal-phase training dataset
# with sequence length 36 (3h block).
#
# Usage:
#   chmod +x scripts/run_v9_training.sh
#   bash scripts/run_v9_training.sh [epochs]
# ==============================================================================

set -euo pipefail

EPOCHS="${1:-500}"
BASE_CKPT="models_ckpt/ImagenFew/ImagenFew_36.ckpt"
CONFIG="regime_training/config_v9.yaml"
BATCH_SIZE=2048
LR=0.0001

echo "═══════════════════════════════════════════════════════════"
echo "  Starting Clean v9 Training Pipeline (seq_len = 36)"
echo "═══════════════════════════════════════════════════════════"
echo "  Base Checkpoint : ${BASE_CKPT}"
echo "  Config File     : ${CONFIG}"
echo "  Training Split  : data/rainfall/splits/train_years_labelled.csv"
echo "  Validation Split: data/rainfall/splits/val_years_labelled.csv"
echo "  Sequence Length : 36 (3 hours)"
echo "  Epochs          : ${EPOCHS}"
echo "  Batch Size      : ${BATCH_SIZE}"
echo "  Learning Rate   : ${LR}"
echo "═══════════════════════════════════════════════════════════"

# Verify files exist
if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: Base checkpoint not found at ${BASE_CKPT}!"
    exit 1
fi

if [ ! -f "data/rainfall/splits/train_years_labelled.csv" ]; then
    echo "ERROR: Clean training split not found at data/rainfall/splits/train_years_labelled.csv!"
    exit 1
fi

python regime_training/train_regime.py \
    --model_ckpt "${BASE_CKPT}" \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  v9 Training Complete!"
echo "═══════════════════════════════════════════════════════════"
echo "Outputs saved under: logs/ImagenFew/Rainfall_v9/<run_id>/"
