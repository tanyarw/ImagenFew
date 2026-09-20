#!/usr/bin/env bash
# ==============================================================================
# Clean v10 Training Pipeline (Strict Holdout 2000–2007, seq_len = 288)
# ==============================================================================
# Fine-tunes ImagenFew (v10) on the clean 2000–2007 seasonal-phase training dataset
# with sequence length 288 (24h full diurnal block).
#
# Usage:
#   chmod +x scripts/run_v10_training.sh
#   bash scripts/run_v10_training.sh [epochs]
# ==============================================================================

set -euo pipefail

EPOCHS="${1:-500}"
BASE_CKPT="models_ckpt/ImagenFew/ImagenFew_24.ckpt"
CONFIG="regime_training/config_v10.yaml"
BATCH_SIZE=2048
LR=0.0001

echo "═══════════════════════════════════════════════════════════"
echo "  Starting Clean v10 Training Pipeline (seq_len = 288)"
echo "═══════════════════════════════════════════════════════════"
echo "  Base Checkpoint : ${BASE_CKPT}"
echo "  Config File     : ${CONFIG}"
echo "  Training Split  : data/rainfall/splits/train_years_labelled.csv"
echo "  Validation Split: data/rainfall/splits/val_years_labelled.csv"
echo "  Sequence Length : 288 (24 hours / full diurnal cycle)"
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
echo "  v10 Training Complete!"
echo "═══════════════════════════════════════════════════════════"
echo "Outputs saved under: logs/ImagenFew/Rainfall_v10/<run_id>/"
