#!/usr/bin/env bash
# ==============================================================================
# Clean v11 Unconditional Training Pipeline (Strict Holdout 2000–2007, seq_len = 24)
# ==============================================================================
# Ablation study: fine-tunes ImagenFew unconditionally (n_classes = 0) on the
# clean 2000–2007 holdout split with sequence length 24 (2.0h block).
#
# Usage:
#   chmod +x scripts/run_v11_training.sh
#   bash scripts/run_v11_training.sh [epochs] [run_id]
# ==============================================================================

set -euo pipefail

EPOCHS="${1:-500}"
RUN_ID="${2:-v11}"
BASE_CKPT="models_ckpt/ImagenFew/ImagenFew_24.ckpt"
CONFIG="regime_training/config_v11.yaml"
BATCH_SIZE=2048
LR=0.0001

echo "═══════════════════════════════════════════════════════════"
echo "  Starting Clean v11 Unconditional Training Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo "  Run ID          : ${RUN_ID}"
echo "  Base Checkpoint : ${BASE_CKPT}"
echo "  Config File     : ${CONFIG}"
echo "  Training Split  : data/rainfall/splits/train_years_labelled.csv"
echo "  Validation Split: data/rainfall/splits/val_years_labelled.csv"
echo "  Sequence Length : 24 (2.0 hours)"
echo "  Conditioning    : Unconditional (n_classes = 0)"
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
    --learning_rate "${LR}" \
    --run_id "${RUN_ID}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  v11 Training Complete!"
echo "═══════════════════════════════════════════════════════════"
echo "Outputs saved under: logs/ImagenFew/Rainfall_Regime/${RUN_ID}/"
