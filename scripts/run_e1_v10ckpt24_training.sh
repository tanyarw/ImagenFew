#!/usr/bin/env bash
# ==============================================================================
# E1 Control Run A — seq_len=64 (v10's config) initialised from ImagenFew_24.ckpt
# ==============================================================================
# Decouples context length from checkpoint choice (code_plan/AUDIT_2026-09-22.md
# §3.1, §6 E1). v10 was fine-tuned at seq_len=64 from ImagenFew_64.ckpt; this run
# uses the identical config_v10.yaml (seq_len=64, same split, same loss) but
# initialises from ImagenFew_24.ckpt instead — the checkpoint v8 used. All four
# base checkpoints are byte-identical in size and every prior run has loaded
# 1006/1008 parameters regardless of which one was used, so this should load
# cleanly; if it doesn't, that itself is new information.
# If this run's storm-count ratio and hourly ACF RMSE land near v10's
# (~1.00, ~0.049), context length is confirmed as the cause of v10's
# improvement over v8. If they land near v8's (~1.21, ~0.086) instead,
# the improvement tracks the checkpoint, not the window.
#
# Usage:
#   chmod +x scripts/run_e1_v10ckpt24_training.sh
#   bash scripts/run_e1_v10ckpt24_training.sh [epochs]
# ==============================================================================

set -euo pipefail

EPOCHS="${1:-500}"
BASE_CKPT="models_ckpt/ImagenFew/ImagenFew_24.ckpt"   # <-- the only thing that differs from v10
CONFIG="regime_training/config_v10.yaml"               # seq_len=64, unchanged from v10
BATCH_SIZE=2048
LR=0.0001
RUN_ID="e1_v10ckpt24"

echo "═══════════════════════════════════════════════════════════"
echo "  E1 Control Run A: seq_len=64 config, ImagenFew_24.ckpt init"
echo "═══════════════════════════════════════════════════════════"
echo "  Base Checkpoint : ${BASE_CKPT}  (v10 normally uses ImagenFew_64.ckpt)"
echo "  Config File     : ${CONFIG}  (unchanged — seq_len=64)"
echo "  Run ID          : ${RUN_ID}"
echo "  Training Split  : data/rainfall/splits/train_years_labelled.csv"
echo "  Validation Split: data/rainfall/splits/val_years_labelled.csv"
echo "  Epochs          : ${EPOCHS}"
echo "  Batch Size      : ${BATCH_SIZE}"
echo "  Learning Rate   : ${LR}"
echo "═══════════════════════════════════════════════════════════"

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
    --run_id "${RUN_ID}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  E1 Control Run A Complete!"
echo "═══════════════════════════════════════════════════════════"
echo "Outputs saved under: logs/ImagenFew/Rainfall_Regime/${RUN_ID}/"
echo "Check the checkpoint-load log line first:"
echo "  'Loaded 1006 parameters, skipped 2' confirms the checkpoints really are"
echo "  interchangeable, as code_plan/AUDIT_2026-09-22.md §3.1 predicts."
echo "Then generate a 10-year realisation the same way v10's was generated"
echo "(scripts/run_v10_generation.sh, pointed at this run's checkpoint/scaler)"
echo "and score it with:"
echo "  python scripts/gate_a_scorecard.py --reference train"
