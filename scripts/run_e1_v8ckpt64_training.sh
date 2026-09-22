#!/usr/bin/env bash
# ==============================================================================
# E1 Control Run B — seq_len=24 (v8's config) initialised from ImagenFew_64.ckpt
# ==============================================================================
# Decouples context length from checkpoint choice (code_plan/AUDIT_2026-09-22.md
# §3.1, §6 E1). v8 was fine-tuned at seq_len=24 from ImagenFew_24.ckpt; this run
# uses the identical config_v8.yaml (seq_len=24, same split, same loss) but
# initialises from ImagenFew_64.ckpt instead — the checkpoint v10 used.
#
# Paired with scripts/run_e1_v10ckpt24_training.sh (Control Run A). If this run's
# storm-count ratio and hourly ACF RMSE land near v8's (~1.21, ~0.086), context
# length is confirmed as the cause of v10's improvement. If they instead land near
# v10's (~1.00, ~0.049), the improvement tracks the checkpoint, not the window,
# and the headline claim must be withdrawn.
#
# Usage:
#   chmod +x scripts/run_e1_v8ckpt64_training.sh
#   bash scripts/run_e1_v8ckpt64_training.sh [epochs]
# ==============================================================================

set -euo pipefail

EPOCHS="${1:-500}"
BASE_CKPT="models_ckpt/ImagenFew/ImagenFew_64.ckpt"   # <-- the only thing that differs from v8
CONFIG="regime_training/config_v8.yaml"                # seq_len=24, unchanged from v8
BATCH_SIZE=2048
LR=0.0001
RUN_ID="e1_v8ckpt64"

echo "═══════════════════════════════════════════════════════════"
echo "  E1 Control Run B: seq_len=24 config, ImagenFew_64.ckpt init"
echo "═══════════════════════════════════════════════════════════"
echo "  Base Checkpoint : ${BASE_CKPT}  (v8 normally uses ImagenFew_24.ckpt)"
echo "  Config File     : ${CONFIG}  (unchanged — seq_len=24)"
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
echo "  E1 Control Run B Complete!"
echo "═══════════════════════════════════════════════════════════"
echo "Outputs saved under: logs/ImagenFew/Rainfall_Regime/${RUN_ID}/"
echo "Check the checkpoint-load log line first:"
echo "  'Loaded 1006 parameters, skipped 2' confirms the checkpoints really are"
echo "  interchangeable, as code_plan/AUDIT_2026-09-22.md §3.1 predicts."
echo "Then generate a 10-year realisation the same way v8's was generated"
echo "(scripts/run_v8_generation.sh, pointed at this run's checkpoint/scaler)"
echo "and score it with:"
echo "  python scripts/gate_a_scorecard.py --reference train"
