#!/usr/bin/env bash
# ==============================================================================
# Clean v12 Unconditional Synthetic Generation (10-Year Realization, seq_len = 64)
# ==============================================================================
# Generates 10-year synthetic rainfall realization from the clean v12
# unconditional checkpoint (fine-tuned without class conditioning on 2000–2007
# holdout split with seq_len = 64 / 5.33h block).
#
# Usage:
#   chmod +x scripts/run_v12_generation.sh
#   bash scripts/run_v12_generation.sh [run_id] [years] [seed]
# ==============================================================================

set -euo pipefail

RUN_ID="${1:-v12}"
YEARS="${2:-10}"
SEED="${3:-42}"

# Support both logs/ImagenFew/Rainfall_Regime/${RUN_ID} and logs/ImagenFew/Rainfall_v12/${RUN_ID}
if [ -f "logs/ImagenFew/Rainfall_v12/${RUN_ID}/best_regime_model.pt" ]; then
    MODEL_CKPT="logs/ImagenFew/Rainfall_v12/${RUN_ID}/best_regime_model.pt"
    SCALER_PATH="logs/ImagenFew/Rainfall_v12/${RUN_ID}/scaler.pkl"
elif [ -f "logs/ImagenFew/Rainfall_Regime/${RUN_ID}/best_regime_model.pt" ]; then
    MODEL_CKPT="logs/ImagenFew/Rainfall_Regime/${RUN_ID}/best_regime_model.pt"
    SCALER_PATH="logs/ImagenFew/Rainfall_Regime/${RUN_ID}/scaler.pkl"
else
    MODEL_CKPT="logs/ImagenFew/Rainfall_v12/${RUN_ID}/best_regime_model.pt"
    SCALER_PATH="logs/ImagenFew/Rainfall_v12/${RUN_ID}/scaler.pkl"
fi

CONFIG="regime_training/config_v12.yaml"
OUTPUT_DIR="results/generated_data"
OUTPUT_NAME="rainfall_synthetic_10y_v12.csv"

echo "═══════════════════════════════════════════════════════════"
echo "  Clean v12 Unconditional Synthetic Generation Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo "  Run ID            : ${RUN_ID}"
echo "  Model Checkpoint  : ${MODEL_CKPT}"
echo "  Scaler Path       : ${SCALER_PATH}"
echo "  Config File       : ${CONFIG}"
echo "  Assembly Mode     : unconditional"
echo "  Years             : ${YEARS}"
echo "  Seed              : ${SEED}"
echo "  Output File       : ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "═══════════════════════════════════════════════════════════"

# Validate required files exist
if [ ! -f "${MODEL_CKPT}" ]; then
    echo "ERROR: Checkpoint not found at ${MODEL_CKPT}!"
    exit 1
fi

if [ ! -f "${SCALER_PATH}" ]; then
    echo "ERROR: Scaler not found at ${SCALER_PATH}!"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "▶ Generating v12 with UNCONDITIONAL assembly (n_classes = 0, seq_len = 64 ablation of v10)..."
python regime_training/generate_hmm_v1.py \
    --model_ckpt "${MODEL_CKPT}" \
    --scaler_path "${SCALER_PATH}" \
    --config "${CONFIG}" \
    --assembly_mode "unconditional" \
    --years "${YEARS}" \
    --freq "5min" \
    --overlap 4 \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "${OUTPUT_NAME}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✓ Generation complete: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "═══════════════════════════════════════════════════════════"
