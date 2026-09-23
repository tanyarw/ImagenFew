#!/usr/bin/env bash
# ==============================================================================
# E1 Control Run A Synthetic Generation (10-Year Realizations, seq_len = 64, ImagenFew_24.ckpt init)
# ==============================================================================
# Generates 10-year synthetic rainfall realizations from E1 Control Run A: the
# checkpoint (seq_len = 64/5.33h, same config and split as v10, but initialised
# from ImagenFew_24.ckpt instead of ImagenFew_64.ckpt). See
# code_plan/AUDIT_2026-09-22.md §3.1 / §6 E1 — this is the decoupling experiment.
#
# Supports both:
#   1. Markov assembly (Option A): stochastic random walk via P_block
#   2. Calendar assembly (Option B): deterministic annual seasonal calendar progression
#
# Usage:
#   chmod +x scripts/run_e1_v10ckpt24_generation.sh
#   bash scripts/run_e1_v10ckpt24_generation.sh [run_id] [years] [seed] [mode: both|markov|calendar]
# ==============================================================================

set -euo pipefail

RUN_ID="${1:-e1_v10ckpt24}"
YEARS="${2:-10}"
SEED="${3:-42}"
MODE="${4:-both}"

MODEL_CKPT="logs/ImagenFew/Rainfall_Regime/${RUN_ID}/best_regime_model.pt"
SCALER_PATH="logs/ImagenFew/Rainfall_Regime/${RUN_ID}/scaler.pkl"
CONFIG="regime_training/config_v10.yaml"
TRANS_MATRIX="data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl"
OUTPUT_DIR="results/generated_data"

echo "═══════════════════════════════════════════════════════════"
echo "  E1 Control Run A Synthetic Rainfall Generation Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo "  Run ID            : ${RUN_ID}"
echo "  Model Checkpoint  : ${MODEL_CKPT}"
echo "  Scaler Path       : ${SCALER_PATH}"
echo "  Config File       : ${CONFIG}"
echo "  Transition Matrix : ${TRANS_MATRIX}"
echo "  Years             : ${YEARS}"
echo "  Seed              : ${SEED}"
echo "  Assembly Mode     : ${MODE}"
echo "  Output Directory  : ${OUTPUT_DIR}"
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

if [ ! -f "${TRANS_MATRIX}" ]; then
    echo "ERROR: Transition matrix not found at ${TRANS_MATRIX}!"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ── 1. Markov Assembly ────────────────────────────────────────────────
if [ "${MODE}" = "both" ] || [ "${MODE}" = "markov" ]; then
    echo ""
    echo "▶ [1/2] Generating E1 Control Run A with MARKOV sequence assembly..."
    python regime_training/generate_hmm_v1.py \
        --model_ckpt "${MODEL_CKPT}" \
        --scaler_path "${SCALER_PATH}" \
        --config "${CONFIG}" \
        --transition_matrix_path "${TRANS_MATRIX}" \
        --assembly_mode "markov" \
        --years "${YEARS}" \
        --freq "5min" \
        --overlap 4 \
        --transition_overlap 8 \
        --bridge_blocks 1 \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --output_name "rainfall_synthetic_10y_e1_v10ckpt24.csv"
    echo "✓ Markov generation complete: ${OUTPUT_DIR}/rainfall_synthetic_10y_e1_v10ckpt24.csv"
fi

# ── 2. Calendar-Ordered Assembly ──────────────────────────────────────
if [ "${MODE}" = "both" ] || [ "${MODE}" = "calendar" ]; then
    echo ""
    echo "▶ [2/2] Generating E1 Control Run A with CALENDAR-ORDERED sequence assembly..."
    python regime_training/generate_hmm_v1.py \
        --model_ckpt "${MODEL_CKPT}" \
        --scaler_path "${SCALER_PATH}" \
        --config "${CONFIG}" \
        --transition_matrix_path "${TRANS_MATRIX}" \
        --assembly_mode "calendar" \
        --years "${YEARS}" \
        --freq "5min" \
        --overlap 4 \
        --transition_overlap 8 \
        --bridge_blocks 1 \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --output_name "rainfall_synthetic_10y_e1_v10ckpt24_cal.csv"
    echo "✓ Calendar generation complete: ${OUTPUT_DIR}/rainfall_synthetic_10y_e1_v10ckpt24_cal.csv"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  E1 Control Run A Generation Pipeline Complete!"
echo "═══════════════════════════════════════════════════════════"
ls -lh "${OUTPUT_DIR}"/rainfall_synthetic_10y_e1_v10ckpt24*.csv
