#!/usr/bin/env bash
# ==============================================================================
# Run Vanilla ARIMA Baseline Generation (10-Year Realization, Holdout 2000–2007)
# ==============================================================================
# Fits classical ARIMA on the 2000–2007 training split and generates a 10-year
# continuous synthetic precipitation realization.
#
# Usage:
#   chmod +x scripts/run_arima_generation.sh
#   bash scripts/run_arima_generation.sh [years] [seed] [order_p]
# ==============================================================================

set -euo pipefail

YEARS="${1:-10}"
SEED="${2:-42}"
P="${3:-36}"

OUTPUT_CSV="results/generated_data/rainfall_synthetic_10y_arima.csv"
CARD_JSON="results/reference/arima_model_card.json"

echo "═══════════════════════════════════════════════════════════"
echo "  Vanilla ARIMA Synthetic Generation Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo "  Training Partition : data/rainfall/splits/train_years_labelled.csv"
echo "  Validation Split   : data/rainfall/splits/val_years_labelled.csv"
echo "  Years to Generate  : ${YEARS}"
echo "  Random Seed        : ${SEED}"
echo "  AR Order (p)       : ${P}"
echo "  Output CSV         : ${OUTPUT_CSV}"
echo "  Model Card         : ${CARD_JSON}"
echo "═══════════════════════════════════════════════════════════"

python3 scripts/baselines/run_vanilla_arima.py \
    --train-csv "data/rainfall/splits/train_years_labelled.csv" \
    --val-csv "data/rainfall/splits/val_years_labelled.csv" \
    --years "${YEARS}" \
    --p "${P}" \
    --seed "${SEED}" \
    --output-csv "${OUTPUT_CSV}" \
    --card-json "${CARD_JSON}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✓ Generation complete: ${OUTPUT_CSV}"
echo "═══════════════════════════════════════════════════════════"
