#!/usr/bin/env bash
# ==============================================================================
# Run Gaussian-Copula AR Baseline Generation (10-Year Realization, Holdout 2000–2007)
# ==============================================================================
# Fits Gaussian-Copula AR on 2000–2007 train split and generates a 10-year synthetic
# realization with exact empirical zero-fraction and physical extremes.
#
# Usage:
#   chmod +x scripts/run_copula_arima_generation.sh
#   bash scripts/run_copula_arima_generation.sh [order_p] [years] [seed]
#
# Examples:
#   bash scripts/run_copula_arima_generation.sh 24   # mirrors v8 (2.0h)
#   bash scripts/run_copula_arima_generation.sh 64   # mirrors v10 (5.33h)
# ==============================================================================

set -euo pipefail

P="${1:-64}"
YEARS="${2:-10}"
SEED="${3:-42}"

echo "═══════════════════════════════════════════════════════════"
echo "  Gaussian-Copula AR(p=${P}) Synthetic Generation Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo "  Training Partition : data/rainfall/splits/train_years_labelled.csv"
echo "  Validation Split   : data/rainfall/splits/val_years_labelled.csv"
echo "  AR Order (p)       : ${P} (matching diffusion receptive field)"
echo "  Years to Generate  : ${YEARS}"
echo "  Random Seed        : ${SEED}"
echo "═══════════════════════════════════════════════════════════"

python3 scripts/baselines/run_copula_arima.py \
    --train-csv "data/rainfall/splits/train_years_labelled.csv" \
    --val-csv "data/rainfall/splits/val_years_labelled.csv" \
    --p "${P}" \
    --years "${YEARS}" \
    --seed "${SEED}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✓ Generation complete for arima_copula_len${P}!"
echo "═══════════════════════════════════════════════════════════"
