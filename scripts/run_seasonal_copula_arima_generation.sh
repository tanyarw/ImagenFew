#!/usr/bin/env bash
# ==============================================================================
# run_seasonal_copula_arima_generation.sh
# ==============================================================================
# Generates 10-year synthetic rainfall realizations using Seasonal Gaussian-Copula AR
# conditioned on the 4 latent seasonal regimes, under both Markov and Calendar assembly.
#
# Usage:
#   bash scripts/run_seasonal_copula_arima_generation.sh [24|64|all] [markov|calendar|both]
# ==============================================================================

set -euo pipefail

P_ORDER="${1:-64}"
ASSEMBLY_MODE="${2:-both}"
YEARS=10
SEED=42

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  SEASONAL GAUSSIAN-COPULA AR GENERATION (p=${P_ORDER}, mode=${ASSEMBLY_MODE})"
echo "═══════════════════════════════════════════════════════════════════════════"

PYTHON=".venv/bin/python"
if [ ! -f "${PYTHON}" ]; then
    PYTHON="python3"
fi

if [ "${P_ORDER}" = "all" ]; then
    ORDERS=(24 64)
else
    ORDERS=("${P_ORDER}")
fi

for p in "${ORDERS[@]}"; do
    echo ""
    echo "▶ Generating Seasonal Copula AR(p=${p}) with assembly mode: ${ASSEMBLY_MODE}"
    "${PYTHON}" scripts/baselines/run_seasonal_copula_arima.py \
        --p "${p}" \
        --assembly-mode "${ASSEMBLY_MODE}" \
        --years "${YEARS}" \
        --seed "${SEED}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  RUNNING GATE A SCORECARD EVALUATION"
echo "═══════════════════════════════════════════════════════════════════════════"

if [ "${P_ORDER}" = "64" ] || [ "${P_ORDER}" = "all" ]; then
    echo "▶ Evaluating p=64 baselines against headline diffusion (v10 / v10_cal):"
    "${PYTHON}" scripts/gate_a_scorecard.py \
        --reference train \
        --versions v10 v10_cal arima_copula_len64 arima_copula_seas_markov_len64 arima_copula_seas_cal_len64
fi

if [ "${P_ORDER}" = "24" ] || [ "${P_ORDER}" = "all" ]; then
    echo ""
    echo "▶ Evaluating p=24 baselines against base diffusion (v8 / v8_cal):"
    "${PYTHON}" scripts/gate_a_scorecard.py \
        --reference train \
        --versions v8 v8_cal arima_copula_len24 arima_copula_seas_markov_len24 arima_copula_seas_cal_len24
fi
