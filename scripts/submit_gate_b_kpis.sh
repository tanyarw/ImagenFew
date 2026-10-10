#!/bin/bash
#SBATCH --job-name=gate_b_kpis
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G

# ==============================================================================
# Strengthening run 4 (code_plan/STRENGTHENING_RUNS.md): the full Gate B table.
#   1. scripts/gate_b_sewer.py: overflow volume and events for the storm-and-cell baseline
#      (bartlett_lewis_relaxed_wet), the one main baseline never put through the sewer.
#   2. scripts/gate_b_kpis.py: the remaining Gate B KPIs (peak spill rate, tank levels,
#      largest event per year, time stressed) for real, v14 and both
#      baselines, under BC and EFD.
# Both use flood-control's code and .venv unchanged (pyswmm), CPU only.
#
#   FLOOD_CONTROL=~/flood-control sbatch scripts/submit_gate_b_kpis.sh
#   FLOOD_CONTROL=~/flood-control sbatch scripts/submit_gate_b_kpis.sh v17_ctx_cal v14_s1_nobridge_cal
#       (extra arguments: more series, added to the KPI file once their CSVs exist)
#
# About 80 simulated years at ~40 s each on 23 workers: ~3 min.
# Results: results/reference/gate_b_sewer.json, results/reference/gate_b_kpis.json
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
IMAGENFEW="$(pwd)"
FLOOD_CONTROL="${FLOOD_CONTROL:-${HOME}/flood-control}"
[ -x "${FLOOD_CONTROL}/.venv/bin/python" ] || {
    echo "ERROR: ${FLOOD_CONTROL}/.venv/bin/python not found. Set FLOOD_CONTROL to the flood-control checkout"
    echo "       (with 'uv sync --frozen' done once on the login node)."; exit 1; }
mkdir -p logs

DEFAULT="v14_nobridge_cal=v14 bartlett_lewis_relaxed_wet arima_copula_occ_len64"
SERIES="${*:-${DEFAULT}}"
for s in ${SERIES}; do
    f="results/generated_data/rainfall_synthetic_10y_${s%%=*}.csv"
    [ -f "${f}" ] || { echo "ERROR: ${f} not found"; exit 1; }
done

echo "Job ID: ${SLURM_JOB_ID:-} | Host: $(hostname -s) | CPUs: $(nproc) | Start: $(date)"
echo "ImagenFew commit $(git rev-parse HEAD); flood-control commit $(git -C "${FLOOD_CONTROL}" rev-parse HEAD)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKERS=$(( $(nproc) - 1 ))

cd "${FLOOD_CONTROL}"
if [ $# -eq 0 ]; then
    PYTHONPATH=. .venv/bin/python "${IMAGENFEW}/scripts/gate_b_sewer.py" bartlett_lewis_relaxed_wet \
        --controllers BC EFD --flood_control "${FLOOD_CONTROL}" --workers "${WORKERS}"
fi
PYTHONPATH=. .venv/bin/python "${IMAGENFEW}/scripts/gate_b_kpis.py" ${SERIES} \
    --controllers BC EFD --flood_control "${FLOOD_CONTROL}" --workers "${WORKERS}"

echo "Job completed at: $(date)"
