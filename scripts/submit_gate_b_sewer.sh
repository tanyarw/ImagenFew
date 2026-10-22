#!/bin/bash
#SBATCH --job-name=gate_b_sewer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G

# ==============================================================================
# scripts/gate_b_sewer.py on the cluster: the named series under BC and EFD, every calendar year from empty tanks,
# with flood-control's code and .venv unchanged (CPU only). For checks too long for the laptop.
#
#   sbatch scripts/submit_gate_b_sewer.sh SERIES [SERIES ...]
#   ... --dependency=afterok:<generation job> ...    to start once the series exist
#
# About 40 s per simulated year on 23 workers. Results: results/reference/gate_b_sewer.json (rows of the named
# series replaced, all others kept). Do not run gate_b_sewer.py elsewhere at the same time: both rewrite that file.
# ==============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
IMAGENFEW="$(pwd)"
FLOOD_CONTROL="${FLOOD_CONTROL:-${HOME}/thesis/flood-control}"
[ -x "${FLOOD_CONTROL}/.venv/bin/python" ] || {
    echo "ERROR: ${FLOOD_CONTROL}/.venv/bin/python not found. Set FLOOD_CONTROL to the flood-control checkout"
    echo "       (with 'uv sync --frozen' done once on the login node)."; exit 1; }
[ $# -gt 0 ] || { echo "ERROR: name the series"; exit 1; }
for s in "$@"; do
    f="results/generated_data/rainfall_synthetic_10y_${s%%=*}.csv"
    [ -f "${f}" ] || { echo "ERROR: ${f} not found"; exit 1; }
done
mkdir -p logs

CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"          # read before OMP_NUM_THREADS=1: nproc obeys it
WORKERS=$(( CPUS > 1 ? CPUS - 1 : 1 ))
echo "Job ID: ${SLURM_JOB_ID:-} | Host: $(hostname -s) | CPUs: ${CPUS} | workers ${WORKERS} | Start: $(date)"
echo "ImagenFew commit $(git rev-parse HEAD); flood-control commit $(git -C "${FLOOD_CONTROL}" rev-parse HEAD)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${FLOOD_CONTROL}"
PYTHONPATH=. .venv/bin/python "${IMAGENFEW}/scripts/gate_b_sewer.py" "$@" \
    --controllers BC EFD --flood_control "${FLOOD_CONTROL}" --workers "${WORKERS}"

echo "Job completed at: $(date)"
