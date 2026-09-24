#!/bin/bash
#SBATCH --job-name=gen_arima
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal

mkdir -p logs
mkdir -p results/generated_data
mkdir -p results/reference
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "Job ID: $SLURM_JOB_ID | Host: $(hostname -s) | Start: $(date)"

bash scripts/run_arima_generation.sh 10 42 36

echo "Job completed at: $(date)"
