#!/bin/bash
#SBATCH --job-name=gen_v11
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --nodelist=xgpg[0-7]

RUN_ID="${1:-v11}"
YEARS="${2:-10}"
SEED="${3:-42}"

mkdir -p logs
mkdir -p results/generated_data
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s)"
echo "Clean v11 Synthetic Generation (Holdout Split 2000–2007, seq_len = 64, Unconditional)"
echo "Run ID:        $RUN_ID"
echo "Years:         $YEARS | Freq: 5min | Seed: $SEED"
echo "Start Time:    $(date)"
echo "=========================================================="

# Fast CUDA sanity check to fail immediately if driver is unhealthy
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

bash scripts/run_v11_generation.sh "$RUN_ID" "$YEARS" "$SEED"

echo ""
echo "=========================================================="
echo "Generation complete! Output file in results/generated_data/:"
ls -lh results/generated_data/rainfall_synthetic_10y_v11.csv
echo "End Time: $(date)"
echo "=========================================================="
