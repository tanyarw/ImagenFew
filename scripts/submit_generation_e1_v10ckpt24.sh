#!/bin/bash
#SBATCH --job-name=gen_e1_v10ckpt24
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --nodelist=xgpg[0-7]

RUN_ID="${1:-e1_v10ckpt24}"
YEARS="${2:-10}"
SEED="${3:-42}"
MODE="${4:-both}"

mkdir -p logs
mkdir -p results/generated_data
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES | Host: $(hostname -s)"
echo "E1 Control Run A Synthetic Generation (seq_len = 64, ImagenFew_24.ckpt init)"
echo "Run ID:        $RUN_ID"
echo "Years:         $YEARS | Freq: 5min | Seed: $SEED | Mode: $MODE"
echo "Start Time:    $(date)"
echo "=========================================================="

# Fail fast if CUDA is not functional on the allocated node
python -c "import torch; assert torch.cuda.is_available(), 'ERROR: CUDA is not available on ' + '$(hostname -s)'"

bash scripts/run_e1_v10ckpt24_generation.sh "$RUN_ID" "$YEARS" "$SEED" "$MODE"

echo ""
echo "=========================================================="
echo "Generation complete! Output files in results/generated_data/:"
ls -lh results/generated_data/rainfall_synthetic_10y_e1_v10ckpt24*.csv
echo "End Time: $(date)"
echo "=========================================================="
