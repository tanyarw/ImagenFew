#!/bin/bash
#SBATCH --job-name=tr_stage
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1

# ==============================================================================
# Two-resolution generator: generate and judge one stage (code_plan/TWO_RESOLUTION_PLAN.md, sections 3-5).
# Sent by tworesolution/slurm/submit.sh; arguments: 1 | 2 | 3, then optionally the coarse run (default
# twores_coarse30; twores_coarse30_long for the follow-up, whose outputs carry "_long"). Results in
# results/tworesolution/.
#
#   1  10 years of coarse totals (4 members) -> stage1.json: does the coarse model carry long storms?
#   2  the fine model and the cascade given the real 2008-09 totals -> stage2.json
#   3  coarse -> fine (and/or cascade, as stage 2 decided) -> 10-year 5-minute series
#      results/generated_data/rainfall_synthetic_10y_twores_{v1,cascade}[_mK].csv -> stage3.json
#      Refuses to run unless stages 1 and 2 passed (tworesolution/evaluate.py, stage3_methods).
# ==============================================================================
set -euo pipefail
STAGE="${1:?usage: stage.sbatch 1|2|3 [COARSE_RUN]}"
RUN="${2:-twores_coarse30}"
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs
echo "job ${SLURM_JOB_ID:-} | $(hostname -s) | GPU ${CUDA_VISIBLE_DEVICES:-} | $(date)"
echo "commit $(git rev-parse HEAD) $([ -z "$(git status --porcelain --untracked-files=no)" ] || echo '(UNCOMMITTED CHANGES)')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
python -m tworesolution build-data > /dev/null
python -m tworesolution check

case "${STAGE}" in
    1)
        python -m tworesolution generate-coarse --coarse-run "${RUN}"
        python -m tworesolution evaluate --stage 1 --workers 8 --coarse-run "${RUN}"
        ;;
    2)
        python -m tworesolution disaggregate --source heldout --method fine
        python -m tworesolution disaggregate --source heldout --method cascade
        python -m tworesolution evaluate --stage 2 --workers 8
        ;;
    3)
        METHODS=$(python -m tworesolution stage3-methods --coarse-run "${RUN}")
        for M in ${METHODS}; do
            python -m tworesolution disaggregate --source coarse --method "${M}" --coarse-run "${RUN}"
        done
        python -m tworesolution evaluate --stage 3 --coarse-run "${RUN}"
        ;;
    *)
        echo "unknown stage ${STAGE} (1 | 2 | 3)"; exit 1
        ;;
esac
echo "done $(date)"
