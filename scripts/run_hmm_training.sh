#!/usr/bin/env bash
# ==============================================================================
# HMM-Conditional Training Pipeline
# ==============================================================================
# Run this on a Linux GPU machine to fine-tune ImagenFew on HMM state labels.
#
# Prerequisites:
#   - Python 3.10+ with PyTorch, omegaconf, scikit-learn installed
#   - Pre-trained checkpoint at models_ckpt/ImagenFew/ImagenFew_24.ckpt
#   - Raw HMM datasets in data/rainfall/astlingen/hmm_datasets/
#
# Usage:
#   chmod +x scripts/run_hmm_training.sh
#   bash scripts/run_hmm_training.sh              # 105120-fit (recommended)
#   bash scripts/run_hmm_training.sh 365           # 365-fit variant
# ==============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
HMM_VARIANT="${1:-105120}"        # 105120, 105120_v2, or 365
EPOCHS="${2:-500}"                # 500 epochs (~12.5h, fully converged) or 1001 (~25.5h)
BASE_CKPT="models_ckpt/ImagenFew/ImagenFew_24.ckpt"
CONFIG="regime_training/config_hmm.yaml"
BATCH_SIZE=2048
LR=0.0001

echo "═══════════════════════════════════════════════════════════"
echo "  HMM-Conditional Training Pipeline (variant=${HMM_VARIANT})"
echo "═══════════════════════════════════════════════════════════"

# ── Step 1: Prepare HMM Dataset ──────────────────────────────────────
echo ""
echo "[Step 1/3] Preparing HMM dataset (5min resolution, transition matrix)..."
python scripts/prepare_hmm_dataset.py \
    --hmm_variant "${HMM_VARIANT}" \
    --resolution 5min \
    --block_size_steps 24 \
    --validate

# If using a variant other than default 105120, update config paths on the fly
if [ "${HMM_VARIANT}" != "105120" ]; then
    echo ""
    echo "[INFO] Using ${HMM_VARIANT} variant — overriding config paths..."
    # Create a temporary config with variant paths
    sed "s/hmm_105120/hmm_${HMM_VARIANT}/g" "${CONFIG}" > "/tmp/config_hmm_${HMM_VARIANT}.yaml"
    CONFIG="/tmp/config_hmm_${HMM_VARIANT}.yaml"
fi

# ── Step 2: Fine-tune Model ──────────────────────────────────────────
echo ""
echo "[Step 2/3] Fine-tuning ImagenFew on HMM states..."
echo "  Checkpoint : ${BASE_CKPT}"
echo "  Config     : ${CONFIG}"
echo "  Epochs     : ${EPOCHS}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  LR         : ${LR}"
echo ""

python regime_training/train_regime.py \
    --model_ckpt "${BASE_CKPT}" \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}"

# ── Step 3: Locate outputs ───────────────────────────────────────────
echo ""
echo "[Step 3/3] Training complete!"
echo ""
echo "Outputs saved under: logs/ImagenFew/Rainfall_HMM/<run_id>/"
echo "  - best_regime_model.pt   (best checkpoint)"
echo "  - final_regime_model.pt  (final checkpoint)"
echo "  - scaler.pkl             (for inverse scaling during generation)"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Next: Generate synthetic data with:"
echo ""
echo "  python regime_training/generate_hmm_v1.py \\"
echo "      --model_ckpt logs/ImagenFew/Rainfall_HMM/<run_id>/best_regime_model.pt \\"
echo "      --scaler_path logs/ImagenFew/Rainfall_HMM/<run_id>/scaler.pkl \\"
echo "      --years 10 --freq 5min --hmm_variant ${HMM_VARIANT}"
echo "═══════════════════════════════════════════════════════════"
