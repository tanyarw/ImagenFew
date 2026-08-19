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
HMM_VARIANT="${1:-105120}"        # 105120 (default) or 365
BASE_CKPT="models_ckpt/ImagenFew/ImagenFew_24.ckpt"
CONFIG="regime_training/config_hmm.yaml"
EPOCHS=1001
BATCH_SIZE=2048
LR=0.0001

echo "═══════════════════════════════════════════════════════════"
echo "  HMM-Conditional Training Pipeline (variant=${HMM_VARIANT})"
echo "═══════════════════════════════════════════════════════════"

# ── Step 1: Prepare HMM Dataset ──────────────────────────────────────
echo ""
echo "[Step 1/3] Preparing HMM dataset (resample 5min→10min, transition matrix)..."
python scripts/prepare_hmm_dataset.py \
    --hmm_variant "${HMM_VARIANT}" \
    --validate

# If using 365-fit, update config paths on the fly
if [ "${HMM_VARIANT}" = "365" ]; then
    echo ""
    echo "[INFO] Using 365-fit variant — overriding config paths..."
    # Create a temporary config with 365 paths
    sed "s/hmm_105120/hmm_365/g" "${CONFIG}" > /tmp/config_hmm_365.yaml
    CONFIG="/tmp/config_hmm_365.yaml"
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
echo "      --years 10 --hmm_variant ${HMM_VARIANT}"
echo "═══════════════════════════════════════════════════════════"
