"""
Regime-Conditional Synthetic Rainfall Generator (v5 — Transition-Aware Markovian Block Assembly)
================================================================================================
Generates synthetic rainfall data conditioned on gmm_regime labels using a regime-trained
ImagenFew checkpoint. Replaces random permutation shuffling (v4) with 1st-order Markovian
sequential regime sampling P(R_{t+1} | R_t) and Overlap-Add (OLA) boundary smoothing.

Usage
-----
  # Transition-Aware Mixed mode (default)
  python regime_training/generate_regime_v5.py \\
      --model_ckpt logs/ImagenFew/Rainfall_Regime/<run_id>/best_regime_model.pt \\
      --scaler_path logs/ImagenFew/Rainfall_Regime/<run_id>/scaler.pkl \\
      --years 10

  # Custom overlap and transition granularity
  python regime_training/generate_regime_v5.py \\
      --model_ckpt ... --scaler_path ... --years 10 --overlap 4 --granularity window
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.ImagenFew.ImagenFew import ImagenFew
from models.ImagenFew.sampler import DiffusionProcess
from scripts.estimate_transition_matrix import estimate_transition_matrix

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Transition-Aware Markovian Rainfall Generator (v5)")
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to the regime-trained checkpoint")
    p.add_argument("--scaler_path", type=str, required=True,
                   help="Path to scaler.pkl saved during training")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    p.add_argument("--years", type=float, default=10.0,
                   help="Years of synthetic data to generate")
    p.add_argument("--regime", type=int, default=None, choices=[0, 1, 2, 3],
                   help="Generate only this regime. If omitted, sample via Markov P_ij.")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "results", "generated_data"))
    p.add_argument("--transition_matrix_path", type=str, default=None,
                   help="Path to pre-computed regime_transition_matrix.pkl")
    p.add_argument("--overlap", type=int, default=4,
                   help="Overlap-add boundary smoothing window in steps (default: 4 steps = 40 mins)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────
# Markov Sampling & Overlap-Add Functions
# ──────────────────────────────────────────────────────────────────────

def sample_markov_regime_sequence(transition_matrix, initial_dist, length, seed=42):
    """
    Samples a sequence of regime labels (R_1, R_2, ..., R_N) of given length
    using a 1st-order discrete-time Markov chain.
    """
    rng = np.random.default_rng(seed)
    n_states = transition_matrix.shape[0]
    
    sequence = np.zeros(length, dtype=int)
    # Sample initial state R_1
    sequence[0] = rng.choice(n_states, p=initial_dist)
    
    # Iteratively sample R_{k+1} ~ Categorical(P_{R_k, :})
    for k in range(length - 1):
        curr_state = sequence[k]
        p_row = transition_matrix[curr_state]
        sequence[k + 1] = rng.choice(n_states, p=p_row)
        
    return sequence


def apply_overlap_add_smoothing(blocks, overlap=4):
    """
    Applies Overlap-Add (OLA) raised-cosine cross-fading across adjacent blocks.

    Args:
        blocks  : ndarray of shape [N, L, 1] or [N, L]
        overlap : int, number of overlap steps between adjacent blocks

    Returns:
        continuous 1D array of length (N - 1) * (L - overlap) + L
    """
    if blocks.ndim == 3:
        blocks = blocks.squeeze(-1)
        
    N, L = blocks.shape
    if overlap <= 0 or N == 1:
        return blocks.reshape(-1)

    stride = L - overlap
    total_len = (N - 1) * stride + L

    output = np.zeros(total_len, dtype=np.float32)
    weights = np.zeros(total_len, dtype=np.float32)

    # Raised cosine cross-fade window ramps over 'overlap' steps
    # w_up ramps from 0 to 1 over overlap steps
    m = np.arange(overlap, dtype=np.float32) + 0.5
    ramp_up = 0.5 * (1.0 - np.cos(np.pi * m / overlap))
    ramp_down = 1.0 - ramp_up

    for i in range(N):
        start = i * stride
        end = start + L

        w = np.ones(L, dtype=np.float32)
        if i > 0:          # Ramp up at block head
            w[:overlap] = ramp_up
        if i < N - 1:      # Ramp down at block tail
            w[-overlap:] = ramp_down

        output[start:end] += blocks[i] * w
        weights[start:end] += w

    # Normalize by exact composite weights buffer
    smoothed = output / (weights + 1e-8)
    return smoothed

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    # Config → Namespace
    cfg = OmegaConf.to_container(OmegaConf.load(cli.config), resolve=True)
    args = argparse.Namespace(**cfg)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model_ckpt = cli.model_ckpt

    # ── Scaler ────────────────────────────────────────────────────────
    with open(cli.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logging.info("Loaded scaler from %s", cli.scaler_path)

    # ── Model ─────────────────────────────────────────────────────────
    logging.info("Building model (n_classes=%d) …", args.n_classes)
    model = ImagenFew(args, args.device).to(args.device)

    loaded = torch.load(cli.model_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(loaded["model"], strict=True)
    if "ema_model" in loaded and args.ema:
        model.model_ema.load_state_dict(loaded["ema_model"], strict=True)
    model.eval()
    logging.info("Model loaded from %s", cli.model_ckpt)

    # ── Calculate steps & num_samples with Overlap ────────────────────
    freq = "10min"
    steps_per_year = len(
        pd.date_range(start="2026-01-01", end="2027-01-01",
                      freq=freq, inclusive="left")
    )
    total_steps = int(cli.years * steps_per_year)
    stride = args.seq_len - cli.overlap
    if stride <= 0:
        raise ValueError(f"Overlap ({cli.overlap}) must be strictly smaller than seq_len ({args.seq_len})")

    num_samples = int(np.ceil((total_steps - cli.overlap) / stride))

    # ── Transition Matrix & Regime Sequence Sampling ──────────────────
    if cli.regime is not None:
        regime_sequence = np.full(num_samples, cli.regime, dtype=int)
        tag = f"regime{cli.regime}"
        logging.info("Single-regime mode: locked to Regime %d", cli.regime)
    else:
        # Load or compute transition matrix
        trans_path = cli.transition_matrix_path
        if trans_path is None:
            trans_path = os.path.join(PROJECT_ROOT, "data", "rainfall", "train", "regime_transition_matrix.pkl")

        if os.path.exists(trans_path):
            with open(trans_path, "rb") as f:
                trans_data = pickle.load(f)
            logging.info("Loaded transition matrix from %s", trans_path)
        else:
            labeled_csv = os.path.join(PROJECT_ROOT, "data", "rainfall", "train", "rainfall_10min_labeled.csv")
            logging.info("Transition matrix pickle not found. Estimating from %s ...", labeled_csv)
            trans_data = estimate_transition_matrix(labeled_csv)

        P_mat = trans_data.get("P_block", trans_data["P"])
        pi_dist = trans_data.get("pi_block", trans_data["pi"])

        logging.info("Using block-level transition matrix:\n%s", np.round(P_mat, 5))
        logging.info("Stationary distribution: %s", np.round(pi_dist, 4))

        regime_sequence = sample_markov_regime_sequence(
            P_mat, pi_dist, num_samples, seed=cli.seed
        )
        tag = "markov_v5_block"

    unique_r, counts_r = np.unique(regime_sequence, return_counts=True)
    regime_alloc = {int(r): int(c) for r, c in zip(unique_r, counts_r)}
    logging.info("Generating %.1f years (%d steps, %d blocks, overlap=%d steps)",
                 cli.years, total_steps, num_samples, cli.overlap)
    logging.info("Sampled regime sequence counts: %s", regime_alloc)

    # ── Batch Generation & Re-ordering ────────────────────────────────
    channels = 1
    process = DiffusionProcess(
        args, model.net,
        (channels, args.img_resolution, args.img_resolution),
    )

    # Dictionary to store generated blocks per regime: regime -> list of 3D arrays [L, 1]
    regime_generated_pool = {r: [] for r in range(args.n_classes)}

    with model.ema_scope():
        for regime, n in regime_alloc.items():
            logging.info("  Regime %d  →  generating %d samples …", regime, n)
            for i in range(0, n, args.batch_size):
                b = min(args.batch_size, n - i)
                cls = torch.full((b,), regime, device=args.device, dtype=torch.long)
                oh = nn.functional.one_hot(cls, num_classes=args.n_classes).float()

                x_img = torch.zeros(
                    b, channels, args.img_resolution, args.img_resolution,
                    device=args.device,
                )
                mask = model.ts_to_img(
                    torch.zeros(b, args.seq_len, channels, device=args.device),
                    pad_val=1,
                )
                with torch.no_grad():
                    sampled = process.interpolate(x_img, mask, class_labels=oh)
                    ts = model.img_to_ts(sampled)[:, :, :channels]
                    for sample in ts.cpu().numpy():
                        regime_generated_pool[regime].append(sample)

    # Re-assemble generated blocks into exact Markov sequence positions
    regime_ptrs = {r: 0 for r in range(args.n_classes)}
    ordered_blocks = []
    for r in regime_sequence:
        block = regime_generated_pool[r][regime_ptrs[r]]
        regime_ptrs[r] += 1
        ordered_blocks.append(block)

    ordered_blocks = np.array(ordered_blocks) # [num_samples, seq_len, 1]

    # ── Overlap-Add Boundary Smoothing ────────────────────────────────
    continuous_scaled = apply_overlap_add_smoothing(ordered_blocks, overlap=cli.overlap)
    continuous_scaled = continuous_scaled[:total_steps].reshape(-1, channels)

    # ── Inverse Scaling & Thresholding ────────────────────────────────
    unscaled = scaler.inverse_transform(continuous_scaled)
    unscaled[unscaled < 0.005] = 0.0

    # ── Save Output ───────────────────────────────────────────────────
    os.makedirs(cli.output_dir, exist_ok=True)
    run_id = os.path.splitext(os.path.basename(cli.model_ckpt))[0]
    out_path = os.path.join(
        cli.output_dir,
        f"rainfall_regime_{tag}_{cli.years}y_{run_id}.csv",
    )

    df = pd.DataFrame(unscaled, columns=["avg_rainfall"])
    df.insert(0, "date", pd.date_range(
        start="2026-01-01", periods=len(df), freq=freq,
    ))
    df.to_csv(out_path, index=False)

    # ── Summary Statistics ────────────────────────────────────────────
    zero_frac = (df["avg_rainfall"] == 0).mean()
    nz = df.loc[df["avg_rainfall"] > 0, "avg_rainfall"]
    logging.info("═" * 60)
    logging.info("Saved v5 Markovian Series → %s", out_path)
    logging.info("Shape      : %s", df.shape)
    logging.info("Zero%%      : %.2f%%", zero_frac * 100)
    logging.info("Mean       : %.4f  (non-zero: %.4f)", df["avg_rainfall"].mean(),
                 nz.mean() if len(nz) else 0.0)
    logging.info("Max 10-min : %.4f mm", df["avg_rainfall"].max())
    logging.info("═" * 60)

if __name__ == "__main__":
    main()
