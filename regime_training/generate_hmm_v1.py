"""
HMM-Conditional Synthetic Rainfall Generator (v1 — Transition-Aware Stitching)
================================================================================
Generates synthetic rainfall data conditioned on HMM state labels using
a state-trained ImagenFew checkpoint.

Key features:
  1. Block-level (4-hour) Markov chain sampling of HMM state sequence
  2. Wider Overlap-Add (OLA) at state transition boundaries
  3. Soft-label bridge blocks at transitions for smooth blending
  4. Supports both 105120-fit and 365-fit HMM variants

Usage
-----
  # Default mixed-state generation with transition-aware stitching
  python regime_training/generate_hmm_v1.py \\
      --model_ckpt logs/ImagenFew/Rainfall_HMM/<run_id>/best_regime_model.pt \\
      --scaler_path logs/ImagenFew/Rainfall_HMM/<run_id>/scaler.pkl \\
      --years 10

  # Single-state generation (for debugging/comparison)
  python regime_training/generate_hmm_v1.py \\
      --model_ckpt ... --scaler_path ... --years 1 --state 2

  # Custom overlap and bridge settings
  python regime_training/generate_hmm_v1.py \\
      --model_ckpt ... --scaler_path ... --years 10 \\
      --overlap 4 --transition_overlap 8 --bridge_blocks 1
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


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="HMM-Conditional Rainfall Generator with Transition-Aware Stitching"
    )
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to the HMM state-trained checkpoint")
    p.add_argument("--scaler_path", type=str, required=True,
                   help="Path to scaler.pkl saved during training")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "config_hmm.yaml"))
    p.add_argument("--years", type=float, default=10.0,
                   help="Years of synthetic data to generate")
    p.add_argument("--state", type=int, default=None, choices=[0, 1, 2, 3],
                   help="Generate only this state. If omitted, uses Markov sampling.")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "results", "generated_data"))
    p.add_argument("--transition_matrix_path", type=str, default=None,
                   help="Path to pre-computed HMM transition matrix pickle")
    p.add_argument("--hmm_variant", type=str, default="105120",
                   choices=["105120", "105120_v2", "365"],
                   help="Which HMM variant to use for transition matrix (default: 105120)")

    # Stitching parameters
    p.add_argument("--freq", type=str, default="5min", choices=["5min", "10min"],
                   help="Temporal resolution of generated data (default: '5min')")
    p.add_argument("--overlap", type=int, default=4,
                   help="Base OLA overlap in steps (default: 4 steps)")
    p.add_argument("--transition_overlap", type=int, default=8,
                   help="Wider OLA overlap at state transition boundaries "
                        "(default: 8 steps)")
    p.add_argument("--bridge_blocks", type=int, default=1,
                   help="Number of soft-label bridge blocks to insert at "
                        "state transitions (default: 1, 0 to disable)")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Markov Sampling
# ──────────────────────────────────────────────────────────────────────

def sample_markov_state_sequence(transition_matrix, initial_dist, length, seed=42):
    """
    Samples a sequence of HMM state labels using a 1st-order Markov chain.

    Returns:
        np.ndarray of int, shape [length]
    """
    rng = np.random.default_rng(seed)
    n_states = transition_matrix.shape[0]

    sequence = np.zeros(length, dtype=int)
    sequence[0] = rng.choice(n_states, p=initial_dist)

    for k in range(length - 1):
        p_row = transition_matrix[sequence[k]]
        sequence[k + 1] = rng.choice(n_states, p=p_row)

    return sequence


# ──────────────────────────────────────────────────────────────────────
# Transition-Aware Block Sequence Builder
# ──────────────────────────────────────────────────────────────────────

def build_generation_plan(state_sequence, n_bridge=1):
    """
    Builds an ordered generation plan from the Markov state sequence.
    Inserts soft-label bridge blocks at state transition boundaries.

    Args:
        state_sequence : np.ndarray of int, the Markov-sampled state sequence
        n_bridge       : Number of bridge blocks to insert at each transition

    Returns:
        plan: list of dicts, each with:
            - 'type': 'normal' or 'bridge'
            - 'state': int (for normal blocks) or None (for bridge)
            - 'label_weights': np.ndarray of shape [n_states] (one-hot or soft)
    """
    n_states = 4
    plan = []

    for i, state in enumerate(state_sequence):
        # Add normal block
        oh = np.zeros(n_states, dtype=np.float32)
        oh[state] = 1.0
        plan.append({
            "type": "normal",
            "state": int(state),
            "label_weights": oh,
        })

        # At transition boundaries, insert bridge blocks
        if n_bridge > 0 and i < len(state_sequence) - 1:
            next_state = state_sequence[i + 1]
            if state != next_state:
                for b in range(n_bridge):
                    # Linear interpolation between states
                    alpha = (b + 1) / (n_bridge + 1)
                    soft = np.zeros(n_states, dtype=np.float32)
                    soft[state] = 1.0 - alpha
                    soft[next_state] = alpha
                    plan.append({
                        "type": "bridge",
                        "state": None,
                        "label_weights": soft,
                    })

    return plan


# ──────────────────────────────────────────────────────────────────────
# Overlap-Add Stitching (Transition-Aware)
# ──────────────────────────────────────────────────────────────────────

def stitch_blocks_transition_aware(blocks, plan, base_overlap=4,
                                    transition_overlap=8):
    """
    Overlap-Add stitching with wider overlap at state transition boundaries.

    At boundaries where the state changes (including bridge blocks),
    uses `transition_overlap` instead of `base_overlap`.

    Args:
        blocks             : list of 1D np.ndarray, each of length L
        plan               : generation plan from build_generation_plan()
        base_overlap       : overlap for same-state boundaries
        transition_overlap : overlap for state-change boundaries

    Returns:
        1D np.ndarray — the stitched continuous series
    """
    if len(blocks) <= 1:
        return blocks[0] if blocks else np.array([])

    L = len(blocks[0])
    N = len(blocks)

    # Determine per-boundary overlap
    overlaps = []
    for i in range(N - 1):
        is_transition = (plan[i]["state"] != plan[i + 1]["state"]) or \
                        plan[i]["type"] == "bridge" or \
                        plan[i + 1]["type"] == "bridge"
        ov = transition_overlap if is_transition else base_overlap
        ov = min(ov, L - 1)  # Can't overlap more than block length - 1
        overlaps.append(ov)

    # Calculate total length
    total_len = L  # First block
    for ov in overlaps:
        total_len += L - ov

    output = np.zeros(total_len, dtype=np.float32)
    weights = np.zeros(total_len, dtype=np.float32)

    pos = 0
    for i in range(N):
        ov_before = overlaps[i - 1] if i > 0 else 0
        ov_after = overlaps[i] if i < N - 1 else 0

        w = np.ones(L, dtype=np.float32)

        # Ramp up at block head (if overlapping with previous)
        if ov_before > 0:
            m = np.arange(ov_before, dtype=np.float32) + 0.5
            w[:ov_before] = 0.5 * (1.0 - np.cos(np.pi * m / ov_before))

        # Ramp down at block tail (if overlapping with next)
        if ov_after > 0:
            m = np.arange(ov_after, dtype=np.float32) + 0.5
            w[-ov_after:] = 0.5 * (1.0 + np.cos(np.pi * m / ov_after))

        output[pos:pos + L] += blocks[i] * w
        weights[pos:pos + L] += w

        if i < N - 1:
            pos += L - overlaps[i]

    smoothed = output / (weights + 1e-8)
    return smoothed


# ──────────────────────────────────────────────────────────────────────
# Diffusion Sampling
# ──────────────────────────────────────────────────────────────────────

def generate_block(model, process, label_weights, args, channels=1):
    """
    Generate a single block conditioned on label_weights.

    Args:
        label_weights : np.ndarray of shape [n_states] — one-hot or soft label

    Returns:
        np.ndarray of shape [seq_len, channels]
    """
    with torch.no_grad():
        oh = torch.FloatTensor(label_weights).unsqueeze(0).to(args.device)  # [1, n_states]

        x_img = torch.zeros(
            1, channels, args.img_resolution, args.img_resolution,
            device=args.device,
        )
        mask = model.ts_to_img(
            torch.zeros(1, args.seq_len, channels, device=args.device),
            pad_val=1,
        )
        sampled = process.interpolate(x_img, mask, class_labels=oh)
        ts = model.img_to_ts(sampled)[:, :, :channels]
        return ts.cpu().numpy()[0]  # [seq_len, channels]


def batch_generate_blocks(model, process, label_weights_list, args,
                           channels=1, batch_size=None):
    """
    Generate multiple blocks in batches for efficiency.

    Args:
        label_weights_list : list of np.ndarray, each [n_states]

    Returns:
        list of np.ndarray, each [seq_len, channels]
    """
    if batch_size is None:
        batch_size = args.batch_size

    results = []
    n = len(label_weights_list)

    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        batch_labels = np.stack(label_weights_list[i:i + b])
        oh = torch.FloatTensor(batch_labels).to(args.device)  # [b, n_states]

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
                results.append(sample)

    return results


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

    # ── Calculate steps needed ────────────────────────────────────────
    freq = cli.freq
    steps_per_year = len(
        pd.date_range(start="2026-01-01", end="2027-01-01",
                      freq=freq, inclusive="left")
    )
    total_steps = int(cli.years * steps_per_year)

    # Estimate how many raw blocks needed (conservative — some become bridges)
    base_stride = args.seq_len - cli.overlap
    if base_stride <= 0:
        raise ValueError(
            f"Overlap ({cli.overlap}) must be < seq_len ({args.seq_len})"
        )
    # Overshoot slightly to account for wider overlaps at transitions
    num_base_blocks = int(np.ceil(total_steps / base_stride)) + 10

    # ── Transition Matrix & State Sequence ────────────────────────────
    if cli.state is not None:
        # Single-state mode
        state_sequence = np.full(num_base_blocks, cli.state, dtype=int)
        tag = f"hmm_state{cli.state}"
        logging.info("Single-state mode: locked to HMM State %d", cli.state)
    else:
        # Load or compute transition matrix
        trans_path = cli.transition_matrix_path
        if trans_path is None:
            trans_path = os.path.join(
                PROJECT_ROOT, "data", "rainfall", "astlingen", "hmm_datasets",
                f"hmm_{cli.hmm_variant}_transition_matrix.pkl",
            )

        if os.path.exists(trans_path):
            with open(trans_path, "rb") as f:
                trans_data = pickle.load(f)
            logging.info("Loaded transition matrix from %s", trans_path)
        else:
            raise FileNotFoundError(
                f"Transition matrix not found at {trans_path}. "
                f"Run scripts/prepare_hmm_dataset.py first."
            )

        P_mat = trans_data["P_block"]
        pi_dist = trans_data["pi_block"]

        logging.info("Block-level transition matrix:\n%s", np.round(P_mat, 5))
        logging.info("Stationary distribution: %s", np.round(pi_dist, 4))

        state_sequence = sample_markov_state_sequence(
            P_mat, pi_dist, num_base_blocks, seed=cli.seed,
        )
        tag = f"hmm_{cli.hmm_variant}_v1"

    # ── Build Generation Plan (with bridge blocks) ────────────────────
    plan = build_generation_plan(state_sequence, n_bridge=cli.bridge_blocks)

    # Count statistics
    n_normal = sum(1 for p in plan if p["type"] == "normal")
    n_bridge = sum(1 for p in plan if p["type"] == "bridge")
    n_transitions = sum(
        1 for i in range(len(state_sequence) - 1)
        if state_sequence[i] != state_sequence[i + 1]
    )

    unique_s, counts_s = np.unique(state_sequence, return_counts=True)
    state_alloc = {int(s): int(c) for s, c in zip(unique_s, counts_s)}

    logging.info("═" * 60)
    logging.info("Generating %.1f years (%d steps)", cli.years, total_steps)
    logging.info("Normal blocks: %d  |  Bridge blocks: %d  |  Transitions: %d",
                 n_normal, n_bridge, n_transitions)
    logging.info("State allocation: %s", state_alloc)
    logging.info("Overlap: base=%d, transition=%d  |  Bridge blocks per transition: %d",
                 cli.overlap, cli.transition_overlap, cli.bridge_blocks)
    logging.info("═" * 60)

    # ── Generate All Blocks ───────────────────────────────────────────
    channels = 1
    process = DiffusionProcess(
        args, model.net,
        (channels, args.img_resolution, args.img_resolution),
    )

    label_weights_list = [p["label_weights"] for p in plan]

    with model.ema_scope():
        logging.info("Generating %d total blocks ...", len(plan))
        all_blocks = batch_generate_blocks(
            model, process, label_weights_list, args, channels=channels,
        )

    # ── Stitch with Transition-Aware OLA ──────────────────────────────
    logging.info("Stitching with transition-aware OLA ...")

    # Flatten each block to 1D
    flat_blocks = [b.squeeze(-1) if b.ndim > 1 else b for b in all_blocks]

    continuous_scaled = stitch_blocks_transition_aware(
        flat_blocks, plan,
        base_overlap=cli.overlap,
        transition_overlap=cli.transition_overlap,
    )

    # Trim to exact length
    continuous_scaled = continuous_scaled[:total_steps].reshape(-1, channels)

    # ── Inverse Scaling & Thresholding ────────────────────────────────
    unscaled = scaler.inverse_transform(continuous_scaled)
    unscaled[unscaled < 0.005] = 0.0

    # ── Save Output ───────────────────────────────────────────────────
    os.makedirs(cli.output_dir, exist_ok=True)
    run_id = os.path.splitext(os.path.basename(cli.model_ckpt))[0]
    out_path = os.path.join(
        cli.output_dir,
        f"rainfall_{tag}_{cli.years}y_{run_id}.csv",
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
    logging.info("Saved HMM-Conditional Series → %s", out_path)
    logging.info("Shape      : %s", df.shape)
    logging.info("Zero%%      : %.2f%%", zero_frac * 100)
    logging.info("Mean       : %.4f  (non-zero: %.4f)",
                 df["avg_rainfall"].mean(), nz.mean() if len(nz) else 0.0)
    logging.info("Std        : %.4f  (non-zero: %.4f)",
                 df["avg_rainfall"].std(), nz.std() if len(nz) else 0.0)
    logging.info("Max intensity: %.4f mm (%s)", df["avg_rainfall"].max(), freq)

    # Per-state statistics (approximate: assign each output step its
    # source block's majority state)
    logging.info("\n--- Per-State Summary ---")
    step_states = []
    pos = 0
    for i, p in enumerate(plan):
        ov_after = 0
        if i < len(plan) - 1:
            is_trans = (p["state"] != plan[i + 1]["state"]) or \
                       p["type"] == "bridge" or plan[i + 1]["type"] == "bridge"
            ov_after = cli.transition_overlap if is_trans else cli.overlap
        block_len = args.seq_len
        effective_len = block_len - (ov_after if i < len(plan) - 1 else 0)
        state_label = p["state"] if p["state"] is not None else -1
        step_states.extend([state_label] * effective_len)
        pos += effective_len

    step_states = np.array(step_states[:total_steps])
    for s in range(args.n_classes):
        mask = step_states == s
        if mask.sum() > 0:
            vals = unscaled[mask, 0]
            logging.info(
                "  State %d: n=%d (%.1f%%), mean=%.4f, zero%%=%.1f%%, max=%.4f",
                s, mask.sum(), mask.mean() * 100,
                vals.mean(), (vals == 0).mean() * 100, vals.max(),
            )
    bridge_mask = step_states == -1
    if bridge_mask.sum() > 0:
        logging.info("  Bridge : n=%d (%.1f%%)", bridge_mask.sum(),
                     bridge_mask.mean() * 100)

    logging.info("═" * 60)


if __name__ == "__main__":
    main()
