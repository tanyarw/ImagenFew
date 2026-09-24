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
import socket
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

    p.add_argument("--assembly_mode", type=str, default="markov",
                   choices=["markov", "calendar", "unconditional"],
                   help="Sequence assembly mode: 'markov' (legacy stochastic random walk), "
                        "'calendar' (deterministic annual seasonal calendar progression), "
                        "or 'unconditional' (ablation: generate without 4-class conditioning)")
    p.add_argument("--uncond_method", type=str, default="zeros",
                   choices=["zeros", "uniform", "marginal", "random"],
                   help="Conditioning method for unconditional ablation mode: "
                        "'zeros' (null class embedding: class vector is all 0s), "
                        "'uniform' (equal soft-label weighting [0.25, 0.25, 0.25, 0.25]), "
                        "'marginal' (empirical stationary distribution weights), "
                        "'random' (i.i.d. sampling from marginal distribution per block)")
    p.add_argument("--output_name", type=str, default=None,
                   help="Explicit output CSV filename (e.g., 'rainfall_synthetic_10y_v6.csv')")
    p.add_argument("--device", type=str, default=None,
                   help="Device ('cuda' or 'cpu'). Defaults to 'cuda' (errors if unavailable).")
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
        if getattr(args, "n_classes", 0) > 0 and batch_labels.shape[-1] > 0:
            oh = torch.FloatTensor(batch_labels).to(args.device)  # [b, n_states]
        else:
            oh = None

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
    device = getattr(cli, "device", None)
    if device is None or device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is not available on {socket.gethostname()}! "
                "Refusing to run 10-year diffusion generation on CPU. "
                "Pass --device cpu explicitly if CPU execution is really intended."
            )
        args.device = "cuda"
    else:
        args.device = device
    args.model_ckpt = cli.model_ckpt

    # ── Scaler ────────────────────────────────────────────────────────
    with open(cli.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logging.info("Loaded scaler from %s", cli.scaler_path)

    # ── Model ─────────────────────────────────────────────────────────
    logging.info("Building model (n_classes=%d) …", args.n_classes)
    model = ImagenFew(args, args.device).to(args.device)

    loaded = torch.load(cli.model_ckpt, map_location=args.device, weights_only=False)
    state_dict = loaded.get("model", loaded)
    has_map_label = any("map_label" in k for k in state_dict.keys())
    if not has_map_label and getattr(args, "n_classes", 0) > 0:
        logging.info("Checkpoint does not contain map_label — switching args.n_classes = 0")
        args.n_classes = 0
        model = ImagenFew(args, args.device).to(args.device)

    cur_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in cur_state and v.shape == cur_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    logging.info("Loaded %d/%d parameters from %s", len(filtered), len(cur_state), cli.model_ckpt)

    if "ema_model" in loaded and args.ema:
        ema_state = loaded["ema_model"]
        cur_ema = model.model_ema.state_dict()
        filtered_ema = {k: v for k, v in ema_state.items() if k in cur_ema and v.shape == cur_ema[k].shape}
        model.model_ema.load_state_dict(filtered_ema, strict=False)
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
        plan = build_generation_plan(state_sequence, n_bridge=cli.bridge_blocks)
    elif cli.assembly_mode == "unconditional":
        # Unconditional ablation mode (generate without 4-class conditioning)
        tag = f"unconditional_{cli.uncond_method}"
        logging.info("═" * 60)
        logging.info("Unconditional Ablation Mode: method='%s', n_classes=%d",
                     cli.uncond_method, args.n_classes)
        logging.info("═" * 60)
        state_sequence = None
        n_states = args.n_classes

        if cli.uncond_method == "zeros":
            # Zero vector: null class conditioning (map_label outputs zeros)
            oh = np.zeros(n_states, dtype=np.float32) if n_states > 0 else np.zeros(1, dtype=np.float32)
            plan = [{
                "type": "normal",
                "state": None,
                "label_weights": oh,
            } for _ in range(num_base_blocks)]
        elif cli.uncond_method == "uniform":
            # Equal soft-label weighting across all classes
            oh = np.full(n_states, 1.0 / max(1, n_states), dtype=np.float32)
            plan = [{
                "type": "normal",
                "state": None,
                "label_weights": oh,
            } for _ in range(num_base_blocks)]
        elif cli.uncond_method == "marginal":
            # Stationary distribution weights across classes
            pi_dist = None
            if cli.transition_matrix_path and os.path.exists(cli.transition_matrix_path):
                with open(cli.transition_matrix_path, "rb") as f:
                    t_data = pickle.load(f)
                if "pi_block" in t_data:
                    pi_dist = np.array(t_data["pi_block"], dtype=np.float32)
            if pi_dist is None:
                props_path = os.path.join(os.path.dirname(cli.model_ckpt), "regime_proportions.pkl")
                if os.path.exists(props_path):
                    with open(props_path, "rb") as f:
                        props = pickle.load(f)
                    pi_dist = np.array([props.get(s, 1.0 / max(1, n_states)) for s in range(n_states)], dtype=np.float32)
                    pi_dist = pi_dist / pi_dist.sum()
            if pi_dist is None or len(pi_dist) != n_states:
                pi_dist = np.full(n_states, 1.0 / max(1, n_states), dtype=np.float32)
            logging.info("Marginal distribution weights: %s", np.round(pi_dist, 4))
            plan = [{
                "type": "normal",
                "state": None,
                "label_weights": pi_dist,
            } for _ in range(num_base_blocks)]
        elif cli.uncond_method == "random":
            # Random i.i.d. sampling per block without Markov memory
            rng = np.random.default_rng(cli.seed)
            pi_dist = None
            if cli.transition_matrix_path and os.path.exists(cli.transition_matrix_path):
                with open(cli.transition_matrix_path, "rb") as f:
                    t_data = pickle.load(f)
                if "pi_block" in t_data:
                    pi_dist = np.array(t_data["pi_block"], dtype=np.float32)
            if pi_dist is None:
                props_path = os.path.join(os.path.dirname(cli.model_ckpt), "regime_proportions.pkl")
                if os.path.exists(props_path):
                    with open(props_path, "rb") as f:
                        props = pickle.load(f)
                    pi_dist = np.array([props.get(s, 1.0 / max(1, n_states)) for s in range(n_states)], dtype=np.float32)
                    pi_dist = pi_dist / pi_dist.sum()
            if pi_dist is None or len(pi_dist) != n_states:
                p_draw = None
            else:
                p_draw = pi_dist

            state_sequence = rng.choice(n_states, size=num_base_blocks, p=p_draw)
            plan = []
            for s in state_sequence:
                oh = np.zeros(n_states, dtype=np.float32)
                oh[s] = 1.0
                plan.append({
                    "type": "normal",
                    "state": int(s),
                    "label_weights": oh,
                })
    elif cli.assembly_mode == "calendar":
        # Calendar-ordered assembly mode (deterministic annual seasonal cycle)
        trans_path = cli.transition_matrix_path
        if trans_path is None:
            trans_path = os.path.join(
                PROJECT_ROOT, "data", "rainfall", "splits", "seasonal_transition_matrix_train_len24.pkl"
            )
        if not os.path.exists(trans_path):
            trans_path = os.path.join(
                PROJECT_ROOT, "data", "rainfall", "splits", "seasonal_transition_matrix_train.pkl"
            )
        if os.path.exists(trans_path):
            with open(trans_path, "rb") as f:
                trans_data = pickle.load(f)
            logging.info("Loaded seasonal profile from %s", trans_path)
        else:
            raise FileNotFoundError(f"Transition matrix/profile not found at {trans_path}")

        if "climatology_profile_1yr" in trans_data:
            clim = trans_data["climatology_profile_1yr"]
            state_sequence = np.array([clim[(k * base_stride) % len(clim)] for k in range(num_base_blocks)], dtype=int)
        elif "calendar_sequence_1yr" in trans_data:
            cal_seq = trans_data["calendar_sequence_1yr"]
            reps = int(np.ceil(num_base_blocks / len(cal_seq)))
            state_sequence = np.tile(cal_seq, reps)[:num_base_blocks].astype(int)
        elif "calendar_sequence_1yr_step24" in trans_data:
            cal_seq = trans_data["calendar_sequence_1yr_step24"]
            reps = int(np.ceil(num_base_blocks / len(cal_seq)))
            state_sequence = np.tile(cal_seq, reps)[:num_base_blocks].astype(int)
        else:
            raise ValueError(f"No calendar sequence profile found in {trans_path}")

        tag = "calendar_v1"
        logging.info("Calendar-Ordered Assembly: %d blocks mapped across %.1f annual seasonal cycles",
                     num_base_blocks, cli.years)
        plan = build_generation_plan(state_sequence, n_bridge=cli.bridge_blocks)
    else:
        # Load or compute transition matrix (stochastic Markov sampling)
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
        plan = build_generation_plan(state_sequence, n_bridge=cli.bridge_blocks)

    # Count statistics
    n_normal = sum(1 for p in plan if p["type"] == "normal")
    n_bridge = sum(1 for p in plan if p["type"] == "bridge")
    if state_sequence is not None:
        n_transitions = sum(
            1 for i in range(len(state_sequence) - 1)
            if state_sequence[i] != state_sequence[i + 1]
        )
        unique_s, counts_s = np.unique(state_sequence, return_counts=True)
        state_alloc = {int(s): int(c) for s, c in zip(unique_s, counts_s)}
    else:
        n_transitions = 0
        state_alloc = {f"unconditional_{cli.uncond_method}": n_normal}

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
    run_id = os.path.basename(os.path.dirname(cli.model_ckpt))
    if not run_id or run_id in ["Rainfall_Regime", "logs", "."]:
        run_id = os.path.splitext(os.path.basename(cli.model_ckpt))[0]

    if cli.output_name:
        out_filename = cli.output_name if cli.output_name.endswith(".csv") else f"{cli.output_name}.csv"
    else:
        out_filename = f"rainfall_synthetic_{cli.years}y_hmm_{cli.hmm_variant}_{run_id}.csv"
    
    out_path = os.path.join(cli.output_dir, out_filename)

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
    if cli.assembly_mode == "unconditional" and cli.uncond_method != "random":
        logging.info("\n--- Unconditional Generation Summary ---")
        logging.info("  Ablation Mode  : unconditional (%s)", cli.uncond_method)
        logging.info("  Total Blocks   : %d (uniform OLA overlap: %d steps)", len(plan), cli.overlap)
        logging.info("  Zero%%          : %.2f%%", zero_frac * 100)
        logging.info("  Mean (wet)     : %.4f mm", nz.mean() if len(nz) else 0.0)
        logging.info("  Max Intensity  : %.4f mm", df["avg_rainfall"].max())
    elif args.n_classes > 0:
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
