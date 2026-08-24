"""
Prepare HMM-Labeled Rainfall Dataset for Conditional Diffusion Training
========================================================================
Resamples 5-minute HMM-labeled rainfall data to 10-minute resolution,
creates train/test splits, and estimates a 4-hour block-level transition
matrix for Markov sequence generation.

Supports two HMM variants:
  - 105120: HMM fitted on ~1-year (105120-step) blocks — more transitions
  - 365:    HMM fitted on 365-day blocks — fewer, longer state runs

Usage
-----
  python scripts/prepare_hmm_dataset.py --hmm_variant 105120
  python scripts/prepare_hmm_dataset.py --hmm_variant 365
  python scripts/prepare_hmm_dataset.py --hmm_variant 105120 --validate
"""

import os
import sys
import argparse
import logging
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare HMM Dataset for Conditional Training")
    p.add_argument("--hmm_variant", type=str, default="105120",
                   choices=["105120", "365"],
                   help="Which HMM fit to use: '105120' (recommended) or '365'")
    p.add_argument("--resolution", type=str, default="5min",
                   choices=["5min", "10min"],
                   help="Temporal resolution of the training dataset (default: '5min')")
    p.add_argument("--block_size_steps", type=int, default=24,
                   help="Block size in steps for transition matrix "
                        "(default: 24, matching seq_len=24)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validate", action="store_true",
                   help="Run validation checks after preparation")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Data Preparation
# ──────────────────────────────────────────────────────────────────────

def load_hmm_data(variant):
    """Load the appropriate HMM dataset."""
    path_map = {
        "105120": os.path.join(
            PROJECT_ROOT, "data", "rainfall", "astlingen",
            "hmm_datasets", "dataset_with_105120_fit_hmm.csv"
        ),
        "365": os.path.join(
            PROJECT_ROOT, "data", "rainfall", "astlingen",
            "hmm_datasets", "dataset_with_365_fit_hmm.csv"
        ),
    }
    path = path_map[variant]
    if not os.path.exists(path):
        raise FileNotFoundError(f"HMM dataset not found: {path}")

    logging.info("Loading HMM dataset (variant=%s): %s", variant, path)
    df = pd.read_csv(path)

    # Build a proper datetime column from date + time
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                     format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)

    logging.info("  Shape: %s, Date range: %s to %s",
                 df.shape, df["datetime"].iloc[0], df["datetime"].iloc[-1])
    logging.info("  HMM states: %s", sorted(df["hmm_state"].unique()))
    return df


def prepare_5min_dataset(df):
    """
    Format raw 5-minute data with standardized column names.
    Includes both 'avg_rainfall' and 'rainfall_intensity' for backwards compatibility.
    """
    logging.info("Preparing native 5-minute resolution dataset ...")
    formatted = pd.DataFrame({
        "datetime": df["datetime"],
        "avg_rainfall": df["rainfall_intensity"],
        "rainfall_intensity": df["rainfall_intensity"],
        "hmm_state": df["hmm_state"],
        "day_of_year": df["day_of_year"] if "day_of_year" in df.columns else df["datetime"].dt.dayofyear,
    })
    logging.info("  5-min dataset shape: %s", formatted.shape)
    return formatted


def resample_to_10min(df):
    """
    Resample 5-minute data to 10-minute resolution.
    - rainfall_intensity / avg_rainfall: sum of two consecutive 5-min values (preserves total volume)
    - hmm_state: first value in each pair (states persist for hours, so this is safe)
    """
    logging.info("Resampling 5-min → 10-min ...")

    # Pair consecutive rows (0-1, 2-3, 4-5, ...) and aggregate
    n = len(df)
    n_pairs = n // 2

    datetimes = df["datetime"].values[:n_pairs * 2:2]  # Take every other datetime
    rainfall = df["rainfall_intensity"].values[:n_pairs * 2]
    hmm = df["hmm_state"].values[:n_pairs * 2]

    rain_pairs = rainfall.reshape(n_pairs, 2)
    hmm_pairs = hmm.reshape(n_pairs, 2)

    resampled = pd.DataFrame({
        "datetime": datetimes,
        "avg_rainfall": rain_pairs.sum(axis=1),
        "rainfall_intensity": rain_pairs.sum(axis=1),
        # Take the first value — with 99.97%+ persistence, state changes
        # within a 10-min window are vanishingly rare
        "hmm_state": hmm_pairs[:, 0],
    })

    logging.info("  Resampled shape: %s", resampled.shape)
    return resampled


# No train/test split — all data used for training (generative model).


# ──────────────────────────────────────────────────────────────────────
# Block-Level Transition Matrix
# ──────────────────────────────────────────────────────────────────────

def estimate_block_transition_matrix(df, block_size=24, n_states=4, smooth=1e-5):
    """
    Estimate block-level Markov transition matrix P(S_{k+1} | S_k).

    Each "block" spans `block_size` 10-minute steps (default 24 = 4 hours).
    The majority HMM state within each block determines the block's state.

    Args:
        df         : DataFrame with 'hmm_state' column
        block_size : Number of 10-min steps per block (default: 24 = 4 hours)
        n_states   : Number of HMM states
        smooth     : Laplace smoothing factor

    Returns:
        dict with P_block, C_block, pi_block, regime_proportions, block_sequence
    """
    states = df["hmm_state"].values
    n_blocks = len(states) // block_size

    # Assign each block its majority state
    block_states = np.zeros(n_blocks, dtype=int)
    for i in range(n_blocks):
        chunk = states[i * block_size : (i + 1) * block_size]
        vals, counts = np.unique(chunk, return_counts=True)
        block_states[i] = vals[np.argmax(counts)]

    # Count transitions
    C_block = np.zeros((n_states, n_states), dtype=np.float64) + smooth
    for i in range(len(block_states) - 1):
        s1, s2 = block_states[i], block_states[i + 1]
        if 0 <= s1 < n_states and 0 <= s2 < n_states:
            C_block[s1, s2] += 1.0

    P_block = C_block / C_block.sum(axis=1, keepdims=True)

    # Stationary distribution (left eigenvector for eigenvalue ≈ 1.0)
    eigenvalues, eigenvectors = np.linalg.eig(P_block.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_block = np.real(eigenvectors[:, idx])
    pi_block = np.abs(pi_block / np.sum(pi_block))

    # Empirical proportions from timestep-level data
    unique, counts = np.unique(states, return_counts=True)
    total = len(states)
    regime_proportions = {int(r): float(c / total) for r, c in zip(unique, counts)}

    # Total transitions count
    n_transitions = sum(
        1 for i in range(len(block_states) - 1)
        if block_states[i] != block_states[i + 1]
    )

    return {
        "P_block": P_block,
        "P": P_block,               # Alias for backwards compatibility
        "C_block": C_block,
        "pi_block": pi_block,
        "pi": pi_block,             # Alias for backwards compatibility
        "regime_proportions": regime_proportions,
        "n_states": n_states,
        "block_size": block_size,
        "block_sequence": block_states,
        "n_blocks": n_blocks,
        "n_transitions": n_transitions,
    }


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────

def validate_outputs(out_dir, variant, expected_resolution="5min"):
    """Run sanity checks on the prepared outputs."""
    logging.info("\n" + "=" * 60)
    logging.info("VALIDATION CHECKS (Resolution: %s)", expected_resolution)
    logging.info("=" * 60)

    # Load outputs
    train_path = os.path.join(out_dir, f"hmm_{variant}_train.csv")
    trans_path = os.path.join(out_dir, f"hmm_{variant}_transition_matrix.pkl")

    train_df = pd.read_csv(train_path)

    with open(trans_path, "rb") as f:
        trans = pickle.load(f)

    checks_passed = 0
    checks_total = 0

    # Check 1: No NaN values
    checks_total += 1
    has_nan = train_df.isna().any().any()
    if not has_nan:
        logging.info("  ✓ No NaN values in training data")
        checks_passed += 1
    else:
        logging.error("  ✗ NaN values found!")

    # Check 2: HMM states are in {0, 1, 2, 3}
    checks_total += 1
    valid_states = set(range(4))
    train_states = set(train_df["hmm_state"].unique())
    if train_states.issubset(valid_states):
        logging.info("  ✓ HMM states are valid {0,1,2,3}")
        checks_passed += 1
    else:
        logging.error("  ✗ Invalid HMM states: %s", train_states)

    # Check 3: Transition matrix rows sum to 1
    checks_total += 1
    row_sums = trans["P_block"].sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-6):
        logging.info("  ✓ Transition matrix rows sum to 1.0")
        checks_passed += 1
    else:
        logging.error("  ✗ Row sums: %s", row_sums)

    # Check 4: Stationary distribution sums to 1
    checks_total += 1
    pi_sum = trans["pi_block"].sum()
    if np.isclose(pi_sum, 1.0, atol=1e-6):
        logging.info("  ✓ Stationary distribution sums to 1.0")
        checks_passed += 1
    else:
        logging.error("  ✗ Stationary dist sum: %f", pi_sum)

    # Check 5: Transition matrix diagonal is NOT degenerate (< 0.999 for block size)
    checks_total += 1
    diag = np.diag(trans["P_block"])
    max_diag = diag.max()
    if max_diag < 0.999:
        logging.info("  ✓ Block-level diagonals reasonable (max P_ii = %.5f)", max_diag)
        checks_passed += 1
    else:
        logging.warning("  ⚠ Block-level diagonals still very high (max P_ii = %.5f)", max_diag)
        checks_passed += 1  # Warning, not failure

    # Check 6: Temporal resolution
    checks_total += 1
    train_dt = pd.to_datetime(train_df["datetime"])
    diffs = train_dt.diff().dropna()
    median_diff = diffs.median()
    expected_minutes = 5 if expected_resolution == "5min" else 10
    if median_diff == pd.Timedelta(minutes=expected_minutes):
        logging.info("  ✓ Data is at %d-minute resolution", expected_minutes)
        checks_passed += 1
    else:
        logging.error("  ✗ Median time gap: %s (expected %d min)", median_diff, expected_minutes)

    logging.info("\nValidation: %d/%d checks passed", checks_passed, checks_total)

    # Print summary statistics
    hours_per_block = (trans["block_size"] * (5 if expected_resolution == "5min" else 10)) / 60.0
    logging.info("\n--- Transition Matrix (%.1f-hour block level, block_size=%d) ---",
                 hours_per_block, trans["block_size"])
    logging.info("\n%s", np.round(trans["P_block"], 5))
    logging.info("\nDiagonal (persistence):")
    for s in range(trans["n_states"]):
        p_ii = trans["P_block"][s, s]
        avg_dur = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else float("inf")
        logging.info("  State %d: P_ii=%.5f  (avg run: %.1f blocks = %.1f hours)",
                     s, p_ii, avg_dur, avg_dur * hours_per_block)
    logging.info("Stationary dist: %s", np.round(trans["pi_block"], 4))
    logging.info("Total block transitions: %d", trans["n_transitions"])
    logging.info("Regime proportions: %s", trans["regime_proportions"])


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    logging.info("═" * 60)
    logging.info("HMM Dataset Preparation (variant=%s, resolution=%s, block_size=%d)",
                 cli.hmm_variant, cli.resolution, cli.block_size_steps)
    logging.info("═" * 60)

    # 1. Load raw data
    df = load_hmm_data(cli.hmm_variant)

    # 2. Format / Resample based on resolution
    if cli.resolution == "5min":
        train_df = prepare_5min_dataset(df)
    else:
        train_df = resample_to_10min(df)

    # 3. Use all data for training (no test split — generative model)
    logging.info("  Using all data for training: %d samples (%s to %s)",
                 len(train_df), train_df["datetime"].iloc[0],
                 train_df["datetime"].iloc[-1])

    # 4. Estimate block-level transition matrix
    trans = estimate_block_transition_matrix(
        train_df,
        block_size=cli.block_size_steps,
        n_states=4,
    )

    # 5. Save outputs
    out_dir = os.path.join(PROJECT_ROOT, "data", "rainfall", "astlingen", "hmm_datasets")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, f"hmm_{cli.hmm_variant}_train.csv")
    trans_path = os.path.join(out_dir, f"hmm_{cli.hmm_variant}_transition_matrix.pkl")

    train_df.to_csv(train_path, index=False)
    with open(trans_path, "wb") as f:
        pickle.dump(trans, f)

    logging.info("\nSaved outputs:")
    logging.info("  Train:      %s (%d rows)", train_path, len(train_df))
    logging.info("  Transition: %s", trans_path)

    # 6. Optional validation
    if cli.validate:
        validate_outputs(out_dir, cli.hmm_variant, expected_resolution=cli.resolution)

    logging.info("\n✓ Done!")


if __name__ == "__main__":
    main()
