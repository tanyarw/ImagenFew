"""
Estimate 1st-Order Markov Transition Matrix for Regime-Conditional Generation
=============================================================================
Calculates the empirical block-level transition probability matrix
P(R_{t+1} | R_t) from labeled rainfall data.

Supports both:
  - GMM regime labels ('gmm_regime' column, grouped by 'block_id')
  - HMM state labels ('hmm_state' column, with configurable block size)

Usage
-----
  # GMM (original 14-day blocks)
  python scripts/estimate_transition_matrix.py \\
      --csv_path data/rainfall/train/rainfall_10min_labeled.csv

  # HMM (4-hour blocks)
  python scripts/estimate_transition_matrix.py \\
      --csv_path data/rainfall/astlingen/hmm_datasets/hmm_105120_train.csv \\
      --state_col hmm_state --block_size 24

Outputs:
  - Prints block-level transition matrix and stationary state distribution
  - Saves transition dictionary as pickle
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def estimate_transition_matrix(csv_path, n_states=4, smooth=1e-5,
                                state_col="gmm_regime", block_size=None):
    """
    Estimates block-level Markov transition matrix P(R_{t+1} | R_t).

    Args:
        csv_path   : Path to labeled CSV
        n_states   : Number of states (default 4)
        smooth     : Laplace smoothing factor
        state_col  : Column name for state labels
                    ('gmm_regime' or 'hmm_state')
        block_size : Number of timesteps per block for aggregation.
                    If None, uses 'block_id' column for grouping (GMM path).
                    If int, segments the time series into fixed-size blocks
                    and assigns each block its majority state (HMM path).

    Returns:
        dict containing P_block, C_block, pi_block, regime_proportions, etc.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if state_col not in df.columns:
        raise ValueError(
            f"CSV missing '{state_col}' column. Found: {df.columns.tolist()}"
        )

    # Build block-level state sequence
    if block_size is not None:
        # Fixed-size block aggregation (HMM path)
        states = df[state_col].values.astype(int)
        n_blocks = len(states) // block_size
        block_seq = np.zeros(n_blocks, dtype=int)
        for i in range(n_blocks):
            chunk = states[i * block_size : (i + 1) * block_size]
            vals, counts = np.unique(chunk, return_counts=True)
            block_seq[i] = vals[np.argmax(counts)]
        block_desc = f"{block_size}-step"
    elif "block_id" in df.columns:
        # Group by block_id (GMM path with 14-day blocks)
        block_seq = (
            df.groupby("block_id", sort=True)[state_col]
            .first().values.astype(int)
        )
        block_desc = "block_id"
    else:
        # Fallback: treat each row as its own block (step-level transitions)
        block_seq = df[state_col].values.astype(int)
        block_desc = "step-level"

    # Count transitions
    C_block = np.zeros((n_states, n_states), dtype=np.float64) + smooth
    for i in range(len(block_seq) - 1):
        r1, r2 = block_seq[i], block_seq[i + 1]
        if 0 <= r1 < n_states and 0 <= r2 < n_states:
            C_block[r1, r2] += 1.0

    P_block = C_block / C_block.sum(axis=1, keepdims=True)

    # Stationary distribution (left eigenvector for eigenvalue ≈ 1.0)
    eigenvalues, eigenvectors = np.linalg.eig(P_block.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_block = np.real(eigenvectors[:, idx])
    pi_block = np.abs(pi_block / np.sum(pi_block))

    # Empirical proportions from timestep-level data
    unique, counts = np.unique(df[state_col].values, return_counts=True)
    total = len(df)
    regime_proportions = {int(r): float(c / total) for r, c in zip(unique, counts)}

    # Count actual transitions (state changes)
    n_transitions = sum(
        1 for i in range(len(block_seq) - 1)
        if block_seq[i] != block_seq[i + 1]
    )

    return {
        "P": P_block,
        "pi": pi_block,
        "P_block": P_block,
        "C_block": C_block,
        "pi_block": pi_block,
        "regime_proportions": regime_proportions,
        "n_states": n_states,
        "block_size": block_size,
        "block_desc": block_desc,
        "n_blocks": len(block_seq),
        "n_transitions": n_transitions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate 1st-Order Block-Level Markov Transition Matrix"
    )
    parser.add_argument(
        "--csv_path", type=str,
        default=os.path.join(
            PROJECT_ROOT, "data", "rainfall", "train",
            "rainfall_10min_labeled.csv"
        ),
    )
    parser.add_argument(
        "--out_path", type=str,
        default=os.path.join(
            PROJECT_ROOT, "data", "rainfall", "train",
            "regime_transition_matrix.pkl"
        ),
    )
    parser.add_argument(
        "--state_col", type=str, default="gmm_regime",
        help="Column name for state labels (default: gmm_regime)",
    )
    parser.add_argument(
        "--block_size", type=int, default=None,
        help="Number of timesteps per block. If omitted, uses block_id column.",
    )
    args = parser.parse_args()

    print(f"Loading {args.csv_path}...")
    res = estimate_transition_matrix(
        args.csv_path,
        state_col=args.state_col,
        block_size=args.block_size,
    )

    print("\n" + "=" * 60)
    print(f"Block-Level ({res['block_desc']}) Transition Matrix P(R_{{t+1}} | R_t):")
    print("=" * 60)
    print(np.round(res["P_block"], 5))

    print(f"\nPersistence P_ii (Diagonal):")
    for r in range(res["n_states"]):
        p_ii = res["P_block"][r, r]
        avg_dur = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else float("inf")
        if args.block_size:
            dur_hours = avg_dur * args.block_size * 10 / 60
            print(f"  State {r}: {p_ii:.5f} (Avg: {avg_dur:.1f} blocks = {dur_hours:.1f} hours)")
        else:
            print(f"  State {r}: {p_ii:.5f} (Avg duration: {avg_dur:.1f} blocks)")

    print(f"\nTotal blocks: {res['n_blocks']}")
    print(f"Total transitions (state changes): {res['n_transitions']}")
    print(f"\nEmpirical Proportions: {res['regime_proportions']}")
    print(f"Stationary Distribution: {np.round(res['pi_block'], 4)}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "wb") as f:
        pickle.dump(res, f)
    print(f"\nSaved transition matrix data to: {args.out_path}")


if __name__ == "__main__":
    main()
