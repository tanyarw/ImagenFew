"""
Estimate 1st-Order Markov Transition Matrix for Regime-Conditional Generation
=============================================================================
Calculates the empirical transition probability matrix P(R_{t+1} | R_t) from
labeled 10-minute rainfall data (`rainfall_10min_labeled.csv`).

Supports both:
  1. Window-level transitions (e.g. seq_len=24 steps = 4 hours)
  2. Block-level transitions (e.g. block_id = 2016 steps = 14 days)

Outputs:
  - Prints matrices and stationary state distributions
  - Saves transition dictionary to data/rainfall/train/regime_transition_matrix.pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def estimate_transition_matrix(csv_path, seq_len=24, n_states=4, smooth=1e-5):
    """
    Estimates window-level and block-level Markov transition matrices.

    Args:
        csv_path  : Path to rainfall_10min_labeled.csv
        seq_len   : Window length in steps (default 24 = 4 hours)
        n_states  : Number of GMM regime states (default 4)
        smooth    : Laplace smoothing factor

    Returns:
        dict containing P_window, P_block, pi_stationary_window, regime_counts
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if "gmm_regime" not in df.columns:
        raise ValueError(f"CSV missing 'gmm_regime' column. Found: {df.columns.tolist()}")

    # ---------------------------------------------------------
    # 1. Window-level transitions (every seq_len steps)
    # ---------------------------------------------------------
    window_regimes = df['gmm_regime'].iloc[::seq_len].values.astype(int)
    C_window = np.zeros((n_states, n_states), dtype=np.float64) + smooth
    for i in range(len(window_regimes) - 1):
        r1, r2 = window_regimes[i], window_regimes[i+1]
        if 0 <= r1 < n_states and 0 <= r2 < n_states:
            C_window[r1, r2] += 1.0

    P_window = C_window / C_window.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------
    # 2. Block-level transitions (grouped by block_id)
    # ---------------------------------------------------------
    if "block_id" in df.columns:
        block_seq = df.groupby("block_id", sort=True)["gmm_regime"].first().values.astype(int)
        C_block = np.zeros((n_states, n_states), dtype=np.float64) + smooth
        for i in range(len(block_seq) - 1):
            r1, r2 = block_seq[i], block_seq[i+1]
            if 0 <= r1 < n_states and 0 <= r2 < n_states:
                C_block[r1, r2] += 1.0
        P_block = C_block / C_block.sum(axis=1, keepdims=True)
    else:
        P_block = P_window
        C_block = C_window

    # ---------------------------------------------------------
    # 3. Stationary Distributions (left eigenvectors for eigenvalue 1)
    # ---------------------------------------------------------
    def get_stationary_dist(P):
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / np.sum(pi)
        return np.abs(pi)

    pi_window = get_stationary_dist(P_window)
    pi_block = get_stationary_dist(P_block)

    # Empirical proportions
    unique, counts = np.unique(df['gmm_regime'].values, return_counts=True)
    total = len(df)
    regime_proportions = {int(r): float(c / total) for r, c in zip(unique, counts)}

    return {
        "P_window": P_window,
        "C_window": C_window,
        "pi_window": pi_window,
        "P_block": P_block,
        "C_block": C_block,
        "pi_block": pi_block,
        "regime_proportions": regime_proportions,
        "seq_len": seq_len,
        "n_states": n_states,
    }

def main():
    parser = argparse.ArgumentParser(description="Estimate 1st-Order Markov Transition Matrix")
    parser.add_argument("--csv_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "rainfall", "train", "rainfall_10min_labeled.csv"))
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--out_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "rainfall", "train", "regime_transition_matrix.pkl"))
    args = parser.parse_args()

    print(f"Loading {args.csv_path}...")
    res = estimate_transition_matrix(args.csv_path, seq_len=args.seq_len)

    print("\n" + "=" * 60)
    print(f"Window-Level (seq_len={args.seq_len}) 1st-Order Markov Transition Matrix P(R_{{t+1}} | R_t):")
    print("=" * 60)
    print(np.round(res["P_window"], 5))

    print("\nWindow-Level Persistence P_ii (Diagonal):")
    for r in range(res["n_states"]):
        print(f"  Regime {r}: {res['P_window'][r, r]:.5f} (Avg duration: {1.0 / (1.0 - res['P_window'][r, r]):.1f} windows)")

    print("\n" + "=" * 60)
    print("Block-Level (14-day) 1st-Order Markov Transition Matrix P(R_{{t+1}} | R_t):")
    print("=" * 60)
    print(np.round(res["P_block"], 4))

    print("\nEmpirical Regime Proportions:", res["regime_proportions"])
    print("Stationary Distribution (Window):", np.round(res["pi_window"], 4))
    print("Stationary Distribution (Block): ", np.round(res["pi_block"], 4))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "wb") as f:
        pickle.dump(res, f)
    print(f"\nSaved transition matrix data to: {args.out_path}")

if __name__ == "__main__":
    main()
