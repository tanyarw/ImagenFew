"""
Estimate 1st-Order Markov Transition Matrix for Regime-Conditional Generation
=============================================================================
Calculates the empirical block-level (14-day) transition probability matrix
P(R_{t+1} | R_t) from labeled 10-minute rainfall data (`rainfall_10min_labeled.csv`).

Outputs:
  - Prints block-level transition matrix and stationary state distribution
  - Saves transition dictionary to data/rainfall/train/regime_transition_matrix.pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def estimate_transition_matrix(csv_path, n_states=4, smooth=1e-5):
    """
    Estimates block-level Markov transition matrix P(R_{t+1} | R_t).

    Args:
        csv_path  : Path to rainfall_10min_labeled.csv
        n_states  : Number of GMM regime states (default 4)
        smooth    : Laplace smoothing factor

    Returns:
        dict containing P, pi, P_block, C_block, pi_block, regime_proportions
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if "gmm_regime" not in df.columns:
        raise ValueError(f"CSV missing 'gmm_regime' column. Found: {df.columns.tolist()}")

    # Group by block_id (14-day blocks) to get sequential regime steps
    if "block_id" in df.columns:
        block_seq = df.groupby("block_id", sort=True)["gmm_regime"].first().values.astype(int)
    else:
        # Fallback if block_id column isn't present
        block_seq = df['gmm_regime'].values.astype(int)

    C_block = np.zeros((n_states, n_states), dtype=np.float64) + smooth
    for i in range(len(block_seq) - 1):
        r1, r2 = block_seq[i], block_seq[i+1]
        if 0 <= r1 < n_states and 0 <= r2 < n_states:
            C_block[r1, r2] += 1.0

    P_block = C_block / C_block.sum(axis=1, keepdims=True)

    # Calculate stationary distribution (left eigenvector for eigenvalue = 1.0)
    eigenvalues, eigenvectors = np.linalg.eig(P_block.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_block = np.real(eigenvectors[:, idx])
    pi_block = np.abs(pi_block / np.sum(pi_block))

    # Calculate empirical regime proportions across all timestamps
    unique, counts = np.unique(df['gmm_regime'].values, return_counts=True)
    total = len(df)
    regime_proportions = {int(r): float(c / total) for r, c in zip(unique, counts)}

    return {
        "P": P_block,                  # Default block-level transition matrix
        "pi": pi_block,                # Default block-level stationary dist
        "P_block": P_block,            # Alias key for backwards compatibility
        "C_block": C_block,
        "pi_block": pi_block,          # Alias key for backwards compatibility
        "regime_proportions": regime_proportions,
        "n_states": n_states,
    }

def main():
    parser = argparse.ArgumentParser(description="Estimate 1st-Order Block-Level Markov Transition Matrix")
    parser.add_argument("--csv_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "rainfall", "train", "rainfall_10min_labeled.csv"))
    parser.add_argument("--out_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "rainfall", "train", "regime_transition_matrix.pkl"))
    args = parser.parse_args()

    print(f"Loading {args.csv_path}...")
    res = estimate_transition_matrix(args.csv_path)

    print("\n" + "=" * 60)
    print("Block-Level (14-day) 1st-Order Markov Transition Matrix P(R_{t+1} | R_t):")
    print("=" * 60)
    print(np.round(res["P_block"], 5))

    print("\nBlock-Level Persistence P_ii (Diagonal):")
    for r in range(res["n_states"]):
        p_ii = res['P_block'][r, r]
        avg_dur = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else float('inf')
        print(f"  Regime {r}: {p_ii:.5f} (Avg duration: {avg_dur:.1f} blocks / {avg_dur * 14:.1f} days)")

    print("\nEmpirical Regime Proportions:", res["regime_proportions"])
    print("Stationary Distribution (Block): ", np.round(res["pi_block"], 4))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "wb") as f:
        pickle.dump(res, f)
    print(f"\nSaved transition matrix data to: {args.out_path}")

if __name__ == "__main__":
    main()
