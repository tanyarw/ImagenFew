#!/usr/bin/env python3
"""
run_vanilla_arima.py — Textbook Vanilla ARIMA Baseline for Rainfall Generation
==============================================================================
Fits a classical AutoRegressive (AR / ARIMA(p, 0, 0)) model on the continuous
precipitation time series and generates a multi-year synthetic realization.

Methodology:
1. Strict Data Partitioning:
   - Fits exclusively on the clean 2000-2007 chronological training partition
     (data/rainfall/splits/train_years_labelled.csv, 840,960 steps).
   - Validates that no validation (2008) or test (2009) data is seen during fitting.
2. Model Estimation:
   - Evaluates stationarity and centers data: y_t = x_t - mean(x_train).
   - Solves the Yule-Walker / Levinson-Durbin recursion on the sample autocovariance.
   - Automatically selects order p via Bayesian Information Criterion (BIC) or
     accepts user-specified order --p.
   - Computes out-of-sample 1-step prediction error on the 2008 validation split.
3. Simulation & Generation:
   - Generates N_gen = years * 365 * 288 steps (10 years = 1,051,200 steps).
   - Simulates y_t from Gaussian innovations N(0, sigma^2) with stationary burn-in.
   - Re-centers: x_t = y_t + mean(x_train).
   - Enforces physical non-negativity: clips negative values to 0.0 mm.
   - Applies the canonical wet threshold: values < 0.005 mm are set to 0.0 mm.
4. Reproducibility & Output:
   - Writes generated time series to results/generated_data/rainfall_synthetic_10y_arima.csv
   - Saves fitted parameters, order selection table, and diagnostics to a JSON model card.

Usage:
    python scripts/baselines/run_vanilla_arima.py --years 10 --seed 42
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN = os.path.join(ROOT, "data", "rainfall", "splits", "train_years_labelled.csv")
DEFAULT_VAL   = os.path.join(ROOT, "data", "rainfall", "splits", "val_years_labelled.csv")
DEFAULT_OUT   = os.path.join(ROOT, "results", "generated_data", "rainfall_synthetic_10y_arima.csv")
DEFAULT_CARD  = os.path.join(ROOT, "results", "reference", "arima_model_card.json")
WET_THR = 0.005  # Frozen project wet threshold (0.005 mm)
STEPS_PER_DAY = 288
STEPS_PER_YEAR = 365 * STEPS_PER_DAY  # 105,120 steps


def levinson_durbin(r: np.ndarray, p: int):
    """
    Fits AR(p) parameters via the Levinson-Durbin recursion.
    Args:
        r: Sample autocovariances r[0], r[1], ..., r[p]
        p: Target AR order
    Returns:
        phi: Array of AR coefficients [phi_1, phi_2, ..., phi_p]
        sigma2: Innovation variance of the AR(p) model
    """
    phi = np.zeros((p + 1, p + 1))
    sig2 = np.zeros(p + 1)
    sig2[0] = r[0]

    for k in range(1, p + 1):
        num = r[k] - np.dot(phi[k - 1, 1:k], r[k - 1:0:-1])
        ref = num / sig2[k - 1]
        phi[k, k] = ref
        phi[k, 1:k] = phi[k - 1, 1:k] - ref * phi[k - 1, k - 1:0:-1]
        sig2[k] = sig2[k - 1] * (1.0 - ref ** 2)

    return phi[p, 1:p + 1], float(sig2[p])


def main():
    parser = argparse.ArgumentParser(
        description="Fit vanilla ARIMA on clean training holdout and generate synthetic rainfall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN, help="Path to training CSV (2000-2007)")
    parser.add_argument("--val-csv", default=DEFAULT_VAL, help="Path to validation CSV (2008)")
    parser.add_argument("--years", type=int, default=10, help="Number of synthetic years to generate")
    parser.add_argument("--p", type=int, default=36, help="AR order (if not using --auto-order)")
    parser.add_argument("--auto-order", action="store_true", help="Auto-select p in 1..48 by minimum BIC")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--output-csv", default=DEFAULT_OUT, help="Path to save synthetic CSV")
    parser.add_argument("--card-json", default=DEFAULT_CARD, help="Path to save model card metadata")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 80)
    print("  VANILLA ARIMA RAINFALL GENERATION PIPELINE")
    print("=" * 80)

    # 1. Load and Verify Training Data
    if not os.path.exists(args.train_csv):
        print(f"ERROR: Training split not found at: {args.train_csv}")
        sys.exit(1)

    print(f"▶ Loading training split: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv, parse_dates=["date"])
    col = "avg_rainfall" if "avg_rainfall" in train_df.columns else "rainfall_intensity"
    x_train = train_df[col].values.astype(np.float64)
    train_dates = train_df["date"]
    N_train = len(x_train)

    # Integrity assertion: ensure no data leakage from 2008 or 2009
    max_year = train_dates.dt.year.max()
    min_year = train_dates.dt.year.min()
    print(f"  Training span   : {min_year}–{max_year} ({N_train:,} steps)")
    if max_year > 2007:
        print(f"ERROR: Training partition leaks years > 2007 (found {max_year})!")
        sys.exit(1)

    # Summary of observed training statistics
    mu_train = float(np.mean(x_train))
    var_train = float(np.var(x_train))
    zero_train_pct = float(np.mean(x_train < WET_THR) * 100.0)
    print(f"  Observed Mean   : {mu_train:.6f} mm / 5-min")
    print(f"  Observed Var    : {var_train:.6f}")
    print(f"  Zero Fraction   : {zero_train_pct:.2f}% (wet threshold < {WET_THR} mm)")

    # 2. Centering & Autocovariance
    y_train = x_train - mu_train
    max_k = max(args.p, 64)
    acov = np.array([np.mean(y_train[:N_train - k] * y_train[k:]) for k in range(max_k + 1)])

    # 3. Order Selection
    candidate_orders = [1, 2, 4, 8, 12, 16, 24, 36, 48, 64]
    bic_table = []
    print("\n▶ Evaluating Candidate AR Orders (Levinson-Durbin Recursion):")
    print(f"  {'p':>3} | {'sigma2':>12} | {'BIC':>14}")
    print("  " + "-" * 35)

    for cand_p in candidate_orders:
        _, s2_cand = levinson_durbin(acov, cand_p)
        bic = N_train * np.log(s2_cand) + cand_p * np.log(N_train)
        bic_table.append({"p": cand_p, "sigma2": float(s2_cand), "bic": float(bic)})
        print(f"  {cand_p:>3} | {s2_cand:>12.8f} | {bic:>14.2f}")

    if args.auto_order:
        selected_p = min(bic_table, key=lambda x: x["bic"])["p"]
        print(f"\n✓ Auto-selected optimal order: p = {selected_p} (minimum BIC)")
    else:
        selected_p = args.p
        print(f"\n✓ Using specified order: p = {selected_p}")

    # 4. Final Model Estimation
    phi, sigma2 = levinson_durbin(acov, selected_p)
    sigma = np.sqrt(sigma2)
    print(f"  Innovation std dev (sigma) : {sigma:.6f}")
    print(f"  Top 3 AR coefficients      : {np.round(phi[:3], 4)}")

    # 5. Out-of-sample validation on 2008 split (if available)
    val_mse = None
    if os.path.exists(args.val_csv):
        val_df = pd.read_csv(args.val_csv)
        val_col = "avg_rainfall" if "avg_rainfall" in val_df.columns else "rainfall_intensity"
        x_val = val_df[val_col].values.astype(np.float64)
        y_val = x_val - mu_train
        # Compute 1-step prediction residuals on validation set
        res_val = lfilter(np.r_[1.0, -phi], [1.0], y_val)[selected_p:]
        val_mse = float(np.mean(res_val ** 2))
        print(f"  Validation 1-step MSE (2008) : {val_mse:.8f}")

    # 6. Generate 10-Year Synthetic Realization
    N_gen = args.years * STEPS_PER_YEAR
    burn_in = 2000
    print(f"\n▶ Simulating {args.years} synthetic years ({N_gen:,} steps) with seed {args.seed}...")

    np.random.seed(args.seed)
    eps = np.random.normal(0, sigma, N_gen + burn_in)
    # Simulate AR process using all-pole IIR filter
    y_sim = lfilter([1.0], np.r_[1.0, -phi], eps)[burn_in:]
    x_raw = y_sim + mu_train

    # Diagnostics of Gaussian assumptions on rainfall
    pct_negative = float(np.mean(x_raw < 0.0) * 100.0)
    x_clipped = np.maximum(x_raw, 0.0)
    pct_zeros = float(np.mean(x_clipped < WET_THR) * 100.0)
    x_clipped[x_clipped < WET_THR] = 0.0

    annual_volume = float(np.sum(x_clipped) / args.years)
    max_burst = float(np.max(x_clipped))

    print(f"  Negative values clipped to 0.0  : {pct_negative:.2f}% (Gaussian artifact)")
    print(f"  Simulated Zero Fraction         : {pct_zeros:.2f}% (Real: {zero_train_pct:.2f}%)")
    print(f"  Simulated Annual Volume         : {annual_volume:.2f} mm/yr (Real: {mu_train * STEPS_PER_YEAR:.2f} mm/yr)")
    print(f"  Simulated Max 5-min Burst       : {max_burst:.4f} mm (Real max: {np.max(x_train):.4f} mm)")

    # 7. Save Synthetic CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    date_index = pd.date_range(start="2026-01-01 00:00:00", periods=N_gen, freq="5min")
    out_df = pd.DataFrame({"date": date_index, "avg_rainfall": x_clipped})
    out_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Saved synthetic series to: {args.output_csv}")

    # 8. Save Model Card Metadata
    card_data = {
        "model_name": "Vanilla ARIMA",
        "model_order": f"ARIMA({selected_p}, 0, 0)",
        "p": int(selected_p),
        "d": 0,
        "q": 0,
        "ar_coefficients": phi.tolist(),
        "innovation_sigma": float(sigma),
        "innovation_sigma2": float(sigma2),
        "mean_offset_mm": float(mu_train),
        "train_partition": os.path.relpath(args.train_csv, ROOT),
        "train_steps": int(N_train),
        "train_years": f"{min_year}-{max_year}",
        "val_partition": os.path.relpath(args.val_csv, ROOT) if os.path.exists(args.val_csv) else None,
        "val_1step_mse": val_mse,
        "bic_table": bic_table,
        "synthetic_realization": {
            "path": os.path.relpath(args.output_csv, ROOT),
            "years": int(args.years),
            "steps": int(N_gen),
            "seed": int(args.seed),
            "negative_clipped_pct": float(pct_negative),
            "synthetic_zero_pct": float(pct_zeros),
            "simulated_annual_volume_mm": float(annual_volume),
            "simulated_max_burst_mm": float(max_burst),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t_start, 2),
    }

    os.makedirs(os.path.dirname(args.card_json), exist_ok=True)
    with open(args.card_json, "w") as f:
        json.dump(card_data, f, indent=2)
    print(f"✓ Saved reproducible model card to: {args.card_json}")

    print("\n" + "=" * 80)
    print("  Generation complete in {:.2f}s".format(time.time() - t_start))
    print("  To score against Gate A and compare with v10:")
    print("    python scripts/gate_a_scorecard.py --reference train --versions v10 arima")
    print("=" * 80)


if __name__ == "__main__":
    main()
