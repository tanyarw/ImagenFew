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
4. Diagnostics (model card): ADF unit-root test, Ljung-Box and Jarque-Bera on the
   training residuals, and 2008 1-step MSE against persistence and the training mean.
5. Optional --transform asinh: fit on y = asinh(x / s) (s = 0.035, v14's transform),
   back-transform with s * sinh(y), then set the lowest p_dry share of steps to 0 so the
   dry fraction matches the training years.
6. Reproducibility & Output:
   - Writes generated time series to results/generated_data/rainfall_synthetic_10y_arima.csv
   - Saves fitted parameters, order selection table, and diagnostics to a JSON model card.

Usage:
    python scripts/baselines/run_vanilla_arima.py --years 10 --seed 42
    python scripts/baselines/run_vanilla_arima.py --p 64 --transform asinh \
        --output-csv results/generated_data/rainfall_synthetic_10y_arima_asinh_len64.csv \
        --card-json results/reference/arima_asinh_len64_model_card.json
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
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


def adf_test(y: np.ndarray, lags: int = 12):
    """Augmented Dickey-Fuller t-statistic (constant, `lags` lagged differences), by least squares.
    Critical values with a constant, large samples (MacKinnon 1996): 1% -3.43, 5% -2.86."""
    dy = np.diff(y)
    T = len(dy) - lags
    X = [np.ones(T), y[lags:-1]] + [dy[lags - i:len(dy) - i] for i in range(1, lags + 1)]
    X = np.column_stack(X)
    target = dy[lags:]
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    s2 = resid @ resid / (T - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(beta[1] / se)


def ljung_box(e: np.ndarray, h: int, n_params: int = 0):
    """Ljung-Box Q over lags 1..h and its chi-square p-value (df = h - n_params)."""
    e = e - e.mean()
    n, v = len(e), e @ e
    r = np.array([e[:n - k] @ e[k:] / v for k in range(1, h + 1)])
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, h + 1)))
    return float(q), float(stats.chi2.sf(q, max(h - n_params, 1)))


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
    parser.add_argument("--transform", default="none", choices=["none", "asinh"],
                        help="'asinh': fit on asinh(x / s) and restore the training dry fraction after back-transform")
    parser.add_argument("--asinh-scale", type=float, default=0.035, help="s in asinh(x / s) (v14 uses 0.035)")
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

    # 2. Transform (optional), Centering & Autocovariance
    fwd = (lambda a: np.arcsinh(a / args.asinh_scale)) if args.transform == "asinh" else (lambda a: a)
    inv = (lambda a: args.asinh_scale * np.sinh(a)) if args.transform == "asinh" else (lambda a: a)
    v_train = fwd(x_train)
    mu_fit = float(np.mean(v_train))
    y_train = v_train - mu_fit
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

    # 5. Diagnostics on the training fit
    res_train = lfilter(np.r_[1.0, -phi], [1.0], y_train)[selected_p:]
    adf_t = adf_test(v_train)
    lb12, lb12_p = ljung_box(res_train, 12)
    lb288, lb288_p = ljung_box(res_train, 288, n_params=selected_p)   # lags 1..p are whitened by construction
    jb = stats.jarque_bera(res_train)
    diagnostics = {
        "adf_t_stat": adf_t, "adf_crit_1pct": -3.43, "adf_crit_5pct": -2.86, "adf_lags": 12,
        "ljung_box_q12": lb12, "ljung_box_p12": lb12_p, "ljung_box_q288": lb288, "ljung_box_p288": lb288_p,
        "residual_skewness": float(stats.skew(res_train)),
        "residual_excess_kurtosis": float(stats.kurtosis(res_train)),
        "jarque_bera_stat": float(jb.statistic), "jarque_bera_p": float(jb.pvalue),
    }
    print(f"  ADF t-stat (constant, 12 lags): {adf_t:.1f} (5% critical -2.86: "
          f"{'unit root rejected' if adf_t < -2.86 else 'unit root NOT rejected'})")
    print(f"  Ljung-Box Q(12) = {lb12:.0f} (p = {lb12_p:.2g}), Q(288) = {lb288:.0f} (p = {lb288_p:.2g})")
    print(f"  Residual skewness {diagnostics['residual_skewness']:.1f}, excess kurtosis "
          f"{diagnostics['residual_excess_kurtosis']:.0f}, Jarque-Bera p = {jb.pvalue:.2g}")

    # 6. Out-of-sample 1-step forecasts on 2008 (in mm), against persistence and the training mean
    val_mse = None
    forecast = None
    if os.path.exists(args.val_csv):
        val_df = pd.read_csv(args.val_csv)
        val_col = "avg_rainfall" if "avg_rainfall" in val_df.columns else "rainfall_intensity"
        x_val = val_df[val_col].values.astype(np.float64)
        y_val = fwd(x_val) - mu_fit
        # Compute 1-step prediction residuals on validation set (in the fitted space)
        res_val = lfilter(np.r_[1.0, -phi], [1.0], y_val)[selected_p:]
        val_mse = float(np.mean(res_val ** 2))
        pred_mm = inv((y_val - lfilter(np.r_[1.0, -phi], [1.0], y_val)) + mu_fit)[selected_p:]
        truth = x_val[selected_p:]
        mse_ar = float(np.mean((pred_mm - truth) ** 2))
        mse_persist = float(np.mean((x_val[selected_p - 1:-1] - truth) ** 2))
        mse_mean = float(np.mean((mu_train - truth) ** 2))
        forecast = {"mse_ar_mm": mse_ar, "mse_persistence_mm": mse_persist, "mse_train_mean_mm": mse_mean,
                    "skill_vs_persistence": 1.0 - mse_ar / mse_persist, "skill_vs_mean": 1.0 - mse_ar / mse_mean}
        print(f"  Validation 1-step MSE (2008) : {val_mse:.8f}")
        print(f"  2008 1-step MSE in mm: AR {mse_ar:.3e} | persistence {mse_persist:.3e} | mean {mse_mean:.3e} "
              f"(skill vs persistence {forecast['skill_vs_persistence']:.1%})")

    # 7. Generate 10-Year Synthetic Realization
    N_gen = args.years * STEPS_PER_YEAR
    burn_in = 2000
    print(f"\n▶ Simulating {args.years} synthetic years ({N_gen:,} steps) with seed {args.seed}...")

    np.random.seed(args.seed)
    eps = np.random.normal(0, sigma, N_gen + burn_in)
    # Simulate AR process using all-pole IIR filter
    y_sim = lfilter([1.0], np.r_[1.0, -phi], eps)[burn_in:]
    x_raw = inv(y_sim + mu_fit)

    # Diagnostics of Gaussian assumptions on rainfall
    pct_negative = float(np.mean(x_raw < 0.0) * 100.0)
    x_clipped = np.maximum(x_raw, 0.0)
    dry_cut = None
    if args.transform == "asinh":
        # Restore the training dry fraction: the lowest p_dry share of steps becomes 0
        p_dry = float(np.mean(x_train < WET_THR))
        dry_cut = float(np.quantile(x_raw, p_dry))
        print(f"  Dry fraction after clipping alone: {np.mean(x_clipped < WET_THR) * 100:.2f}%; "
              f"cut at {dry_cut:.4f} mm restores {p_dry * 100:.2f}%")
        x_clipped[x_raw <= dry_cut] = 0.0
    pct_zeros = float(np.mean(x_clipped < WET_THR) * 100.0)
    x_clipped[x_clipped < WET_THR] = 0.0

    annual_volume = float(np.sum(x_clipped) / args.years)
    max_burst = float(np.max(x_clipped))

    print(f"  Negative values clipped to 0.0  : {pct_negative:.2f}% (Gaussian artifact)")
    print(f"  Simulated Zero Fraction         : {pct_zeros:.2f}% (Real: {zero_train_pct:.2f}%)")
    print(f"  Simulated Annual Volume         : {annual_volume:.2f} mm/yr (Real: {mu_train * STEPS_PER_YEAR:.2f} mm/yr)")
    print(f"  Simulated Max 5-min Burst       : {max_burst:.4f} mm (Real max: {np.max(x_train):.4f} mm)")

    # 8. Save Synthetic CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    date_index = pd.date_range(start="2026-01-01 00:00:00", periods=N_gen, freq="5min")
    out_df = pd.DataFrame({"date": date_index, "avg_rainfall": x_clipped})
    out_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Saved synthetic series to: {args.output_csv}")

    # 9. Save Model Card Metadata
    card_data = {
        "model_name": "Vanilla ARIMA" if args.transform == "none" else f"ARIMA on asinh(x / {args.asinh_scale})",
        "transform": args.transform,
        "asinh_scale": args.asinh_scale if args.transform == "asinh" else None,
        "dry_cut_mm": dry_cut,
        "model_order": f"ARIMA({selected_p}, 0, 0)",
        "p": int(selected_p),
        "d": 0,
        "q": 0,
        "ar_coefficients": phi.tolist(),
        "innovation_sigma": float(sigma),
        "innovation_sigma2": float(sigma2),
        "mean_offset_mm": float(mu_train),
        "mean_offset_fitted_space": mu_fit,
        "train_partition": os.path.relpath(args.train_csv, ROOT),
        "train_steps": int(N_train),
        "train_years": f"{min_year}-{max_year}",
        "val_partition": os.path.relpath(args.val_csv, ROOT) if os.path.exists(args.val_csv) else None,
        "val_1step_mse": val_mse,
        "val_1step_forecast_mm": forecast,
        "diagnostics": diagnostics,
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
