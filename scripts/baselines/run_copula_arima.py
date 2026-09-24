#!/usr/bin/env python3
"""
run_copula_arima.py — Latent Gaussian-Copula AR Baseline for Rainfall Generation
================================================================================
Combines a stationary linear Autoregressive (AR) process in a latent Gaussian space
with an Empirical Probability Integral Transform (PIT / Quantile Mapping) observation
model, decoupling temporal persistence from non-Gaussian marginal rainfall physics.

Methodology:
1. Strict Data Partitioning:
   - Fits exclusively on the clean 2000-2007 chronological training partition
     (data/rainfall/splits/train_years_labelled.csv, 840,960 steps).
   - Validates that no validation (2008) or test (2009) data is accessed during fitting.
2. Observation Model (Empirical Quantile Translator):
   - Maps latent Gaussian variable z to empirical rainfall x:
       x_t = Q_empirical(Phi(z_t))
   - Because 90.8% of historical rain is 0.0 mm, any z_t below z_threshold = Phi^-1(0.908)
     automatically outputs 0.0 mm, guaranteeing the exact zero fraction.
   - For z_t above the threshold, values map monotonically to the empirical positive rain
     distribution, perfectly matching gauge resolution and extreme cloudbursts.
3. Latent Autoregressive Fitting:
   - Wet observations have exact latent coordinates z_t = Phi^-1(u_t).
   - Dry observations are interval-censored in (-inf, z_threshold].
   - Runs a Stochastic EM / Gibbs procedure to impute latent autocorrelation during dry spells,
     solving Yule-Walker recursion on the stationary latent autocovariance.
4. Generation:
   - Simulates N_gen = years * 365 * 288 steps (10 years = 1,051,200 steps) of the latent AR(p).
   - Pushes the latent sequence through the EmpiricalQuantile translator.
   - Saves realization to results/generated_data/rainfall_synthetic_10y_arima_copula_len{P}.csv
   - Saves fitted coefficients and diagnostic metadata to a JSON model card.

Usage:
    python scripts/baselines/run_copula_arima.py --p 24 --years 10 --seed 42
    python scripts/baselines/run_copula_arima.py --p 64 --years 10 --seed 42
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.signal import lfilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN = os.path.join(ROOT, "data", "rainfall", "splits", "train_years_labelled.csv")
DEFAULT_VAL   = os.path.join(ROOT, "data", "rainfall", "splits", "val_years_labelled.csv")
WET_THR = 0.005  # Frozen project wet threshold (0.005 mm)
STEPS_PER_DAY = 288
STEPS_PER_YEAR = 365 * STEPS_PER_DAY  # 105,120 steps


class EmpiricalQuantile:
    """
    Invertible Empirical Quantile Translator:
    u in [0, 1) -> x in [0, max_rain] via the empirical CDF.
    """

    def __init__(self, x_train: np.ndarray, wet_threshold: float = WET_THR):
        self.wet_threshold = wet_threshold
        self.sorted_x = np.sort(x_train.astype(np.float64))
        self.N = len(self.sorted_x)
        self.p_zero = float(np.mean(self.sorted_x < self.wet_threshold))
        self.z_threshold = float(ndtri(self.p_zero))

    def z_to_x(self, z: np.ndarray) -> np.ndarray:
        """Translates latent Gaussian values z to physical rainfall x."""
        u = ndtr(z)
        idx = np.clip((u * self.N).astype(int), 0, self.N - 1)
        x = self.sorted_x[idx].copy()
        x[x < self.wet_threshold] = 0.0
        return x

    def x_to_initial_z(self, x: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        Maps real rainfall observations x to initial latent values z.
        Wet values map to exact ranks; dry values map below z_threshold.
        """
        rng = np.random.default_rng(seed)
        N = len(x)
        z = np.zeros(N, dtype=np.float64)

        dry_mask = (x < self.wet_threshold)
        n_dry = np.sum(dry_mask)
        n_wet = N - n_dry

        # Smooth rank ordering for dry states
        ranks = np.zeros(N, dtype=np.float64)
        ranks[dry_mask] = (rng.permutation(n_dry) + 0.5) / N * self.p_zero

        # Rank ordering for wet states
        wet_vals = x[~dry_mask]
        wet_order = np.argsort(np.argsort(wet_vals))
        ranks[~dry_mask] = self.p_zero + (wet_order + 0.5) / n_wet * (1.0 - self.p_zero - 1e-6)

        # Numerical safeguards against infinite quantiles
        ranks = np.clip(ranks, 1e-7, 1.0 - 1e-7)
        z = ndtri(ranks)
        return z


def levinson_durbin(r: np.ndarray, p: int):
    """Fits AR(p) parameters via the Levinson-Durbin recursion."""
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


def fit_copula_ar(eq: EmpiricalQuantile, x_train: np.ndarray, p: int, em_iters: int = 15, seed: int = 42):
    """
    Fits latent Gaussian AR(p) via Stochastic EM / Gibbs refinement over censored dry sites.
    """
    print(f"\n▶ Fitting Latent Gaussian AR(p={p}) on {len(x_train):,} steps...")
    rng = np.random.default_rng(seed)
    N = len(x_train)
    z = eq.x_to_initial_z(x_train, seed=seed)
    dry_mask = (x_train < eq.wet_threshold)

    # Initial sample autocovariance
    max_k = max(p, 64)
    acov = np.array([np.mean(z[:N - k] * z[k:]) for k in range(max_k + 1)])
    phi, sigma2 = levinson_durbin(acov, p)
    sigma = np.sqrt(max(sigma2, 1e-6))

    # Fast chromatic Gibbs sweeps to impute dry spell persistence
    z_hi = eq.z_threshold
    colors = p + 1

    for it in range(1, em_iters + 1):
        # Color sweep: update dry sites conditionally given neighbors
        for c in range(colors):
            idx = np.arange(c, N, colors)
            idx_dry = idx[dry_mask[idx]]
            if len(idx_dry) == 0:
                continue

            # Conditional mean from past and future AR neighbors
            # For general AR(p): mu_cond approximates AR prediction
            prev_idx = np.clip(idx_dry - 1, 0, N - 1)
            next_idx = np.clip(idx_dry + 1, 0, N - 1)
            mu_cond = 0.5 * phi[0] * (z[prev_idx] + z[next_idx])
            sd_cond = sigma

            # Sample from truncated normal: Z <= z_hi
            alpha = (z_hi - mu_cond) / sd_cond
            u_hi = ndtr(alpha)
            u_samp = rng.uniform(1e-7, np.maximum(u_hi, 1e-6))
            z[idx_dry] = mu_cond + sd_cond * ndtri(u_samp)

        # Re-estimate autocovariance on updated latent series
        acov = np.array([np.mean(z[:N - k] * z[k:]) for k in range(max_k + 1)])
        phi, sigma2 = levinson_durbin(acov, p)
        sigma = np.sqrt(max(sigma2, 1e-6))

        if it % 5 == 0 or it == em_iters:
            print(f"  EM Iteration {it:2d}/{em_iters} | sigma = {sigma:.4f} | phi[:3] = {np.round(phi[:3], 4)}")

    return phi, float(sigma2), z


def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian-Copula AR on clean training holdout and generate synthetic rainfall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN, help="Path to training CSV (2000-2007)")
    parser.add_argument("--val-csv", default=DEFAULT_VAL, help="Path to validation CSV (2008)")
    parser.add_argument("--p", type=int, default=64, choices=[24, 64], help="AR order matching v8 (24) or v10 (64)")
    parser.add_argument("--years", type=int, default=10, help="Number of synthetic years to generate")
    parser.add_argument("--em-iters", type=int, default=15, help="Stochastic-EM Gibbs refinement iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--output-csv", default=None, help="Path to save synthetic CSV (auto-named if None)")
    parser.add_argument("--card-json", default=None, help="Path to save model card metadata (auto-named if None)")
    args = parser.parse_args()

    t_start = time.time()
    p = args.p
    default_out = os.path.join(ROOT, "results", "generated_data", f"rainfall_synthetic_10y_arima_copula_len{p}.csv")
    default_card = os.path.join(ROOT, "results", "reference", f"arima_copula_len{p}_model_card.json")
    out_csv = args.output_csv or default_out
    card_json = args.card_json or default_card

    print("=" * 80)
    print(f"  GAUSSIAN-COPULA AR(p={p}) RAINFALL GENERATION PIPELINE")
    print(f"  Matching Receptive Field: {p} steps ({(p * 5) / 60:.2f} hours)")
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

    max_year = train_dates.dt.year.max()
    min_year = train_dates.dt.year.min()
    print(f"  Training span   : {min_year}–{max_year} ({N_train:,} steps)")
    if max_year > 2007:
        print(f"ERROR: Training partition leaks years > 2007 (found {max_year})!")
        sys.exit(1)

    # 2. Build Empirical Quantile Translator
    eq = EmpiricalQuantile(x_train, wet_threshold=WET_THR)
    print(f"  Observed Mean   : {np.mean(x_train):.6f} mm / 5-min")
    print(f"  Zero Fraction   : {eq.p_zero * 100:.2f}% (wet threshold < {WET_THR} mm)")
    print(f"  Latent Threshold: z = {eq.z_threshold:.4f} (values below this map to 0.0 mm)")

    # 3. Fit Latent Autoregressive Model
    phi, sigma2, z_fit = fit_copula_ar(eq, x_train, p=p, em_iters=args.em_iters, seed=args.seed)
    sigma = np.sqrt(sigma2)
    print(f"\n✓ Fitted Latent AR({p}):")
    print(f"  Innovation sigma: {sigma:.6f}")
    print(f"  Top 3 AR weights: {np.round(phi[:3], 4)}")

    # 4. Out-of-sample validation on 2008 split
    val_mse = None
    if os.path.exists(args.val_csv):
        val_df = pd.read_csv(args.val_csv)
        val_col = "avg_rainfall" if "avg_rainfall" in val_df.columns else "rainfall_intensity"
        x_val = val_df[val_col].values.astype(np.float64)
        z_val = eq.x_to_initial_z(x_val, seed=args.seed)
        res_val = lfilter(np.r_[1.0, -phi], [1.0], z_val)[p:]
        val_mse = float(np.mean(res_val ** 2))
        print(f"  Validation 1-step Latent MSE (2008): {val_mse:.6f}")

    # 5. Generate 10-Year Synthetic Realization
    N_gen = args.years * STEPS_PER_YEAR
    burn_in = 2000
    print(f"\n▶ Simulating {args.years} synthetic years ({N_gen:,} steps) with seed {args.seed}...")

    np.random.seed(args.seed)
    eps_sim = np.random.normal(0, sigma, N_gen + burn_in)
    z_sim = lfilter([1.0], np.r_[1.0, -phi], eps_sim)[burn_in:]

    # Standardize simulated latent series to exact standard normal N(0, 1)
    z_sim = (z_sim - np.mean(z_sim)) / (np.std(z_sim) + 1e-9)

    # Pass through Empirical Quantile Translator
    x_sim = eq.z_to_x(z_sim)
    pct_zeros = float(np.mean(x_sim < WET_THR) * 100.0)
    annual_volume = float(np.sum(x_sim) / args.years)
    max_burst = float(np.max(x_sim))

    print(f"  Simulated Zero Fraction   : {pct_zeros:.2f}% (Real train: {eq.p_zero * 100:.2f}%)")
    print(f"  Simulated Annual Volume   : {annual_volume:.2f} mm/yr (Real train: {np.mean(x_train) * STEPS_PER_YEAR:.2f} mm/yr)")
    print(f"  Simulated Max 5-min Burst : {max_burst:.4f} mm (Real max: {np.max(x_train):.4f} mm)")

    # 6. Save Synthetic CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    date_index = pd.date_range(start="2026-01-01 00:00:00", periods=N_gen, freq="5min")
    out_df = pd.DataFrame({"date": date_index, "avg_rainfall": x_sim})
    out_df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved synthetic series to: {out_csv}")

    # 7. Save Model Card Metadata
    card_data = {
        "model_name": f"Gaussian-Copula AR({p})",
        "p": int(p),
        "horizon_hours": float((p * 5) / 60),
        "matched_diffusion_version": "v8" if p == 24 else "v10",
        "ar_coefficients": phi.tolist(),
        "innovation_sigma": float(sigma),
        "innovation_sigma2": float(sigma2),
        "latent_zero_threshold": float(eq.z_threshold),
        "train_partition": os.path.relpath(args.train_csv, ROOT),
        "train_steps": int(N_train),
        "train_years": f"{min_year}-{max_year}",
        "val_partition": os.path.relpath(args.val_csv, ROOT) if os.path.exists(args.val_csv) else None,
        "val_latent_mse": val_mse,
        "synthetic_realization": {
            "path": os.path.relpath(out_csv, ROOT),
            "years": int(args.years),
            "steps": int(N_gen),
            "seed": int(args.seed),
            "synthetic_zero_pct": float(pct_zeros),
            "simulated_annual_volume_mm": float(annual_volume),
            "simulated_max_burst_mm": float(max_burst),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t_start, 2),
    }

    os.makedirs(os.path.dirname(card_json), exist_ok=True)
    with open(card_json, "w") as f:
        json.dump(card_data, f, indent=2)
    print(f"✓ Saved reproducible model card to: {card_json}")

    print("\n" + "=" * 80)
    print(f"  Generation complete in {time.time() - t_start:.2f}s")
    print("  To score against Gate A and compare with diffusion:")
    version_label = f"arima_copula_len{p}"
    print(f"    python scripts/gate_a_scorecard.py --reference train --versions v10 {version_label}")
    print("=" * 80)


if __name__ == "__main__":
    main()
