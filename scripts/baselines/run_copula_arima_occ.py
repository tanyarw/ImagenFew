#!/usr/bin/env python3
"""
run_copula_arima_occ.py — Occurrence-Matched Gaussian-Copula AR Baseline
========================================================================
Same model as run_copula_arima.py (a latent Gaussian AR(p) pushed through the empirical
quantile function of the training rain), fitted differently. The latent autocorrelation is
chosen lag by lag so that the generated series rains at two steps k apart exactly as often
as real rain does. This replaces the censored-imputation loop of run_copula_arima.py,
which settled at a latent lag-1 correlation of 0.47 where the data need 0.976.

Methodology:
1. Strict Data Partitioning:
   - Fits exclusively on the clean 2000-2007 chronological training partition
     (data/rainfall/splits/train_years_labelled.csv, 840,960 steps).
   - 2008 is used only to report how far real wet-wet probabilities move between years.
2. Observation Model: as run_copula_arima.py, x_t = Q_emp(Phi(z_t)); z_t <= c = Phi^-1(p_dry)
   gives 0.0 mm. With --assembly calendar, one quantile table per HMM state (as
   run_seasonal_copula_arima.py) and the state timeline is the 365-day climatology profile.
3. Latent Autocorrelation (the change):
   - For k = 1..p, the target is the observed joint wet probability q_k = P(wet_t, wet_{t+k}).
   - For a standard bivariate normal with correlation rho, P(Z1 > c, Z2 > c) is
       (1 - Phi(c))^2 + integral_0^rho exp(-c^2 / (1 + r)) / (2 pi sqrt(1 - r^2)) dr,
     which increases with rho, so rho_k is the unique root of P(rho) = q_k.
     (Calendar mode: each state's threshold, weighted by its share of time.)
   - Levinson-Durbin on (1, rho_1, ..., rho_p) gives the AR(p) whose autocorrelation equals
     the targets exactly at lags 1..p (Yule-Walker). The Toeplitz matrix is checked for
     positive definiteness first.
4. Generation: as run_copula_arima.py (Gaussian innovations, 2,000-step burn-in, latent series
   standardised to N(0, 1), quantile mapping). Saves the CSV and a JSON model card.

Usage:
    python scripts/baselines/run_copula_arima_occ.py --p 64 --years 10 --seed 42
    python scripts/baselines/run_copula_arima_occ.py --p 64 --assembly calendar --years 10 --seed 42
"""

import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_copula_arima import EmpiricalQuantile, levinson_durbin  # noqa: E402
from run_seasonal_copula_arima import SeasonalEmpiricalQuantile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN = os.path.join(ROOT, "data", "rainfall", "splits", "train_years_labelled.csv")
DEFAULT_VAL   = os.path.join(ROOT, "data", "rainfall", "splits", "val_years_labelled.csv")
WET_THR = 0.005  # Frozen project wet threshold (0.005 mm)
STEPS_PER_DAY = 288
STEPS_PER_YEAR = 365 * STEPS_PER_DAY  # 105,120 steps
REPORT_LAGS = [1, 2, 3, 6, 12, 24, 36, 64]


def joint_wet(wet: np.ndarray, k: int) -> float:
    """Observed P(wet_t and wet_{t+k})."""
    return float(np.mean(wet[:-k] & wet[k:]))


def both_above(c: float, rho: float) -> float:
    """P(Z1 > c, Z2 > c) for a standard bivariate normal with correlation rho (rho in [0, 1))."""
    tail = 1.0 - ndtr(c)
    integral, _ = quad(lambda r: np.exp(-c * c / (1.0 + r)) / (2.0 * np.pi * np.sqrt(1.0 - r * r)),
                       0.0, rho, limit=200)
    return tail * tail + integral


def latent_targets(wet: np.ndarray, thresholds: np.ndarray, weights: np.ndarray, p: int):
    """Latent correlation rho_k (k = 1..p) that reproduces the observed joint wet probabilities.
    `thresholds` / `weights`: one latent threshold per state and its share of time (one state
    in plain mode)."""
    q = np.array([joint_wet(wet, k) for k in range(1, p + 1)])
    rho = np.zeros(p)
    for i, qk in enumerate(q):
        f = lambda r: sum(w * both_above(c, r) for c, w in zip(thresholds, weights)) - qk
        if f(0.0) >= 0:      # no more joint rain than independence: no positive correlation needed
            rho[i] = 0.0
            continue
        rho[i] = brentq(f, 0.0, 1.0 - 1e-9, xtol=1e-10)
    return q, rho


def main():
    parser = argparse.ArgumentParser(
        description="Fit an occurrence-matched Gaussian-copula AR on 2000-2007 and generate synthetic rainfall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN, help="Path to training CSV (2000-2007)")
    parser.add_argument("--val-csv", default=DEFAULT_VAL, help="Path to validation CSV (2008)")
    parser.add_argument("--p", type=int, default=64, help="AR order (64 matches v10/v14 block length)")
    parser.add_argument("--assembly", default="none", choices=["none", "calendar"],
                        help="'none': one quantile table; 'calendar': per-state tables on the climatology profile")
    parser.add_argument("--years", type=int, default=10, help="Number of synthetic years to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--output-csv", default=None, help="Path to save synthetic CSV (auto-named if None)")
    parser.add_argument("--card-json", default=None, help="Path to save model card metadata (auto-named if None)")
    args = parser.parse_args()

    t_start = time.time()
    p = args.p
    name = f"arima_copula_occ{'_cal' if args.assembly == 'calendar' else ''}_len{p}"
    out_csv = args.output_csv or os.path.join(ROOT, "results", "generated_data", f"rainfall_synthetic_10y_{name}.csv")
    card_json = args.card_json or os.path.join(ROOT, "results", "reference", f"{name}_model_card.json")

    print("=" * 80)
    print(f"  OCCURRENCE-MATCHED GAUSSIAN-COPULA AR(p={p}) [{args.assembly}]")
    print("=" * 80)

    # 1. Load and Verify Training Data
    train_df = pd.read_csv(args.train_csv, parse_dates=["date"])
    col = "avg_rainfall" if "avg_rainfall" in train_df.columns else "rainfall_intensity"
    x_train = train_df[col].values.astype(np.float64)
    min_year, max_year = train_df["date"].dt.year.min(), train_df["date"].dt.year.max()
    print(f"  Training span   : {min_year}–{max_year} ({len(x_train):,} steps)")
    if max_year > 2007:
        print(f"ERROR: Training partition leaks years > 2007 (found {max_year})!")
        sys.exit(1)
    wet = x_train >= WET_THR

    # 2. Observation model and latent thresholds
    if args.assembly == "calendar":
        states_train = train_df["hmm_state"].values.astype(int)
        seq = SeasonalEmpiricalQuantile(x_train, states_train, wet_threshold=WET_THR)
        thresholds = np.array([seq.z_threshold[s] for s in range(4)])
        weights = np.array([np.mean(states_train == s) for s in range(4)])
    else:
        eq = EmpiricalQuantile(x_train, wet_threshold=WET_THR)
        thresholds, weights = np.array([eq.z_threshold]), np.array([1.0])
    print(f"  Latent thresholds: {np.round(thresholds, 4)} (weights {np.round(weights, 3)})")

    # 3. Latent correlation targets and AR fit
    q, rho = latent_targets(wet, thresholds, weights, p)
    r = np.r_[1.0, rho]
    T = r[np.abs(np.subtract.outer(np.arange(p + 1), np.arange(p + 1)))]
    min_eig = float(np.linalg.eigvalsh(T).min())
    print(f"\n▶ Latent targets: rho at lags {REPORT_LAGS} = "
          f"{np.round([rho[k - 1] for k in REPORT_LAGS if k <= p], 4)}")
    print(f"  Toeplitz min eigenvalue: {min_eig:.3e} ({'valid' if min_eig > 0 else 'NOT positive definite'})")
    if min_eig <= 0:
        print("ERROR: target autocorrelations do not define a valid AR process.")
        sys.exit(1)
    phi, sigma2 = levinson_durbin(r, p)
    sigma = np.sqrt(sigma2)
    print(f"  Innovation sigma: {sigma:.6f}  |  phi[:3] = {np.round(phi[:3], 4)}  |  sum(phi) = {phi.sum():.4f}")

    # 4. Year-to-year reference: the same targets on 2008
    q_val = None
    if os.path.exists(args.val_csv):
        val_df = pd.read_csv(args.val_csv)
        wet_val = val_df[col].values.astype(np.float64) >= WET_THR
        q_val = {k: joint_wet(wet_val, k) for k in REPORT_LAGS if k <= p}

    # 5. Generate
    N_gen = args.years * STEPS_PER_YEAR
    burn_in = 2000
    np.random.seed(args.seed)
    eps_sim = np.random.normal(0, sigma, N_gen + burn_in)
    z_sim = lfilter([1.0], np.r_[1.0, -phi], eps_sim)[burn_in:]
    z_sim = (z_sim - np.mean(z_sim)) / (np.std(z_sim) + 1e-9)
    if args.assembly == "calendar":
        with open(os.path.join(ROOT, "data", "rainfall", "splits", f"seasonal_transition_matrix_train_len{p}.pkl"), "rb") as f:
            clim = np.asarray(pickle.load(f)["climatology_profile_1yr"])
        x_sim = seq.z_to_x(z_sim, np.tile(clim, args.years)[:N_gen])
    else:
        x_sim = eq.z_to_x(z_sim)

    # 6. Check: generated vs target joint wet probabilities
    wet_sim = x_sim >= WET_THR
    check = []
    print(f"\n  {'lag':>4} | {'target q_k':>10} | {'generated':>10} | {'2008 real':>10}")
    for k in REPORT_LAGS:
        if k > p:
            continue
        g = joint_wet(wet_sim, k)
        check.append({"lag": k, "target": q[k - 1], "generated": g, "real_2008": q_val[k] if q_val else None,
                      "rho": float(rho[k - 1])})
        print(f"  {k:>4} | {q[k - 1]:>10.5f} | {g:>10.5f} | {q_val[k] if q_val else float('nan'):>10.5f}")
    pct_zeros = float(np.mean(~wet_sim) * 100.0)
    annual_volume = float(np.sum(x_sim) / args.years)
    max_burst = float(np.max(x_sim))
    print(f"\n  Simulated Zero Fraction   : {pct_zeros:.2f}% (Real train: {np.mean(~wet) * 100:.2f}%)")
    print(f"  Simulated Annual Volume   : {annual_volume:.2f} mm/yr (Real train: {np.mean(x_train) * STEPS_PER_YEAR:.2f} mm/yr)")
    print(f"  Simulated Max 5-min Burst : {max_burst:.4f} mm (Real max: {np.max(x_train):.4f} mm)")

    # 7. Save Synthetic CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    date_index = pd.date_range(start="2026-01-01 00:00:00", periods=N_gen, freq="5min")
    pd.DataFrame({"date": date_index, "avg_rainfall": x_sim}).to_csv(out_csv, index=False)
    print(f"\n✓ Saved synthetic series to: {out_csv}")

    # 8. Save Model Card Metadata
    card = {
        "model_name": f"Occurrence-Matched Gaussian-Copula AR({p})" + (" [Calendar Mode]" if args.assembly == "calendar" else ""),
        "p": int(p),
        "horizon_hours": float((p * 5) / 60),
        "assembly_mode": args.assembly,
        "matched_diffusion_version": "v14_nobridge_cal" if args.assembly == "calendar" else "v14",
        "fit_method": "latent correlation solved per lag from observed joint wet probabilities; Levinson-Durbin",
        "latent_rho_targets": rho.tolist(),
        "joint_wet_targets": q.tolist(),
        "toeplitz_min_eigenvalue": min_eig,
        "ar_coefficients": phi.tolist(),
        "innovation_sigma": float(sigma),
        "innovation_sigma2": float(sigma2),
        "latent_thresholds": thresholds.tolist(),
        "threshold_weights": weights.tolist(),
        "occurrence_check": check,
        "train_partition": os.path.relpath(args.train_csv, ROOT),
        "train_steps": int(len(x_train)),
        "train_years": f"{min_year}-{max_year}",
        "synthetic_realization": {
            "path": os.path.relpath(out_csv, ROOT),
            "years": int(args.years),
            "steps": int(N_gen),
            "seed": int(args.seed),
            "synthetic_zero_pct": pct_zeros,
            "simulated_annual_volume_mm": annual_volume,
            "simulated_max_burst_mm": max_burst,
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t_start, 2),
    }
    os.makedirs(os.path.dirname(card_json), exist_ok=True)
    with open(card_json, "w") as f:
        json.dump(card, f, indent=2)
    print(f"✓ Saved reproducible model card to: {card_json}")
    print(f"  Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
