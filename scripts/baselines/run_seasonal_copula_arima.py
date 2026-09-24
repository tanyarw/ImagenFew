#!/usr/bin/env python3
"""
run_seasonal_copula_arima.py — Seasonal Gaussian-Copula AR Baseline for Rainfall Generation
==========================================================================================
Integrates a stationary linear Autoregressive (AR) process in a latent Gaussian space
with a 4-regime Empirical Probability Integral Transform (PIT / Quantile Mapping)
observation model conditioned on seasonal states (hmm_state in {0, 1, 2, 3}).

Supports both sequence assembly paradigms used across the ImagenFew benchmark:
  1. Markov Mode: Block-level stochastic Markov chain sampling from the empirical
     block transition matrix P_block and stationary distribution pi_block.
  2. Calendar Mode: Deterministic annual seasonal climatology replay from the 365-day
     profile (climatology_profile_1yr), locking wet/dry seasons to real calendar months.

Methodology:
1. Strict Data Partitioning:
   - Fits exclusively on the clean 2000-2007 chronological training partition
     (data/rainfall/splits/train_years_labelled.csv, 840,960 steps).
   - Validates that no validation (2008) or test (2009) data is accessed during fitting.
2. 4-Regime Observation Model:
   - For each regime s in {0, 1, 2, 3}, constructs an empirical quantile translator Q_s(u):
       x_t = Q_{S_t}(Phi(z_t))
   - Regime 0 (dry): ~91.9% zeros, max burst ~1.98 mm.
   - Regime 1 (moderate): ~90.3% zeros, max burst ~5.55 mm.
   - Regime 2 (wet): ~89.6% zeros, max burst ~3.65 mm.
   - Regime 3 (extreme convective): ~91.8% zeros, high P99_wet, max burst ~4.10 mm.
   - Any latent Gaussian z_t <= z_threshold[s] maps to 0.0 mm, guaranteeing the regime's
     exact zero fraction.
3. Latent Autoregressive Fitting:
   - Inverts training observations to standard normal latent coordinates z_t:
     Wet observations map to within-regime empirical rank quantiles;
     Dry observations are interval-censored in (-inf, z_threshold[S_t]].
   - Runs a Chromatic Stochastic EM / Gibbs refinement to impute latent autocorrelation
     during dry spells, bounded above by each step's active regime threshold z_threshold[S_t].
   - Solves Levinson-Durbin recursion on the imputed latent autocovariance.
4. Sequence Assembly & Generation:
   - Simulates N_gen = years * 365 * 288 steps (10 years = 1,051,200 steps) of the latent AR(p).
   - Assembles the regime sequence S_t via Markov random walk or Calendar mode.
   - Translates z_t -> x_t step-by-step using Q_{S_t}(Phi(z_t)).
   - Saves realizations to results/generated_data/
   - Saves reproducible model cards to results/reference/

Usage:
    python scripts/baselines/run_seasonal_copula_arima.py --p 64 --assembly-mode both --years 10 --seed 42
    python scripts/baselines/run_seasonal_copula_arima.py --p 24 --assembly-mode both --years 10 --seed 42
"""

import argparse
import json
import os
import pickle
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


class SeasonalEmpiricalQuantile:
    """
    4-Regime Empirical Quantile Translator:
    Translates latent Gaussian coordinates z_t to physical rainfall x_t conditioned on
    active seasonal regime S_t in {0, 1, 2, 3}.
    """

    def __init__(self, x_train: np.ndarray, states_train: np.ndarray, wet_threshold: float = WET_THR):
        self.wet_threshold = wet_threshold
        self.n_states = 4
        self.translators = {}
        self.p_zero = {}
        self.z_threshold = {}
        self.N_s = {}

        for s in range(self.n_states):
            mask = (states_train == s)
            sub_x = np.sort(x_train[mask].astype(np.float64))
            p_0 = float(np.mean(sub_x < self.wet_threshold))
            z_thr = float(ndtri(p_0))
            self.translators[s] = sub_x
            self.p_zero[s] = p_0
            self.z_threshold[s] = z_thr
            self.N_s[s] = len(sub_x)

    def z_to_x(self, z: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Translates latent Gaussian values z conditioned on regime sequence to physical rainfall x."""
        u = ndtr(z)
        x = np.zeros_like(z, dtype=np.float64)

        for s in range(self.n_states):
            mask = (states == s)
            if not np.any(mask):
                continue
            sub_u = u[mask]
            sub_sorted = self.translators[s]
            N_s = self.N_s[s]
            idx = np.clip((sub_u * N_s).astype(int), 0, N_s - 1)
            sub_x = sub_sorted[idx].copy()
            sub_x[sub_x < self.wet_threshold] = 0.0
            x[mask] = sub_x

        return x

    def x_to_initial_z(self, x: np.ndarray, states: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        Maps physical rainfall observations x conditioned on states to initial latent values z.
        Wet observations map to exact rank quantiles; dry observations map below regime threshold.
        """
        rng = np.random.default_rng(seed)
        N = len(x)
        z = np.zeros(N, dtype=np.float64)

        for s in range(self.n_states):
            mask = (states == s)
            if not np.any(mask):
                continue
            sub_x = x[mask]
            n_sub = len(sub_x)
            p_0 = self.p_zero[s]

            dry_mask = (sub_x < self.wet_threshold)
            n_dry = np.sum(dry_mask)
            n_wet = n_sub - n_dry

            ranks = np.zeros(n_sub, dtype=np.float64)
            ranks[dry_mask] = (rng.permutation(n_dry) + 0.5) / n_sub * p_0
            if n_wet > 0:
                wet_vals = sub_x[~dry_mask]
                wet_order = np.argsort(np.argsort(wet_vals))
                ranks[~dry_mask] = p_0 + (wet_order + 0.5) / n_wet * (1.0 - p_0 - 1e-6)

            ranks = np.clip(ranks, 1e-7, 1.0 - 1e-7)
            z[mask] = ndtri(ranks)

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


def fit_seasonal_copula_ar(
    seq: SeasonalEmpiricalQuantile,
    x_train: np.ndarray,
    states_train: np.ndarray,
    p: int,
    em_iters: int = 15,
    seed: int = 42,
):
    """
    Fits latent Gaussian AR(p) via Stochastic EM / Gibbs refinement over censored dry sites,
    incorporating time-varying regime thresholds z_threshold[S_t].
    """
    print(f"\n▶ Fitting Latent Gaussian AR(p={p}) conditioned on 4 Seasonal Regimes ({len(x_train):,} steps)...")
    rng = np.random.default_rng(seed)
    N = len(x_train)
    z = seq.x_to_initial_z(x_train, states_train, seed=seed)
    dry_mask = (x_train < seq.wet_threshold)

    # Time-varying upper bounds for dry values
    z_thr_all = np.array([seq.z_threshold[s] for s in states_train], dtype=np.float64)

    # Initial sample autocovariance
    max_k = max(p, 64)
    acov = np.array([np.mean(z[:N - k] * z[k:]) for k in range(max_k + 1)])
    phi, sigma2 = levinson_durbin(acov, p)
    sigma = np.sqrt(max(sigma2, 1e-6))

    # Fast chromatic Gibbs sweeps
    colors = p + 1
    for it in range(1, em_iters + 1):
        for c in range(colors):
            idx = np.arange(c, N, colors)
            idx_dry = idx[dry_mask[idx]]
            if len(idx_dry) == 0:
                continue

            prev_idx = np.clip(idx_dry - 1, 0, N - 1)
            next_idx = np.clip(idx_dry + 1, 0, N - 1)
            mu_cond = 0.5 * phi[0] * (z[prev_idx] + z[next_idx])
            sd_cond = sigma

            alpha = (z_thr_all[idx_dry] - mu_cond) / sd_cond
            u_hi = ndtr(alpha)
            u_samp = rng.uniform(1e-7, np.maximum(u_hi, 1e-6))
            z[idx_dry] = mu_cond + sd_cond * ndtri(u_samp)

        acov = np.array([np.mean(z[:N - k] * z[k:]) for k in range(max_k + 1)])
        phi, sigma2 = levinson_durbin(acov, p)
        sigma = np.sqrt(max(sigma2, 1e-6))

        if it % 5 == 0 or it == em_iters:
            print(f"  EM Iteration {it:2d}/{em_iters} | sigma = {sigma:.4f} | phi[:3] = {np.round(phi[:3], 4)}")

    return phi, float(sigma2), z


def sample_markov_sequence(P_block: np.ndarray, pi_block: np.ndarray, n_blocks: int, seed: int = 42) -> np.ndarray:
    """Samples block state sequence from 1st-order Markov chain."""
    rng = np.random.default_rng(seed)
    seq = np.zeros(n_blocks, dtype=int)
    seq[0] = rng.choice(len(pi_block), p=pi_block)
    for i in range(n_blocks - 1):
        seq[i + 1] = rng.choice(len(pi_block), p=P_block[seq[i]])
    return seq


def generate_realization(
    mode: str,
    z_sim: np.ndarray,
    seq: SeasonalEmpiricalQuantile,
    trans_data: dict,
    p: int,
    years: int,
    N_gen: int,
    seed: int,
    out_csv: str,
    card_json: str,
    phi: np.ndarray,
    sigma: float,
    sigma2: float,
    N_train: int,
    min_year: int,
    max_year: int,
    args: argparse.Namespace,
    val_mse: float = None,
    t_pipeline_start: float = 0.0,
):
    """Generates and saves a synthetic rainfall realization under Markov or Calendar assembly."""
    t_start = time.time()
    print("-" * 80)
    print(f"▶ Generating Realization: ASSEMBLY MODE = {mode.upper()} (p={p}, years={years})")
    print("-" * 80)

    # 1. Assemble Seasonal State Sequence
    if mode == "markov":
        P_block = trans_data["P_block"]
        pi_block = trans_data["pi_block"]
        n_blocks = int(np.ceil(N_gen / p))
        block_states = sample_markov_sequence(P_block, pi_block, n_blocks, seed=seed)
        states_gen = np.repeat(block_states, p)[:N_gen]
        desc_mode = f"1st-order Markov chain sampled over {n_blocks:,} blocks of size {p}"
    elif mode == "calendar":
        if "climatology_profile_1yr" in trans_data:
            clim = trans_data["climatology_profile_1yr"]
            states_gen = np.tile(clim, years)[:N_gen]
        else:
            cal_seq = trans_data["calendar_sequence_1yr"]
            states_gen = np.repeat(np.tile(cal_seq, years), p)[:N_gen]
        desc_mode = f"Deterministic annual climatology cycle tiled over {years} years"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. Translate Latent Trajectory Through Regime-Specific Quantiles
    x_sim = seq.z_to_x(z_sim, states_gen)
    pct_zeros = float(np.mean(x_sim < WET_THR) * 100.0)
    annual_volume = float(np.sum(x_sim) / years)
    max_burst = float(np.max(x_sim))

    print(f"  Assembly description      : {desc_mode}")
    print(f"  Simulated Zero Fraction   : {pct_zeros:.2f}% (Target: ~90.8%)")
    print(f"  Simulated Annual Volume   : {annual_volume:.2f} mm/yr (Target: ~703 mm/yr)")
    print(f"  Simulated Max 5-min Burst : {max_burst:.4f} mm")

    # Regime distribution in generated series
    regime_counts = pd.Series(states_gen).value_counts().sort_index()
    print("  Regime distribution in realization:")
    for s_idx in range(4):
        cnt = regime_counts.get(s_idx, 0)
        pct = (cnt / N_gen) * 100.0
        sub_rain = x_sim[states_gen == s_idx]
        sub_mean = np.mean(sub_rain) * STEPS_PER_YEAR if len(sub_rain) > 0 else 0.0
        print(f"    State {s_idx}: {cnt:7d} steps ({pct:5.2f}%) | mean vol = {sub_mean:6.1f} mm/yr")

    # 3. Save Synthetic CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    date_index = pd.date_range(start="2026-01-01 00:00:00", periods=N_gen, freq="5min")
    out_df = pd.DataFrame({"date": date_index, "avg_rainfall": x_sim})
    out_df.to_csv(out_csv, index=False)
    print(f"✓ Saved synthetic realization to: {out_csv}")

    # 4. Save Model Card Metadata
    card_data = {
        "model_name": f"Seasonal Gaussian-Copula AR({p}) [{mode.capitalize()} Mode]",
        "p": int(p),
        "horizon_hours": float((p * 5) / 60),
        "assembly_mode": mode,
        "assembly_description": desc_mode,
        "matched_diffusion_version": f"v10{'_cal' if mode == 'calendar' else ''}" if p == 64 else f"v8{'_cal' if mode == 'calendar' else ''}",
        "ar_coefficients": phi.tolist(),
        "innovation_sigma": float(sigma),
        "innovation_sigma2": float(sigma2),
        "regime_thresholds": {int(s): float(seq.z_threshold[s]) for s in range(4)},
        "regime_p_zero": {int(s): float(seq.p_zero[s]) for s in range(4)},
        "train_partition": os.path.relpath(args.train_csv, ROOT),
        "train_steps": int(N_train),
        "train_years": f"{min_year}-{max_year}",
        "val_partition": os.path.relpath(args.val_csv, ROOT) if os.path.exists(args.val_csv) else None,
        "val_latent_mse": val_mse,
        "synthetic_realization": {
            "path": os.path.relpath(out_csv, ROOT),
            "years": int(years),
            "steps": int(N_gen),
            "seed": int(seed),
            "synthetic_zero_pct": float(pct_zeros),
            "simulated_annual_volume_mm": float(annual_volume),
            "simulated_max_burst_mm": float(max_burst),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation_time_sec": round(time.time() - t_start, 2),
    }

    os.makedirs(os.path.dirname(card_json), exist_ok=True)
    with open(card_json, "w") as f:
        json.dump(card_data, f, indent=2)
    print(f"✓ Saved reproducible model card to: {card_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Seasonal Gaussian-Copula AR conditioned on 4 latent seasonal states and generate rainfall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN, help="Path to training CSV (2000-2007)")
    parser.add_argument("--val-csv", default=DEFAULT_VAL, help="Path to validation CSV (2008)")
    parser.add_argument("--p", type=int, default=64, choices=[24, 64], help="AR order matching v8 (24) or v10 (64)")
    parser.add_argument("--assembly-mode", default="both", choices=["markov", "calendar", "both"],
                        help="Assembly mode: 'markov' (random walk), 'calendar' (annual cycle), or 'both'")
    parser.add_argument("--years", type=int, default=10, help="Number of synthetic years to generate")
    parser.add_argument("--em-iters", type=int, default=15, help="Stochastic-EM Gibbs refinement iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--transition-matrix", default=None, help="Path to seasonal_transition_matrix_train_len{P}.pkl")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "results", "generated_data"),
                        help="Directory to save synthetic CSV files")
    args = parser.parse_args()

    t_start = time.time()
    p = args.p

    print("=" * 80)
    print(f"  SEASONAL GAUSSIAN-COPULA AR(p={p}) RAINFALL GENERATION PIPELINE")
    print(f"  Conditioned on 4 Seasonal Regimes (hmm_state 0..3)")
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
    states_train = train_df["hmm_state"].values.astype(int)
    train_dates = train_df["date"]
    N_train = len(x_train)

    max_year = train_dates.dt.year.max()
    min_year = train_dates.dt.year.min()
    print(f"  Training span   : {min_year}–{max_year} ({N_train:,} steps)")
    if max_year > 2007:
        print(f"ERROR: Training partition leaks years > 2007 (found {max_year})!")
        sys.exit(1)

    # 2. Load Precomputed Seasonal Transition Matrix
    trans_path = args.transition_matrix
    if trans_path is None:
        trans_path = os.path.join(ROOT, "data", "rainfall", "splits", f"seasonal_transition_matrix_train_len{p}.pkl")
    if not os.path.exists(trans_path):
        print(f"ERROR: Seasonal transition matrix not found at: {trans_path}")
        sys.exit(1)

    print(f"▶ Loading seasonal transition profile: {trans_path}")
    with open(trans_path, "rb") as f:
        trans_data = pickle.load(f)

    # 3. Build 4-Regime Empirical Quantile Translator
    seq = SeasonalEmpiricalQuantile(x_train, states_train, wet_threshold=WET_THR)
    print("\n✓ 4-Regime Empirical Quantile Translators fitted:")
    for s in range(4):
        print(f"  State {s}: N={seq.N_s[s]:7d} | zero={seq.p_zero[s]*100:5.2f}% | z_thr={seq.z_threshold[s]:.4f}")

    # 4. Fit Latent Autoregressive Process
    phi, sigma2, z_fit = fit_seasonal_copula_ar(seq, x_train, states_train, p=p, em_iters=args.em_iters, seed=args.seed)
    sigma = np.sqrt(sigma2)
    print(f"\n✓ Fitted Latent AR({p}):")
    print(f"  Innovation sigma: {sigma:.6f}")
    print(f"  Top 5 AR weights: {np.round(phi[:5], 4)}")

    # 5. Out-of-sample validation on 2008 split
    val_mse = None
    if os.path.exists(args.val_csv):
        val_df = pd.read_csv(args.val_csv)
        val_col = "avg_rainfall" if "avg_rainfall" in val_df.columns else "rainfall_intensity"
        x_val = val_df[val_col].values.astype(np.float64)
        states_val = val_df["hmm_state"].values.astype(int)
        z_val = seq.x_to_initial_z(x_val, states_val, seed=args.seed)
        res_val = lfilter(np.r_[1.0, -phi], [1.0], z_val)[p:]
        val_mse = float(np.mean(res_val ** 2))
        print(f"  Validation 1-step Latent MSE (2008): {val_mse:.6f}")

    # 6. Simulate Latent Gaussian Trajectory
    N_gen = args.years * STEPS_PER_YEAR
    burn_in = 2000
    print(f"\n▶ Simulating {args.years} synthetic years of latent AR({p}) ({N_gen:,} steps) with seed {args.seed}...")
    np.random.seed(args.seed)
    eps_sim = np.random.normal(0, sigma, N_gen + burn_in)
    z_sim = lfilter([1.0], np.r_[1.0, -phi], eps_sim)[burn_in:]

    # Standardize simulated latent series to exact standard normal N(0, 1)
    z_sim = (z_sim - np.mean(z_sim)) / (np.std(z_sim) + 1e-9)

    # 7. Generate Realizations According to Requested Modes
    modes_to_run = ["markov", "calendar"] if args.assembly_mode == "both" else [args.assembly_mode]

    for mode in modes_to_run:
        tag = "markov" if mode == "markov" else "cal"
        out_csv = os.path.join(args.output_dir, f"rainfall_synthetic_10y_arima_copula_seas_{tag}_len{p}.csv")
        card_json = os.path.join(ROOT, "results", "reference", f"arima_copula_seas_{tag}_len{p}_model_card.json")

        generate_realization(
            mode=mode,
            z_sim=z_sim,
            seq=seq,
            trans_data=trans_data,
            p=p,
            years=args.years,
            N_gen=N_gen,
            seed=args.seed,
            out_csv=out_csv,
            card_json=card_json,
            phi=phi,
            sigma=sigma,
            sigma2=sigma2,
            N_train=N_train,
            min_year=min_year,
            max_year=max_year,
            args=args,
            val_mse=val_mse,
            t_pipeline_start=t_start,
        )

    print("\n" + "=" * 80)
    print(f"  All realizations completed in {time.time() - t_start:.2f}s")
    print("  To score against Gate A and compare with diffusion models:")
    vers = [f"arima_copula_seas_{'markov' if m == 'markov' else 'cal'}_len{p}" for m in modes_to_run]
    vers_str = " ".join(vers)
    print(f"    python scripts/gate_a_scorecard.py --reference train --versions v10 v10_cal {vers_str}")
    print("=" * 80)


if __name__ == "__main__":
    main()
