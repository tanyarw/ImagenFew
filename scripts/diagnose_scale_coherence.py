#!/usr/bin/env python
"""
Scale-coherence diagnostics for block-generated synthetic time series.
=====================================================================

Motivation
----------
ImagenFew (and ImagenTime before it) learns a distribution over length-L
windows.  A record longer than L is produced OUTSIDE the model, by drawing
blocks and concatenating them.  This script measures the consequence.

If blocks of length L are drawn independently and concatenated, the resulting
process has autocovariance

    gamma_blk(tau) = (1 - |tau|/L) * gamma(tau)   for |tau| < L,   0 otherwise

i.e. the true autocovariance multiplied by a Bartlett (Fejer) lag window and
truncated at L.  Consequently the variance of a D-step accumulation is

    Var_blk[S_D] = sum_{|tau| < min(D,L)} (D - |tau|) (1 - |tau|/L) gamma(tau)

which is a PARAMETER-FREE prediction: it needs only the REAL record's
autocovariance and the block length L.  Nothing about the network enters it.

Four diagnostics are produced:

  A. multi-scale standard-deviation ratio, observed vs predicted
  B. effective block length recovered by fitting L to the observed curve
     (the falsification control: the fit must recover each model's own L)
  C. delay-embedding column-seam test (intra-block representation defect)
  D. coherence horizon D* -- the largest aggregation scale at which the
     synthetic record still carries >= 90% of the real standard deviation

Requires numpy / pandas only.  Runs on a laptop in well under a minute.

Usage
-----
    python scripts/diagnose_scale_coherence.py
    python scripts/diagnose_scale_coherence.py --versions v8 v10 --max-lag 4200
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data/rainfall/splits/train_years_labelled.csv")
GEN = os.path.join(ROOT, "results/generated_data/rainfall_synthetic_10y_{v}.csv")

# nominal training block length and generation overlap per version
VERSION_SPEC = {
    "v8":      dict(L=24, overlap=4),
    "v9":      dict(L=36, overlap=4),
    "v10":     dict(L=64, overlap=4),
    "v8_cal":  dict(L=24, overlap=4),
    "v9_cal":  dict(L=36, overlap=4),
    "v10_cal": dict(L=64, overlap=4),
}
EMBEDDING = 8  # delay == embedding == 8 in every regime_training config

SCALES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288,
          384, 576, 864, 1152, 2016, 4032]


# --------------------------------------------------------------------------
# core statistics
# --------------------------------------------------------------------------
def autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocovariance gamma(0..max_lag) via FFT."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1].real
    return ac / n


def var_aggregate(gamma: np.ndarray, D: int) -> float:
    """Var[S_D] for a stationary process with autocovariance gamma."""
    if D == 1:
        return float(gamma[0])
    t = np.arange(1, D)
    return float(D * gamma[0] + 2.0 * np.sum((D - t) * gamma[t]))


def var_aggregate_blocked(gamma: np.ndarray, D: int, L: int) -> float:
    """Var[S_D] under independent length-L block concatenation (Bartlett taper)."""
    m = min(D, L)
    if m <= 1:
        return float(D * gamma[0])
    t = np.arange(1, m)
    return float(D * gamma[0] + 2.0 * np.sum((D - t) * (1.0 - t / L) * gamma[t]))


def empirical_var_aggregate(x: np.ndarray, D: int) -> float:
    """Variance of non-overlapping D-step sums of an observed series."""
    n = x.size // D
    return float(x[: n * D].reshape(n, D).sum(axis=1).var())


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def diagnostic_A_B(real: np.ndarray, synth: dict, gamma: np.ndarray,
                   scales: list[int]) -> pd.DataFrame:
    """Observed vs predicted SD ratio, plus best-fit effective block length."""
    real_var = {D: var_aggregate(gamma, D) for D in scales}
    rows = []
    for v, x in synth.items():
        L, ov = VERSION_SPEC[v]["L"], VERSION_SPEC[v]["overlap"]
        L_eff = L - ov
        # normalise at D=1 so a marginal-intensity bias does not contaminate
        # the dependence test
        base = np.sqrt(empirical_var_aggregate(x, 1) / real_var[1])
        for D in scales:
            obs = np.sqrt(empirical_var_aggregate(x, D) / real_var[D]) / base
            pred = np.sqrt(var_aggregate_blocked(gamma, D, L_eff) / real_var[D])
            rows.append(dict(version=v, L=L, L_eff=L_eff, D=D,
                             hours=D * 5 / 60, observed=obs, predicted=pred))
    return pd.DataFrame(rows)


def fit_effective_block_length(df: pd.DataFrame, gamma: np.ndarray,
                               grid=np.arange(2, 601)) -> pd.DataFrame:
    """Recover L from the observed curve alone. The control for diagnostic A."""
    out = []
    for v, g in df.groupby("version", sort=False):
        g = g[g.D >= 8]
        Ds = g.D.values
        obs = g.observed.values
        real_var = np.array([var_aggregate(gamma, int(D)) for D in Ds])
        errs = np.array([
            np.abs(obs - np.sqrt(np.array([var_aggregate_blocked(gamma, int(D), int(Lc))
                                           for D in Ds]) / real_var)).mean()
            for Lc in grid
        ])
        null = np.abs(obs - 1.0).mean()          # a perfect generator
        out.append(dict(version=v,
                        L_nominal=int(g.L.iloc[0]),
                        L_eff_true=int(g.L_eff.iloc[0]),
                        L_eff_fitted=int(grid[errs.argmin()]),
                        mae_at_fit=float(errs.min()),
                        mae_null=float(null),
                        error_reduction=float(null / max(errs.min(), 1e-12))))
    return pd.DataFrame(out)


def diagnostic_C_seams(synth: dict, embedding: int = EMBEDDING) -> pd.DataFrame:
    """Is lag-1 structure degraded at delay-embedding column boundaries?

    With delay == embedding == E, the length-L window is written column-major
    into an E x q grid.  The temporally adjacent pair (t, t+1) with
    t mod E == E-1 lands at grid rows E-1 and 0 of neighbouring columns,
    E-1 rows apart, outside the reach of a 3x3 convolution at full resolution.
    """
    rows = []
    for v, x in synth.items():
        L, ov = VERSION_SPEC[v]["L"], VERSION_SPEC[v]["overlap"]
        stride = L - ov
        a, b = x[:-1], x[1:]
        phase = np.arange(a.size) % stride
        corr, dinc = np.full(stride, np.nan), np.full(stride, np.nan)
        for p in range(stride):
            sel = phase == p
            aa, bb = a[sel], b[sel]
            if aa.std() > 0 and bb.std() > 0:
                corr[p] = np.corrcoef(aa, bb)[0, 1]
            dinc[p] = np.abs(bb - aa).mean()
        seam = np.array([p for p in range(stride) if p % embedding == embedding - 1])
        interior = np.array([p for p in range(stride)
                             if p % embedding != embedding - 1 and ov <= p < stride - ov])
        if seam.size == 0 or interior.size == 0:
            continue
        z = ((np.nanmean(corr[seam]) - np.nanmean(corr[interior]))
             / (np.nanstd(corr[interior], ddof=1) / np.sqrt(seam.size)))
        rows.append(dict(version=v, L=L, stride=stride,
                         n_seam_phases=seam.size,
                         lag1_seam=np.nanmean(corr[seam]),
                         lag1_interior=np.nanmean(corr[interior]),
                         lag1_drop=np.nanmean(corr[interior]) - np.nanmean(corr[seam]),
                         absinc_ratio=np.nanmean(dinc[seam]) / np.nanmean(dinc[interior]),
                         z=z,
                         ragged=(L % embedding != 0)))
    return pd.DataFrame(rows)


def diagnostic_D_horizon(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Coherence horizon: largest D with observed SD ratio >= threshold."""
    out = []
    for v, g in df.groupby("version", sort=False):
        g = g.sort_values("D")
        ok = g[g.observed >= threshold]
        D_star = int(ok.D.max()) if len(ok) else 0
        out.append(dict(version=v, L=int(g.L.iloc[0]), D_star=D_star,
                        hours=D_star * 5 / 60,
                        ratio_to_L=D_star / g.L.iloc[0] if D_star else 0.0))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=list(VERSION_SPEC),
                    help="which synthetic versions to score")
    ap.add_argument("--max-lag", type=int, default=4200)
    ap.add_argument("--real", default=REAL)
    ap.add_argument("--out", default=None, help="optional CSV path for the scale table")
    args = ap.parse_args()

    real = pd.read_csv(args.real, usecols=["avg_rainfall"])["avg_rainfall"].to_numpy(np.float64)
    gamma = autocovariance(real, args.max_lag)
    print(f"reference : {args.real}")
    print(f"            n={real.size}  mean={real.mean():.5f}  "
          f"zero-fraction={(real <= 0).mean():.4f}  rho(1)={gamma[1]/gamma[0]:.4f}")

    lo = np.log(np.array(SCALES, float))
    lv = np.log(np.array([var_aggregate(gamma, D) for D in SCALES]))
    H = np.polyfit(lo, lv, 1)[0] / 2
    print(f"            Var[S_D] ~ D^{2*H:.3f}  ->  Hurst H = {H:.3f}"
          f"  ({'long-range dependent' if H > 0.5 else 'short memory'})\n")

    synth = {}
    for v in args.versions:
        p = GEN.format(v=v)
        if not os.path.exists(p):
            print(f"  [skip] {v}: {p} not found")
            continue
        synth[v] = pd.read_csv(p, usecols=["avg_rainfall"])["avg_rainfall"].to_numpy(np.float64)
    if not synth:
        raise SystemExit("no synthetic series found")

    df = diagnostic_A_B(real, synth, gamma, SCALES)

    print("A. Multi-scale SD ratio, normalised at D=1.  obs = measured, pred = Bartlett-taper")
    print("   prediction from the REAL autocovariance and that version's own block length.\n")
    wide = df.pivot(index="D", columns="version", values=["observed", "predicted"])
    wide.columns = [f"{v}.{k[:4]}" for k, v in wide.columns]
    print(wide.round(3).to_string())

    print("\nB. Falsification control: recover the block length from the curve alone.")
    fit = fit_effective_block_length(df, gamma)
    print(fit.round(4).to_string(index=False))
    print("   `mae_null` is the error of assuming a perfect generator (ratio 1.0 at all D).")

    print("\nC. Delay-embedding column-seam test (embedding = %d)." % EMBEDDING)
    seams = diagnostic_C_seams(synth)
    print(seams.round(4).to_string(index=False))
    print("   `ragged` is True when seq_len %% embedding != 0, which leaves a partially")
    print("   filled final column whose unwritten cells the loss still counts as signal.")

    print("\nD. Coherence horizon (largest D with SD ratio >= 0.90).")
    print(diagnostic_D_horizon(df).round(3).to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
