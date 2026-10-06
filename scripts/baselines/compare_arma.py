#!/usr/bin/env python3
"""
compare_arma.py — Does q > 0 help? ARMA comparison for the AR baselines
=======================================================================
Every AR baseline is ARIMA(p, 0, 0). d = 0 is backed by an ADF test (run_vanilla_arima.py);
this script tests q = 0. All fits on the 2000-2007 training split; 2008 for out-of-sample error.

  A. ARMA(64, q), q = 0..4, on raw mm, by Hannan-Rissanen: a long AR(128) gives estimated
     shocks; then least squares of y_t on 64 lags of y and q lags of those shocks.
  B. ARMA(p, q), p, q <= 4, on raw mm, by exact maximum likelihood (statsmodels state space).
     ~1-2 min per model; each finished model is appended to a .partial.jsonl and skipped on rerun.
  C. Copula latent process: occurrence-matched latent correlation targets solved out to lag 144
     (12 h). AR(64) fitted to lags 1-64 (as run_copula_arima_occ.py) is compared with the targets
     at lags 65-144, where it extrapolates; short ARMA(p, q), p, q <= 4, are fitted by least
     squares to lags 1-64 and scored on both ranges. (ARMA(64, q) is not identifiable from 64
     correlations: 64 + q unknowns.)

Every model in A and B is scored identically: shocks from the ARMA recursion on the centred
series (first BURN steps dropped), sigma2 = their mean square, BIC* = N ln sigma2 + (p + q) ln N,
and the same recursion on 2008 for the 1-step MSE (mm^2).

Usage:
    python scripts/baselines/compare_arma.py --parts A C      # seconds
    python scripts/baselines/compare_arma.py --parts B        # ~40 min; rerun to resume
    python scripts/baselines/compare_arma.py --parts report   # combine into the JSON
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.special import ndtri

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_copula_arima import levinson_durbin  # noqa: E402
from run_copula_arima_occ import latent_targets  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(ROOT, "data", "rainfall", "splits", "train_years_labelled.csv")
VAL = os.path.join(ROOT, "data", "rainfall", "splits", "val_years_labelled.csv")
OUT = os.path.join(ROOT, "results", "reference", "arma_comparison.json")
PARTIAL_B = OUT.replace(".json", "_B.partial.jsonl")
WET_THR = 0.005
BURN = 200          # steps dropped before scoring shocks (recursion start-up)
P_LONG = 128        # Hannan-Rissanen stage-1 AR order
MAX_LAG_C = 144     # 12 h


def load():
    x = pd.read_csv(TRAIN, usecols=["avg_rainfall"])["avg_rainfall"].to_numpy(np.float64)
    xv = pd.read_csv(VAL, usecols=["avg_rainfall"])["avg_rainfall"].to_numpy(np.float64)
    return x, xv


def shocks(y, phi, theta):
    """e_t from (1 - sum phi_i L^i) y_t = (1 + sum theta_j L^j) e_t (statsmodels' MA sign)."""
    return lfilter(np.r_[1.0, -np.asarray(phi)], np.r_[1.0, np.asarray(theta)], y)


def score(name, y, yv, phi, theta, extra=None):
    e = shocks(y, phi, theta)[BURN:]
    ev = shocks(yv, phi, theta)[BURN:]
    n, k = len(e), len(phi) + len(theta)
    s2 = float(np.mean(e ** 2))
    ma_roots = np.roots(np.r_[1.0, theta][::-1]) if len(theta) else np.array([])
    row = {"model": name, "p": len(phi), "q": len(theta), "sigma2": s2,
           "bic_star": float(n * np.log(s2) + k * np.log(n)), "val_mse_mm2": float(np.mean(ev ** 2)),
           "ma_invertible": bool(np.all(np.abs(ma_roots) > 1.0)) if len(theta) else True,
           "phi_head": [float(v) for v in np.asarray(phi)[:4]], "theta": [float(v) for v in theta]}
    row.update(extra or {})
    return row


def acov(y, K):
    n = len(y)
    return np.array([y[:n - k] @ y[k:] / n for k in range(K + 1)])


# ──────────────────────────────────────────────────────────────────────
# A. ARMA(64, q) by Hannan-Rissanen
# ──────────────────────────────────────────────────────────────────────

def part_a(y, yv, p=64, qs=(0, 1, 2, 3, 4)):
    rows = []
    phi_yw, _ = levinson_durbin(acov(y, p), p)
    rows.append(score("AR(64) Yule-Walker (baseline)", y, yv, phi_yw, [], {"method": "yule-walker"}))
    phi_long, _ = levinson_durbin(acov(y, P_LONG), P_LONG)
    e1 = shocks(y, phi_long, [])                       # stage-1 shock estimates
    start = max(P_LONG, p) + max(qs)
    T = len(y) - start
    for q in qs:
        cols = [y[start - i:start - i + T] for i in range(1, p + 1)] + \
               [e1[start - j:start - j + T] for j in range(1, q + 1)]
        X = np.column_stack(cols)
        beta = np.linalg.solve(X.T @ X, X.T @ y[start:start + T])
        del X
        rows.append(score(f"ARMA(64,{q}) Hannan-Rissanen", y, yv, beta[:p], beta[p:], {"method": "hannan-rissanen"}))
        print(f"  A  q={q}: sigma2={rows[-1]['sigma2']:.4e}  BIC*={rows[-1]['bic_star']:.1f}  "
              f"2008 MSE={rows[-1]['val_mse_mm2']:.4e}", flush=True)
    return rows


# ──────────────────────────────────────────────────────────────────────
# B. Short ARMA by exact maximum likelihood (resumable)
# ──────────────────────────────────────────────────────────────────────

def part_b(y, yv, pmax=4, qmax=4):
    from statsmodels.tsa.arima.model import ARIMA
    done = {}
    if os.path.exists(PARTIAL_B):
        for line in open(PARTIAL_B):
            r = json.loads(line)
            done[(r["p"], r["q"])] = r
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            if (p, q) == (0, 0) or (p, q) in done:
                continue
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(y, order=(p, 0, q), trend="n").fit()
            par = dict(zip(res.model.param_names, np.asarray(res.params)))
            phi = [par[f"ar.L{i}"] for i in range(1, p + 1)]
            theta = [par[f"ma.L{j}"] for j in range(1, q + 1)]
            row = score(f"ARMA({p},{q}) MLE", y, yv, phi, theta,
                        {"method": "mle-statespace", "llf": float(res.llf), "converged": bool(res.mle_retvals.get("converged")),
                         "fit_sec": round(time.time() - t, 1)})
            with open(PARTIAL_B, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  B  ({p},{q}): sigma2={row['sigma2']:.4e}  BIC*={row['bic_star']:.1f}  "
                  f"2008 MSE={row['val_mse_mm2']:.4e}  {row['fit_sec']}s", flush=True)
    return [json.loads(line) for line in open(PARTIAL_B)] if os.path.exists(PARTIAL_B) else []


# ──────────────────────────────────────────────────────────────────────
# C. Copula latent targets: AR(64) extrapolation and short ARMA
# ──────────────────────────────────────────────────────────────────────

def arma_acf(phi, theta, nlags):
    from statsmodels.tsa.arima_process import arma_acf as _acf
    return _acf(np.r_[1.0, -np.asarray(phi)], np.r_[1.0, np.asarray(theta)], lags=nlags + 1)


def part_c(x):
    wet = x >= WET_THR
    c = ndtri(1.0 - wet.mean())
    _, rho = latent_targets(wet, np.array([c]), np.array([1.0]), MAX_LAG_C)
    lags_in, lags_out = np.arange(1, 65), np.arange(65, MAX_LAG_C + 1)
    out = {"targets": rho.tolist(), "rows": []}

    def errs(acf):
        d_in, d_out = acf[lags_in] - rho[lags_in - 1], acf[lags_out] - rho[lags_out - 1]
        return {"rmse_lags_1_64": float(np.sqrt(np.mean(d_in ** 2))), "max_abs_lags_1_64": float(np.abs(d_in).max()),
                "rmse_lags_65_144": float(np.sqrt(np.mean(d_out ** 2))), "max_abs_lags_65_144": float(np.abs(d_out).max()),
                "mean_diff_lags_65_144": float(d_out.mean())}

    phi64, _ = levinson_durbin(np.r_[1.0, rho[:64]], 64)
    out["rows"].append({"model": "AR(64) (run_copula_arima_occ.py)", "p": 64, "q": 0, **errs(arma_acf(phi64, [], MAX_LAG_C))})
    rng = np.random.default_rng(0)
    for p in range(1, 5):
        for q in range(0, 5):
            def resid(v):
                phi, theta = v[:p], v[p:]
                if np.any(np.abs(np.roots(np.r_[1.0, -phi])) >= 0.999):   # roots of z^p - phi1 z^(p-1) - ...
                    return np.full(64, 10.0)
                return arma_acf(phi, theta, 64)[1:] - rho[:64]
            best = None
            for _ in range(8):                                             # multi-start
                v0 = np.r_[rng.uniform(0.2, 0.9) / p * np.ones(p), rng.uniform(-0.5, 0.5, q)]
                try:
                    r = least_squares(resid, v0, method="lm" if p + q <= 64 else "trf")
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
            phi, theta = best.x[:p], best.x[p:]
            out["rows"].append({"model": f"ARMA({p},{q}) fit to lags 1-64", "p": p, "q": q,
                                "phi": phi.tolist(), "theta": theta.tolist(), **errs(arma_acf(phi, theta, MAX_LAG_C))})
    for r in out["rows"]:
        print(f"  C  {r['model']:<34} RMSE 1-64 {r['rmse_lags_1_64']:.4f}   RMSE 65-144 {r['rmse_lags_65_144']:.4f}  "
              f"(mean diff {r['mean_diff_lags_65_144']:+.4f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=["A", "C"], choices=["A", "B", "C", "report"])
    args = ap.parse_args()
    x, xv = load()
    mu = x.mean()
    y, yv = x - mu, xv - mu
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "A" in args.parts:
        res["A_arma64_hannan_rissanen"] = part_a(y, yv)
    if "B" in args.parts:
        res["B_short_arma_mle"] = part_b(y, yv)
    if "C" in args.parts:
        res["C_copula_latent_targets"] = part_c(x)
    if "report" in args.parts and os.path.exists(PARTIAL_B):
        res["B_short_arma_mle"] = [json.loads(line) for line in open(PARTIAL_B)]
    res["protocol"] = ("Fit 2000-2007, score 2008. BIC* = N ln sigma2 + (p+q) ln N from 1-step shocks of the same "
                       "recursion (first %d steps dropped); val_mse_mm2 on 2008. C: occurrence-matched latent "
                       "correlation targets to lag %d." % (BURN, MAX_LAG_C))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
