#!/usr/bin/env python3
"""
run_bartlett_lewis.py — Randomised Bartlett-Lewis rectangular-pulse rainfall generator
=====================================================================================
The classical "storm and cell" model (Rodriguez-Iturbe, Cox & Isham 1988; the standard
competitor for sub-hourly point rainfall, see docs/LITERATURE_RAINFALL_GENERATORS.md):

  * storms arrive at random (Poisson, rate lambda per hour);
  * each storm gets its own time scale eta ~ Gamma(alpha, rate nu)  ("randomised" variant);
  * while a storm is active (exponential duration, rate phi*eta) it sets off rain cells
    (Poisson, rate kappa*eta), plus one cell at the storm's start;
  * each cell is a rectangular pulse with exponential duration (rate eta) and exponential
    intensity. Default variant "rblx" (Kaczmarska, Isham & Onof 2014, fitted to German 5-min
    data): the mean intensity is iota * eta, so short cells are more intense, which is what
    produces sub-hourly bursts. Variant "rbl": a fixed mean intensity mu_x (classic RBL, which
    under-produces 5-min extremes). Rain is the sum of all active cells.

Fitting: one parameter set per calendar month, by the simulated method of moments. The
model is simulated for a candidate parameter set and its statistics are matched to the
observed ones (2000-2007 training years, same series as every other version): mean at 1 h,
and coefficient of variation, lag-1 autocorrelation and dry fraction at 5 min, 1 h, 6 h and
24 h, plus skewness at 5 min and 1 h. Differential evolution searches the six parameters
(log scale); a fixed random seed per evaluation keeps the objective repeatable. Only numpy /
scipy (no rainfall package needed).

Generation: 10 years on a 365-day calendar; storms start with the parameters of the month
they start in, and cells may run on into the next month. 5-min depths are exact integrals of
the pulses over each interval; values below the 0.005 mm wet threshold are set to 0.

Outputs:
  results/generated_data/rainfall_synthetic_10y_bartlett_lewis.csv
  results/reference/bartlett_lewis_model_card.json   (parameters and fit quality per month)

Usage:
    python scripts/baselines/run_bartlett_lewis.py                 # fit + generate (~15 min, 8 cores)
    python scripts/baselines/run_bartlett_lewis.py --variant rbl   # classic variant, for comparison
    python scripts/baselines/run_bartlett_lewis.py --generate_only  # reuse the model card
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN = os.path.join(ROOT, "data", "rainfall", "splits", "train_years_labelled.csv")
OUT_CSV = os.path.join(ROOT, "results", "generated_data", "rainfall_synthetic_10y_bartlett_lewis{}.csv")
CARD = os.path.join(ROOT, "results", "reference", "bartlett_lewis{}_model_card.json")
WET_THR = 0.005
DT = 1 / 12                      # 5 minutes, in hours
STEPS_PER_DAY = 288
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
PARAMS = {'rblx': ['lambda', 'iota', 'alpha', 'nu', 'kappa', 'phi'],
          'rbl': ['lambda', 'mu_x', 'alpha', 'nu', 'kappa', 'phi']}
# search box (log10 except alpha): lambda (storms/h), iota (mm) or mu_x (mm/h), alpha, nu (h), kappa, phi
BOUNDS = {'rblx': [(np.log10(0.002), np.log10(0.3)), (np.log10(0.005), np.log10(5.0)), (2.05, 40.0),
                   (np.log10(0.02), np.log10(40.0)), (np.log10(0.01), np.log10(10.0)), (np.log10(0.002), np.log10(1.0))],
          'rbl': [(np.log10(0.002), np.log10(0.3)), (np.log10(0.2), np.log10(80.0)), (2.05, 40.0),
                  (np.log10(0.02), np.log10(40.0)), (np.log10(0.01), np.log10(10.0)), (np.log10(0.002), np.log10(1.0))]}
FIT_YEARS = 100                  # years of a month simulated per objective evaluation (30 overfits the seed)
MAX_CELLS_PER_STORM = 300        # mean cells per storm 1 + kappa/phi is capped (guards runaway sims)


def unpack(v, variant):
    return dict(lam=10 ** v[0], mu=10 ** v[1], alpha=v[2], nu=10 ** v[3], kappa=10 ** v[4], phi=10 ** v[5],
                dependent=(variant == 'rblx'))


# ──────────────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────────────

def storm_cells(rng, t0, t1, p):
    """Cells (start, end, intensity) of all storms starting in [t0, t1) hours."""
    n = rng.poisson(p['lam'] * (t1 - t0))
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    origin = rng.uniform(t0, t1, n)
    eta = rng.gamma(p['alpha'], 1.0 / p['nu'], n)               # per-storm time scale (1/h)
    active = rng.exponential(1.0 / (p['phi'] * eta))             # storm activity duration (h)
    n_extra = rng.poisson(p['kappa'] * eta * active)             # cells after the first
    n_cells = 1 + n_extra
    storm = np.repeat(np.arange(n), n_cells)
    first = np.zeros(len(storm), dtype=bool)
    first[np.r_[0, np.cumsum(n_cells)[:-1]]] = True
    offset = np.where(first, 0.0, rng.uniform(0, 1, len(storm)) * active[storm])
    start = origin[storm] + offset
    dur = rng.exponential(1.0, len(storm)) / eta[storm]
    inten = rng.exponential(p['mu'] * eta[storm] if p['dependent'] else p['mu'], len(storm))
    return start, start + dur, inten


def to_steps(start, end, inten, n_steps):
    """Exact 5-min depths (mm) of a sum of rectangular pulses on [0, n_steps * DT)."""
    t = np.concatenate([start, end])
    dr = np.concatenate([inten, -inten])
    order = np.argsort(t, kind='stable')
    t, dr = t[order], dr[order]
    rate = np.cumsum(dr)                                         # rate after each event
    depth = np.concatenate([[0.0], np.cumsum(rate[:-1] * np.diff(t))])
    grid = np.arange(n_steps + 1) * DT
    D = np.interp(grid, t, depth, left=0.0, right=depth[-1])
    x = np.diff(D)
    x[x < WET_THR] = 0.0
    return x


def simulate_month(v, days, years, seed, variant):
    """`years` copies of one month, back to back, with one parameter set (used for fitting)."""
    p = unpack(v, variant)
    if 1 + p['kappa'] / p['phi'] > MAX_CELLS_PER_STORM:
        return None
    rng = np.random.default_rng(seed)
    H = days * 24 * years
    warm = 48.0
    s, e, x = storm_cells(rng, -warm, H, p)
    if len(s) == 0:
        return np.zeros(int(round(H / DT)))
    keep = e > 0
    return to_steps(np.clip(s[keep], 0, None), e[keep], x[keep], int(round(H / DT)))


# ──────────────────────────────────────────────────────────────────────
# Statistics and fit
# ──────────────────────────────────────────────────────────────────────

def agg(x, k):
    n = len(x) // k * k
    return x[:n].reshape(-1, k).sum(1)


def stats(x):
    """The statistics matched by the fit (from a thresholded 5-min series)."""
    out = {}
    for name, k in [('5min', 1), ('1h', 12), ('6h', 72), ('24h', 288)]:
        a = agg(x, k)
        m, sd = a.mean(), a.std()
        out[f'cv_{name}'] = sd / m if m > 0 else 0.0
        out[f'ac1_{name}'] = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if sd > 0 else 0.0
        out[f'pdry_{name}'] = float((a == 0).mean())
        if name == '1h':
            out['mean_1h'] = m
        if name in ('5min', '1h'):
            out[f'skew_{name}'] = float(((a - m) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    return out


# volume matters most; skewness and day-scale autocorrelation are noisy, so they count less
WEIGHTS = {'mean_1h': 5.0, 'skew_5min': 0.5, 'skew_1h': 0.5, 'ac1_24h': 0.5}


def objective(v, target, days, seed, variant):
    x = simulate_month(v, days, FIT_YEARS, seed, variant)
    if x is None or x.sum() == 0:
        return 1e6
    s = stats(x)
    return float(sum(WEIGHTS.get(k, 1.0) * ((s[k] - t) / t) ** 2 for k, t in target.items() if t != 0))


def fit_month(args):
    month, target, seed, variant = args
    days = MONTH_DAYS[month - 1]
    t0 = time.time()
    res = differential_evolution(objective, BOUNDS[variant], args=(target, days, seed + month, variant), popsize=12,
                                 maxiter=60, tol=1e-3, seed=seed + month, polish=False, updating='immediate')
    fresh = seed + 1000 + month                                                    # fresh seed: honest check
    fitted = stats(simulate_month(res.x, days, FIT_YEARS, fresh, variant))
    return dict(month=month, params=dict(zip(PARAMS[variant], map(float, [10 ** res.x[0], 10 ** res.x[1], res.x[2],
                                                                          10 ** res.x[3], 10 ** res.x[4], 10 ** res.x[5]]))),
                vector=list(map(float, res.x)), objective=float(res.fun),
                objective_fresh_seed=objective(res.x, target, days, fresh, variant), evaluations=int(res.nfev),
                seconds=round(time.time() - t0, 1), target=target, fitted=fitted)


def observed_targets(train):
    """Per-month statistics of the 2000-2007 training years (365-day calendar)."""
    doy = (np.arange(len(train)) // STEPS_PER_DAY) % 365
    month = np.repeat(np.arange(1, 13), MONTH_DAYS)[doy]
    return {m: stats(train[month == m]) for m in range(1, 13)}


# ──────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────

def match_means(card, seed, years=300):
    """Final calibration: scale the intensity parameter so each month's long-run mean equals the
    observed one. Rain depth is proportional to that parameter, so nothing else moves; the
    simulated fit alone leaves a few-percent seed bias in the mean."""
    for m in card['months']:
        if 'mean_factor' in m:
            continue
        days = MONTH_DAYS[m['month'] - 1]
        sim = stats(simulate_month(np.array(m['vector']), days, years, seed + 5000 + m['month'], card['variant']))
        f = m['target']['mean_1h'] / sim['mean_1h']
        m['vector'][1] += float(np.log10(f))
        key = 'iota' if card['variant'] == 'rblx' else 'mu_x'
        m['params'][key] = float(10 ** m['vector'][1])
        m['mean_factor'] = float(f)
    return card


def generate(vectors, years, seed, variant):
    """Continuous record: storms take the parameters of the month they start in."""
    rng = np.random.default_rng(seed)
    starts, ends, intens = [], [], []
    t = 0.0
    for _ in range(years):
        for m in range(12):
            h = MONTH_DAYS[m] * 24.0
            s, e, x = storm_cells(rng, t, t + h, unpack(vectors[m], variant))
            starts.append(s); ends.append(e); intens.append(x)
            t += h
    n_steps = years * 365 * STEPS_PER_DAY
    return to_steps(np.concatenate(starts), np.concatenate(ends), np.concatenate(intens), n_steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', default=DEFAULT_TRAIN)
    ap.add_argument('--years', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--generate_only', action='store_true')
    ap.add_argument('--variant', default='rblx', choices=['rblx', 'rbl'],
                    help='rblx: intensity scales with cell speed (default); rbl: classic')
    args = ap.parse_args()
    tag = '' if args.variant == 'rblx' else '_classic'
    card_path, out_csv = CARD.format(tag), OUT_CSV.format(tag)

    if args.generate_only:
        card = json.load(open(card_path))
    else:
        train = pd.read_csv(args.train, usecols=['avg_rainfall'])['avg_rainfall'].to_numpy(np.float64)
        train = np.clip(train, 0, None)
        train[train < WET_THR] = 0.0
        assert len(train) == 8 * 365 * STEPS_PER_DAY, 'expected the 2000-2007 training years'
        targets = observed_targets(train)
        print(f'fitting 12 months on {args.workers} workers ...', flush=True)
        t0 = time.time()
        with Pool(args.workers) as pool:
            fits = []
            for f in pool.imap_unordered(fit_month, [(m, targets[m], args.seed, args.variant) for m in range(1, 13)]):
                print(f"  month {f['month']:2d}: objective {f['objective']:.3f} (fresh seed {f['objective_fresh_seed']:.3f}), "
                      f"{f['evaluations']} sims, {f['seconds']} s; mean 1h obs {f['target']['mean_1h']:.3f} "
                      f"fit {f['fitted']['mean_1h']:.3f}", flush=True)
                fits.append(f)
        fits.sort(key=lambda f: f['month'])
        card = dict(model='randomised Bartlett-Lewis rectangular pulse, ' +
                    ('intensity mean iota*eta (Kaczmarska et al. 2014)' if args.variant == 'rblx'
                     else 'fixed mean intensity mu_x (classic)'), variant=args.variant,
                    fitted_on='data/rainfall/splits/train_years_labelled.csv (2000-2007, avg_rainfall)',
                    method='simulated method of moments, differential evolution, one parameter set per month',
                    units={'lambda': 'storms per hour', 'mu_x': 'mm/h', 'alpha': '-', 'nu': 'hours',
                           'kappa': '-', 'phi': '-'},
                    fit_years_per_evaluation=FIT_YEARS, weights=WEIGHTS, seed=args.seed,
                    fit_seconds=round(time.time() - t0, 1), months=fits)
        os.makedirs(os.path.dirname(card_path), exist_ok=True)
        json.dump(card, open(card_path, 'w'), indent=2)
        print(f'model card -> {os.path.relpath(card_path, ROOT)}')

    card.setdefault('variant', args.variant)
    card = match_means(card, args.seed)
    json.dump(card, open(card_path, 'w'), indent=2)
    print('mean correction factors:', [round(m['mean_factor'], 3) for m in card['months']])
    x = generate([m['vector'] for m in card['months']], args.years, args.seed, card['variant'])
    dates = pd.date_range('2026-01-01', periods=len(x), freq='5min')
    pd.DataFrame({'date': dates, 'avg_rainfall': np.round(x, 6)}).to_csv(out_csv, index=False)
    print(f'{os.path.relpath(out_csv, ROOT)}: {len(x)} steps | {x.sum() / args.years:.1f} mm/yr | '
          f'dry {100 * (x == 0).mean():.2f}% | max 5-min {x.max():.2f} mm')


if __name__ == '__main__':
    main()
