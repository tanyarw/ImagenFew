#!/usr/bin/env python
"""
score_context_generation.py — score E2 (context-chained generation) against the pass / fail
rules written down in advance in code_plan/NEXT_STEPS_AFTER_v14.md §3.

For every version given:

  SUCCESS        storms > 320 min per year                 >= 10    (real 14.25, v14 1.90)
                 share of total rain in those storms       >= 15 %  (real 21.4 %, v14 2.2 %)
  MUST NOT BREAK wet-run length histogram (1/2/3/4-6/7-12/13+ steps), every bin within 2 pp of real
                 zero fraction within 1.0 pp of real (90.80 %)
                 lag-1 autocorrelation within 0.02 of real
                 annual volume ratio in 0.95-1.05
  DRIFT          each generated year scored separately and compared with the SAME year of the
                 baseline (v14, same plan seed), which cancels the wet/dry years the plan
                 produces. Rule: drifting if the slope of (version - baseline) across years is
                 significant (|t| > 2) AND large: over the whole run it changes the annual volume
                 ratio by more than 0.10, or the hourly ACF RMSE by more than 0.02. For scale:
                 v14 itself varies by about +/-0.12 in volume ratio from year to year, with a
                 slope of -0.002 per year.

Storms are the canonical definition (contiguous wet steps >= 0.005 mm, at least 3 steps), as in
gate_a_scorecard.py and diagnose_storm_duration_cliff.py. Reference = 2000-2007 training split.
Pure numpy / pandas, runs locally.

    python scripts/score_context_generation.py --versions v14_ctx v14_ctx_m1 v14_ctx_m2 v14_ctx_m3
    python scripts/score_context_generation.py --versions v14_ctx_cal --baseline v14_cal
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import gate_a_scorecard as sc  # noqa: E402  (load, spells, storms, acf, hourly: Gate A definitions)

GEN = os.path.join(ROOT, 'results', 'generated_data')
REAL_CSV = os.path.join(ROOT, 'data', 'rainfall', 'splits', 'train_years_labelled.csv')
Y = 105120
LONG = 64                                   # 320 min
RUN_BINS = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 10 ** 9)]
RUN_LABELS = ['1', '2', '3', '4-6', '7-12', '13+']


def series(name_or_path):
    path = name_or_path if name_or_path.endswith('.csv') else os.path.join(GEN, f'rainfall_synthetic_10y_{name_or_path}.csv')
    return sc.load(path)['avg_rainfall'].to_numpy()


def long_storms(a):
    runs = sc.storms(a)
    long_ = [r for r in runs if len(r) > LONG]
    return len(long_) / (len(a) / Y), 100 * sum(r.sum() for r in long_) / a.sum()


def run_hist(a):
    wet, _ = sc.spells(a)
    return np.array([100 * np.mean((wet >= lo) & (wet <= hi)) for lo, hi in RUN_BINS])


def per_year(a, real_mean_vol, real_hourly_acf):
    n = len(a) // Y
    vol = np.array([a[y * Y:(y + 1) * Y].sum() / real_mean_vol for y in range(n)])
    acf = np.array([np.sqrt(np.mean((sc.acf(sc.hourly(a[y * Y:(y + 1) * Y]), 24)[1:] - real_hourly_acf) ** 2))
                    for y in range(n)])
    return vol, acf


def trend(d):
    """Slope per year of d, its standard error, and the change it implies over the run."""
    x = np.arange(len(d))
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, d, 1)
    resid = d - (a + b * x)
    se = np.sqrt(np.sum(resid ** 2) / (len(d) - 2) / np.sum((x - x.mean()) ** 2))
    return b, se, b * (len(d) - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--versions', nargs='+', required=True, help='names (rainfall_synthetic_10y_<name>.csv) or CSV paths')
    ap.add_argument('--baseline', default='v14', help='paired baseline for the drift test (same plan seed)')
    ap.add_argument('--json', default=None, help='write the results here')
    args = ap.parse_args()

    real = sc.load(REAL_CSV)['avg_rainfall'].to_numpy()
    real_vol = real.sum() / (len(real) / Y)
    real_acf = sc.acf(sc.hourly(real), 24)[1:]
    r_long, r_share = long_storms(real)
    r_hist, r_zero, r_lag1 = run_hist(real), 100 * np.mean(real == 0), sc.acf(real, 1)[1]
    base = series(args.baseline)
    b_vol, b_acf = per_year(base, real_vol, real_acf)
    b_long, b_share = long_storms(base)

    print(f'reference: 2000-2007 train split | {real_vol:.1f} mm/yr | zero {r_zero:.2f}% | '
          f'storms > 320 min {r_long:.2f}/yr carrying {r_share:.1f}% of rain')
    print(f'baseline {args.baseline}: storms > 320 min {b_long:.2f}/yr carrying {b_share:.1f}% of rain\n')
    results = {}
    for v in args.versions:
        a = series(v)
        n_long, share = long_storms(a)
        hist = run_hist(a)
        zero, lag1 = 100 * np.mean(a == 0), sc.acf(a, 1)[1]
        vol_ratio = a.sum() / (len(a) / Y) / real_vol
        vol, acf = per_year(a, real_vol, real_acf)
        n = min(len(vol), len(b_vol))
        dv, dv_se, dv_run = trend(vol[:n] - b_vol[:n])
        da, da_se, da_run = trend(acf[:n] - b_acf[:n])
        drift_vol = bool(abs(dv) > 2 * dv_se and abs(dv_run) > 0.10)
        drift_acf = bool(abs(da) > 2 * da_se and abs(da_run) > 0.02)
        checks = {
            'storms > 320 min >= 10 /yr': n_long >= 10,
            'rain share in them >= 15%': share >= 15,
            'wet-run histogram within 2 pp': bool(np.all(np.abs(hist - r_hist) <= 2)),
            'zero fraction within 1.0 pp': abs(zero - r_zero) <= 1.0,
            'lag-1 ACF within 0.02': abs(lag1 - r_lag1) <= 0.02,
            'annual volume ratio 0.95-1.05': 0.95 <= vol_ratio <= 1.05,
            'no volume drift vs baseline': not drift_vol,
            'no hourly-ACF drift vs baseline': not drift_acf,
        }
        print(f'=== {v}')
        print(f'  storms > 320 min {n_long:.2f}/yr (real {r_long:.2f}) | rain in them {share:.1f}% (real {r_share:.1f}%)')
        print(f'  zero {zero:.2f}% (real {r_zero:.2f}) | lag-1 ACF {lag1:.3f} (real {r_lag1:.3f}) | '
              f'volume ratio {vol_ratio:.3f}')
        print('  wet-run length (steps) ' + '  '.join(f'{l:>5s}' for l in RUN_LABELS))
        print('  % of runs, generated   ' + '  '.join(f'{x:5.1f}' for x in hist))
        print('  % of runs, real        ' + '  '.join(f'{x:5.1f}' for x in r_hist))
        print(f'  per-year volume ratio   {np.round(vol, 2)}')
        print(f'  same years, {args.baseline:<11s} {np.round(b_vol, 2)}')
        print(f'  drift of the difference: volume {dv:+.4f}/yr (SE {dv_se:.4f}, {dv_run:+.3f} over the run) | '
              f'hourly ACF RMSE {da:+.4f}/yr (SE {da_se:.4f}, {da_run:+.3f} over the run)')
        for k, ok in checks.items():
            print(f'  {"PASS" if ok else "FAIL"}  {k}')
        checks = {k: bool(ok) for k, ok in checks.items()}
        verdict = 'SUCCESS' if all(checks.values()) else (
            'DRIFTING' if drift_vol or drift_acf else 'NOT YET')
        print(f'  VERDICT: {verdict}\n')
        results[v] = dict(storms_long_per_yr=n_long, rain_share_long_pct=share, zero_pct=zero, lag1=lag1,
                          volume_ratio=vol_ratio, wet_run_hist_pct=hist.tolist(), per_year_volume=vol.tolist(),
                          per_year_hourly_acf_rmse=acf.tolist(), drift_volume_slope=dv, drift_acf_slope=da,
                          checks=checks, verdict=verdict)
    if args.json:
        with open(os.path.join(ROOT, args.json) if not os.path.isabs(args.json) else args.json, 'w') as f:
            json.dump(dict(baseline=args.baseline, results=results), f, indent=2, default=float)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()
