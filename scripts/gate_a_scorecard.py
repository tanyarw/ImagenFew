#!/usr/bin/env python
"""
gate_a_scorecard.py — score every synthetic version against Gate A exactly as written in
`code_plan/ACCEPTANCE_CRITERIA.md`, including the Tier 4 (AMS/IDF) and Tier 5
(seasonality/diurnal) tiers that `scripts/run_evaluation.py` never computes.

Differences from run_evaluation.py, all deliberate:
  * `--reference {train,full,test}` makes the comparison partition explicit. `train`
    (2000-2007) is the canonical thesis baseline frozen in `my notes/memo/day_1.md` §3.1.
  * P99 / P99.9 are reported WET-ONLY, matching day_1.md §3.2 and EVALUATION_GUIDE.md
    (0.595 / 1.613 mm). run_evaluation.py reports all-steps quantiles (0.160 / 0.565)
    under the same metric name.
  * Every metric carries a machine-readable pass/fail against its band (T2.2 / "what
    compare_all_versions.py still needs").
  * Each 10-year realisation is also split into its 10 constituent years so that a
    mean +/- sd is reported against the interannual spread of the observed record. This
    is a within-realisation proxy for the >= 30-member ensemble of T2.1, not a
    substitute for it.

Pure numpy/pandas/scipy — runs in the local .venv, no torch.

    python scripts/gate_a_scorecard.py --reference train --json results/reference/gate_a.json
"""
import argparse, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, 'data', 'rainfall', 'real_rainfall_data.csv')
GEN = os.path.join(ROOT, 'results', 'generated_data')
WET_THR = 0.005            # day_1.md §2.1, frozen
STEPS_PER_YEAR = 365 * 288
DURATIONS = {'15min': 3, '1h': 12, '3h': 36, '6h': 72, '24h': 288}
RETURN_PERIODS = [2, 5, 10]


def load(path):
    df = pd.read_csv(path, parse_dates=['date'])
    col = 'avg_rainfall' if 'avg_rainfall' in df.columns else df.columns[1]
    df = df[['date', col]].rename(columns={col: 'avg_rainfall'})
    df['avg_rainfall'] = df['avg_rainfall'].clip(lower=0)
    df.loc[df['avg_rainfall'] < WET_THR, 'avg_rainfall'] = 0.0
    return df


def hourly(a):
    n = len(a) // 12 * 12
    return a[:n].reshape(-1, 12).sum(1)


def daily(a):
    n = len(a) // 288 * 288
    return a[:n].reshape(-1, 288).sum(1)


def acf(a, max_lag):
    m, v = a.mean(), a.var()
    if v == 0:
        return np.zeros(max_lag + 1)
    N = len(a)
    return np.array([np.mean((a[:N - k] - m) * (a[k:] - m)) / v for k in range(max_lag + 1)])


def spells(a):
    """(wet_spell_lengths, dry_spell_lengths) in steps."""
    wet = (a > 0).astype(int)
    ch = np.diff(wet, prepend=-1)
    starts = np.where(ch != 0)[0]
    lengths = np.diff(np.append(starts, len(a)))
    types = wet[starts]
    return lengths[types == 1], lengths[types == 0]


def storms(a, min_steps=3):
    """Contiguous wet runs of >= 15 physical minutes (day_1.md §3.2 def. 6)."""
    wet = (a > 0).astype(int)
    ch = np.diff(wet, prepend=-1)
    starts = np.where(ch != 0)[0]
    lengths = np.diff(np.append(starts, len(a)))
    types = wet[starts]
    return [a[s:s + l] for s, l, t in zip(starts, lengths, types) if t == 1 and l >= min_steps]


def ams(a, win, n_years):
    """Annual maximum series of the rolling `win`-step accumulation."""
    out = []
    for y in range(n_years):
        s = a[y * STEPS_PER_YEAR:(y + 1) * STEPS_PER_YEAR]
        if len(s) < win:
            continue
        c = np.concatenate([[0.0], np.cumsum(s)])
        out.append((c[win:] - c[:-win]).max())
    return np.array(out)


def gumbel_quantile(x, T):
    """Method-of-moments Gumbel quantile at return period T. 10 points is thin; see CI."""
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float(x.mean()) if len(x) else 0.0
    b = x.std(ddof=1) * np.sqrt(6) / np.pi
    a = x.mean() - 0.5772 * b
    return float(a - b * np.log(-np.log(1 - 1.0 / T)))


def core_stats(a):
    ws, ds = spells(a)
    st = storms(a)
    w = a[a > 0]
    ny = len(a) / STEPS_PER_YEAR
    return dict(
        volume=a.sum() / ny,
        zero_pct=100 * (a == 0).mean(),
        wet_spell_min=ws.mean() * 5 if len(ws) else 0.0,
        dry_spell_min=ds.mean() * 5 if len(ds) else 0.0,
        p99_wet=float(np.percentile(w, 99)) if len(w) else 0.0,
        p999_wet=float(np.percentile(w, 99.9)) if len(w) else 0.0,
        max_burst=float(a.max()),
        acf1_native=float(acf(a, 1)[1]),
        storm_count_yr=len(st) / ny,
        storm_dur_min=float(np.mean([len(s) * 5 for s in st])) if st else 0.0,
        storm_vol_mm=float(np.mean([s.sum() for s in st])) if st else 0.0,
        daily_max=float(daily(a).max()),
    )


def band(x, lo, hi):
    return 'PASS' if lo <= x <= hi else 'FAIL'


def within(x, tol):
    return 'PASS' if abs(x) <= tol else 'FAIL'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', default='train', choices=['train', 'full', 'test'],
                    help="observed partition used as ground truth (default: train, "
                         "the canonical thesis baseline of day_1.md §3.1)")
    ap.add_argument('--versions', nargs='*',
                    default=['v7', 'v8', 'v8_cal', 'v9', 'v9_cal', 'v10', 'v10_cal', 'v11', 'v12'])
    ap.add_argument('--json', default=None, help='write machine-readable results here')
    args = ap.parse_args()

    real = load(REAL)
    real['year'] = real['date'].dt.year
    sel = {'full': real,
           'train': real[real.year <= 2007],
           'test': real[real.year == 2009]}[args.reference]
    ref = sel['avg_rainfall'].values
    R = core_stats(ref)
    ref_h = hourly(ref)
    ref_acf_h = acf(ref_h, 24)
    ref_h_nz = ref_h[ref_h > 0]
    sel_m = sel.groupby(sel['date'].dt.month)['avg_rainfall'].sum()
    sel_d = sel.groupby(sel['date'].dt.hour)['avg_rainfall'].sum()

    data = {}
    for v in args.versions:
        p = os.path.join(GEN, f'rainfall_synthetic_10y_{v}.csv')
        if os.path.exists(p):
            data[v] = load(p)
        else:
            print(f'MISSING: {p}')
    names = list(data)

    print('=' * 100)
    print(f'GATE A SCORECARD   reference = {args.reference} '
          f'({sel.year.min()}-{sel.year.max()}, {len(ref)} steps)')
    print(f'  volume={R["volume"]:.1f} mm/yr  zero={R["zero_pct"]:.2f}%  '
          f'wet_spell={R["wet_spell_min"]:.1f} min  storm_dur={R["storm_dur_min"]:.1f} min  '
          f'N/yr={R["storm_count_yr"]:.1f}')
    print(f'  P99_wet={R["p99_wet"]:.3f}  P99.9_wet={R["p999_wet"]:.3f}  '
          f'max={R["max_burst"]:.3f}  daily_max={R["daily_max"]:.2f}')
    print('=' * 100)

    results = {v: {} for v in names}
    stats_cache = {v: core_stats(data[v]['avg_rainfall'].values) for v in names}

    def emit(tier, label, bandtxt, fn, chk):
        vals, flags = [], []
        for v in names:
            x = fn(v)
            vals.append(x)
            flags.append(chk(x))
            results[v][label] = {'value': float(x), 'band': bandtxt, 'verdict': flags[-1],
                                 'tier': tier}
        print(f'  {label:<28}{bandtxt:<12}' + ''.join(f'{x:>9.3f}' for x in vals))
        print(f'  {"":<28}{"":<12}' + ''.join(f'{f:>9}' for f in flags))

    print(f'  {"metric":<28}{"band":<12}' + ''.join(f'{v:>9}' for v in names))

    S = lambda v: stats_cache[v]
    A = lambda v: data[v]['avg_rainfall'].values

    emit(1, 'T1 volume ratio', '0.95-1.05', lambda v: S(v)['volume'] / R['volume'],
         lambda x: band(x, .95, 1.05))
    emit(1, 'T1 zero frac diff pp', '<=1.0', lambda v: S(v)['zero_pct'] - R['zero_pct'],
         lambda x: within(x, 1.0))
    emit(1, 'T1 wet-spell ratio', '0.90-1.10',
         lambda v: S(v)['wet_spell_min'] / R['wet_spell_min'], lambda x: band(x, .90, 1.10))
    emit(1, 'T1 dry-spell ratio', '0.90-1.10',
         lambda v: S(v)['dry_spell_min'] / R['dry_spell_min'], lambda x: band(x, .90, 1.10))
    emit(2, 'T2 P99 wet ratio', '0.90-1.10', lambda v: S(v)['p99_wet'] / R['p99_wet'],
         lambda x: band(x, .90, 1.10))
    emit(2, 'T2 P99.9 wet ratio', '0.85-1.15', lambda v: S(v)['p999_wet'] / R['p999_wet'],
         lambda x: band(x, .85, 1.15))
    emit(2, 'T2 max ratio', '0.80-1.25', lambda v: S(v)['max_burst'] / R['max_burst'],
         lambda x: band(x, .80, 1.25))

    def jsd_h(v):
        h = hourly(A(v)); h_nz = h[h > 0]
        if not len(h_nz):
            return 1.0
        bins = np.linspace(0.001, max(ref_h_nz.max(), h_nz.max()), 150)
        p, _ = np.histogram(ref_h_nz, bins=bins, density=True)
        q, _ = np.histogram(h_nz, bins=bins, density=True)
        p = p + 1e-9; p /= p.sum(); q = q + 1e-9; q /= q.sum()
        return float(jensenshannon(p, q))

    emit(2, 'T2 JSD hourly', '<=0.05', jsd_h, lambda x: 'PASS' if x <= 0.05 else 'FAIL')
    emit(2, 'T2 KS hourly', '<=0.05',
         lambda v: float(stats.ks_2samp(ref_h, hourly(A(v))).statistic),
         lambda x: 'PASS' if x <= 0.05 else 'FAIL')
    emit(3, 'T3 lag1 ACF diff', '<=0.02',
         lambda v: S(v)['acf1_native'] - R['acf1_native'], lambda x: within(x, 0.02))
    emit(3, 'T3 ACF RMSE 1-24h', '<=0.05',
         lambda v: float(np.sqrt(np.mean((ref_acf_h - acf(hourly(A(v)), 24)) ** 2))),
         lambda x: 'PASS' if x <= 0.05 else 'FAIL')
    emit(3, 'T3 storm duration ratio', '0.90-1.10',
         lambda v: S(v)['storm_dur_min'] / R['storm_dur_min'], lambda x: band(x, .90, 1.10))
    emit(3, 'T3 storm count ratio', '0.90-1.10',
         lambda v: S(v)['storm_count_yr'] / R['storm_count_yr'], lambda x: band(x, .90, 1.10))
    emit(3, 'T3 storm volume ratio', '0.90-1.10',
         lambda v: S(v)['storm_vol_mm'] / R['storm_vol_mm'], lambda x: band(x, .90, 1.10))
    emit(4, 'T4 daily max ratio', '0.85-1.15',
         lambda v: S(v)['daily_max'] / R['daily_max'], lambda x: band(x, .85, 1.15))
    emit(5, 'T5 monthly Pearson r', '>=0.90',
         lambda v: float(np.corrcoef(
             sel_m.values,
             data[v].groupby(data[v]['date'].dt.month)['avg_rainfall'].sum().values)[0, 1]),
         lambda x: 'PASS' if x >= 0.90 else 'FAIL')
    emit(5, 'T5 diurnal Pearson r', '>=0.80',
         lambda v: float(np.corrcoef(
             sel_d.values,
             data[v].groupby(data[v]['date'].dt.hour)['avg_rainfall'].sum().values)[0, 1]),
         lambda x: 'PASS' if x >= 0.80 else 'FAIL')

    # ---- Tier 4: AMS / IDF ------------------------------------------------
    n_ref_years = max(1, len(ref) // STEPS_PER_YEAR)
    print('\n' + '=' * 100)
    print('TIER 4 — ANNUAL MAXIMUM SERIES / IDF   band: ratio in 0.80-1.20 at EVERY (D,T) cell')
    if n_ref_years < 5:
        print(f'  NOTE: reference has only {n_ref_years} year(s); Gumbel fit is not meaningful.')
    print('=' * 100)
    idf = {v: [] for v in names}
    for d, win in DURATIONS.items():
        rx = ams(ref, win, n_ref_years)
        print(f'\n  D={d:<6} real AMS mean={rx.mean():.3f} sd={rx.std(ddof=1) if len(rx)>1 else 0:.3f}  '
              + '  '.join(f'T{t}={gumbel_quantile(rx, t):.2f}' for t in RETURN_PERIODS))
        print(f'    {"model":<9}' + ''.join(f'{"r(T"+str(t)+")":>10}' for t in RETURN_PERIODS) + '   verdict')
        for v in names:
            sx = ams(A(v), win, 10)
            rr = [gumbel_quantile(sx, t) / gumbel_quantile(rx, t) if gumbel_quantile(rx, t) else 0
                  for t in RETURN_PERIODS]
            idf[v].extend(rr)
            ok = all(0.80 <= r <= 1.20 for r in rr)
            results[v][f'T4 IDF {d}'] = {'ratios': rr, 'band': '0.80-1.20',
                                         'verdict': 'PASS' if ok else 'FAIL', 'tier': 4}
            print(f'    {v:<9}' + ''.join(f'{r:>10.2f}' for r in rr)
                  + f'   {"PASS" if ok else "FAIL"}')

    print(f'\n  TIER 4 OVERALL ({len(DURATIONS)*len(RETURN_PERIODS)} cells must all lie in 0.80-1.20)')
    print(f'    {"model":<9}{"cells PASS":>12}{"worst":>9}{"verdict":>10}')
    for v in names:
        rr = idf[v]
        n_ok = sum(1 for r in rr if 0.80 <= r <= 1.20)
        worst = min(rr) if rr else 0
        results[v]['T4 IDF overall'] = {'cells_pass': n_ok, 'cells_total': len(rr),
                                        'worst_ratio': float(worst),
                                        'verdict': 'PASS' if n_ok == len(rr) else 'FAIL',
                                        'tier': 4}
        print(f'    {v:<9}{n_ok:>9}/{len(rr)}{worst:>9.2f}'
              f'{"PASS" if n_ok == len(rr) else "FAIL":>10}')

    # ---- Interannual spread ----------------------------------------------
    print('\n' + '=' * 100)
    print('INTERANNUAL SPREAD — each realisation split into its 10 years (mean +/- sd).')
    print('  A within-realisation proxy for the >=30-member ensemble of T2.1, NOT a substitute.')
    print('=' * 100)
    per_year = {'REAL(2000-09)': [core_stats(real[real.year == y]['avg_rainfall'].values)
                                  for y in range(2000, 2010)]}
    for v in names:
        a = A(v)
        per_year[v] = [core_stats(a[i * STEPS_PER_YEAR:(i + 1) * STEPS_PER_YEAR])
                       for i in range(10)]
    for key, lbl in [('volume', 'Annual volume (mm)'), ('storm_count_yr', 'Storm count (N/yr)'),
                     ('storm_dur_min', 'Mean storm duration (min)'),
                     ('max_burst', 'Max 5-min burst (mm)')]:
        print(f'\n  {lbl}')
        rv = np.array([r[key] for r in per_year['REAL(2000-09)']])
        print(f'    {"REAL":<14}{rv.mean():>9.2f} +/-{rv.std(ddof=1):>7.2f}   '
              f'[{rv.min():.2f}, {rv.max():.2f}]')
        for v in names:
            x = np.array([r[key] for r in per_year[v]])
            z = (x.mean() - rv.mean()) / (rv.std(ddof=1) / np.sqrt(10))
            results[v].setdefault('interannual', {})[key] = {
                'mean': float(x.mean()), 'sd': float(x.std(ddof=1)), 'z_vs_real': float(z)}
            print(f'    {v:<14}{x.mean():>9.2f} +/-{x.std(ddof=1):>7.2f}   '
                  f'[{x.min():.2f}, {x.max():.2f}]  z={z:+.1f}')

    # ---- Tally ------------------------------------------------------------
    print('\n' + '=' * 100)
    print('GATE A TALLY (scalar-band metrics + Tier 4 overall)')
    print('=' * 100)
    for v in names:
        flags = [m['verdict'] for k, m in results[v].items()
                 if isinstance(m, dict) and 'verdict' in m and not k.startswith('T4 IDF 1')
                 and not k.startswith('T4 IDF 3') and not k.startswith('T4 IDF 6')
                 and not k.startswith('T4 IDF 2')]
        print(f'  {v:<10} PASS {flags.count("PASS"):>2}/{len(flags)}   '
              f'FAIL {flags.count("FAIL"):>2}')

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({'reference': args.reference,
                       'reference_stats': R,
                       'wet_threshold_mm': WET_THR,
                       'versions': results}, f, indent=2)
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()
