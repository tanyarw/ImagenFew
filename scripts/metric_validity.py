#!/usr/bin/env python
"""
metric_validity.py — do the usual generative-model metrics predict what the sewer sees?

For every version with both cached metrics (results/evaluation/eval_cache.pkl) and a sewer run
(results/reference/gate_b_sewer.json, BC controller), rank-correlate each metric's distance from
real with the sewer error |ln(overflow / real overflow)|. A metric that tracks downstream use
should have a large positive Spearman rho. The ImagenFew paper's own metrics are added when
results/evaluation/base_paper_metrics.json exists (scripts/base_paper_metrics.py, cluster).

Every metric is oriented so that 0 is "like real" and larger is worse (e.g. classifier AUC as
|AUC - 0.5|, Gate A as checks failed, ratios as |ln ratio|).

Rows: all versions; the diffusion family alone (v1-v15, e1); the classical families alone (AR /
copula, storm-and-cell); 95% intervals from resampling versions (2,000 draws). Versions within a
family are near-copies, so the "all" row overstates the effective sample size: read the family
rows and the intervals before the point estimates.

Real-vs-real floor for the sewer error: 10-year means resampled from the 9 real sewer years.

    python scripts/metric_validity.py          # -> results/evaluation/metric_validity.json
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'results', 'evaluation', 'eval_cache.pkl')
SEWER = os.path.join(ROOT, 'results', 'reference', 'gate_b_sewer.json')
BASE = os.path.join(ROOT, 'results', 'evaluation', 'base_paper_metrics.json')
OUT = os.path.join(ROOT, 'results', 'evaluation', 'metric_validity.json')
# sewer-run label -> eval-cache name (gate_b_sewer.json's "v14" is flood-control's saved v14_nobridge_cal)
ALIAS = {'v14': 'v14_nobridge_cal', 'v14_markov': 'v14'}
RNG = np.random.default_rng(0)


def family(v):
    if v.startswith('v16'):
        return 'two-level'
    if v.startswith('arima'):
        return 'AR / copula'
    if v.startswith('bartlett'):
        return 'storm-and-cell'
    return 'diffusion'


def lnr(x, ref):
    return abs(np.log(max(x, 1e-9) / ref))


def sewer():
    rows = pd.DataFrame(json.load(open(SEWER))['rows'])
    rows = rows[rows.controller == 'BC'].copy()
    rows['m3'] = rows.total_m3 * 365 / rows.days
    per = rows.groupby('source').m3.mean()
    real = rows[rows.source == 'real'].m3.to_numpy()
    floor = np.abs(np.log(RNG.choice(real, (20000, 10)).mean(1) / per['real']))
    per = per.drop('real').rename(index=lambda s: ALIAS.get(s, s))
    return per / per_real(per, rows), float(np.percentile(floor, 95)), len(real)


def per_real(per, rows):
    return rows[rows.source == 'real'].m3.mean()


def metrics(cache, base):
    hy, R = cache['hydro'], cache['hydro']['real 2000-2007']
    T_real = cache['tstr']['real 2000-2007']
    out = {}
    for v in cache['windows']:
        if v.startswith('real'):
            continue
        h, w, t = hy[v], cache['windows'][v], cache['tstr'][v]
        m = {
            # machine-learning metrics
            'C2ST AUC, 5 h': abs(w['c2st_auc_5h'] - 0.5),
            'C2ST AUC, 1 day': abs(w['c2st_auc_1day'] - 0.5),
            '1 - precision': 1 - w['precision'],
            '1 - recall': 1 - w['recall'],
            '|density - 1|': abs(w['density'] - 1),
            '1 - coverage': 1 - w['coverage'],
            'TSTR gap, rain next 1 h': abs(T_real['auc_next_1h'] - t['auc_next_1h']),
            'TSTR gap, >= 2 mm next 6 h': abs(T_real['auc_next_6h_2mm'] - t['auc_next_6h_2mm']),
            # hydrology metrics
            'Gate A checks failed (/18)': sum(not x for x in cache['gate'][v].values()),
            'Tier 6 checks failed (/3)': sum(not x for x in cache['tier6'][v].values()),
            'IDF cells out of band (/15)': h['idf_cells'] - h['idf_cells_in_band'],
            '|ln volume ratio|': lnr(h['volume_mm_yr'], R['volume_mm_yr']),
            'hourly ACF RMSE': h['hourly_acf_rmse'],
            'W1 daily totals': h['w1_daily'],
            'variance-scaling RMSE': h['var_scaling_rmse'],
            '|ln storms >= 10 mm ratio|': lnr(h['storms_ge10mm_yr'], R['storms_ge10mm_yr']),
            '|ln rain in storms > 320 min|': lnr(h['rain_in_320min_pct'], R['rain_in_320min_pct']),
            '|ln top-10 storm depth|': lnr(h['top10_storm_mm'], R['top10_storm_mm']),
            '|ln max 5-min ratio|': lnr(h['max_5min'], R['max_5min']),
        }
        if v in base:
            m['ImagenFew: discriminative score'] = base[v]['disc_mean']
            m['ImagenFew: predictive score'] = base[v]['pred_mean']
            m['ImagenFew: context-FID'] = base[v]['context_fid']
        out[v] = m
    return pd.DataFrame(out).T


ML = ['C2ST AUC, 5 h', 'C2ST AUC, 1 day', '1 - precision', '1 - recall', '|density - 1|', '1 - coverage',
      'TSTR gap, rain next 1 h', 'TSTR gap, >= 2 mm next 6 h', 'ImagenFew: discriminative score',
      'ImagenFew: predictive score', 'ImagenFew: context-FID']


def rho_ci(x, y, n=2000):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) == 0:              # too few versions, or the metric is constant in the group
        return np.nan, np.nan, np.nan, len(x)
    r = spearmanr(x, y).statistic
    bs = []
    for _ in range(n):
        i = RNG.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(r), float(lo), float(hi), int(len(x))


def main():
    cache = pickle.load(open(CACHE, 'rb'))
    base = json.load(open(BASE)) if os.path.exists(BASE) else {}
    ratio, floor95, n_real = sewer()
    M = metrics(cache, base)
    common = [v for v in M.index if v in ratio.index]
    M = M.loc[common]
    err = np.abs(np.log(ratio[common].to_numpy()))
    fam = np.array([family(v) for v in common])
    groups = {'all': np.ones(len(common), bool), 'diffusion only': fam == 'diffusion',
              'classical only': np.isin(fam, ['AR / copula', 'storm-and-cell'])}

    res = {}
    for col in M.columns:
        x = M[col].to_numpy(float)
        res[col] = {g: rho_ci(x[mask], err[mask]) for g, mask in groups.items()}

    pd.set_option('display.width', 200)
    print(f'{len(common)} versions with both metrics and a sewer run '
          f'({", ".join(f"{f} {int((fam == f).sum())}" for f in dict.fromkeys(fam))})')
    print(f'sewer error |ln(overflow / real)|: real-vs-real 95th percentile {floor95:.3f} '
          f'(10-year means from {n_real} real years); versions inside it: '
          f'{", ".join(v for v, e in zip(common, err) if e <= floor95) or "none"}\n')
    tab = pd.DataFrame({col: {g: f'{r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] n={r[3]}' for g, r in d.items()}
                        for col, d in res.items()}).T
    tab.insert(0, 'kind', ['ML' if c in ML else 'hydrology' for c in tab.index])
    tab['_s'] = [res[c]['all'][0] for c in tab.index]
    print(tab.sort_values('_s', ascending=False).drop(columns='_s').to_string())

    print('\nPer version: sewer ratio and the ML metrics')
    show = pd.DataFrame({'family': fam, 'sewer ratio': ratio[common].round(2).to_numpy(),
                         'C2ST 5h': (M['C2ST AUC, 5 h'] + 0.5).round(2).to_numpy(),
                         'TSTR 6h gap': M['TSTR gap, >= 2 mm next 6 h'].round(3).to_numpy(),
                         'Gate A fails': M['Gate A checks failed (/18)'].to_numpy(),
                         'Tier 6 fails': M['Tier 6 checks failed (/3)'].to_numpy()}, index=common)
    print(show.sort_values('sewer ratio').to_string())

    json.dump({'n_versions': len(common), 'sewer_floor95': floor95, 'versions': common,
               'families': dict(zip(common, fam.tolist())),
               'sewer_ratio': dict(zip(common, ratio[common].round(4).tolist())),
               'metrics': json.loads(M.to_json(orient='index')),
               'spearman': {c: {g: dict(zip(['rho', 'lo', 'hi', 'n'], r)) for g, r in d.items()} for c, d in res.items()},
               'base_paper_metrics_included': bool(base)},
              open(OUT, 'w'), indent=1)
    print(f'\n-> {os.path.relpath(OUT, ROOT)}')


if __name__ == '__main__':
    main()
