#!/usr/bin/env python
"""
eval_suite.py — the metrics behind notebooks/research_evaluation.ipynb.

Scores every generated version (v1-v15, the ablations, E2 and the classical baselines) against
the real record with two families of metrics:

  hydrology  water balance, occurrence (wet / dry), intensity, storms (including the
             block-boundary cliff), extremes (annual maxima, IDF), temporal structure across
             time scales, seasonality and year-to-year variability; storms at the sewer's
             scale (wet steps separated by < 2 h dry), scored as Gate A Tier 6
  ML         distribution distances against a real-vs-real noise floor; precision, recall,
             density and coverage of 5-hour windows; classifier two-sample tests on 5-hour and
             1-day windows; memorisation; train-on-synthetic-test-on-real forecasting utility;
             conditioning fidelity (does the state label steer the rain?)

Reference: the 2000-2007 training years (the canonical baseline, as in gate_a_scorecard.py, whose
definitions are reused). Noise floor: real 2-year samples that are not in the reference: the
held-out years 2008-2009 (never trained on, except by v7) and leave-two-years-out chunks of
2000-2007. A version is "as close as real" on a metric when it sits inside that spread.

Results are cached in results/evaluation/eval_cache.pkl (refreshed when a data file changes).

    python scripts/eval_suite.py              # compute or refresh the cache (~10 min on a laptop)
    python scripts/eval_suite.py --refresh    # recompute everything
"""
import argparse
import os
import pickle
import sys
import time
from argparse import Namespace

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import gate_a_scorecard as sc  # noqa: E402  (Gate A definitions: storms, spells, ACF, AMS, Gumbel)

GEN = os.path.join(ROOT, 'results', 'generated_data')
CACHE = os.path.join(ROOT, 'results', 'evaluation', 'eval_cache.pkl')
WET = 0.005                     # mm per 5 min, frozen wet threshold
Y = 105120                      # 5-min steps per (365-day) year
DOY_MONTH = np.repeat(np.arange(1, 13), [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
SURV_K = [12, 24, 36, 48, 64, 96]          # storm-duration thresholds in steps (1 h ... 8 h)
SCALES = [1, 3, 6, 12, 36, 72, 144, 288, 864, 2016]   # aggregation scales, 5 min ... 1 week
STORM_GAP = 24                  # Tier 6 storms: a dry run of 2 h ends one (flood-control's src/rain/split.py)
TOP_RATE = 1.25                 # Tier 6 "top-10": the largest 1.25 storms per year (10 in the 8 reference years)
RNG = np.random.default_rng(0)

# name, group, short label, what changed, block length (steps), plan key, clean 2000-2007 split
REGISTRY = [
    ('v1', 'early', 'v1', 'first 5-min model (unconditional)', None, None, False),
    ('v2', 'early', 'v2', 'v1 + negative-value thresholding', None, None, False),
    ('v3', 'early', 'v3', 'unconditional, 10-min resolution', None, None, False),
    ('v4', 'early', 'v4', '4 GMM regime classes, 10-min resolution', None, None, False),
    ('v5', 'early', 'v5', '4 seasonal-phase classes (daily features)', None, None, False),
    ('v6', 'early', 'v6', '4 seasonal-phase classes (day of year)', None, None, False),
    ('v7', 'early', 'v7', '4 seasonal-phase classes, trained on 2000-2009 (incl. held-out years)', None, None, False),
    ('v8', 'context length', 'v8 L24', 'clean 2000-2007 split, 2-h blocks', 24, 'markov24', True),
    ('v8_cal', 'context length', 'v8 L24 cal', 'v8, calendar assembly', 24, 'cal24', True),
    ('v9', 'context length', 'v9 L36', '3-h blocks', 36, 'markov36', True),
    ('v9_cal', 'context length', 'v9 L36 cal', 'v9, calendar assembly', 36, 'cal36', True),
    ('v10', 'context length', 'v10 L64', '5h20 blocks (previous best)', 64, 'markov64', True),
    ('v10_cal', 'context length', 'v10 L64 cal', 'v10, calendar assembly', 64, 'cal64', True),
    ('e1_v10ckpt24', 'checkpoint control', 'v10 init24', 'v10 initialised from the 24-step checkpoint', 64, 'markov64', True),
    ('e1_v10ckpt24_cal', 'checkpoint control', 'v10 init24 cal', 'as above, calendar assembly', 64, 'cal64', True),
    ('v11', 'conditioning', 'v11 no label', 'v8 without the state label', 24, 'uncond24', True),
    ('v12', 'conditioning', 'v12 no label', 'v10 without the state label', 64, 'uncond64', True),
    ('v13', 'transform', 'v13 log1p', 'v10 + log1p transform', 64, 'markov64', True),
    ('v13_cal', 'transform', 'v13 log1p cal', 'v13, calendar assembly', 64, 'cal64', True),
    ('v14', 'transform', 'v14 asinh', 'v10 + asinh transform (current baseline)', 64, 'markov64', True),
    ('v14_cal', 'transform', 'v14 asinh cal', 'v14, calendar assembly', 64, 'cal64', True),
    ('v15', 'attention (E1)', 'v15 attn (E1)', 'v14 + 8x8 attention, 200 more epochs', 64, 'markov64', True),
    ('v15_cal', 'attention (E1)', 'v15 attn cal (E1)', 'v15, calendar assembly', 64, 'cal64', True),
    ('v15_ctrl', 'attention (E1)', 'v15 ctrl (E1 control)', 'v14 + 200 more epochs, no change (control)', 64, 'markov64', True),
    ('v15_ctrl_cal', 'attention (E1)', 'v15 ctrl cal', 'v15_ctrl, calendar assembly', 64, 'cal64', True),
    ('v14_nobridge_cal', 'assembly', 'v14 cal no bridge', 'v14_cal without bridge blocks (no calendar drift)', 64, 'cal64nb', True),
    ('v14_ctx', 'context chaining (E2)', 'v14 ctx (E2)', 'v14 + context chaining; partial: 4 chains x 2 y', 64, 'markov64', True),
    ('v14_ctx_cal', 'context chaining (E2)', 'v14 ctx cal (E2)', 'as above, exact calendar; partial: 4 chains x 2 y', 64, 'calexact', True),
    ('arima_len24', 'classical baseline', 'AR(24)', 'Gaussian AR(24)', 24, None, True),
    ('arima_len64', 'classical baseline', 'AR(64)', 'Gaussian AR(64)', 64, None, True),
    ('arima_copula_len24', 'classical baseline', 'copula AR(24)', 'Gaussian-copula AR(24) + quantile mapping', 24, None, True),
    ('arima_copula_len64', 'classical baseline', 'copula AR(64)', 'Gaussian-copula AR(64) + quantile mapping', 64, None, True),
    ('arima_copula_seas_markov_len24', 'classical baseline', 'seas. copula AR(24)', 'copula AR(24), 4 states, Markov assembly', 24, None, True),
    ('arima_copula_seas_markov_len64', 'classical baseline', 'seas. copula AR(64)', 'copula AR(64), 4 states, Markov assembly', 64, None, True),
    ('arima_copula_seas_cal_len24', 'classical baseline', 'seas. copula AR(24) cal', 'copula AR(24), 4 states, calendar', 24, None, True),
    ('arima_copula_seas_cal_len64', 'classical baseline', 'seas. copula AR(64) cal', 'copula AR(64), 4 states, calendar', 64, None, True),
    ('arima_copula_occ_len64', 'classical baseline', 'copula AR(64), occurrence-matched', 'copula AR(64), latent correlation solved from joint wet probabilities', 64, None, True),
    ('arima_copula_occ_cal_len64', 'classical baseline', 'copula AR(64), occurrence-matched cal', 'as above, 4 state quantile tables, calendar', 64, None, True),
    ('bartlett_lewis_classic', 'classical baseline', 'storm-and-cell (classic)', 'randomised Bartlett-Lewis pulses, fixed cell intensity, per month', None, None, True),
    ('bartlett_lewis', 'classical baseline', 'storm-and-cell', 'randomised Bartlett-Lewis pulses, intensity scales with cell speed (Kaczmarska 2014), per month', None, None, True),
    ('bartlett_lewis_relaxed', 'classical baseline', 'storm-and-cell (alpha relaxed)', 'as storm-and-cell, alpha allowed down to 0.2 (Onof & Wang 2020)', None, None, True),
    ('bartlett_lewis_relaxed_wet', 'classical baseline', 'storm-and-cell (relaxed, wet fraction)', 'relaxed alpha, objective scores the wet fraction', None, None, True),
    ('bartlett_lewis_relaxed_ivw', 'classical baseline', 'storm-and-cell (relaxed, inverse variance)', 'relaxed alpha, inverse-variance weights (pyBL)', None, None, True),
    ('v16', 'two-level', 'v16 two-level', 'storm renewal model (2-h gap, 2000-2007) + v14 5-min texture', None, None, True),
    ('v16_m1', 'two-level', 'v16 seed 1', 'v16, generator seed 1', None, None, True),
    ('v16_m2', 'two-level', 'v16 seed 2', 'v16, generator seed 2', None, None, True),
    ('v16_m3', 'two-level', 'v16 seed 3', 'v16, generator seed 3', None, None, True),
    ('v16_m4', 'two-level', 'v16 seed 4', 'v16, generator seed 4', None, None, True),
]
META = pd.DataFrame(REGISTRY, columns=['name', 'group', 'label', 'change', 'L', 'plan', 'clean']).set_index('name')


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def _clean(a):
    a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
    a[a < WET] = 0.0
    return a


def load_real():
    """(train 2000-2007, held-out 2008-2009) at 5 min. The record has 365-day years."""
    a = pd.read_csv(os.path.join(ROOT, 'data', 'rainfall', 'real_rainfall_data.csv'),
                    usecols=['avg_rainfall'])['avg_rainfall'].to_numpy()
    a = _clean(a)
    return a[:8 * Y], a[8 * Y:]


PARTIAL = ('v14_ctx', 'v14_ctx_cal')      # E2 runs stopped early: only their progress files exist


def data_path(name):
    if name in PARTIAL:
        return os.path.join(GEN, f'rainfall_synthetic_10y_{name}.ctx_state.npz')
    return os.path.join(GEN, f'rainfall_synthetic_10y_{name}.csv')


def load_series(name):
    """Returns (series, resolution in minutes). E2 progress files: each chain cut to whole
    years and the chains placed one after another, so 'year' slices stay calendar-aligned."""
    path = data_path(name)
    if path.endswith('.npz'):
        out = np.load(path)['out']
        full = out.shape[1] // Y
        return _clean(np.concatenate([m[:full * Y] for m in out])), 5
    a = pd.read_csv(path, usecols=['avg_rainfall'])['avg_rainfall'].to_numpy()
    return _clean(a), (5 if len(a) > 600_000 else 10)


def to_hourly(a, res=5):
    k = 60 // res
    n = len(a) // k * k
    return a[:n].reshape(-1, k).sum(1)


def month_hour(n, res=5):
    """Month (1-12) and hour (0-23) of every step, on a 365-day calendar from 1 January."""
    per_day = 1440 // res
    p = np.arange(n)
    return DOY_MONTH[(p // per_day) % 365], (p % per_day) // (60 // res)


# ──────────────────────────────────────────────────────────────────────
# Hydrology
# ──────────────────────────────────────────────────────────────────────

def reference(train):
    """Everything about the reference that the metrics compare against."""
    s = sc.core_stats(train)
    h = to_hourly(train)
    mo, hr = month_hour(len(train))
    dur = np.array([len(r) for r in sc.storms(train)])
    return dict(
        stats=s, hourly=h, hourly_wet=h[h > 0], acf_h=sc.acf(h, 48),
        wet=train[train > 0], daily=sc.daily(train),
        monthly=np.bincount(mo, weights=train, minlength=13)[1:],
        diurnal=np.bincount(hr, weights=train, minlength=24),
        surv=np.array([(dur > k).mean() for k in SURV_K]),
        var_scale=var_scaling(train),
        idf={d: [sc.gumbel_quantile(sc.ams(train, w, 8), t) for t in sc.RETURN_PERIODS]
             for d, w in sc.DURATIONS.items()},
        annual=np.array([train[y * Y:(y + 1) * Y].sum() for y in range(len(train) // Y)]),
        sewer_storms=sewer_storm_stats(train),
    )


def sewer_storms(a):
    """Duration (steps) and depth (mm) of every storm at the sewer's scale: wet steps with no dry
    run of STORM_GAP steps or more between them."""
    wet = np.flatnonzero(a > 0)
    if not len(wet):
        return np.zeros(0, int), np.zeros(0)
    br = np.flatnonzero(np.diff(wet) > STORM_GAP)
    s, e = np.r_[wet[0], wet[br + 1]], np.r_[wet[br], wet[-1]]
    c = np.r_[0.0, np.cumsum(a)]
    return e - s + 1, c[e + 1] - c[s]


def sewer_storm_stats(a):
    """Tier 6 quantities. top10_mm: mean depth of the largest TOP_RATE storms per year, so records
    of different lengths are compared at the same frequency (10 storms in 8 years)."""
    ny = len(a) / Y
    dur, dep = sewer_storms(a)
    top = np.sort(dep)[::-1][:max(1, int(round(TOP_RATE * ny)))]
    return dict(storms_2h_yr=len(dep) / ny, storms_ge10mm_yr=(dep >= 10).sum() / ny,
                storms_320min_yr=(dur > 64).sum() / ny,
                rain_in_320min_pct=100 * dep[dur > 64].sum() / a.sum(),
                top10_storm_mm=float(top.mean()), max_storm_mm=float(dep.max()))


def var_scaling(a):
    """Variance of rainfall totals at each aggregation scale (5 min ... 1 week)."""
    out = []
    for s in SCALES:
        n = len(a) // s * s
        out.append(a[:n].reshape(-1, s).sum(1).var())
    return np.array(out)


def idf_ratios(a, R, res=5):
    """Gumbel T-year level of annual maxima, generated / real, for every (duration, T) cell."""
    ny = len(a) // (Y if res == 5 else Y // 2)
    out = {}
    for d, w in sc.DURATIONS.items():
        if res == 10 and w % 2:                      # 15 min cannot be built from 10-min steps
            continue
        win = w if res == 5 else w // 2
        steps_year = Y if res == 5 else Y // 2
        ams = []
        for y in range(ny):
            seg = a[y * steps_year:(y + 1) * steps_year]
            c = np.concatenate([[0.0], np.cumsum(seg)])
            ams.append((c[win:] - c[:-win]).max())
        ams = np.array(ams)
        for t, rq in zip(sc.RETURN_PERIODS, R['idf'][d]):
            out[(d, t)] = sc.gumbel_quantile(ams, t) / rq if rq else np.nan
    return out


def hydro(a, R, res=5):
    """Scalar hydrological metrics for one series (NaN where the resolution cannot support it)."""
    h = to_hourly(a, res)
    hw = h[h > 0]
    ny = len(a) / (Y if res == 5 else Y // 2)
    mo, hr = month_hour(len(a), res)
    m = dict(
        volume_mm_yr=a.sum() / ny,
        wet_hours_pct=100 * (h > 0).mean(),
        max_1h=h.max(),
        daily_max=sc.daily(a).max() if res == 5 else h[: len(h) // 24 * 24].reshape(-1, 24).sum(1).max(),
        hourly_acf_rmse=float(np.sqrt(np.mean((sc.acf(h, 24)[1:] - R['acf_h'][1:25]) ** 2))),
        ks_hourly=float(stats.ks_2samp(R['hourly'], h).statistic),
        w1_wet_hours=float(stats.wasserstein_distance(R['hourly_wet'], hw)) if len(hw) else np.nan,
        monthly_r=float(np.corrcoef(R['monthly'], np.bincount(mo, weights=a, minlength=13)[1:])[0, 1]),
        diurnal_r=float(np.corrcoef(R['diurnal'], np.bincount(hr, weights=a, minlength=24))[0, 1]),
        interannual_cv=float(np.std([a[y * int(len(a) / ny):(y + 1) * int(len(a) / ny)].sum()
                                     for y in range(int(ny))], ddof=1)
                             / np.mean([a[y * int(len(a) / ny):(y + 1) * int(len(a) / ny)].sum()
                                        for y in range(int(ny))])),
    )
    bins = np.linspace(0.001, max(R['hourly_wet'].max(), hw.max() if len(hw) else 1), 150)
    p, _ = np.histogram(R['hourly_wet'], bins=bins, density=True)
    q, _ = np.histogram(hw, bins=bins, density=True)
    p = p + 1e-9; q = q + 1e-9
    m['jsd_hourly'] = float(jensenshannon(p / p.sum(), q / q.sum()))
    idf = idf_ratios(a, R, res)
    m['idf_cells_in_band'] = int(sum(0.8 <= v <= 1.2 for v in idf.values()))
    m['idf_cells'] = len(idf)
    m['idf_worst'] = float(max(idf.values(), key=lambda v: abs(np.log(v))))   # furthest from 1
    if res != 5:
        return m, idf, None

    s = sc.core_stats(a)
    w = a > 0
    wet = a[w]
    runs = sc.storms(a)
    dur = np.array([len(r) for r in runs])
    long_ = [r for r in runs if len(r) > 64]
    wet_runs, _ = sc.spells(a)
    m.update(
        single_step_showers_pct=100 * float((wet_runs == 1).mean()),
        zero_pct=s['zero_pct'],
        p_wet_after_wet=float(w[1:][w[:-1]].mean()),
        p_wet_after_dry=float(w[1:][~w[:-1]].mean()),
        wet_spell_min=s['wet_spell_min'], dry_spell_h=s['dry_spell_min'] / 60,
        mean_wet_mm=float(wet.mean()), p50_wet=float(np.percentile(wet, 50)),
        p90_wet=float(np.percentile(wet, 90)), p99_wet=s['p99_wet'], p999_wet=s['p999_wet'],
        max_5min=s['max_burst'], very_light_pct=100 * float(((a >= WET) & (a < 0.02)).mean()),
        storms_yr=s['storm_count_yr'], storm_dur_min=s['storm_dur_min'], storm_depth_mm=s['storm_vol_mm'],
        storm_peak_mm=float(np.mean([r.max() for r in runs])),
        long_storms_yr=len(long_) / ny,
        long_storm_rain_pct=100 * sum(r.sum() for r in long_) / a.sum(),
        acf1=s['acf1_native'],
        w1_wet_5min=float(stats.wasserstein_distance(R['wet'], wet)),
        w1_daily=float(stats.wasserstein_distance(R['daily'], sc.daily(a))),
        var_scaling_rmse=float(np.sqrt(np.mean(np.log10(var_scaling(a) / R['var_scale']) ** 2))),
        **sewer_storm_stats(a),
    )
    for k, sr in zip(SURV_K, R['surv']):
        m[f'surv_ratio_{k * 5}min'] = float((dur > k).mean() / sr)
    curves = dict(
        surv=np.array([(dur > k).mean() for k in range(1, 200)]),
        wet_quantiles=np.quantile(wet, QPROBS),
        acf_h=sc.acf(h, 48), var_scale=var_scaling(a),
        monthly=np.bincount(mo, weights=a, minlength=13)[1:] / ny,
        diurnal=np.bincount(hr, weights=a, minlength=24) / ny,
    )
    return m, idf, curves


QPROBS = 1 - np.geomspace(0.5, 1e-4, 60)       # wet-intensity exceedance curve, P50 ... P99.99


def gate_a(m, R):
    """The 18 Gate A checks (gate_a_scorecard.py), from the metrics above. None if not computable."""
    S = R['stats']
    if 'zero_pct' not in m:
        return None
    checks = {
        'T1 volume': 0.95 <= m['volume_mm_yr'] / S['volume'] <= 1.05,
        'T1 zero fraction': abs(m['zero_pct'] - S['zero_pct']) <= 1.0,
        'T1 wet spell': 0.90 <= m['wet_spell_min'] / S['wet_spell_min'] <= 1.10,
        'T1 dry spell': 0.90 <= m['dry_spell_h'] * 60 / S['dry_spell_min'] <= 1.10,
        'T2 P99 wet': 0.90 <= m['p99_wet'] / S['p99_wet'] <= 1.10,
        'T2 P99.9 wet': 0.85 <= m['p999_wet'] / S['p999_wet'] <= 1.15,
        'T2 max': 0.80 <= m['max_5min'] / S['max_burst'] <= 1.25,
        'T2 JSD hourly': m['jsd_hourly'] <= 0.05,
        'T2 KS hourly': m['ks_hourly'] <= 0.05,
        'T3 lag-1 ACF': abs(m['acf1'] - S['acf1_native']) <= 0.02,
        'T3 hourly ACF': m['hourly_acf_rmse'] <= 0.05,
        'T3 storm duration': 0.90 <= m['storm_dur_min'] / S['storm_dur_min'] <= 1.10,
        'T3 storm count': 0.90 <= m['storms_yr'] / S['storm_count_yr'] <= 1.10,
        'T3 storm volume': 0.90 <= m['storm_depth_mm'] / S['storm_vol_mm'] <= 1.10,
        'T4 daily max': 0.85 <= m['daily_max'] / S['daily_max'] <= 1.15,
        'T4 IDF (all 15 cells)': m['idf_cells_in_band'] == m['idf_cells'] == 15,
        'T5 monthly cycle': m['monthly_r'] >= 0.90,
        'T5 diurnal cycle': m['diurnal_r'] >= 0.80,
    }
    return {k: bool(v) for k, v in checks.items()}


def tier6(m, R):
    """Gate A Tier 6, storm scale (code_plan/ACCEPTANCE_CRITERIA.md): kept apart from the 18 checks
    above so earlier tallies stay comparable. Bands: the 95% range of a 10-year resample of real
    years against the 8-year reference, rounded out. None if not computable."""
    if 'storms_ge10mm_yr' not in m:
        return None
    S = R['sewer_storms']
    checks = {
        'S1 storms >= 10 mm': 0.85 <= m['storms_ge10mm_yr'] / S['storms_ge10mm_yr'] <= 1.15,
        'S2 rain in storms > 320 min': 0.90 <= m['rain_in_320min_pct'] / S['rain_in_320min_pct'] <= 1.10,
        'S3 top-10 storm depths': 0.70 <= m['top10_storm_mm'] / S['top10_storm_mm'] <= 1.45,
    }
    return {k: bool(v) for k, v in checks.items()}


# ──────────────────────────────────────────────────────────────────────
# Noise floor: 2-year samples, generated vs real
# ──────────────────────────────────────────────────────────────────────

def chunk_metrics(c, ref):
    """Distances of one 2-year sample `c` from a reference sample `ref` (both 5 min)."""
    hc, hr = to_hourly(c), to_hourly(ref)
    dc = np.array([len(r) for r in sc.storms(c)])
    dr = np.array([len(r) for r in sc.storms(ref)])
    st_c, st_r = sewer_storm_stats(c), sewer_storm_stats(ref)
    return dict(
        w1_wet_5min=stats.wasserstein_distance(ref[ref > 0], c[c > 0]),
        w1_wet_hours=stats.wasserstein_distance(hr[hr > 0], hc[hc > 0]),
        w1_daily=stats.wasserstein_distance(sc.daily(ref), sc.daily(c)),
        hourly_acf_rmse=float(np.sqrt(np.mean((sc.acf(hc, 24)[1:] - sc.acf(hr, 24)[1:]) ** 2))),
        zero_gap_pp=abs(100 * (c == 0).mean() - 100 * (ref == 0).mean()),
        volume_ratio=c.sum() / len(c) / (ref.sum() / len(ref)),
        surv_ratio_320min=(dc > 64).mean() / (dr > 64).mean(),
        long_storms_yr=(dc > 64).sum() / (len(c) / Y),
        storms_ge10mm_ratio=st_c['storms_ge10mm_yr'] / st_r['storms_ge10mm_yr'],
        rain_in_320min_ratio=st_c['rain_in_320min_pct'] / st_r['rain_in_320min_pct'],
    )


def noise_floor(train, held):
    rows = []
    for i in range(4):                                    # leave two training years out
        c = train[2 * i * Y:(2 * i + 2) * Y]
        ref = np.concatenate([train[:2 * i * Y], train[(2 * i + 2) * Y:]])
        rows.append(dict(sample=f'real {2000 + 2 * i}-{2001 + 2 * i}', **chunk_metrics(c, ref)))
    rows.append(dict(sample='real 2008-2009 (held out)', **chunk_metrics(held, train)))
    return pd.DataFrame(rows).set_index('sample')


def chunked(a, train):
    n = len(a) // (2 * Y)
    return pd.DataFrame([chunk_metrics(a[2 * i * Y:(2 * i + 2) * Y], train) for i in range(n)])


# ──────────────────────────────────────────────────────────────────────
# Window-level ML metrics
# ──────────────────────────────────────────────────────────────────────

def wet_windows(a, L, agg=1):
    n = len(a) // L
    W = a[:n * L].reshape(n, L)
    W = W[W.max(1) > 0]
    if agg > 1:
        W = W.reshape(len(W), L // agg, agg).sum(2)
    return np.log1p(W / 0.1), n


def prdc(real, fake, k=5):
    """Precision, recall, density, coverage (Naeem et al., ICML 2020) with k-NN balls."""
    from sklearn.metrics import pairwise_distances
    rr = pairwise_distances(real)
    r_rad = np.sort(rr, axis=1)[:, k]                    # k-th neighbour, column 0 is the point itself
    ff = pairwise_distances(fake)
    f_rad = np.sort(ff, axis=1)[:, k]
    fr = pairwise_distances(fake, real)                  # [n_fake, n_real]
    inside = fr <= r_rad[None, :]
    return dict(precision=inside.any(1).mean(),
                recall=(fr.T <= f_rad[None, :]).any(1).mean(),
                density=inside.sum(1).mean() / k,
                coverage=(fr.min(0) <= r_rad).mean())


def c2st_auc(real, fake, seed=0):
    """Classifier two-sample test: 5-fold ROC AUC of telling real from generated (0.5 = can't)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    n = min(len(real), len(fake))
    rng = np.random.default_rng(seed)
    X = np.vstack([real[rng.choice(len(real), n, replace=False)], fake[rng.choice(len(fake), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                 scoring='roc_auc').mean())


class WindowRef:
    """Fixed real samples and embeddings shared by every version (so results are comparable)."""
    def __init__(self, train, held, n_real=2000, n_fake=1000, n_real_day=1000, n_fake_day=300):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        self.n_fake, self.n_fake_day = n_fake, n_fake_day
        W, self.n_win_train = wet_windows(train, 64)
        self.wet_share_train = len(W) / self.n_win_train
        self.pca = PCA(16, random_state=0).fit(W)
        self.real = W[RNG.choice(len(W), n_real, replace=False)]
        self.real_emb = self.pca.transform(self.real)
        self.nn = NearestNeighbors(n_neighbors=1).fit(W)  # all training windows, for memorisation
        D, _ = wet_windows(train, 288, agg=3)
        self.pca_day = PCA(16, random_state=0).fit(D)
        self.real_day = D[RNG.choice(len(D), n_real_day, replace=False)]
        self.real_day_emb = self.pca_day.transform(self.real_day)
        Wh, _ = wet_windows(held, 64)
        self.held_nn = self.nn.kneighbors(Wh)[0][:, 0]  # how close genuinely new real windows get

    def score(self, a):
        W, n = wet_windows(a, 64)
        D, _ = wet_windows(a, 288, agg=3)
        f = W[RNG.choice(len(W), min(self.n_fake, len(W)), replace=False)]
        fd = D[RNG.choice(len(D), min(self.n_fake_day, len(D)), replace=False)]
        out = prdc(self.real_emb, self.pca.transform(f))
        out['c2st_auc_5h'] = c2st_auc(self.real, f)
        out['c2st_auc_1day'] = c2st_auc(self.real_day, fd)
        out['wet_window_share'] = len(W) / n
        out['wet_steps_per_wet_window'] = float((np.expm1(W) > 0).sum(1).mean())
        nn = self.nn.kneighbors(f)[0][:, 0]
        out['nn_dist_ratio'] = float(np.median(nn) / np.median(self.held_nn))
        out['too_close_pct'] = 100 * float((nn < np.percentile(self.held_nn, 5)).mean())
        return out


# ──────────────────────────────────────────────────────────────────────
# Utility: train on synthetic, test on real (TSTR)
# ──────────────────────────────────────────────────────────────────────

def forecast_xy(h):
    """Features: last 6 hourly totals and the 24-h total (log1p). Targets: rain >= 0.1 mm in the
    next hour; >= 2 mm over the next 6 hours (storm persistence beyond a 5h20 block)."""
    from numpy.lib.stride_tricks import sliding_window_view
    t = np.arange(24, len(h) - 6)
    last6 = sliding_window_view(h, 6)[t - 6]
    last24 = sliding_window_view(h, 24)[t - 24].sum(1, keepdims=True)
    X = np.log1p(np.hstack([last6, last24]))
    nxt6 = sliding_window_view(h, 6)[t].sum(1)
    return X, h[t] >= 0.1, nxt6 >= 2.0


def tstr(h_train_syn, test_xy, seed=0, n=40000):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X, y1, y6 = forecast_xy(h_train_syn)
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    Xt, y1t, y6t = test_xy
    out = {}
    for name, y, yt in [('auc_next_1h', y1, y1t), ('auc_next_6h_2mm', y6, y6t)]:
        if y[idx].all() or not y[idx].any():
            out[name] = np.nan
            continue
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed).fit(X[idx], y[idx])
        out[name] = float(roc_auc_score(yt, clf.predict_proba(Xt)[:, 1]))
    return out


# ──────────────────────────────────────────────────────────────────────
# Conditioning fidelity (Markov L=64 versions; seed-42 production timeline)
# ──────────────────────────────────────────────────────────────────────

def markov64_timeline(total):
    from regime_training.generate_context import state_timeline   # imports torch (local venv has it)
    cli = Namespace(mode='markov', seed=42, overlap=4, transition_overlap=8, bridge_blocks=1,
                    transition_matrix_path='data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl')
    return state_timeline(cli, total, 64)


def per_state(a, states):
    return pd.DataFrame({'rain_mm_yr': [a[states == s].mean() * Y for s in range(4)],
                         'wet_pct': [100 * (a[states == s] > 0).mean() for s in range(4)]})


# ──────────────────────────────────────────────────────────────────────
# Year by year
# ──────────────────────────────────────────────────────────────────────

def per_year(a, R):
    rows = []
    for y in range(len(a) // Y):
        c = a[y * Y:(y + 1) * Y]
        ws, _ = sc.spells(c)
        runs = sc.storms(c)
        h = to_hourly(c)
        dur2, dep2 = sewer_storms(c)
        rows.append(dict(
            year=y + 1,
            volume_ratio=c.sum() / R['stats']['volume'],
            zero_pct=100 * (c == 0).mean(),
            wet_spell_min=ws.mean() * 5,
            storm_dur_min=np.mean([len(r) for r in runs]) * 5,
            long_storms=sum(len(r) > 64 for r in runs),
            p99_wet=np.percentile(c[c > 0], 99),
            max_1h=h.max(),
            hourly_acf_rmse=float(np.sqrt(np.mean((sc.acf(h, 24)[1:] - R['acf_h'][1:25]) ** 2))),
            storms_ge10mm=int((dep2 >= 10).sum()),
            rain_in_320min_pct=100 * dep2[dur2 > 64].sum() / c.sum() if c.sum() else np.nan,
        ))
    return pd.DataFrame(rows).set_index('year')


def paired(a_years, b_years, metric):
    """Mean of (b - a) over shared years, 95% t-interval, and how many years b > a."""
    n = min(len(a_years), len(b_years))
    d = (b_years[metric].to_numpy()[:n] - a_years[metric].to_numpy()[:n]).astype(float)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    t = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    return dict(metric=metric, mean_diff=d.mean(), ci_low=d.mean() - t * se, ci_high=d.mean() + t * se,
                years_up=int((d > 0).sum()), years=n,
                p_sign=float(stats.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue) if (d != 0).any() else 1.0)


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def compute(refresh=False):
    names = [n for n in META.index if os.path.exists(data_path(n))]
    stamp = {n: os.path.getmtime(data_path(n)) for n in names}
    if os.path.exists(CACHE) and not refresh:
        with open(CACHE, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('stamp') == stamp:
            return cached
    t0 = time.time()
    train, held = load_real()
    R = reference(train)
    wref = WindowRef(train, held)
    test_xy = forecast_xy(to_hourly(held))
    timeline = None
    real_states = pd.read_csv(os.path.join(ROOT, 'data/rainfall/splits/train_years_labelled.csv'),
                              usecols=['hmm_state'])['hmm_state'].to_numpy()

    out = dict(stamp=stamp, hydro={}, idf={}, curves={}, gate={}, tier6={}, windows={}, tstr={}, floor_gen={},
               per_year={}, states={}, years={})
    # real rows: training years scored against themselves are not informative, so the real
    # comparisons use the held-out years (an honest "fresh real data" score)
    for nm, a in [('real 2000-2007', train), ('real 2008-2009', held)]:
        m, idf, cur = hydro(a, R)
        out['hydro'][nm], out['idf'][nm], out['curves'][nm] = m, idf, cur
    out['gate']['real 2008-2009'] = gate_a(out['hydro']['real 2008-2009'], R)
    out['tier6']['real 2008-2009'] = tier6(out['hydro']['real 2008-2009'], R)
    out['windows']['real 2008-2009'] = wref.score(held)
    out['tstr']['real 2000-2007'] = tstr(to_hourly(train), test_xy)
    out['per_year']['real 2000-2007'] = per_year(train, R)
    out['states']['real 2000-2007'] = per_state(train, real_states[:len(train)])
    out['floor'] = noise_floor(train, held)
    print(f'reference ready ({time.time() - t0:.0f}s)', flush=True)

    for nm in names:
        t1 = time.time()
        a, res = load_series(nm)
        out['years'][nm] = len(a) / (Y if res == 5 else Y // 2)
        m, idf, cur = hydro(a, R, res)
        out['hydro'][nm], out['idf'][nm], out['curves'][nm] = m, idf, cur
        out['tstr'][nm] = tstr(to_hourly(a, res), test_xy)
        if res == 5:
            out['gate'][nm] = gate_a(m, R)
            out['tier6'][nm] = tier6(m, R)
            out['windows'][nm] = wref.score(a)
            out['floor_gen'][nm] = chunked(a, train)
            out['per_year'][nm] = per_year(a, R)
            if META.loc[nm, 'plan'] == 'markov64' and META.loc[nm, 'L'] == 64:
                if timeline is None:
                    timeline = markov64_timeline(10 * Y)
                st = np.tile(timeline[:2 * Y], len(a) // (2 * Y)) if nm == 'v14_ctx' else timeline[:len(a)]
                out['states'][nm] = per_state(a, st)
        print(f'{nm:32s} {time.time() - t1:5.1f}s', flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(out, f)
    print(f'done in {time.time() - t0:.0f}s -> {os.path.relpath(CACHE, ROOT)}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--refresh', action='store_true')
    compute(refresh=ap.parse_args().refresh)
