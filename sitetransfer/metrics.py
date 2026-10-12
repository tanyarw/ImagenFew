"""Hydrological metrics with the site's parameters passed in, never read from globals.

Definitions follow scripts/gate_a_scorecard.py and scripts/eval_suite.py exactly; the regression test
(tests/test_metrics_regression.py) checks every number against the existing evaluation cache on
Astlingen. Two additions: an optional outage mask (volume, zero fraction and the monthly and daily
cycles are computed over observed steps and scaled to full length; with no mask the formulas reduce to
the existing ones), and the daily-cycle peak/trough ratio.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from .rules import LONG_STEPS, storms_2h
from .timegrid import STEPS_PER_DAY, STEPS_PER_HOUR, STEPS_PER_YEAR, month_hour

DURATIONS = {"15min": 3, "1h": 12, "3h": 36, "6h": 72, "24h": 288}
RETURN_PERIODS = [2, 5, 10]
SCALES = [1, 3, 6, 12, 36, 72, 144, 288, 864, 2016]           # 5 min ... 1 week
MIN_STORM_STEPS = 3                                           # Gate A storm: wet run of >= 15 min
TOP_RATE = 1.25                                               # Tier 6 "top-10": the largest 1.25 storms per year


# ── building blocks (gate_a_scorecard.py) ────────────────────────────────────────────────────────

def hourly(a):
    n = len(a) // STEPS_PER_HOUR * STEPS_PER_HOUR
    return a[:n].reshape(-1, STEPS_PER_HOUR).sum(1)


def daily(a):
    n = len(a) // STEPS_PER_DAY * STEPS_PER_DAY
    return a[:n].reshape(-1, STEPS_PER_DAY).sum(1)


def acf(a, max_lag):
    m, v = a.mean(), a.var()
    if v == 0:
        return np.zeros(max_lag + 1)
    n = len(a)
    return np.array([np.mean((a[:n - k] - m) * (a[k:] - m)) / v for k in range(max_lag + 1)])


def spells(a):
    """(wet spell lengths, dry spell lengths) in steps."""
    wet = (a > 0).astype(int)
    ch = np.diff(wet, prepend=-1)
    starts = np.where(ch != 0)[0]
    lengths = np.diff(np.append(starts, len(a)))
    types = wet[starts]
    return lengths[types == 1], lengths[types == 0]


def storms(a, min_steps=MIN_STORM_STEPS):
    """Unbroken wet runs of at least min_steps."""
    wet = (a > 0).astype(int)
    ch = np.diff(wet, prepend=-1)
    starts = np.where(ch != 0)[0]
    lengths = np.diff(np.append(starts, len(a)))
    types = wet[starts]
    return [a[s:s + l] for s, l, t in zip(starts, lengths, types) if t == 1 and l >= min_steps]


def annual_maxima(a, win):
    out = []
    for y in range(len(a) // STEPS_PER_YEAR):
        s = a[y * STEPS_PER_YEAR:(y + 1) * STEPS_PER_YEAR]
        c = np.concatenate([[0.0], np.cumsum(s)])
        out.append((c[win:] - c[:-win]).max())
    return np.array(out)


def gumbel_quantile(x, T):
    """Method-of-moments Gumbel quantile at return period T."""
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float(x.mean()) if len(x) else 0.0
    b = x.std(ddof=1) * np.sqrt(6) / np.pi
    a = x.mean() - 0.5772 * b
    return float(a - b * np.log(-np.log(1 - 1.0 / T)))


def var_scaling(a):
    out = []
    for s in SCALES:
        n = len(a) // s * s
        out.append(a[:n].reshape(-1, s).sum(1).var())
    return np.array(out)


def _observed(a, mask):
    return np.ones(len(a), bool) if mask is None else ~np.asarray(mask, bool)


def coverage_sums(a, groups, n_groups, mask=None):
    """Sum per group over observed steps, scaled up by the group's observed share (plain sums if no mask)."""
    obs = _observed(a, mask)
    total = np.bincount(groups, minlength=n_groups).astype(float)
    seen = np.bincount(groups[obs], minlength=n_groups).astype(float)
    sums = np.bincount(groups[obs], weights=a[obs], minlength=n_groups)
    return np.where(seen > 0, sums * total / np.maximum(seen, 1), 0.0)


def core_stats(a, mask=None):
    """gate_a_scorecard.core_stats; volume and zero fraction over observed steps when a mask is given."""
    ws, ds = spells(a)
    st = storms(a)
    w = a[a > 0]
    ny = len(a) / STEPS_PER_YEAR
    obs = _observed(a, mask)
    return dict(
        volume=a[obs].sum() / (obs.sum() / STEPS_PER_YEAR),
        zero_pct=100 * (a[obs] == 0).mean(),
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


def storm_scale_stats(a, depth_mm):
    """Tier 6 quantities (eval_suite.sewer_storm_stats) with the site's storm-depth threshold."""
    ny = len(a) / STEPS_PER_YEAR
    dur, dep = storms_2h(a)
    top = np.sort(dep)[::-1][:max(1, int(round(TOP_RATE * ny)))]
    return dict(storms_2h_yr=len(dep) / ny, storms_ge_depth_yr=(dep >= depth_mm).sum() / ny,
                storms_320min_yr=(dur > LONG_STEPS).sum() / ny,
                rain_in_320min_pct=100 * dep[dur > LONG_STEPS].sum() / a.sum(),
                top_storm_mm=float(top.mean()), max_storm_mm=float(dep.max()))


# ── reference and per-series metrics ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Params:
    storm_depth_mm: float


def reference(x, params, mask=None):
    """Everything the metrics compare a series against (eval_suite.reference, with the site's params)."""
    s = core_stats(x, mask)
    h = hourly(x)
    mo, hr = month_hour(len(x))
    diurnal = coverage_sums(x, hr, 24, mask)
    return dict(
        stats=s, hourly=h, hourly_wet=h[h > 0], acf_h=acf(h, 48), wet=x[x > 0], daily=daily(x),
        monthly=coverage_sums(x, mo, 13, mask)[1:], diurnal=diurnal,
        diurnal_peak_trough=float(diurnal.max() / max(diurnal.min(), 1e-12)),
        var_scale=var_scaling(x),
        idf={d: [gumbel_quantile(annual_maxima(x, w), t) for t in RETURN_PERIODS] for d, w in DURATIONS.items()},
        storm_scale=storm_scale_stats(x, params.storm_depth_mm),
        long_storms_yr=sum(len(r) > LONG_STEPS for r in storms(x)) / (len(x) / STEPS_PER_YEAR),
    )


def idf_ratios(a, R):
    out = {}
    for d, w in DURATIONS.items():
        am = annual_maxima(a, w)
        for t, rq in zip(RETURN_PERIODS, R["idf"][d]):
            out[f"{d}_T{t}"] = gumbel_quantile(am, t) / rq if rq else np.nan
    return out


def hydro(a, R, params, mask=None):
    """Scalar metrics for one series against reference R (eval_suite.hydro subset)."""
    s = core_stats(a, mask)
    h = hourly(a)
    hw = h[h > 0]
    ny = len(a) / STEPS_PER_YEAR
    mo, hr = month_hour(len(a))
    monthly = coverage_sums(a, mo, 13, mask)[1:]
    diurnal = coverage_sums(a, hr, 24, mask)
    wet_runs, _ = spells(a)
    runs = storms(a)
    long_ = [r for r in runs if len(r) > LONG_STEPS]
    bins = np.linspace(0.001, max(R["hourly_wet"].max(), hw.max() if len(hw) else 1), 150)
    p, _ = np.histogram(R["hourly_wet"], bins=bins, density=True)
    q, _ = np.histogram(hw, bins=bins, density=True)
    p, q = p + 1e-9, q + 1e-9
    idf = idf_ratios(a, R)
    m = dict(
        volume_mm_yr=s["volume"], zero_pct=s["zero_pct"], wet_spell_min=s["wet_spell_min"],
        dry_spell_h=s["dry_spell_min"] / 60, p99_wet=s["p99_wet"], p999_wet=s["p999_wet"], max_5min=s["max_burst"],
        acf1=s["acf1_native"], storms_yr=s["storm_count_yr"], storm_dur_min=s["storm_dur_min"],
        storm_depth_mm=s["storm_vol_mm"], daily_max=float(daily(a).max()),
        jsd_hourly=float(jensenshannon(p / p.sum(), q / q.sum())),
        ks_hourly=float(stats.ks_2samp(R["hourly"], h).statistic),
        hourly_acf_rmse=float(np.sqrt(np.mean((acf(h, 24)[1:] - R["acf_h"][1:25]) ** 2))),
        monthly_r=float(np.corrcoef(R["monthly"], monthly)[0, 1]),
        diurnal_r=float(np.corrcoef(R["diurnal"], diurnal)[0, 1]),
        diurnal_peak_trough=float(diurnal.max() / max(diurnal.min(), 1e-12)),
        diurnal_peak_hour=int(np.argmax(diurnal)),
        single_step_showers_pct=100 * float((wet_runs == 1).mean()),
        long_storms_yr=len(long_) / ny,
        long_storm_rain_pct=100 * sum(r.sum() for r in long_) / a.sum(),
        w1_daily=float(stats.wasserstein_distance(R["daily"], daily(a))),
        var_scaling_rmse=float(np.sqrt(np.mean(np.log10(var_scaling(a) / R["var_scale"]) ** 2))),
        idf_cells_in_band=int(sum(0.8 <= v <= 1.2 for v in idf.values())),
        idf_cells=len(idf),
        **storm_scale_stats(a, params.storm_depth_mm),
    )
    m.update({f"idf_{k}": v for k, v in idf.items()})
    return m
