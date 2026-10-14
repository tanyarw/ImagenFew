"""Checks, metric families, pass bands and the transfer verdict.

Two kinds of bands:
  fixed      the existing Gate A / Tier 6 bands (eval_suite.gate_a, eval_suite.tier6). Reproduced exactly
             on Astlingen (regression test); reported for comparability with every earlier tally.
  resampled  each check's 95% range under "both series come from the same climate": in every draw a
             10-year sample and a reference-sized sample are drawn independently (whole years, with
             replacement) from the reference years, and the check is computed for the first against the
             second. This is how the Tier 6 bands were made (10 years against an 8-year resample).
             Computed the same way at every site, so the transfer verdict compares like with like.

Families (code_plan/SINGAPORE_TRANSFER_PLAN.md, section 8): A amount and occurrence, B intensity and
extremes, C short-scale timing, D long-scale structure, E_monthly and E_daily cycles. A family passes when
at least half of its checks with a band pass.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import metrics as M
from .timegrid import STEPS_PER_YEAR


@dataclass(frozen=True)
class Check:
    name: str
    family: str
    kind: str                    # ratio | diff | upper | lower | cells
    metric: str                  # key in hydro(); for ratios also the reference key path
    ref: tuple = ()              # path into the reference dict for ratios / diffs
    fixed: tuple = None          # (lo, hi) for ratio/diff, (thr,) for upper/lower, None = no fixed band


def _ref(R, path):
    v = R
    for k in path:
        v = v[k]
    return v


CHECKS = [
    Check("T1 volume", "A", "ratio", "volume_mm_yr", ("stats", "volume"), (0.95, 1.05)),
    Check("T1 zero fraction", "A", "diff", "zero_pct", ("stats", "zero_pct"), (-1.0, 1.0)),
    Check("T1 dry spell", "A", "ratio", "dry_spell_min", ("stats", "dry_spell_min"), (0.90, 1.10)),
    Check("T1 wet spell", "C", "ratio", "wet_spell_min", ("stats", "wet_spell_min"), (0.90, 1.10)),
    Check("T2 P99 wet", "B", "ratio", "p99_wet", ("stats", "p99_wet"), (0.90, 1.10)),
    Check("T2 P99.9 wet", "B", "ratio", "p999_wet", ("stats", "p999_wet"), (0.85, 1.15)),
    Check("T2 max", "B", "ratio", "max_5min", ("stats", "max_burst"), (0.80, 1.25)),
    Check("T2 JSD hourly", "B", "upper", "jsd_hourly", (), (0.05,)),
    Check("T2 KS hourly", "B", "upper", "ks_hourly", (), (0.05,)),
    Check("T3 lag-1 ACF", "C", "diff", "acf1", ("stats", "acf1_native"), (-0.02, 0.02)),
    Check("T3 hourly ACF", "C", "upper", "hourly_acf_rmse", (), (0.05,)),
    Check("T3 storm duration", "C", "ratio", "storm_dur_min", ("stats", "storm_dur_min"), (0.90, 1.10)),
    Check("T3 storm count", "C", "ratio", "storms_yr", ("stats", "storm_count_yr"), (0.90, 1.10)),
    Check("T3 storm volume", "C", "ratio", "storm_depth_mm", ("stats", "storm_vol_mm"), (0.90, 1.10)),
    Check("T4 daily max", "B", "ratio", "daily_max", ("stats", "daily_max"), (0.85, 1.15)),
    Check("T4 IDF (all cells)", "B", "cells", "idf", (), (0.80, 1.20)),
    Check("T5 monthly cycle", "E_monthly", "lower", "monthly_r", (), (0.90,)),
    Check("T5 diurnal cycle", "E_daily", "lower", "diurnal_r", (), (0.80,)),
    Check("S1 deep storms", "D", "ratio", "storms_ge_depth_yr", ("storm_scale", "storms_ge_depth_yr"), (0.85, 1.15)),
    Check("S2 rain in storms > 320 min", "D", "ratio", "rain_in_320min_pct", ("storm_scale", "rain_in_320min_pct"),
          (0.90, 1.10)),
    Check("S3 top storm depths", "D", "ratio", "top_storm_mm", ("storm_scale", "top_storm_mm"), (0.70, 1.45)),
    Check("D unbroken runs > 5h20", "D", "ratio", "long_storms_yr", ("long_storms_yr",)),
    Check("D variance across scales", "D", "upper", "var_scaling_rmse"),
    Check("D daily totals", "D", "upper", "w1_daily"),
    Check("E daily peak/trough", "E_daily", "ratio", "diurnal_peak_trough", ("diurnal_peak_trough",)),
]
GATE_A_18 = [c.name for c in CHECKS if c.name[:2] in ("T1", "T2", "T3", "T4", "T5")]
TIER_6 = [c.name for c in CHECKS if c.name[:2] in ("S1", "S2", "S3")]
FAMILIES = ["A", "B", "C", "D", "E_monthly", "E_daily"]
CORE_FAMILIES = ["A", "B", "C", "E_monthly"]       # required to transfer
PREDICTED_FAMILIES = ["D", "E_daily"]               # predicted in advance, not required


def check_value(check, m, R):
    if check.kind == "ratio":
        ref = _ref(R, check.ref)
        if check.metric == "dry_spell_min":
            return m["dry_spell_h"] * 60 / ref
        return m[check.metric] / ref if ref else np.nan
    if check.kind == "diff":
        return m[check.metric] - _ref(R, check.ref)
    if check.kind == "cells":
        return {k[4:]: v for k, v in m.items() if k.startswith("idf_") and "_T" in k}
    return m[check.metric]


def passes(check, value, band):
    if band is None:
        return None
    if check.kind == "cells":
        return all(band[k][0] <= v <= band[k][1] for k, v in value.items())
    if check.kind in ("ratio", "diff"):
        return bool(band[0] <= value <= band[1])
    if check.kind == "upper":
        return bool(value <= band[0])
    return bool(value >= band[0])


def fixed_band(check):
    if check.fixed is None:
        return None
    if check.kind == "cells":
        cells = [f"{d}_T{t}" for d in M.DURATIONS for t in M.RETURN_PERIODS]
        return {c: check.fixed for c in cells}
    return check.fixed


def score(m, R, bands):
    """{check name: {value, pass}} with the given bands ({name: band} or 'fixed')."""
    out = {}
    for c in CHECKS:
        v = check_value(c, m, R)
        b = fixed_band(c) if bands == "fixed" else bands.get(c.name)
        out[c.name] = {"value": v, "pass": passes(c, v, b)}
    return out


def families(scores):
    out = {}
    for f in FAMILIES:
        res = [s["pass"] for c in CHECKS if c.family == f for s in [scores[c.name]] if s["pass"] is not None]
        out[f] = {"passed": int(sum(res)), "of": len(res), "pass": (sum(res) >= len(res) / 2) if res else None}
    return out


# ── resampled bands ──────────────────────────────────────────────────────────────────────────────

_POOL = {}


def _init(years_x, years_m, depth):
    _POOL.update(years_x=years_x, years_m=years_m, P=M.Params(depth))


def _concat(pick):
    x = np.concatenate([_POOL["years_x"][i] for i in pick])
    m = np.concatenate([_POOL["years_m"][i] for i in pick])
    return x, (m if m.any() else None)


def _draw(args):
    seed, n_years, n_ref = args
    rng = np.random.default_rng(seed)
    k = len(_POOL["years_x"])
    sx, sm = _concat(rng.integers(0, k, n_years))           # the stand-in for a generated series
    rx, rm = _concat(rng.integers(0, k, n_ref or k))        # the stand-in for the reference
    R = M.reference(rx, _POOL["P"], mask=rm)
    m = M.hydro(sx, R, _POOL["P"], mask=sm)
    return {c.name: check_value(c, m, R) for c in CHECKS}


def resampled_bands(x, mask, params, n_draws=200, n_years=10, seed=0, workers=1, n_ref=None):
    """95% range of every check over n_draws two-sample resamples of whole reference years.

    n_ref: years in the reference stand-in; None (default) = as many as there are reference years."""
    years_x = [x[k * STEPS_PER_YEAR:(k + 1) * STEPS_PER_YEAR] for k in range(len(x) // STEPS_PER_YEAR)]
    years_m = [mask[k * STEPS_PER_YEAR:(k + 1) * STEPS_PER_YEAR] for k in range(len(x) // STEPS_PER_YEAR)]
    jobs = [(seed * 1_000_003 + i, n_years, n_ref) for i in range(n_draws)]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(years_x, years_m, params.storm_depth_mm)) as ex:
            draws = list(ex.map(_draw, jobs, chunksize=max(1, n_draws // (4 * workers))))
    else:
        _init(years_x, years_m, params.storm_depth_mm)
        draws = [_draw(j) for j in jobs]
    bands = {}
    for c in CHECKS:
        vals = [d[c.name] for d in draws]
        if c.kind == "cells":
            bands[c.name] = {k: (float(np.nanpercentile([v[k] for v in vals], 2.5)),
                                 float(np.nanpercentile([v[k] for v in vals], 97.5))) for k in vals[0]}
        elif c.kind in ("ratio", "diff"):
            bands[c.name] = (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
        elif c.kind == "upper":
            bands[c.name] = (float(np.nanpercentile(vals, 97.5)),)
        else:
            bands[c.name] = (float(np.nanpercentile(vals, 2.5)),)
    return bands


# ── verdicts (plan section 8.4) ──────────────────────────────────────────────────────────────────

def recipe_verdict(home_families, site_families, site_volume_ratio):
    """TRANSFERS / PARTLY / DOES NOT TRANSFER, comparing the core families of v14 at both sites."""
    expected = [f for f in CORE_FAMILIES if home_families[f]["pass"]]
    lost = [f for f in expected if not site_families[f]["pass"]]
    if abs(site_volume_ratio - 1) > 0.20:
        return "DOES NOT TRANSFER", f"generated volume {site_volume_ratio:.2f} of real (training failed)", lost
    if not lost:
        return "TRANSFERS", f"every core family that passes at home also passes here ({', '.join(expected)})", lost
    if len(lost) == 1:
        return "PARTLY", f"family {lost[0]} passes at home but not here", lost
    return "DOES NOT TRANSFER", f"families {', '.join(lost)} pass at home but not here", lost
