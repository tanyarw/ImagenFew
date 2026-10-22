"""Per-year quantities, percentiles and the verdict (code_plan/TEST_YEAR_PLAN.md sections 3-4). Pure functions."""
import numpy as np

from sitetransfer.metrics import daily, hourly, spells, storm_scale_stats
from sitetransfer.timegrid import STEPS_PER_YEAR

from . import config as K


def years(a):
    """A series of whole 365-day years as rows (N, 105120)."""
    a = np.asarray(a, dtype=np.float64)
    if len(a) % STEPS_PER_YEAR:
        raise ValueError(f"{len(a)} steps is not a whole number of 365-day years")
    return a.reshape(-1, STEPS_PER_YEAR)


def rain_quantities(x, storm_depth_mm):
    """The ten rain quantities of one year (x: 105,120 cleaned 5-min values, mm)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) != STEPS_PER_YEAR:
        raise ValueError(f"one year is {STEPS_PER_YEAR} steps, got {len(x)}")
    wet = x[x > 0]
    if not len(wet):
        raise ValueError("a year without rain")
    ws, _ = spells(x)
    s = storm_scale_stats(x, storm_depth_mm)          # one year: per-year rates are counts
    return {"annual_mm": float(x.sum()), "wet_pct": 100 * float((x > 0).mean()),
            "wet_spell_min": float(ws.mean() * 5), "p999_wet": float(np.percentile(wet, 99.9)),
            "max_1h_mm": float(hourly(x).max()), "max_day_mm": float(daily(x).max()),
            "storms_ge_10mm": int(round(s["storms_ge_depth_yr"])), "long_storms": int(round(s["storms_320min_yr"])),
            "rain_in_long_pct": float(s["rain_in_320min_pct"]), "max_storm_mm": float(s["max_storm_mm"])}


def sewer_quantities(rows):
    """{(controller, field): value} of one year from its sewer rows (one per controller), scaled to 365 days as
    gate_b_sewer.summary does, so leap years (29 February simulated dry) compare with the others."""
    out = {}
    for r in rows:
        k = 365 / r["days"]
        for f in K.SEWER_FIELDS:
            out[(r["controller"], f)] = float(r[f]) * k
    return out


def sewer_key(controller, field):
    return f"{controller} {field}"


def percentile(value, sample):
    """Share of `sample` below `value`, ties counted half: the real year's place among the generated years."""
    s = np.asarray(sample, dtype=np.float64)
    if not len(s) or np.isnan(s).any() or np.isnan(value):
        raise ValueError("percentile needs a non-empty sample and values without NaN")
    return float(((s < value).sum() + 0.5 * (s == value).sum()) / len(s))


def inside(p):
    return K.BAND[0] <= p <= K.BAND[1]


def place(real, generated):
    """{quantity: {value, percentile, inside, generated 5/50/95%}} of one real year against the generated years.
    real: {quantity: value}; generated: {quantity: [one value per generated year]}."""
    out = {}
    for q, v in real.items():
        g = np.asarray(generated[q], dtype=np.float64)
        p = percentile(v, g)
        out[q] = {"value": float(v), "percentile": p, "inside": inside(p),
                  "generated_p5_p50_p95": [float(x) for x in np.percentile(g, [5, 50, 95])]}
    return out


def verdict(rain_place, sewer_place, too_close_pct):
    """The rule of TEST_YEAR_PLAN.md section 4."""
    primary = sewer_key(*K.PRIMARY)
    rain_out = sorted(q for q, r in rain_place.items() if not r["inside"])
    tests = {f"sewer: {primary} inside {K.BAND}": bool(sewer_place[primary]["inside"]),
             f"rain: at most {K.MAX_RAIN_OUTSIDE} of {len(rain_place)} outside": len(rain_out) <= K.MAX_RAIN_OUTSIDE,
             f"copying: too-close <= {K.TOO_CLOSE_MAX_PCT}%": too_close_pct <= K.TOO_CLOSE_MAX_PCT}
    return {"tests": tests, "rain_outside": rain_out, "pass": all(tests.values()),
            "verdict": "PASS" if all(tests.values()) else "FAIL"}


def coverage(places):
    """Share of (year, quantity) pairs inside, over several real years' `place` results."""
    flags = [r["inside"] for p in places for r in p.values()]
    return float(np.mean(flags)) if flags else float("nan")
