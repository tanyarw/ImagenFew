"""Stage verdicts (plan sections 3-5; section 10 for what was settled while coding).

    stage 1  the coarse model alone, at 30 minutes: every series is replaced by its 30-minute view (grid.view30)
    stage 2  the fine model and the cascade given the real 2008-2009 half-hour totals, against real 2008-2009
    stage 3  the full chain at 5 minutes, by v17's rule and Gate A; the sewer check runs on the laptop

Numeric rules use the mean over the 4 members (seeds 42-45), as v17's rule did; a family must pass in at least
3 of the 4 members. Each stage writes results/tworesolution/stage<k>.json and prints its table and verdict.
"""
import json
import os
from pathlib import Path

import numpy as np

from sitetransfer import checks as C
from sitetransfer import metrics as M
from sitetransfer.pilot import perfect_window
from sitetransfer.provenance import ROOT, rel, write_json
from sitetransfer.site import load_generated

from . import data as D
from .grid import FACTOR, KEEP, L, aggregate, clean, spread, two_level, view30

OUT = Path(os.environ.get("TWORES_OUT", D.OUT))           # outputs; override only for smoke tests
GENERATED = ROOT / "results" / "generated_data" if "TWORES_OUT" not in os.environ else OUT / "generated_data"
V14 = ROOT / "results" / "generated_data" / "rainfall_synthetic_10y_v14_nobridge_cal.csv"
ASTLINGEN_BANDS = ROOT / "results" / "transfer" / "astlingen" / "bands.json"
SERIES = {"fine": "twores_v1", "cascade": "twores_cascade"}      # the 10-year 5-minute series, by method
RULES = {
    1: {"family_D_members_min": 3, "rain_in_320min_pct_min": 60.0, "volume_ratio": [0.95, 1.05],
        "too_close_pct_max": 10.0},
    2: {"families": ["B", "C"], "family_members_min": 3, "join_gap_max": 0.05},
    3: {"long_runs_yr_min": 10.0, "long_run_rain_pct_min": 15.0, "volume_ratio": [0.95, 1.05],
        "wet_spell_ratio": [0.90, 1.10], "gate_a_min": 12, "sewer_ratio": [0.85, 1.15]},
}


def params():
    return M.Params(float(D.site().params["storm_depth_mm"]))


def member_name(base, k):
    return base if k == 0 else f"{base}_m{k}"


def tag(coarse_run):
    """'' for the first coarse model, '_long' for twores_coarse30_long: every output of a coarse run carries it."""
    return "" if coarse_run == D.RUN_COARSE else "_" + coarse_run.removeprefix(D.RUN_COARSE).lstrip("_")


def series_name(method, coarse_run=D.RUN_COARSE):
    return SERIES[method] + tag(coarse_run)


def series_csv(base, k):
    return GENERATED / f"rainfall_synthetic_10y_{member_name(base, k)}.csv"


def disaggregated_path(source, method, coarse_run=D.RUN_COARSE):
    return OUT / "disaggregated" / f"{source}{tag(coarse_run) if source == 'coarse' else ''}_{method}.npz"


def coarse_path(coarse_run=D.RUN_COARSE):
    return OUT / "coarse" / f"coarse30{tag(coarse_run)}.npz"


def stage_path(stage, coarse_run=D.RUN_COARSE):
    return OUT / f"stage{stage}{tag(coarse_run) if stage != 2 else ''}.json"


def scored(a, R, P, bands):
    m = M.hydro(a, R, P)
    s = C.score(m, R, bands)
    return m, s, C.families(s)


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def _family_members(rows, family):
    return int(sum(bool(r["families"][family]["pass"]) for r in rows))


# ── memorisation (eval_suite.WindowRef's test, at 30 minutes) ───────────────────────────────────────

def wet_windows(x, length=L, agg=1, min_total=0.0):
    """Non-overlapping windows with rain (at least `min_total` mm in all), summed over `agg` steps, as
    log1p(mm / 0.1) (eval_suite.wet_windows)."""
    n = len(x) // length
    W = np.asarray(x[:n * length]).reshape(n, length)
    W = W[(W.max(1) > 0) & (W.sum(1) >= min_total)]
    if agg > 1:
        W = W.reshape(len(W), length // agg, agg).sum(2)
    return np.log1p(W / 0.1)


class Memorisation:
    """Is a generated window closer to a training window than genuinely new real windows get?

    too_close_pct: share of generated windows nearer to their nearest 2000-2007 window than the closest 5% of
    the 2008-2009 windows are; real held-out windows give 5% by construction, a copying generator far more.
    Default: 32-hour windows of half-hour totals (stage 1); oneday/ uses one-day windows of 15-minute totals with
    at least 1 mm (near-dry days are often identical to a training day, which would set the threshold to 0)."""

    def __init__(self, train, held, length=L, agg=1, min_total=0.0):
        from sklearn.neighbors import NearestNeighbors
        self.length, self.agg, self.min_total = length, agg, min_total
        self.nn = NearestNeighbors(n_neighbors=1).fit(wet_windows(train, length, agg, min_total))
        self.held = self.nn.kneighbors(wet_windows(held, length, agg, min_total))[0][:, 0]
        self.threshold = float(np.percentile(self.held, 5))

    def score(self, x):
        d = self.nn.kneighbors(wet_windows(x, self.length, self.agg, self.min_total))[0][:, 0]
        return {"too_close_pct": float(100 * (d < self.threshold).mean()),
                "nn_dist_ratio": float(np.median(d) / np.median(self.held))}


# ── joins between fine windows ──────────────────────────────────────────────────────────────────────

def join_continuation(x5):
    """P(wet | previous step wet) across fine-window joins (every 60 steps) and across the other half-hour
    boundaries. Fine windows do not see each other, so a model that starts or ends windows dry breaks storms at
    the joins only; in real rain the two numbers differ by chance alone."""
    wet = np.asarray(x5) > 0
    t = np.arange(FACTOR, len(wet), FACTOR)
    join = t % KEEP == 0

    def p(ts):
        prev = wet[ts - 1]
        return float(wet[ts][prev].mean()) if prev.any() else float("nan")

    return {"at_joins": p(t[join]), "at_other_half_hours": p(t[~join])}


# ── stage 1 ─────────────────────────────────────────────────────────────────────────────────────────

def stage1(draws=1000, workers=1, coarse_run=D.RUN_COARSE):
    d = D.load_derived()
    t5, t30 = d["tau5_mm"], d["tau30_mm"]
    P = params()
    train5, held5 = D.train_record(), D.heldout_record()
    ref = view30(train5, t5, t30)
    R = M.reference(ref, P)
    bands = C.resampled_bands(ref, np.zeros(len(ref), bool), P, n_draws=draws, n_years=10, seed=0, workers=workers)
    mem = Memorisation(two_level(train5, t5, t30)[1], two_level(held5, t5, t30)[1])
    coarse = np.load(coarse_path(coarse_run))["x30"]
    series = {f"coarse m{k}": spread(coarse[k]) for k in range(len(coarse))}
    series["real 2008-09 (held out)"] = view30(held5, t5, t30)
    series["perfect 32-h windows"] = perfect_window(ref, None, L * FACTOR, 10, seed=0)[0]
    if V14.exists():
        series["v14"] = view30(load_generated(V14, t5), t5, t30)
    rows = {}
    for name, a in series.items():
        m, s, f = scored(a, R, P, bands)
        rows[name] = {"volume_ratio": m["volume_mm_yr"] / R["stats"]["volume"],
                      "rain_in_320min_pct": m["rain_in_320min_pct"], "long_storms_yr": m["long_storms_yr"],
                      "storms_ge_depth_yr": m["storms_ge_depth_yr"], "families": f, "checks": s,
                      **mem.score(aggregate(a))}
    members = [rows[f"coarse m{k}"] for k in range(len(coarse))]
    r = RULES[1]
    tests = {
        "family D passes in >= 3 of 4 members": _family_members(members, "D") >= r["family_D_members_min"],
        "rain in storms > 5h20 >= 60% (mean)": _mean(members, "rain_in_320min_pct") >= r["rain_in_320min_pct_min"],
        "volume within 5% (mean)": r["volume_ratio"][0] <= _mean(members, "volume_ratio") <= r["volume_ratio"][1],
        "too-close windows <= 10% (mean)": _mean(members, "too_close_pct") <= r["too_close_pct_max"],
    }
    verdict = "PASS: continue to stage 3" if all(tests.values()) else "STOP: the coarse model fails stage 1"
    means = {k: _mean(members, k) for k in ("rain_in_320min_pct", "volume_ratio", "too_close_pct")}
    out = {"stage": 1, "coarse_run": coarse_run, "rules": r, "tests": tests, "verdict": verdict,
           "member_means": means | {"family_D_members_passing": _family_members(members, "D")},
           "real_reference": {"rain_in_320min_pct": R["storm_scale"]["rain_in_320min_pct"],
                              "long_storms_yr": R["long_storms_yr"], "volume_mm_yr": R["stats"]["volume"]},
           "bands": {"n_draws": draws, "n_years": 10, "reference": "2000-2007, 30-minute view"},
           "memorisation_threshold": mem.threshold, "series": rows}
    write_json(stage_path(1, coarse_run), out)
    print(f"{'':26s}{'volume':>8s}{'rain>5h20':>11s}{'runs>5h20/yr':>14s}{'family D':>10s}{'too close':>11s}")
    print(f"{'real 2000-07 (reference)':26s}{1:8.2f}{R['storm_scale']['rain_in_320min_pct']:10.1f}%"
          f"{R['long_storms_yr']:14.1f}")
    for name, v in rows.items():
        fd = v["families"]["D"]
        print(f"{name:26s}{v['volume_ratio']:8.2f}{v['rain_in_320min_pct']:10.1f}%{v['long_storms_yr']:14.1f}"
              f"{fd['passed']:>6d}/{fd['of']}{v['too_close_pct']:10.1f}%")
    for k, ok in tests.items():
        print(f"  {'PASS' if ok else 'fail'}  {k}")
    print(verdict)
    return out


# ── stage 2 ─────────────────────────────────────────────────────────────────────────────────────────

def stage2(draws=1000, workers=1):
    d = D.load_derived()
    t5, t30 = d["tau5_mm"], d["tau30_mm"]
    P = params()
    real = clean(D.heldout_record(), t5)
    R = M.reference(real, P)
    train5 = clean(D.train_record(), t5)
    bands = C.resampled_bands(train5, np.zeros(len(train5), bool), P, n_draws=draws, n_years=2, n_ref=2, seed=0,
                              workers=workers)
    series = {"real 2008-09, half-hours < tau30 dry": two_level(D.heldout_record(), t5, t30)[0]}
    for method in ("fine", "cascade"):
        x5 = np.load(disaggregated_path("heldout", method))["x5"]
        series |= {f"{method} m{k}": x5[k] for k in range(len(x5))}
    rows = {}
    for name, a in series.items():
        m, s, f = scored(a, R, P, bands)
        rows[name] = {"families": f, "checks": s, "join": join_continuation(a),
                      "p999_ratio": m["p999_wet"] / R["stats"]["p999_wet"], "acf1": m["acf1"],
                      "wet_spell_ratio": m["wet_spell_min"] / R["stats"]["wet_spell_min"],
                      "single_step_showers_pct": m["single_step_showers_pct"]}
    r = RULES[2]
    result = {}
    for method in ("fine", "cascade"):
        mem = [v for k, v in rows.items() if k.startswith(method)]
        t = {f"family {f} passes in >= 3 of 4 members": _family_members(mem, f) >= r["family_members_min"]
             for f in r["families"]}
        gap = float(np.mean([v["join"]["at_other_half_hours"] - v["join"]["at_joins"] for v in mem]))
        if method == "fine":
            t["rain continues across joins within 0.05 of other half-hours (mean)"] = gap <= r["join_gap_max"]
        result[method] = {"tests": t, "pass": all(t.values()), "join_gap": gap}
    if result["fine"]["pass"]:
        use, verdict = ["fine", "cascade"], "PASS: the fine model disaggregates; stage 3 uses it (cascade compared)"
    elif result["cascade"]["pass"]:
        use, verdict = ["cascade"], "PARTLY: only the cascade disaggregates; stage 3 uses the cascade"
    else:
        use, verdict = [], "STOP: neither the fine model nor the cascade disaggregates"
    out = {"stage": 2, "rules": r, "result": result, "verdict": verdict, "stage3_uses": use,
           "real": {"join": join_continuation(real), "p999_wet": R["stats"]["p999_wet"],
                    "acf1": R["stats"]["acf1_native"], "wet_spell_min": R["stats"]["wet_spell_min"]},
           "bands": {"n_draws": draws, "n_years": 2, "n_ref": 2, "reference": "2000-2007 years, two 2-year samples"},
           "series": rows}
    write_json(stage_path(2), out)
    rj = out["real"]["join"]
    print(f"real 2008-09: P99.9 {R['stats']['p999_wet']:.3f} mm, lag-1 ACF {R['stats']['acf1_native']:.3f}, "
          f"continuation at window joins {rj['at_joins']:.3f} / other half-hours {rj['at_other_half_hours']:.3f}")
    print(f"{'':40s}{'B':>6s}{'C':>6s}{'P99.9':>8s}{'wet spell':>11s}{'joins':>8s}{'others':>8s}")
    for name, v in rows.items():
        fb, fc = v["families"]["B"], v["families"]["C"]
        print(f"{name:40s}{fb['passed']:>3d}/{fb['of']}{fc['passed']:>3d}/{fc['of']}{v['p999_ratio']:8.2f}"
              f"{v['wet_spell_ratio']:11.2f}{v['join']['at_joins']:8.3f}{v['join']['at_other_half_hours']:8.3f}")
    for method, res in result.items():
        for k, ok in res["tests"].items():
            print(f"  {method:8s}{'PASS' if ok else 'fail'}  {k}")
    print(verdict)
    return out


# ── stage 3 ─────────────────────────────────────────────────────────────────────────────────────────

def stage3_methods(coarse_run=D.RUN_COARSE):
    """The stop rules of stages 1 and 2: the disaggregators stage 3 may use, or SystemExit."""
    for k in (1, 2):
        if not stage_path(k, coarse_run).exists():
            raise SystemExit(f"{rel(stage_path(k, coarse_run))} missing: run stage {k} first")
    s1 = json.loads(stage_path(1, coarse_run).read_text())
    s2 = json.loads(stage_path(2).read_text())
    if not s1["verdict"].startswith("PASS"):
        raise SystemExit(f"stage 1 did not pass ({s1['verdict']}): the plan stops here")
    if not s2["stage3_uses"]:
        raise SystemExit(f"stage 2 did not pass ({s2['verdict']}): the plan stops here")
    return s2["stage3_uses"]


def stage3(coarse_run=D.RUN_COARSE):
    uses = stage3_methods(coarse_run)
    t5 = D.load_derived()["tau5_mm"]
    P = params()
    ref = clean(D.train_record(), t5)
    R = M.reference(ref, P)
    b = json.loads(ASTLINGEN_BANDS.read_text())
    if b["reference_years"] != list(range(2000, 2008)) or b["n_years"] != 10:
        raise ValueError(f"{rel(ASTLINGEN_BANDS)} was not made for 10 years against 2000-2007")
    series = {}
    for method in uses:
        x5 = np.load(disaggregated_path("coarse", method, coarse_run))["x5"]
        series |= {member_name(series_name(method, coarse_run), k): x5[k] for k in range(len(x5))}
    series["perfect 32-h windows"] = perfect_window(ref, None, L * FACTOR, 10, seed=0)[0]
    series["perfect 5h20 windows"] = perfect_window(ref, None, L, 10, seed=0)[0]
    if V14.exists():
        series["v14"] = load_generated(V14, t5)
    rows = {}
    for name, a in series.items():
        m, s, f = scored(a, R, P, b["bands"])
        fixed = C.score(m, R, "fixed")
        rows[name] = {"long_runs_yr": m["long_storms_yr"], "long_run_rain_pct": m["long_storm_rain_pct"],
                      "volume_ratio": m["volume_mm_yr"] / R["stats"]["volume"],
                      "wet_spell_ratio": m["wet_spell_min"] / R["stats"]["wet_spell_min"],
                      "gate_a": int(sum(bool(fixed[n]["pass"]) for n in C.GATE_A_18)),
                      "rain_in_320min_pct": m["rain_in_320min_pct"], "families": f, "checks": s}
    r = RULES[3]
    result = {}
    for method in uses:
        mem = [v for k, v in rows.items() if k.startswith(series_name(method, coarse_run))]
        mean = {k: _mean(mem, k) for k in ("long_runs_yr", "long_run_rain_pct", "volume_ratio", "wet_spell_ratio",
                                            "gate_a")}
        t = {"unbroken runs > 5h20 >= 10 / yr": mean["long_runs_yr"] >= r["long_runs_yr_min"],
             "share of rain in them >= 15%": mean["long_run_rain_pct"] >= r["long_run_rain_pct_min"],
             "volume 0.95-1.05": r["volume_ratio"][0] <= mean["volume_ratio"] <= r["volume_ratio"][1],
             "wet spell 0.90-1.10": r["wet_spell_ratio"][0] <= mean["wet_spell_ratio"] <= r["wet_spell_ratio"][1],
             "Gate A >= 12 / 18": mean["gate_a"] >= r["gate_a_min"]}
        result[method] = {"mean": mean, "tests": t, "statistics_pass": all(t.values())}
    verdict = "; ".join(f"{m}: statistics {'PASS' if res['statistics_pass'] else 'FAIL'}" for m, res in result.items())
    out = {"stage": 3, "coarse_run": coarse_run, "rules": r, "result": result, "uses": uses,
           "verdict": verdict + "; sewer pending",
           "real_reference": {"long_runs_yr": R["long_storms_yr"],
                              "rain_in_320min_pct": R["storm_scale"]["rain_in_320min_pct"]},
           "sewer": "pending: run scripts/gate_b_sewer.py on the laptop (README.md)", "series": rows}
    write_json(stage_path(3, coarse_run), out)
    print(f"{'':24s}{'runs>5h20':>10s}{'their rain':>11s}{'volume':>8s}{'wet spell':>10s}{'Gate A':>8s}"
          f"{'rain>5h20 (2h gap)':>20s}")
    print(f"{'real 2000-07':24s}{R['long_storms_yr']:10.1f}{'':11s}{1:8.2f}{1:10.2f}{'':8s}"
          f"{R['storm_scale']['rain_in_320min_pct']:19.1f}%")
    for name, v in rows.items():
        print(f"{name:24s}{v['long_runs_yr']:10.1f}{v['long_run_rain_pct']:10.1f}%{v['volume_ratio']:8.2f}"
              f"{v['wet_spell_ratio']:10.2f}{v['gate_a']:>6d}/18{v['rain_in_320min_pct']:19.1f}%")
    for method, res in result.items():
        for k, ok in res["tests"].items():
            print(f"  {method:8s}{'PASS' if ok else 'fail'}  {k}")
    print(out["verdict"])
    names = " ".join(k for k in rows if k.startswith(tuple(series_name(m, coarse_run) for m in uses)))
    print(f"sewer (laptop, from flood-control): PYTHONPATH=. .venv/bin/python ../ImagenFew/scripts/gate_b_sewer.py "
          f"{names} --controllers BC EFD")
    return out
