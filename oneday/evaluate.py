"""v19 verdict (code_plan/ONE_DAY_WINDOW_PLAN.md, section 4): the rule v17 and the two-resolution stage 3 were held
to, plus memorisation at the one-day scale, member means over 4 generation seeds. The sewer check runs on the
laptop (scripts/gate_b_sewer.py) and is added to the diary by hand.

Comparators: v14, the perfect one-day and 5h20 window generators (sitetransfer.pilot, seed 0), real 2008-2009.
Bands: the committed Astlingen resampled bands (10 years against 2000-2007), as everywhere since the transfer test.
"""
import json

import numpy as np

from sitetransfer import checks as C
from sitetransfer import metrics as M
from sitetransfer.pilot import perfect_window
from sitetransfer.provenance import ROOT, rel, write_json
from sitetransfer.site import load_generated
from tworesolution import data as TD
from tworesolution.evaluate import ASTLINGEN_BANDS, V14, Memorisation, member_name, params

from .config import RUN_ID

MEMBERS = 4
SERIES = f"{RUN_ID}_nobridge_cal"
OUT = ROOT / "results" / "oneday" / "evaluation.json"
RULES = {"long_runs_yr_min": 10.0, "long_run_rain_pct_min": 15.0, "volume_ratio": [0.95, 1.05],
         "wet_spell_ratio": [0.90, 1.10], "gate_a_min": 12, "too_close_pct_max": 10.0, "sewer_ratio": [0.85, 1.15]}


def series_csv(k):
    return ROOT / "results" / "generated_data" / f"rainfall_synthetic_10y_{member_name(SERIES, k)}.csv"


def run(members=MEMBERS):
    tau = TD.tau5()
    P = params()
    ref = TD.clean(TD.train_record(), tau)
    held = TD.clean(TD.heldout_record(), tau)
    R = M.reference(ref, P)
    b = json.loads(ASTLINGEN_BANDS.read_text())
    if b["reference_years"] != list(range(2000, 2008)) or b["n_years"] != 10:
        raise ValueError(f"{rel(ASTLINGEN_BANDS)} was not made for 10 years against 2000-2007")
    mem = Memorisation(ref, held, length=288, agg=3, min_total=1.0)    # one-day windows, 15-minute totals, >= 1 mm
    series = {member_name(SERIES, k): load_generated(series_csv(k), tau) for k in range(members)}
    series["perfect one-day windows"] = perfect_window(ref, None, 288, 10, seed=0)[0]
    series["perfect 5h20 windows"] = perfect_window(ref, None, 64, 10, seed=0)[0]
    series["real 2008-09 (held out)"] = held
    if V14.exists():
        series["v14"] = load_generated(V14, tau)
    rows = {}
    for name, a in series.items():
        m = M.hydro(a, R, P)
        fixed = C.score(m, R, "fixed")
        res = C.score(m, R, b["bands"])
        rows[name] = {"long_runs_yr": m["long_storms_yr"], "long_run_rain_pct": m["long_storm_rain_pct"],
                      "volume_ratio": m["volume_mm_yr"] / R["stats"]["volume"],
                      "wet_spell_ratio": m["wet_spell_min"] / R["stats"]["wet_spell_min"],
                      "gate_a": int(sum(bool(fixed[n]["pass"]) for n in C.GATE_A_18)),
                      "tier6": int(sum(bool(fixed[n]["pass"]) for n in C.TIER_6)),
                      "rain_in_320min_pct": m["rain_in_320min_pct"], "storms_ge_depth_yr": m["storms_ge_depth_yr"],
                      "p999_ratio": m["p999_wet"] / R["stats"]["p999_wet"],
                      "families": C.families(res), "checks": res, **mem.score(a)}
    mrows = [rows[member_name(SERIES, k)] for k in range(members)]
    mean = {k: float(np.mean([r[k] for r in mrows])) for k in ("long_runs_yr", "long_run_rain_pct", "volume_ratio",
                                                                "wet_spell_ratio", "gate_a", "too_close_pct", "tier6")}
    r = RULES
    tests = {"unbroken runs > 5h20 >= 10 / yr": mean["long_runs_yr"] >= r["long_runs_yr_min"],
             "share of rain in them >= 15%": mean["long_run_rain_pct"] >= r["long_run_rain_pct_min"],
             "volume 0.95-1.05": r["volume_ratio"][0] <= mean["volume_ratio"] <= r["volume_ratio"][1],
             "wet spell 0.90-1.10": r["wet_spell_ratio"][0] <= mean["wet_spell_ratio"] <= r["wet_spell_ratio"][1],
             "Gate A >= 12 / 18": mean["gate_a"] >= r["gate_a_min"],
             "too-close one-day windows <= 10%": mean["too_close_pct"] <= r["too_close_pct_max"]}
    verdict = ("statistics PASS" if all(tests.values()) else "statistics FAIL") + "; sewer pending"
    out = {"run": RUN_ID, "rules": r, "member_means": mean, "tests": tests, "verdict": verdict,
           "real_reference": {"long_runs_yr": R["long_storms_yr"], "rain_in_320min_pct":
                              R["storm_scale"]["rain_in_320min_pct"], "volume_mm_yr": R["stats"]["volume"]},
           "memorisation_threshold": mem.threshold, "series": rows}
    write_json(OUT, out)
    print(f"{'':28s}{'runs>5h20':>10s}{'their rain':>11s}{'volume':>8s}{'wet spell':>10s}{'Gate A':>8s}"
          f"{'Tier 6':>7s}{'P99.9':>7s}{'too close':>10s}")
    print(f"{'real 2000-07':28s}{R['long_storms_yr']:10.1f}{'':11s}{1:8.2f}{1:10.2f}")
    for name, v in rows.items():
        print(f"{name:28s}{v['long_runs_yr']:10.1f}{v['long_run_rain_pct']:10.1f}%{v['volume_ratio']:8.2f}"
              f"{v['wet_spell_ratio']:10.2f}{v['gate_a']:>6d}/18{v['tier6']:>5d}/3{v['p999_ratio']:7.2f}"
              f"{v['too_close_pct']:9.1f}%")
    for k, ok in tests.items():
        print(f"  {'PASS' if ok else 'fail'}  {k}")
    print(verdict)
    names = " ".join(member_name(SERIES, k) for k in range(members))
    print(f"sewer (laptop, from flood-control): PYTHONPATH=. .venv/bin/python ../ImagenFew/scripts/gate_b_sewer.py "
          f"{names} --controllers BC EFD")
    return out
