"""register, rehearse and test (code_plan/TEST_YEAR_PLAN.md section 5). The rehearsal and the test run the same
function, `evaluate_year`, on 2008 and 2009; only the guard differs."""
import datetime as dt
import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

from sitetransfer.provenance import git_state, rel, sha256_file, write_json
from sitetransfer.site import load_generated
from tworesolution import data as TD
from tworesolution.evaluate import Memorisation, params

from . import config as K
from . import guard
from . import score as S

RAIN_SEWER_TOLERANCE = 0.005    # a stored sewer row must have the series' own rain (else it is from an older file)


def rule():
    """The rule as stored in the preregistration (JSON-normalised, so a stored copy compares equal)."""
    return json.loads(json.dumps({
        "band": K.BAND, "primary": S.sewer_key(*K.PRIMARY), "max_rain_outside": K.MAX_RAIN_OUTSIDE,
        "too_close_max_pct": K.TOO_CLOSE_MAX_PCT, "rain_quantities": list(K.RAIN_QUANTITIES),
        "sewer_quantities": [S.sewer_key(*q) for q in K.SEWER_QUANTITIES], "memorisation": K.MEMORISATION}))


def code_sha256():
    return {p.name: sha256_file(p) for p in sorted(K.PACKAGE.glob("*.py"))}


def flood_control_git(flood_control):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=flood_control, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=flood_control,
                           capture_output=True, text=True)
    return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


# ── inputs ──────────────────────────────────────────────────────────────────────────────────────────

def real_year(year, key=None):
    """One real year, cleaned like every generated series. The test year needs the guard's Key."""
    guard.require(year, key)
    if year not in K.SPLIT_FILE:
        raise ValueError(f"{year} is not a held-out year of this test")
    x, _ = TD.load_split(K.SPLIT_FILE[year])
    return TD.clean(x, TD.tau5())


def train_years():
    return S.years(TD.clean(TD.train_record(), TD.tau5()))


def flood_control_real_rows(flood_control=K.FLOOD_CONTROL):
    """flood-control's saved real rows (2000-2008), as {year: [row per controller]}."""
    df = pd.read_csv(flood_control / K.FC_REAL_ROWS)
    out = defaultdict(list)
    for r in df[df.source == "real"].to_dict("records"):
        out[int(r["year"])].append(r)
    return out


def generated_sewer_rows(names):
    """{name: {year: [row per controller]}} from gate_b_sewer.json; every name must have both controllers in every
    year it has."""
    rows = json.loads(K.SEWER_GENERATED.read_text())["rows"]
    out = {n: defaultdict(list) for n in names}
    for r in rows:
        if r["source"] in out and r["controller"] in K.CONTROLLERS:
            out[r["source"]][int(r["year"])].append(r)
    for n, by_year in out.items():
        if not by_year:
            raise SystemExit(f"{n}: no sewer rows in {rel(K.SEWER_GENERATED)}; run scripts/gate_b_sewer.py "
                             f"{n} --controllers BC EFD from flood-control first")
        for y, rs in by_year.items():
            if sorted(r["controller"] for r in rs) != sorted(K.CONTROLLERS):
                raise SystemExit(f"{n} {y}: sewer rows for {sorted(r['controller'] for r in rs)}, need "
                                 f"{list(K.CONTROLLERS)}")
    return out


# ── register ────────────────────────────────────────────────────────────────────────────────────────

def register(series=K.SERIES, members=K.MEMBERS, flood_control=K.FLOOD_CONTROL):
    """Fix everything about the generator before any held-out year is read. Refuses to overwrite."""
    if K.PREREG.exists():
        raise SystemExit(f"{rel(K.PREREG)} exists; a preregistration is not replaced")
    tau, depth = TD.tau5(), params().storm_depth_mm
    names = K.member_names(series, members)
    missing = [K.series_csv(n).name for n in names if not K.series_csv(n).exists()]
    if missing:
        raise SystemExit(f"missing series: {missing}")
    sewer = generated_sewer_rows(names)
    per_year, generated = [], defaultdict(list)
    for n in names:
        ys = S.years(load_generated(K.series_csv(n), tau))
        cal = sorted(sewer[n])
        if len(cal) != len(ys) or cal != list(range(cal[0], cal[0] + len(ys))):
            raise SystemExit(f"{n}: {len(ys)} years of rain but sewer years {cal}")
        for x, y in zip(ys, cal):
            rain = S.rain_quantities(x, depth)
            sw = S.sewer_quantities(sewer[n][y])
            for r in sewer[n][y]:
                if abs(r["rain_mm"] - rain["annual_mm"]) > RAIN_SEWER_TOLERANCE * rain["annual_mm"]:
                    raise SystemExit(f"{n} {y}: sewer row has {r['rain_mm']:.1f} mm, the series {rain['annual_mm']:.1f}"
                                     f" mm; rerun the sewer check on the current file")
            row = {**rain, **{S.sewer_key(*q): sw[q] for q in K.SEWER_QUANTITIES}}
            per_year.append({"member": n, "year": y, **row})
            for q, v in row.items():
                generated[q].append(v)
    fc_rows = flood_control_real_rows(flood_control)
    calibration = {}
    for y, x in zip(K.TRAIN_YEARS, train_years()):
        sw = S.sewer_quantities(fc_rows[y])
        calibration[y] = {"rain": S.place(S.rain_quantities(x, depth), generated),
                          "sewer": S.place({S.sewer_key(*q): sw[q] for q in K.SEWER_QUANTITIES}, generated)}
    prereg = {
        "written": dt.datetime.now().isoformat(timespec="seconds"), "git": git_state(),
        "plan": "code_plan/TEST_YEAR_PLAN.md", "series": series, "members": names,
        "series_sha256": {n: sha256_file(K.series_csv(n)) for n in names},
        "inputs_sha256": {rel(K.SEWER_GENERATED): sha256_file(K.SEWER_GENERATED),
                          f"flood-control/{K.FC_REAL_ROWS}": sha256_file(flood_control / K.FC_REAL_ROWS)},
        "rule": rule(), "code_sha256": code_sha256(), "flood_control_git": flood_control_git(flood_control),
        "storm_depth_mm": depth, "tau_mm": tau,
        "generated_years": per_year,
        "calibration_train_years": {
            "coverage_rain": S.coverage([c["rain"] for c in calibration.values()]),
            "coverage_sewer": S.coverage([c["sewer"] for c in calibration.values()]),
            "years": calibration},
    }
    return write_json(K.PREREG, prereg)


def generated_values(prereg):
    out = defaultdict(list)
    for r in prereg["generated_years"]:
        for q, v in r.items():
            if q not in ("member", "year"):
                out[q].append(v)
    return out


# ── one held-out year ───────────────────────────────────────────────────────────────────────────────

def sewer_real(year, prereg_sha, flood_control=K.FLOOD_CONTROL):
    """The real year under BC and EFD, run by flood-control's environment (testyear/sewer_real.py)."""
    out = K.sewer_real_path(year)
    py = flood_control / ".venv" / "bin" / "python"
    env = dict(os.environ, PYTHONPATH=f"{flood_control}{os.pathsep}{K.ROOT}")
    subprocess.run([str(py), "-m", "testyear.sewer_real", "--year", str(year), "--prereg-sha256", prereg_sha,
                    "--out", str(out)], cwd=flood_control, env=env, check=True)
    r = json.loads(out.read_text())
    if r["year"] != year or r["prereg_sha256"] != prereg_sha:
        raise RuntimeError(f"{out.name} is not this run's")
    return r["rows"]


def evaluate_year(year, key=None, flood_control=K.FLOOD_CONTROL):
    prereg, sha = guard.load_prereg()
    if key is not None and key.prereg_sha256 != sha:
        raise guard.Locked("the preregistration changed after unlocking")
    problems = guard.series_unchanged(prereg)
    if prereg["rule"] != rule():
        problems.append("the rule in testyear/config.py differs from the registered one")
    if problems:
        raise SystemExit("; ".join(problems))
    depth, generated = prereg["storm_depth_mm"], generated_values(prereg)
    code_now = code_sha256()

    x = real_year(year, key)
    rows = sewer_real(year, sha, flood_control)
    sw = S.sewer_quantities(rows)
    rain_place = S.place(S.rain_quantities(x, depth), generated)
    sewer_place = S.place({S.sewer_key(*q): sw[q] for q in K.SEWER_QUANTITIES}, generated)

    mem = Memorisation(TD.clean(TD.train_record(), TD.tau5()), x, **K.MEMORISATION)
    per_member = {n: mem.score(load_generated(K.series_csv(n), TD.tau5())) for n in prereg["members"]}
    too_close = float(np.mean([m["too_close_pct"] for m in per_member.values()]))

    checks = {}
    if year == K.REHEARSAL_YEAR:          # the real-year sewer run must reproduce flood-control's saved row
        saved = {r["controller"]: r["total_m3"] for r in flood_control_real_rows(flood_control)[year]}
        checks["sewer_reproduces_flood_control"] = all(
            abs(r["total_m3"] - saved[r["controller"]]) <= 1e-6 * saved[r["controller"]] for r in rows)
    out = {"year": year, "role": "test" if year == K.TEST_YEAR else "rehearsal", "prereg_sha256": sha,
           "series": prereg["series"], "members": prereg["members"],
           "finished": dt.datetime.now().isoformat(timespec="seconds"), "git": git_state(),
           "flood_control_git": flood_control_git(flood_control),
           "code_changed_since_registration": sorted(n for n in set(code_now) | set(prereg["code_sha256"])
                                                     if code_now.get(n) != prereg["code_sha256"].get(n)),
           "rain": rain_place, "sewer": sewer_place, "sewer_rows": rows,
           "memorisation": {"too_close_pct_mean": too_close, "threshold": mem.threshold,
                            "real_windows": len(mem.held), "members": per_member},
           **S.verdict(rain_place, sewer_place, too_close), "checks": checks,
           "calibration_train_years": {k: prereg["calibration_train_years"][k]
                                       for k in ("coverage_rain", "coverage_sewer")}}
    path = K.result_path(year)
    if year == K.TEST_YEAR and path.exists():
        raise guard.Locked(f"{path.name} appeared during the run")
    write_json(path, out)
    return out, path


def rehearse(flood_control=K.FLOOD_CONTROL):
    return evaluate_year(K.REHEARSAL_YEAR, None, flood_control)


def test(spend_test_year, flood_control=K.FLOOD_CONTROL):
    _, key = guard.unlock(spend_test_year)
    return evaluate_year(K.TEST_YEAR, key, flood_control)


# ── report ──────────────────────────────────────────────────────────────────────────────────────────

def report(out):
    print(f"real {out['year']} ({out['role']}) against {len(out['members'])} members of {out['series']}")
    print(f"  {'quantity':32s}{'real':>12s}{'percentile':>11s}   generated 5% / 50% / 95%")
    for kind in ("sewer", "rain"):
        for q, r in out[kind].items():
            g = " / ".join(f"{v:,.1f}" for v in r["generated_p5_p50_p95"])
            print(f"  {q:32s}{r['value']:12,.1f}{r['percentile']:11.2f}{'' if r['inside'] else ' *'}  {g}")
    m = out["memorisation"]
    print(f"  too close (one-day windows): {m['too_close_pct_mean']:.1f}% (members "
          + ", ".join(f"{v['too_close_pct']:.1f}" for v in m["members"].values()) + f"); {m['real_windows']} real "
          "windows set the threshold")
    for name, ok in out["tests"].items():
        print(f"  {'PASS' if ok else 'fail'}  {name}")
    for name, ok in out["checks"].items():
        print(f"  {'ok  ' if ok else 'FAIL'}  check: {name}")
    if out["code_changed_since_registration"]:
        print(f"  note: testyear code changed since registration: {out['code_changed_since_registration']}")
    c = out["calibration_train_years"]
    print(f"  training years inside: rain {100 * c['coverage_rain']:.0f}%, sewer {100 * c['coverage_sewer']:.0f}%"
          f" (90% if the generated years are like real ones)")
    print(f"{out['verdict']}  (* = outside {K.BAND})")


def status():
    rows = [("preregistration", K.PREREG), ("rehearsal", K.result_path(K.REHEARSAL_YEAR)),
            ("test claim", K.claim_path(K.TEST_YEAR)), ("test result", K.result_path(K.TEST_YEAR))]
    for name, p in rows:
        print(f"  {name:16s}{'present' if p.exists() else '-':9s}{rel(p)}")
    if K.PREREG.exists():
        problems = guard.committed([K.PREREG, K.PACKAGE])
        print("  committed       " + ("yes" if not problems else "; ".join(problems)))
