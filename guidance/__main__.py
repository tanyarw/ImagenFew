"""Command line: `python -m guidance <command>` from the repository root.

  pilot-task --index i   generate and score pilot setting i (see guidance.pilot.TASKS; one SLURM array task)
  pilot-summary          collect every task, apply the pass rules, choose a setting -> results/guidance/pilot.json
  pilot-context2h        the chosen setting with 2 hours of context, compared with 80 minutes
                         -> results/guidance/context2h.json (does nothing if no setting passed)
  tasks                  print the settings table
"""
import argparse
import datetime as dt
import json
import sys

import numpy as np

from sitetransfer.provenance import RunRecord, write_json

from . import pilot as P


def _record(command, params):
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RunRecord(command, "astlingen", params, P.OUT / "records" / f"{stamp}_{command}.json")


def run_task(name, context_cols, variant, w, a):
    params = dict(name=name, context_cols=context_cols, variant=variant, w=w, days=a.days, starts=a.starts,
                  chains_per_start=a.chains_per_start, seed=P.SEED, wet_weight=P.WET_WEIGHT, device=a.device)
    with _record(f"pilot-task-{name}", params) as rec:
        model, scaler, net, t_steps = P.load_v14(a.device)
        offsets = P.pilot_offsets(a.starts, a.chains_per_start)
        out, seams, trace = P.generate(model, scaler, net, t_steps, offsets, a.days * P.STEPS_PER_DAY,
                                       context_cols, variant, w, device=a.device)
        m = P.score(out, seams) | dict(params, correction_over_guess_by_step=trace)
        npz, js = P.task_paths(name)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, mm=out.astype(np.float32), seams=seams, offsets=offsets)
        write_json(js, m)
        rec.output(npz)
        rec.output(js)
        rec.result(**{k: m[k] for k in ("continuation", "wet_share_pct", "mm_per_year", "p999_mm")})
    print(f"{name}: continuation {m['continuation']:.3f} (inside {m['continuation_inside']:.3f}), wet "
          f"{m['wet_share_pct']:.2f}%, {m['mm_per_year']:.0f} mm/yr, P99.9 {m['p999_mm']:.3f}", flush=True)
    return m


def summary():
    with _record("pilot-summary", {"rules": P.RULES}) as rec:
        res = {}
        for name, *_ in P.TASKS:
            _, js = P.task_paths(name)
            if not js.exists():
                raise SystemExit(f"{js} missing: task {name} did not finish")
            res[name] = json.loads(js.read_text())
            rec.input(js)
        ref = res["independent"]
        for name, r in res.items():
            if name != "independent":
                r["pass"] = P.judge(r, ref)
        chosen = P.choose(res)
        out = {"rules": P.RULES, "reference": "independent", "chosen": chosen,
               "verdict": "a setting passes" if chosen else "NO SETTING PASSES: stop (negative result)",
               "results": {k: {kk: v[kk] for kk in ("variant", "w", "context_cols", "continuation", "continuation_inside",
                                                     "wet_share_pct", "mm_per_year", "p999_mm",
                                                     "storm_survival_after_wet_seam", "pass") if kk in v}
                           for k, v in res.items()}}
        path = write_json(P.OUT / "pilot.json", out)
        rec.output(path)
        rec.result(chosen=chosen)
    for k, v in out["results"].items():
        flag = "" if "pass" not in v else ("PASS" if all(v["pass"].values()) else "fail " + ",".join(
            c for c, ok in v["pass"].items() if not ok))
        print(f"{k:16s} cont {v['continuation']:.3f}  wet {v['wet_share_pct']:.2f}%  {v['mm_per_year']:6.0f} mm/yr  "
              f"P99.9 {v['p999_mm']:.3f}  {flag}")
    print(f"chosen: {chosen}")
    return out


def context2h(a):
    s = json.loads((P.OUT / "pilot.json").read_text())
    chosen = s["chosen"]
    if not chosen:
        print("no setting passed the pilot: the 2-hour comparison is not run")
        return None
    r = s["results"][chosen]
    m2 = run_task(f"{chosen}_ctx2h", 3, r["variant"], r["w"], a)
    m2["pass"] = P.judge(m2, s["results"]["independent"])
    better = all(m2["pass"].values()) and m2["continuation"] >= r["continuation"] + 0.03
    out = {"rule": P.TWO_HOUR_RULE, "chosen_80min": chosen, "result_80min": r,
           "result_2h": {k: m2[k] for k in ("continuation", "wet_share_pct", "mm_per_year", "p999_mm",
                                            "storm_survival_after_wet_seam", "pass")},
           "context_for_full_run": "2 h" if better else "80 min"}
    write_json(P.OUT / "context2h.json", out)
    print(f"80 min: continuation {r['continuation']:.3f} | 2 h: {m2['continuation']:.3f} -> full run uses "
          f"{out['context_for_full_run']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m guidance", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot-task", "pilot-summary", "pilot-context2h", "tasks"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--chains-per-start", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.command == "tasks":
        for i, t in enumerate(P.TASKS):
            print(i, *t)
    elif a.command == "pilot-task":
        if a.index is None or not 0 <= a.index < len(P.TASKS):
            ap.error(f"--index must be 0..{len(P.TASKS) - 1}")
        run_task(*P.TASKS[a.index], a)
    elif a.command == "pilot-summary":
        summary()
    else:
        context2h(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
