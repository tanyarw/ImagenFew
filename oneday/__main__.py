"""Command line: `python -m oneday <command>` from the repository root (README.md has the order).

  config [--epochs N --batch-size B --reason TEXT]   write regime_training/config_v19.yaml (v14's, window changed)
  check [--no-checkpoint]                             the config is v14's with allowed changes; the checkpoint is
                                                      the recorded one; the one-day calendar file exists
  smoke                                               GPU: speed and memory -> results/oneday/smoke.json
  evaluate [--members 4] [--run-id v19]               verdict -> results/oneday/evaluation.json (v19) or
                                                      evaluation_<run-id>.json (other runs of the recipe)
"""
import argparse
import datetime as dt
import sys

from sitetransfer.provenance import ROOT, RunRecord, rel

from . import config as K


def _record(command, params):
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RunRecord(command, "astlingen", params, ROOT / "results" / "oneday" / "records" / f"{stamp}_{command}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m oneday", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["config", "check", "smoke", "evaluate"])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--reason", default="v14's recipe (500 epochs, batch 2048)")
    ap.add_argument("--no-checkpoint", action="store_true", help="check: skip the checkpoint (login nodes)")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--run-id", default=K.RUN_ID, help="evaluate: the run to score (default v19)")
    a = ap.parse_args(argv)
    if a.command == "config":
        print(f"wrote {rel(K.write(a.epochs, a.batch_size, a.reason))}")
    elif a.command == "check":
        res = K.check(with_checkpoint=not a.no_checkpoint)
        for name, ok, detail in res:
            print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        return 0 if all(ok for _, ok, _ in res) else 1
    elif a.command == "smoke":
        from .smoke import OUT, run
        with _record("smoke", {}) as rec:
            res = run()
            rec.output(OUT)
            rec.result(**res)
        print(f"GPU {res['gpu']} ({res['gpu_memory_gb']:.0f} GB) | partition {res['partition']} time limit "
              f"{res['partition_time_limit']}")
        for b, r in res["batch"].items():
            print(f"  batch {b}: " + ("out of memory" if r.get("out_of_memory") else
                  f"{r['s_per_step']:.3f} s/step, peak {r['peak_memory_gb']:.1f} GB, {r['steps_per_epoch']} "
                  f"steps/epoch -> 500 epochs = {r['hours_500_epochs']:.1f} h"))
        s = res["sampling"]
        print(f"  sampling: {s['s_per_block']*1000:.1f} ms/block -> 4 members x 10 years "
              f"{s['minutes_4_members_10_years']:.0f} min; training checks {s['minutes_training_checks_500_epochs']:.0f} min")
    else:
        from .evaluate import out_path, run
        with _record("evaluate", vars(a)) as rec:
            out = run(a.members, a.run_id)
            rec.output(out_path(a.run_id))
            rec.result(verdict=out["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
