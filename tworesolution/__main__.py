"""Command line: `python -m tworesolution <command>` from the repository root (README.md has the order).

  build-data         30-minute split files (+ results/tworesolution/derived.json the first time; afterwards the
                     split files must hash as recorded there)
  configs            tworesolution/configs/{coarse30,coarse30_long,fine}.yaml from regime_training/config_v14.yaml
  check [--torch]    invariants on data and configs; --torch: the fine model starts as v14 exactly
  train-fine         train the fine model (GPU)
  generate-coarse    10-year coarse series, 4 members -> results/tworesolution/coarse/
  disaggregate       --source heldout|coarse --method fine|cascade -> results/tworesolution/disaggregated/
                     (source coarse also writes results/generated_data/rainfall_synthetic_10y_twores_*.csv)
  evaluate --stage k stage verdict -> results/tworesolution/stage<k>.json
  stage3-methods     print the disaggregators stage 3 may use (fails if stage 1 or 2 stopped the plan)
Every command except check writes a run record to results/tworesolution/records/. TWORES_OUT=<folder> sends
generated data, verdicts and records there instead (smoke tests only).
"""
import argparse
import datetime as dt
import pickle
import sys

import numpy as np
import pandas as pd

from sitetransfer.provenance import RunRecord, rel, write_json

from . import data as D
from . import evaluate as E
from .grid import STEPS_PER_YEAR_30, two_level

MEMBERS, SEED = 4, 42


def _record(command, params):
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RunRecord(command, "astlingen", params, E.OUT / "records" / f"{stamp}_{command}.json")


def _model(kind, run_id, device):
    from .network import load_args, load_fine, load_plain
    run = D.RUNS / run_id
    if kind == "coarse":
        model = load_plain(load_args(run / "config.yaml", device), run / "best_regime_model.pt", device)
    else:
        model = load_fine(load_args(run / "config.yaml", device), run / "best_fine_model.pt", device)
    with open(run / "scaler.pkl", "rb") as f:
        return model, pickle.load(f), run


def generate_coarse(a):
    from .sampling import generate_coarse as gen
    d = D.load_derived()
    n30 = int(round(a.years * STEPS_PER_YEAR_30))
    seeds = [a.seed + k for k in range(a.members)]
    with _record("generate-coarse", vars(a)) as rec:
        model, scaler, run = _model("coarse", a.coarse_run, a.device)
        rec.input(run / "best_regime_model.pt")
        steps = a.steps or model.args.diffusion_steps
        x30 = np.stack([gen(model, scaler, D.calendar30(), n30, s, d["tau30_mm"], a.batch, a.device, steps)
                        for s in seeds])
        path = E.coarse_path(a.coarse_run)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x30=x30, seeds=seeds)
        rec.output(path)
        stats = {"mm_per_year": (x30.sum(1) / a.years).tolist(), "dry_pct": (100 * (x30 == 0).mean(1)).tolist()}
        rec.result(**stats)
    print(f"{rel(path)}: {a.members} x {n30} half-hours | mm/yr {np.round(stats['mm_per_year'], 0)} | "
          f"dry {np.round(stats['dry_pct'], 1)}%  (real 2000-07: {d['mm_per_year_two_level']:.0f} mm/yr, "
          f"{d['dry_half_hours_pct']:.1f}% dry)")


def disaggregate(a):
    d = D.load_derived()
    t5, t30 = d["tau5_mm"], d["tau30_mm"]
    seeds = [a.seed + k for k in range(a.members)]
    with _record(f"disaggregate-{a.source}-{a.method}", vars(a)) as rec:
        if a.source == "heldout":
            x30s = [two_level(D.heldout_record(), t5, t30)[1]] * a.members
        else:
            rec.input(E.coarse_path(a.coarse_run))
            x30s = np.load(E.coarse_path(a.coarse_run))["x30"][:a.members]
        out, counts = [], []
        if a.method == "fine":
            from .sampling import disaggregate as fine
            model, scaler, run = _model("fine", a.fine_run, a.device)
            rec.input(run / "best_fine_model.pt")
            steps = a.steps or model.args.diffusion_steps
            for x30, s in zip(x30s, seeds):
                x5, c = fine(model, scaler, D.calendar5(), x30, s, t5, a.batch, a.device, steps)
                out.append(x5)
                counts.append(c)
        else:
            from . import cascade
            from .mass import finish
            levels = cascade.fit(two_level(D.train_record(), t5, t30)[0])
            write_json(E.OUT / "cascade_fit.json", cascade.summary(levels))
            for x30, s in zip(x30s, seeds):
                x5, c = finish(cascade.disaggregate(levels, x30, s), x30, t5)
                out.append(x5.reshape(-1))
                counts.append(c)
        x5 = np.stack(out)
        path = E.disaggregated_path(a.source, a.method, a.coarse_run)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x5=x5, seeds=seeds)
        rec.output(path)
        if a.source == "coarse":
            dates = pd.date_range(start="2026-01-01", periods=x5.shape[1], freq="5min")     # as production
            for k, x in enumerate(x5):
                csv = E.series_csv(E.series_name(a.method, a.coarse_run), k)
                csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"date": dates, "avg_rainfall": x}).to_csv(csv, index=False)
                rec.output(csv)
        rec.result(mass_matching=counts)
    for s, c in zip(seeds, counts):
        print(f"seed {s}: {c['wet_half_hours']} wet half-hours, {c['empty']} filled by the fallback, "
              f"{c['moved_values']} values below 0.005 mm moved ({c['moved_mm']:.2f} of {c['total_mm']:.0f} mm)")
    print(f"wrote {rel(path)}")


def evaluate(a):
    with _record(f"evaluate-stage{a.stage}", vars(a)) as rec:
        out = E.stage1(a.draws, a.workers, a.coarse_run) if a.stage == 1 else \
            E.stage2(a.draws, a.workers) if a.stage == 2 else E.stage3(a.coarse_run)
        rec.output(E.stage_path(a.stage, a.coarse_run))
        rec.result(verdict=out.get("verdict"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tworesolution", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build-data", "configs", "check", "train-fine", "generate-coarse",
                                        "disaggregate", "evaluate", "stage3-methods"])
    ap.add_argument("--torch", action="store_true", help="check: also compare the fine model's start with v14")
    ap.add_argument("--refresh", action="store_true", help="build-data: rewrite derived.json from the data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--members", type=int, default=MEMBERS)
    ap.add_argument("--seed", type=int, default=None, help=f"generation: {SEED}; train-fine: 0")
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=2048, help="generation batch size")
    ap.add_argument("--steps", type=int, default=None, help="sampler steps (smoke tests only; default: the "
                                                            "model's diffusion_steps, 36)")
    ap.add_argument("--source", choices=["heldout", "coarse"])
    ap.add_argument("--method", choices=["fine", "cascade"])
    ap.add_argument("--stage", type=int, choices=[1, 2, 3])
    ap.add_argument("--draws", type=int, default=1000, help="resampled-band draws (stages 1 and 2)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", default=str(D.FINE_CONFIG), help="train-fine: config")
    ap.add_argument("--run-id", default=D.RUN_FINE, help="train-fine: run folder name")
    ap.add_argument("--epochs", type=int, default=None, help="train-fine: override (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=None, help="train-fine: override (smoke tests)")
    ap.add_argument("--limit-batches", type=int, default=None, help="train-fine: batches per epoch (smoke tests)")
    ap.add_argument("--overwrite", action="store_true", help="train-fine: replace an existing run")
    ap.add_argument("--coarse-run", default=D.RUN_COARSE, help=f"coarse run for generate-coarse, disaggregate "
                    f"--source coarse, evaluate --stage 1|3 and stage3-methods ({D.RUN_COARSE} or "
                    f"{D.RUN_COARSE_LONG}); its outputs carry the run's tag")
    ap.add_argument("--fine-run", default=D.RUN_FINE, help="generation: fine run folder (smoke tests)")
    a = ap.parse_args(argv)
    if a.command == "build-data":
        with _record("build-data", {"refresh": a.refresh}) as rec:
            d = D.build(a.refresh)
            rec.output(D.OUT / "derived.json")
            rec.result(**d)
        for k, v in d.items():
            print(f"  {k}: {v}")
    elif a.command == "configs":
        for p in D.write_configs():
            print(f"wrote {rel(p)}")
    elif a.command == "check":
        from .invariants import check_data, check_v14_start, report
        res = check_data() + (check_v14_start("cpu" if a.device == "cpu" else a.device) if a.torch else [])
        if not report(res):
            return 1
    elif a.command == "train-fine":
        from .train_fine import run
        a.seed = 0 if a.seed is None else a.seed
        run(a)
    elif a.command == "generate-coarse":
        a.seed = SEED if a.seed is None else a.seed
        generate_coarse(a)
    elif a.command == "stage3-methods":
        print(" ".join(E.stage3_methods(a.coarse_run)))
    elif a.command == "disaggregate":
        if not (a.source and a.method):
            ap.error("disaggregate needs --source and --method")
        if a.source == "coarse" and a.method not in E.stage3_methods(a.coarse_run):
            raise SystemExit(f"stage 2 did not select {a.method} for stage 3")
        a.seed = SEED if a.seed is None else a.seed
        disaggregate(a)
    else:
        if a.stage is None:
            ap.error("evaluate needs --stage")
        evaluate(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
