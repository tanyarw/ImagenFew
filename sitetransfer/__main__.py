"""Command line: `python -m sitetransfer <command> --site <name>` from the repository root.

  build-data    raw files -> target series with fill flags        (transfer site)
  derive        data-derived parameters by rule -> derived.json   (home site: checks invariant 1)
  labels        seasonal phases, split files, transition bundle   (transfer site)
  train-config  regime_training/config_<run-id>.yaml from v14     (transfer site)
  baseline      occurrence-matched latent Gaussian AR, 10 years   (transfer site)
  pilot         perfect-window generators, 3 seeds
  bands         resampled pass bands
  evaluate      score every available series
  verdict       recipe and findings verdict from both sites' evaluations
  check         invariants on the built outputs
  prepare       derive, labels, train-config, baseline, pilot, bands, evaluate, check (in order)
"""
import argparse
import sys

from . import invariants, pipeline
from .site import Site


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sitetransfer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build-data", "derive", "labels", "train-config", "baseline", "pilot", "bands",
                                        "evaluate", "verdict", "check", "prepare"])
    ap.add_argument("--site", default="singapore")
    ap.add_argument("--home", default="astlingen")
    ap.add_argument("--run-id", default="sg_v14")
    ap.add_argument("--draws", type=int, default=1000, help="bands: resamples")
    ap.add_argument("--workers", type=int, default=1, help="bands: processes")
    ap.add_argument("--dataset", action="store_true", help="check: also build the training dataset (slow)")
    a = ap.parse_args(argv)
    site = Site.load(a.site)

    if a.command == "build-data":
        pipeline.build_data(site)
    elif a.command == "derive":
        print(pipeline.derive(site)["params"])
    elif a.command == "labels":
        print(pipeline.make_labels(site)["phases"])
    elif a.command == "train-config":
        out, changes, _ = pipeline.train_config(site, a.run_id)
        print(f"wrote {out}: {changes}")
    elif a.command == "baseline":
        print(pipeline.baseline(site))
    elif a.command == "pilot":
        pipeline.run_pilot(site)
    elif a.command == "bands":
        pipeline.bands(site, n_draws=a.draws, workers=a.workers)
    elif a.command == "evaluate":
        pipeline.evaluate(site)
    elif a.command == "verdict":
        print(f"wrote {pipeline.verdict(a.home, a.site)}")
    elif a.command == "check":
        return _check(site, a.run_id, a.dataset)
    elif a.command == "prepare":
        if not site.is_home:
            pipeline.derive(site)
            pipeline.make_labels(site)
            pipeline.train_config(site, a.run_id)
            pipeline.baseline(site)
        else:
            pipeline.derive(site)
        pipeline.run_pilot(site)
        pipeline.bands(site, n_draws=a.draws, workers=a.workers)
        pipeline.evaluate(site)
        return _check(site, a.run_id, a.dataset)
    return 0


def _check(site, run_id, dataset):
    res = invariants.run_all(site, None if site.is_home else run_id, build_dataset=dataset)
    for r in res:
        print(f"{'ok  ' if r['ok'] else 'FAIL'}  {r['invariant']}  {r['detail']}")
    bad = [r for r in res if not r["ok"]]
    print(f"{len(res) - len(bad)}/{len(res)} invariants hold")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
