"""Invariants on built outputs (plan section 7.4). `python -m sitetransfer check --site X` runs them.

The code-level invariants (rules reproduce Astlingen, metrics and labels reproduce the existing pipeline,
the dataset switch changes nothing when off) are the unit tests in sitetransfer/tests/.
"""
import json

import numpy as np
import pandas as pd
import yaml

from . import rules, training
from . import timegrid as tg
from .provenance import ROOT
from .site import Site


def _ok(name, cond, detail=""):
    return {"invariant": name, "ok": bool(cond), "detail": detail}


def check_series(site):
    s = site.cfg["series"]
    df = pd.read_csv(site.path(s["csv"]))
    x = df[s["value_col"]].to_numpy()
    out = [_ok("whole 365-day years", len(x) % tg.STEPS_PER_YEAR == 0, f"{len(x)} steps"),
           _ok("non-negative", (x >= 0).all())]
    if s.get("outage_col"):
        o = df[s["outage_col"]].to_numpy().astype(bool)
        grid = site.cfg["ingest"]["gauge_resolution_mm"] / site.cfg["ingest"]["cluster_size"]
        out += [_ok("values on the gauge grid", np.allclose(x / grid, np.round(x / grid), atol=1e-6), f"grid {grid} mm"),
                _ok("outage placeholders are zero", (x[o] == 0).all()),
                _ok("outage flag matches fill_flag", np.array_equal(o, df["fill_flag"].to_numpy() == "outage")),
                _ok("every step carries a fill flag", df["fill_flag"].notna().all())]
        hr = (np.arange(len(x)) % tg.STEPS_PER_DAY) // tg.STEPS_PER_HOUR
        mean = np.bincount(hr[~o], weights=x[~o], minlength=24) / np.bincount(hr[~o], minlength=24)
        lo, hi = site.cfg.get("expect", {}).get("diurnal_peak_hour", [0, 23])
        out.append(_ok("local time: daily-cycle peak where expected", lo <= int(np.argmax(mean)) <= hi,
                       f"peak at {int(np.argmax(mean))}:00, expected {lo}-{hi}"))
    return out


def check_derived(site):
    d = json.loads(site.derived_path.read_text())
    train = site.years("train")
    out = [_ok("rules saw training years only", d["facts"]["training_years"] == train, str(d["facts"]["training_years"])),
           _ok("reference years are training years", set(d["params"]["reference_years"]) <= set(train),
               str(d["params"]["reference_years"]))]
    if "invariant_1" in d:
        out.append(_ok("invariant 1: rules reproduce the home site's frozen values", all(d["invariant_1"].values()),
                       str(d["invariant_1"])))
    return out


def check_splits(site, build_dataset=False):
    out = []
    split_dir = site.path("splits_dir")
    for split, fname in (("train", "train_years_labelled.csv"), ("val", "val_years_labelled.csv"),
                         ("test", "test_years_labelled.csv")):
        df = pd.read_csv(split_dir / fname, usecols=["date", "outage"])
        years = sorted(pd.to_datetime(df["date"]).dt.year.unique().tolist())
        out.append(_ok(f"{split} split holds exactly its years", years == site.years(split), str(years)))
    if build_dataset:
        import sys
        sys.path.insert(0, str(ROOT))
        from regime_training.regime_dataset import RegimeDataset
        cfg = yaml.safe_load((ROOT / "regime_training" / f"config_{site.cfg.get('run_id', 'sg_v14')}.yaml").read_text())
        ds = RegimeDataset(str(ROOT / cfg["train_csv"]), seq_len=cfg["seq_len"], regime_col="hmm_state",
                           data_col="avg_rainfall", transform="asinh", asinh_scale=cfg["asinh_scale"],
                           mask_col=cfg["mask_col"])
        mask = pd.read_csv(ROOT / cfg["train_csv"], usecols=["outage"])["outage"].to_numpy()
        out.append(_ok("training windows contain no outage step", len(ds) == rules.usable_windows(mask, cfg["seq_len"]),
                       f"{len(ds)} windows, {ds.n_skipped} skipped"))
    return out


def check_config(run_id):
    path = ROOT / "regime_training" / f"config_{run_id}.yaml"
    new = yaml.safe_load(path.read_text())
    base = yaml.safe_load(training.TEMPLATE.read_text())
    extra = set(training.diff_keys(base, new)) - training.ALLOWED_CHANGES
    return [_ok("invariant 9: training config differs from v14 only in data-derived keys", not extra,
                f"differs in {sorted(training.diff_keys(base, new))}")]


def run_all(site, run_id=None, build_dataset=False):
    res = []
    if not site.is_home:
        res += check_series(site)
    if site.derived_path.exists():
        res += check_derived(site)
    if not site.is_home and site.path("splits_dir").exists():
        res += check_splits(site, build_dataset)
    if run_id and (ROOT / "regime_training" / f"config_{run_id}.yaml").exists():
        res += check_config(run_id)
    return res
