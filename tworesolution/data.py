"""Data for both levels, built from the 5-minute split files (the cluster has these, not the full record).

    train    2000-2007   data/rainfall/splits/train_years_labelled.csv
    heldout  2008-2009   val_years_labelled.csv + test_years_labelled.csv  (2008 = val, 2009 = test)

The 5-minute wet threshold is Astlingen's (sitetransfer/sites/astlingen.yaml, 0.005 mm). The coarse threshold
and asinh scale come from the sitetransfer rules applied to the 2000-2007 half-hour totals (plan section 2).
`build` writes the 30-minute split files the coarse model trains on (regime_training/train_regime.py reads them
like any other split) and results/tworesolution/derived.json; `configs` writes both training configs from v14's.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sitetransfer import rules
from sitetransfer.provenance import ROOT, rel, sha256_file, write_json
from sitetransfer.site import Site
from sitetransfer.timegrid import slot_starts
from sitetransfer.training import diff_keys

from .grid import FACTOR, L, STEPS_PER_YEAR_5, STEPS_PER_YEAR_30, aggregate, clean, train_starts, two_level

SPLITS = ROOT / "data" / "rainfall" / "splits"
SPLITS30 = ROOT / "data" / "rainfall" / "splits_30min"
CALENDAR = SPLITS / "seasonal_transition_matrix_train_len64.pkl"
OUT = ROOT / "results" / "tworesolution"
RUNS = ROOT / "logs" / "ImagenFew" / "Rainfall_Regime"
V14_CONFIG = ROOT / "regime_training" / "config_v14.yaml"
V14_RUN = RUNS / "v14"
CONFIGS = Path(__file__).resolve().parent / "configs"
COARSE_CONFIG = CONFIGS / "coarse30.yaml"
COARSE_LONG_CONFIG = CONFIGS / "coarse30_long.yaml"
FINE_CONFIG = CONFIGS / "fine.yaml"
RUN_COARSE = "twores_coarse30"
RUN_COARSE_LONG = "twores_coarse30_long"      # follow-up (plan section 11): v14's number of optimiser steps
RUN_FINE = "twores_fine"
SPLIT_YEARS = {"train": 2000, "val": 2008, "test": 2009}      # first year of each split file
ALLOWED_CHANGES = {"coarse": {"train_csv", "test_csv", "asinh_scale", "run_dir"},
                   "coarse_long": {"train_csv", "test_csv", "asinh_scale", "run_dir", "epochs"},
                   "fine": {"run_dir", "init_from", "scaler_from"}}


def site():
    return Site.load("astlingen")


def tau5():
    return float(site().tau)


def load_split(name):
    """(5-minute values as recorded, seasonal phase of every step) of one split file."""
    df = pd.read_csv(SPLITS / f"{name}_years_labelled.csv", usecols=["avg_rainfall", "hmm_state"])
    x = df["avg_rainfall"].to_numpy(dtype=np.float64)
    if len(x) % STEPS_PER_YEAR_5:
        raise ValueError(f"{name}: {len(x)} steps is not a whole number of 365-day years")
    return x, df["hmm_state"].to_numpy(dtype=int)


def train_record():
    return load_split("train")[0]


def heldout_record():
    return np.r_[load_split("val")[0], load_split("test")[0]]


def calendar5():
    """Seasonal phase of every 5-minute step of one calendar year (the profile v14 is generated with)."""
    with open(CALENDAR, "rb") as f:
        return np.asarray(pickle.load(f)["climatology_profile_1yr"], dtype=int)


def calendar30():
    return calendar5()[::FACTOR]


def derive():
    """Every data-derived setting, from 2000-2007 only."""
    x = train_record()
    t5 = tau5()
    c30 = aggregate(clean(x, t5))
    tau30 = float(rules.wet_threshold(c30))
    s30 = float(rules.asinh_scale(c30, tau30))
    x5c, x30 = two_level(x, t5, tau30)
    dropped = (c30 > 0) & (x30 == 0)
    batch = yaml.safe_load(V14_CONFIG.read_text())["batch_size"]
    coarse_windows = len(x30) - L + 1
    fine_windows = len(train_starts(len(x)))
    return dict(
        tau5_mm=t5, tau30_mm=tau30, asinh_scale30_mm=s30,
        dry_half_hours_pct=float(100 * (x30 == 0).mean()),
        half_hours_set_dry=int(dropped.sum()), half_hours_set_dry_pct_of_wet=float(100 * dropped.sum() / (c30 > 0).sum()),
        mm_set_dry_pct=float(100 * c30[dropped].sum() / c30.sum()),
        coarse_windows=int(coarse_windows), coarse_steps_per_epoch=int(coarse_windows // batch),
        fine_windows=int(fine_windows), fine_steps_per_epoch=int(fine_windows // batch),
        mm_per_year_5min=float(clean(x, t5).sum() / 8), mm_per_year_two_level=float(x30.sum() / 8),
    )


def load_derived():
    path = OUT / "derived.json"
    if not path.exists():
        raise FileNotFoundError(f"{rel(path)} missing: run `python -m tworesolution build-data`")
    return json.loads(path.read_text())["derived"]


def build(refresh=False):
    """30-minute split files, and derived.json the first time (or with refresh).

    Once derived.json exists (it is committed), a rebuild, as on the cluster, must reproduce it: the 5-minute
    split files must hash as recorded (byte-identical input) and the derived settings must agree. The half-hour
    totals are written rounded to 10 decimals (the gauge grid is 0.0025 mm), so that last-digit differences in
    floating-point addition between machines (ARM laptop, x86 cluster: found 2026-10-14) do not change the files;
    a different output hash is reported, not fatal, and `check` compares the values numerically."""
    d = derive()
    path = OUT / "derived.json"
    saved = json.loads(path.read_text()) if path.exists() and not refresh else None
    SPLITS30.mkdir(parents=True, exist_ok=True)
    inputs, outputs = {}, {}
    profile = calendar30()
    for name, first_year in SPLIT_YEARS.items():
        src = SPLITS / f"{name}_years_labelled.csv"
        x, states = load_split(name)
        _, x30 = two_level(x, d["tau5_mm"], d["tau30_mm"])
        lab = np.tile(profile, len(x30) // STEPS_PER_YEAR_30)
        if not np.array_equal(states[::FACTOR], lab):
            raise ValueError(f"{name}: seasonal phases in the split file are not the calendar profile")
        dst = SPLITS30 / f"{name}_years_labelled.csv"
        pd.DataFrame({"date": slot_starts(len(x), first_year)[::FACTOR], "avg_rainfall": np.round(x30, 10),
                      "hmm_state": lab}).to_csv(dst, index=False)
        inputs[rel(src)] = sha256_file(src)
        outputs[rel(dst)] = sha256_file(dst)
    inputs[rel(CALENDAR)] = sha256_file(CALENDAR)
    if saved is None:
        write_json(path, {"derived": d, "inputs": inputs, "outputs": outputs,
                          "plan": "code_plan/TWO_RESOLUTION_PLAN.md, section 2"})
        return d
    bad = [k for k, h in inputs.items() if saved["inputs"].get(k) != h]
    if bad:
        raise ValueError(f"these files differ from the ones {rel(path)} was built from: {bad} (sync the split files "
                         f"from the laptop, or rebuild with --refresh and commit derived.json)")
    off = [k for k in d if not np.isclose(d[k], saved["derived"][k])]
    if off:
        raise ValueError(f"the data give different settings from {rel(path)}: {off}")
    for k, h in outputs.items():
        if saved["outputs"].get(k) != h:
            print(f"note: {k} is not byte-identical to the laptop's (last-digit rounding); values checked by `check`")
    return saved["derived"]


# ── training configs: v14's, with only the listed keys changed ─────────────────────────────────────

def v14_optimizer_steps(base):
    """v14's optimiser steps: (5-minute training windows // batch) * epochs, as DataLoader(drop_last=True) ran."""
    windows = len(site().years("train")) * STEPS_PER_YEAR_5 - base["seq_len"] + 1
    return (windows // base["batch_size"]) * base["epochs"]


def config_dicts(d):
    """{name: config} for base (v14), coarse, coarse_long and fine."""
    base = yaml.safe_load(V14_CONFIG.read_text())
    coarse = dict(base) | {
        "train_csv": str(SPLITS30.relative_to(ROOT) / "train_years_labelled.csv"),
        "test_csv": str(SPLITS30.relative_to(ROOT) / "val_years_labelled.csv"),
        "asinh_scale": round(d["asinh_scale30_mm"], 6),
        "run_dir": f"ImagenFew/Rainfall_{RUN_COARSE}/",
    }
    fine = dict(base) | {
        "run_dir": f"ImagenFew/Rainfall_{RUN_FINE}/",
        "init_from": str((V14_RUN / "best_regime_model.pt").relative_to(ROOT)),
        "scaler_from": str((V14_RUN / "scaler.pkl").relative_to(ROOT)),
    }
    coarse_long = dict(coarse) | {
        "epochs": rules.matched_epochs(d["coarse_windows"], base["batch_size"], v14_optimizer_steps(base)),
        "run_dir": f"ImagenFew/Rainfall_{RUN_COARSE_LONG}/",
    }
    cfgs = {"base": base, "coarse": coarse, "coarse_long": coarse_long, "fine": fine}
    for name, cfg in cfgs.items():
        extra = set(diff_keys(base, cfg)) - ALLOWED_CHANGES.get(name, set())
        if extra:
            raise ValueError(f"{name} config changes keys outside the allowed list: {sorted(extra)}")
    return cfgs


def _render(cfg, base, what):
    changed = diff_keys(base, cfg)
    head = ["# " + "=" * 98,
            f"# {what} WRITTEN BY `python -m tworesolution configs`; do not edit.",
            f"# Template: {rel(V14_CONFIG)} (sha256 {sha256_file(V14_CONFIG)[:16]}...). Only these keys differ:"]
    head += [f"#   {k}: {cfg[k]}" for k in changed]
    head += ["# Plan: code_plan/TWO_RESOLUTION_PLAN.md (sections 2, 10 and 11).", "# " + "=" * 98, ""]
    return "\n".join(head) + yaml.safe_dump(cfg, sort_keys=False)


def write_configs():
    c = config_dicts(load_derived())
    CONFIGS.mkdir(exist_ok=True)
    COARSE_CONFIG.write_text(_render(c["coarse"], c["base"],
                                     f"Coarse model ({RUN_COARSE}): v14's recipe on 30-minute totals."))
    COARSE_LONG_CONFIG.write_text(_render(c["coarse_long"], c["base"],
                                          f"Coarse model ({RUN_COARSE_LONG}): as {RUN_COARSE}, trained for v14's "
                                          f"{v14_optimizer_steps(c['base']):,} optimiser steps."))
    FINE_CONFIG.write_text(_render(c["fine"], c["base"], f"Fine model ({RUN_FINE}): v14 fine-tuned with the half-hour "
                                                         "totals as an extra input (tworesolution/train_fine.py)."))
    return COARSE_CONFIG, COARSE_LONG_CONFIG, FINE_CONFIG
