"""Training config for a transfer site, written from the home recipe's config by rule.

The new config is the v14 config with only the keys in ALLOWED_CHANGES replaced (invariant 9). The epoch
count gives the same number of optimiser steps as v14 had at home: the home training windows, batch size
and epochs fix the step count, and the transfer site's usable windows turn it into epochs.
"""
from pathlib import Path

import numpy as np
import yaml

from . import rules
from .provenance import ROOT, rel, sha256_file

TEMPLATE = ROOT / "regime_training" / "config_v14.yaml"
ALLOWED_CHANGES = {"train_csv", "test_csv", "asinh_scale", "epochs", "run_dir", "mask_col"}


def home_optimizer_steps(home_site, template_cfg):
    """v14's optimiser steps at home: (training windows // batch size) * epochs (drop_last=True)."""
    x, _ = home_site.select_years(home_site.years("train"), clean=False)
    n_windows = len(x) - template_cfg["seq_len"] + 1
    return (n_windows // template_cfg["batch_size"]) * template_cfg["epochs"], n_windows


def build_config(site, home_site, run_id, template=TEMPLATE):
    base = yaml.safe_load(Path(template).read_text())
    steps, home_windows = home_optimizer_steps(home_site, base)
    train_csv = Path(site.cfg["splits_dir"]) / "train_years_labelled.csv"
    val_csv = Path(site.cfg["splits_dir"]) / "val_years_labelled.csv"
    _, mask = site.select_years(site.years("train"), clean=False)
    windows = rules.usable_windows(mask, base["seq_len"])
    changes = {
        "train_csv": str(train_csv), "test_csv": str(val_csv),
        "asinh_scale": float(site.params["asinh_scale_mm"]),
        "epochs": rules.matched_epochs(windows, base["batch_size"], steps),
        "run_dir": f"ImagenFew/Rainfall_{run_id}/",
        "mask_col": site.cfg["series"]["outage_col"],
    }
    unexpected = set(changes) - ALLOWED_CHANGES
    if unexpected:
        raise ValueError(f"changes outside the allowed list: {unexpected}")
    cfg = dict(base)
    cfg.update(changes)
    facts = dict(template=rel(template), template_sha256=sha256_file(template), home_optimizer_steps=int(steps),
                 home_windows=int(home_windows), site_windows=int(windows),
                 site_steps=int((windows // base["batch_size"]) * changes["epochs"]))
    return cfg, changes, facts


def render(cfg, changes, facts, site, run_id):
    head = [
        "# " + "=" * 98,
        f"# {run_id}: the v14 recipe at {site.name}. WRITTEN BY `python -m sitetransfer train-config`; do not edit.",
        f"# Template: {facts['template']} (sha256 {facts['template_sha256'][:16]}...). Only these keys differ:",
    ]
    head += [f"#   {k}: {v}" for k, v in changes.items()]
    head += [f"# Epochs: v14 took {facts['home_optimizer_steps']:,} optimiser steps at home "
             f"({facts['home_windows']:,} windows); {site.name} has {facts['site_windows']:,} usable windows, "
             f"so {changes['epochs']} epochs give {facts['site_steps']:,} steps.",
             "# Plan: code_plan/SINGAPORE_TRANSFER_PLAN.md (sections 5 and 6).", "# " + "=" * 98, ""]
    return "\n".join(head) + yaml.safe_dump(cfg, sort_keys=False)


def diff_keys(cfg_a, cfg_b):
    keys = set(cfg_a) | set(cfg_b)
    return sorted(k for k in keys if cfg_a.get(k, np.nan) != cfg_b.get(k, np.nan))
