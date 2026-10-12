"""Site descriptions: one YAML file per site (sitetransfer/sites/<name>.yaml) plus, for a transfer site,
the machine-written derived values (<results_dir>/derived.json).

`Site.params` is the single place the rest of the package reads data-derived parameters from:
the frozen values for the home site (Astlingen), the derived values for a transfer site.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .provenance import ROOT
from .timegrid import STEPS_PER_YEAR

SITES_DIR = Path(__file__).resolve().parent / "sites"
PARAM_KEYS = ("wet_threshold_mm", "asinh_scale_mm", "storm_depth_mm", "reference_years")


class SiteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Site:
    name: str
    cfg: dict = field(repr=False)

    # -- loading ---------------------------------------------------------------------------------
    @classmethod
    def load(cls, name):
        path = SITES_DIR / f"{name}.yaml"
        if not path.exists():
            raise SiteError(f"no site file {path}; known: {sorted(p.stem for p in SITES_DIR.glob('*.yaml'))}")
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("name") != name:
            raise SiteError(f"{path}: name field {cfg.get('name')!r} does not match the file name")
        for key in ("series", "years", "splits_dir", "results_dir"):
            if key not in cfg:
                raise SiteError(f"{path}: missing '{key}'")
        return cls(name=name, cfg=cfg)

    @property
    def config_path(self):
        return SITES_DIR / f"{self.name}.yaml"

    def path(self, key_or_rel):
        """Absolute path of a repository-relative path (or of a top-level config key holding one)."""
        rel = self.cfg.get(key_or_rel, key_or_rel) if isinstance(key_or_rel, str) else key_or_rel
        return ROOT / rel

    @property
    def results_dir(self):
        return ROOT / self.cfg["results_dir"]

    @property
    def derived_path(self):
        return self.results_dir / "derived.json"

    @property
    def is_home(self):
        return "frozen" in self.cfg

    # -- parameters ------------------------------------------------------------------------------
    @property
    def params(self):
        """Data-derived parameters: frozen for the home site, from derived.json for a transfer site."""
        if self.is_home:
            src = self.cfg["frozen"]
        else:
            if not self.derived_path.exists():
                raise SiteError(f"{self.derived_path} missing: run `python -m sitetransfer derive --site {self.name}`")
            src = json.loads(self.derived_path.read_text())["params"]
        missing = [k for k in PARAM_KEYS if k not in src]
        if missing:
            raise SiteError(f"{self.name}: parameters missing {missing}")
        return {k: src[k] for k in PARAM_KEYS}

    @property
    def tau(self):
        return float(self.params["wet_threshold_mm"])

    def years(self, split):
        lo, hi = self.cfg["years"][split]
        return list(range(lo, hi + 1))

    @property
    def first_year(self):
        return int(self.cfg["series"]["first_year"])

    # -- data ------------------------------------------------------------------------------------
    def load_record(self, clean=True, tau=None):
        """The whole real record: (values, outage mask, years). Values below tau set to 0 when clean."""
        s = self.cfg["series"]
        path = self.path(s["csv"])
        if not path.exists():
            raise SiteError(f"{path} missing" + ("" if self.is_home else
                            f": run `python -m sitetransfer build-data --site {self.name}`"))
        cols = [s["value_col"]] + ([s["outage_col"]] if s.get("outage_col") else [])
        df = pd.read_csv(path, usecols=cols)
        x = df[s["value_col"]].to_numpy(dtype=np.float64)
        mask = df[s["outage_col"]].to_numpy(dtype=bool) if s.get("outage_col") else np.zeros(len(x), bool)
        if len(x) % STEPS_PER_YEAR:
            raise SiteError(f"{path}: {len(x)} steps is not a whole number of 365-day years")
        if clean:
            x = clean_series(x, self.tau if tau is None else tau)
        years = list(range(self.first_year, self.first_year + len(x) // STEPS_PER_YEAR))
        return x, mask, years

    def select_years(self, years_wanted, clean=True):
        """Concatenate whole calendar years from the record: (values, outage mask)."""
        x, mask, years = self.load_record(clean=clean)
        idx = []
        for y in years_wanted:
            if y not in years:
                raise SiteError(f"{self.name}: year {y} not in the record ({years[0]}-{years[-1]})")
            k = years.index(y)
            idx.append(np.arange(k * STEPS_PER_YEAR, (k + 1) * STEPS_PER_YEAR))
        idx = np.concatenate(idx)
        return x[idx], mask[idx]

    def reference(self):
        return self.select_years(self.params["reference_years"])

    def heldout(self):
        return self.select_years(self.years("heldout"))

    def series_to_score(self):
        return {k: ROOT / v for k, v in self.cfg.get("series_to_score", {}).items()}


def clean_series(x, tau):
    """Non-negative, and zero below the wet threshold (eval_suite._clean with the site's threshold)."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0, None)
    x[x < tau] = 0.0
    return x


def load_generated(path, tau):
    """A generated series (date, avg_rainfall) cleaned with the site's wet threshold."""
    path = Path(path)
    df = pd.read_csv(path, usecols=["avg_rainfall"])
    return clean_series(df["avg_rainfall"].to_numpy(), tau)
