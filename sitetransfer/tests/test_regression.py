"""Regression tests on Astlingen: the new code must reproduce the existing pipeline exactly.

Each test is skipped (with the reason) when the project data it compares against is not present.

    python -m unittest discover -s sitetransfer/tests -t .
"""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sitetransfer import baselines, checks, labels, rules
from sitetransfer import metrics as M
from sitetransfer.provenance import ROOT
from sitetransfer.site import Site, load_generated

HOME = Site.load("astlingen")
HAVE_RECORD = HOME.path(HOME.cfg["series"]["csv"]).exists()
CACHE = ROOT / "results" / "evaluation" / "eval_cache.pkl"


@unittest.skipUnless(HAVE_RECORD, "Astlingen record not present")
class Invariant1Rules(unittest.TestCase):
    def test_rules_reproduce_frozen_values(self):
        x, _ = HOME.select_years(HOME.years("train"), clean=False)
        tau = rules.wet_threshold(x)
        self.assertEqual(tau, 0.005)
        self.assertEqual(round(rules.asinh_scale(x, tau), 6), 0.035)
        _, dep = rules.storms_2h(np.where(x >= tau, x, 0))
        thr = rules.storm_depth_threshold(dep, rules.depth_share(dep, 10.0))
        self.assertEqual(int((dep >= thr).sum()), int((dep >= 10.0).sum()))


@unittest.skipUnless(HAVE_RECORD and HOME.path("transition_bundle").exists(), "Astlingen labels not present")
class LabelsPort(unittest.TestCase):
    def test_profile_labels_and_matrix(self):
        x, _ = HOME.select_years(HOME.years("train"), clean=False)
        ph = labels.fit_phases(labels.slot_climatology(x))
        with open(HOME.path("transition_bundle"), "rb") as f:
            old = pickle.load(f)
        np.testing.assert_array_equal(ph["profile"], np.asarray(old["climatology_profile_1yr"]))
        self.assertEqual(ph["min_days"], old["post_processing_min_days"])
        st = labels.label_steps(ph["profile"], len(x))
        tr = pd.read_csv(HOME.path("splits_dir") / "train_years_labelled.csv", usecols=["hmm_state"])["hmm_state"]
        np.testing.assert_array_equal(st, tr.to_numpy())
        b = labels.transition_bundle(st, ph["profile"], ph["min_days"])
        np.testing.assert_allclose(b["P_block"], old["P_block"], rtol=0, atol=1e-15)


@unittest.skipUnless(HAVE_RECORD and CACHE.exists(), "evaluation cache not present")
class MetricsPort(unittest.TestCase):
    """Every shared metric and every fixed Gate A / Tier 6 result equals scripts/eval_suite.py's cache."""

    @classmethod
    def setUpClass(cls):
        with open(CACHE, "rb") as f:
            cls.cache = pickle.load(f)
        cls.P = M.Params(HOME.params["storm_depth_mm"])
        x, _ = HOME.reference()
        cls.R = M.reference(x, cls.P)

    def series(self):
        out = {"real 2008-2009": HOME.heldout()[0]}
        for key, name in (("v14", "v14_nobridge_cal"), ("lgar", "arima_copula_occ_len64")):
            path = HOME.series_to_score()[key]
            if path.exists():
                out[name] = load_generated(path, HOME.tau)
        return out

    def test_metrics(self):
        keymap = {"storms_ge_depth_yr": "storms_ge10mm_yr", "top_storm_mm": "top10_storm_mm"}
        for name, a in self.series().items():
            m = M.hydro(a, self.R, self.P)
            old = self.cache["hydro"][name]
            shared = [k for k in m if keymap.get(k, k) in old]
            self.assertGreaterEqual(len(shared), 30)
            for k in shared:
                self.assertTrue(np.isclose(m[k], old[keymap.get(k, k)], rtol=1e-9, atol=1e-12), (name, k))
            for (d, t), v in self.cache["idf"][name].items():
                self.assertTrue(np.isclose(m[f"idf_{d}_T{t}"], v, rtol=1e-9), (name, d, t))

    def test_fixed_bands(self):
        rename = {"T4 IDF (all cells)": "T4 IDF (all 15 cells)", "S1 deep storms": "S1 storms >= 10 mm",
                  "S3 top storm depths": "S3 top-10 storm depths"}
        for name, a in self.series().items():
            sc = checks.score(M.hydro(a, self.R, self.P), self.R, "fixed")
            for c in checks.GATE_A_18:
                self.assertEqual(sc[c]["pass"], self.cache["gate"][name][rename.get(c, c)], (name, c))
            for c in checks.TIER_6:
                self.assertEqual(sc[c]["pass"], self.cache["tier6"][name][rename.get(c, c)], (name, c))


@unittest.skipUnless(HAVE_RECORD and HOME.series_to_score()["lgar"].exists(), "latent Gaussian AR series not present")
class BaselinePort(unittest.TestCase):
    def test_identical_series(self):
        x, _ = HOME.select_years(HOME.years("train"), clean=False)
        sim = baselines.generate(baselines.fit(x, HOME.tau, 64), 10, 42)
        old = pd.read_csv(HOME.series_to_score()["lgar"], usecols=["avg_rainfall"])["avg_rainfall"].to_numpy()
        np.testing.assert_array_equal(sim, old)


class DatasetMask(unittest.TestCase):
    """The opt-in mask in RegimeDataset changes nothing when off, and skips exactly the masked windows."""

    def test_mask(self):
        sys.path.insert(0, str(ROOT))
        from regime_training.regime_dataset import RegimeDataset
        rng = np.random.default_rng(0)
        n = 600
        df = pd.DataFrame({"avg_rainfall": np.where(rng.random(n) < 0.1, rng.random(n), 0.0),
                           "hmm_state": rng.integers(0, 4, n), "none": 0, "outage": 0})
        df.loc[100:130, "outage"] = 1
        df.loc[100:130, "avg_rainfall"] = 0.0
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.csv"
            df.to_csv(p, index=False)
            kw = dict(seq_len=64, regime_col="hmm_state", transform="asinh", asinh_scale=0.035)
            plain, zero = RegimeDataset(p, **kw), RegimeDataset(p, mask_col="none", **kw)
            masked = RegimeDataset(p, mask_col="outage", **kw)
        self.assertEqual(len(plain), len(zero))
        for i in (0, 100, len(plain) - 1):
            self.assertTrue(np.array_equal(plain[i][0].numpy(), zero[i][0].numpy()))
            self.assertEqual(int(plain[i][1]), int(zero[i][1]))
        self.assertEqual(len(masked), rules.usable_windows(df["outage"].to_numpy(), 64))
        self.assertEqual(masked.n_skipped, len(plain) - len(masked))
        fit = df.loc[df["outage"] == 0, "avg_rainfall"].to_numpy()
        self.assertAlmostEqual(float(masked.scaler._std.mean_[0]), float(np.arcsinh(fit / 0.035).mean()), places=5)


if __name__ == "__main__":
    unittest.main()
