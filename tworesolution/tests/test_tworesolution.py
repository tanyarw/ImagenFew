"""Tests for the two-resolution generator.

    python -m unittest discover -s tworesolution/tests -t .

The numpy tests run anywhere; the torch tests need torch (not on the cluster's login nodes: the GPU jobs run
them before training). Data-dependent tests skip when the split files are absent.
"""
import argparse
import json
import unittest

import numpy as np

from tworesolution import cascade
from tworesolution import data as D
from tworesolution import grid as G
from tworesolution.evaluate import Memorisation, join_continuation
from tworesolution.mass import finish

try:
    import torch
    HAVE_TORCH = True
except Exception:                       # login nodes cannot load libtorch
    HAVE_TORCH = False
HAVE_DATA = (D.SPLITS / "train_years_labelled.csv").exists()


def toy_rain(n30, seed=0, tau5=0.005, tau30=0.0075):
    """A consistent toy record: wet half-hours in runs, 5-minute values on a 0.0025 mm grid."""
    rng = np.random.default_rng(seed)
    wet = np.repeat(rng.random(n30 // 4) < 0.3, 4)[:n30]
    x5 = np.where(np.repeat(wet, 6) & (rng.random(6 * n30) < 0.6), 0.0025 * rng.integers(1, 40, 6 * n30), 0.0)
    return G.two_level(x5, tau5, tau30)


class Grid(unittest.TestCase):
    def test_aggregate_spread(self):
        x30 = np.array([0.0, 0.6, 1.2, 0.0075])
        self.assertTrue(np.allclose(G.aggregate(G.spread(x30)), x30))
        self.assertTrue(np.allclose(G.spread(x30)[6:12], 0.1))
        with self.assertRaises(ValueError):
            G.aggregate(np.zeros(7))

    def test_two_level_consistent(self):
        x5c, x30 = toy_rain(400)
        self.assertTrue(np.allclose(G.aggregate(x5c), x30, rtol=0, atol=1e-12))
        self.assertTrue((x30[x30 > 0] >= 0.0075).all())
        self.assertTrue((x5c.reshape(-1, 6)[x30 == 0] == 0).all())

    def test_generation_windows_tile_the_series_once(self):
        for n30 in (10, 23, 175_200):
            starts = G.generation_starts(n30)
            self.assertEqual(starts[0], -G.PAD)
            kept_steps = (starts[:, None] + G.PAD + np.arange(G.KEEP)).ravel()[:6 * n30]
            self.assertTrue(np.array_equal(kept_steps, np.arange(6 * n30)))
            self.assertTrue(((starts + G.PAD) % 6 == 0).all())

    def test_training_windows_have_the_generation_layout(self):
        s = G.train_starts(6 * 1000)
        self.assertTrue(((s + G.PAD) % 6 == 0).all() and s.min() >= 0 and s.max() + G.L <= 6000)
        self.assertEqual(np.diff(s).max(), 6)

    def test_condition_windows(self):
        x30 = np.arange(1, 25, dtype=float)
        c = G.condition_windows(x30, G.generation_starts(len(x30)))
        self.assertTrue(np.allclose(c[0, :2], 0))                        # before the series: dry
        self.assertTrue(np.allclose(c[0, 2:8], 1 / 6))                   # first kept half-hour
        self.assertTrue(np.allclose(c[0, 62:], 11 / 6))                  # the next window's first half-hour
        self.assertTrue(np.allclose(c[1, :2], 10 / 6))                   # the previous window's last half-hour
        self.assertTrue(np.allclose(c[-1, 2 + 6 * 4:], 0))               # after the series: dry
        raw = np.tile(np.arange(G.L, dtype=float), (len(c), 1))
        k = G.kept(raw, len(x30))
        self.assertEqual(k.shape, (24, 6))
        self.assertTrue(np.array_equal(k[0], np.arange(2, 8)) and np.array_equal(k[10], np.arange(2, 8)))

    def test_window_labels_match_regime_dataset(self):
        rng = np.random.default_rng(1)
        profile = np.repeat(rng.integers(0, 4, 50), 37)
        starts = rng.integers(-5, 3000, 200)
        got = G.window_labels(profile, starts)
        for s, g in zip(starts, got):
            w = profile[(s + np.arange(G.L)) % len(profile)]
            vals, counts = np.unique(w, return_counts=True)
            self.assertEqual(g, vals[np.argmax(counts)])


class MassMatching(unittest.TestCase):
    def test_conserves_and_no_drizzle(self):
        rng = np.random.default_rng(2)
        t = np.where(rng.random(5000) < 0.3, 0.0075 + rng.exponential(0.5, 5000), 0.0)
        raw = rng.normal(0.05, 0.2, (5000, 6))
        f, c = finish(raw, t, 0.005)
        self.assertTrue(np.allclose(f.sum(1), t, rtol=0, atol=1e-12))
        self.assertFalse(((f > 0) & (f < 0.005)).any())
        self.assertTrue((f[t == 0] == 0).all() and (f >= 0).all())
        self.assertEqual(c["wet_half_hours"], int((t > 0).sum()))

    def test_empty_goes_to_largest_raw_value(self):
        f, c = finish(np.array([[0.001, 0.004, -1, 0, 0.002, 0.0]]), np.array([0.3]), 0.005)
        self.assertTrue(np.allclose(f, [[0, 0.3, 0, 0, 0, 0]]) and c["empty"] == 1)

    def test_small_values_move_to_the_largest(self):
        f, c = finish(np.array([[1.0, 0.004, 0.006, 0, 0, 0]]), np.array([0.5]), 0.005)
        self.assertAlmostEqual(f.sum(), 0.5)
        self.assertEqual(int((f > 0).sum()), 1)                           # 0.006 scaled to 0.003 -> moved
        self.assertEqual(c["moved_values"], 1)

    def test_rejects_totals_below_the_threshold(self):
        with self.assertRaises(ValueError):
            finish(np.ones((1, 6)), np.array([0.003]), 0.005)


class Cascade(unittest.TestCase):
    def test_positions(self):
        v = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0])
        self.assertEqual(cascade.position_class(v)[[0, 2, 3, 5, 6, 7]].tolist(), [0, 1, 3, 1, 2, 3])

    def test_conserves_and_is_reproducible(self):
        x5c, x30 = toy_rain(4000)
        levels = cascade.fit(x5c)
        a = cascade.disaggregate(levels, x30, seed=3)
        self.assertTrue(np.allclose(a.sum(1), x30, rtol=0, atol=1e-12))
        self.assertTrue((a >= 0).all())
        self.assertTrue(np.array_equal(a, cascade.disaggregate(levels, x30, seed=3)))
        self.assertFalse(np.array_equal(a, cascade.disaggregate(levels, x30, seed=4)))

    def test_weights_are_observed_splits(self):
        x5c, x30 = toy_rain(4000)
        lv = cascade.fit(x5c)[0]
        W = np.concatenate([w for w in lv.weights.values() if len(w)])
        self.assertTrue(np.allclose(W.sum(1), 1))
        self.assertEqual(len(W), int((x30 > 0).sum()))


class Diagnostics(unittest.TestCase):
    def test_join_continuation(self):
        x = np.ones(6000)
        x[np.arange(60, 6000, 60)] = 0                                   # every join starts dry
        j = join_continuation(x)
        self.assertEqual(j["at_joins"], 0.0)
        self.assertEqual(j["at_other_half_hours"], 1.0)

    def test_memorisation_flags_copies(self):
        _, x30 = toy_rain(64 * 300, seed=5)
        _, held = toy_rain(64 * 100, seed=6)
        mem = Memorisation(x30, held)
        self.assertEqual(mem.score(x30)["too_close_pct"], 100.0)
        self.assertLess(mem.score(toy_rain(64 * 200, seed=7)[1])["too_close_pct"], 20.0)

    def test_resampled_bands_default_unchanged(self):
        from sitetransfer import checks as C
        from sitetransfer import metrics as M
        rng = np.random.default_rng(0)
        x = np.where(rng.random(3 * G.STEPS_PER_YEAR_5) < 0.09, rng.exponential(0.1, 3 * G.STEPS_PER_YEAR_5), 0.0)
        m = np.zeros(len(x), bool)
        a = C.resampled_bands(x, m, M.Params(10.0), n_draws=3, n_years=2)
        b = C.resampled_bands(x, m, M.Params(10.0), n_draws=3, n_years=2, n_ref=3)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))     # NaN-safe


@unittest.skipUnless(HAVE_DATA, "split files not present")
class Data(unittest.TestCase):
    def test_rules_give_the_planned_settings(self):
        d = D.derive()
        self.assertEqual(d["tau30_mm"], 0.0075)
        self.assertAlmostEqual(d["asinh_scale30_mm"], 0.1025)
        self.assertEqual(d["coarse_windows"], 140_097)


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class Network(unittest.TestCase):
    """The condition path changes nothing while it is zero, and learns once trained."""

    @classmethod
    def setUpClass(cls):
        from models.ImagenFew.ImagenFew import ImagenFew
        from tworesolution.network import attach_condition, load_args
        torch.manual_seed(0)
        cls.args = load_args(D.V14_CONFIG, "cpu")
        cls.plain = ImagenFew(cls.args, "cpu").eval()
        cls.fine = attach_condition(ImagenFew(argparse.Namespace(**vars(cls.args)), "cpu")).eval()
        cls.fine.net.model.unet.load_state_dict(cls.plain.net.model.state_dict())
        g = torch.Generator().manual_seed(1)
        cls.x = torch.randn(5, 1, 8, 8, generator=g)
        cls.c = torch.randn(5, 1, 8, 8, generator=g)
        cls.lab = torch.eye(4)[[0, 1, 2, 3, 0]]

    def test_zero_condition_layer_is_the_plain_network(self):
        s = torch.tensor([0.01, 0.3, 1.0, 5.0, 80.0]).view(-1, 1, 1, 1)
        with torch.no_grad():
            self.assertTrue(torch.equal(self.plain.net(self.x, s, self.lab), self.fine.net(self.x, s, self.lab, cond=self.c)))

    def test_sampler_is_production(self):
        from models.ImagenFew.sampler import DiffusionProcess
        from tworesolution.sampling import heun, schedule
        args = argparse.Namespace(**vars(self.args))
        proc = DiffusionProcess(args, self.plain.net, (1, 8, 8))
        zeros = torch.zeros_like(self.x)
        with torch.no_grad():
            a = proc.impute(zeros, self.x, zeros, class_labels=self.lab)
        b = heun(self.plain.net, self.x, self.lab, schedule(self.plain.net, args.diffusion_steps, "cpu"))
        c = heun(self.fine.net, self.x, self.lab, schedule(self.fine.net, args.diffusion_steps, "cpu"), cond=self.c)
        self.assertTrue(torch.equal(a, b) and torch.equal(b, c))

    def test_training_forward_is_imagenfew_forward(self):
        from tworesolution.train_fine import edm_forward
        self.plain.train()
        self.fine.train()
        try:
            mask = self.plain.ts_to_img(torch.zeros(5, 64, 1), pad_val=1)
            self.assertEqual(float(mask.abs().sum()), 0.0)                      # 8 x 8: nothing padded, nothing known
            torch.manual_seed(7)
            a, wa = self.plain(self.x, mask, labels=self.lab)
            torch.manual_seed(7)
            b, wb = edm_forward(self.fine, self.x, self.c, self.lab)
            self.assertTrue(torch.equal(a, b) and torch.equal(wa, wb))
        finally:
            self.plain.eval()
            self.fine.eval()

    def test_condition_layer_learns(self):
        from tworesolution.network import attach_condition
        from models.ImagenFew.ImagenFew import ImagenFew
        m = attach_condition(ImagenFew(argparse.Namespace(**vars(self.args)), "cpu")).train()
        with torch.no_grad():                    # a new U-Net's output layer starts at zero (v14's is trained)
            m.net.model.unet.out_conv.dynamic_weights.normal_()
        s = torch.full((5, 1, 1, 1), 0.5)
        loss = (m.net(self.x, s, self.lab, cond=self.c) - self.x).square().mean()
        loss.backward()
        self.assertGreater(float(m.net.model.cond_conv.weight.grad.abs().sum()), 0.0)

    def test_save_and_load(self):
        import tempfile
        from pathlib import Path
        from tworesolution.network import load_fine, save_fine
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.pt"
            with torch.no_grad():
                self.fine.net.model.cond_conv.weight.normal_()
            try:
                save_fine(self.fine, p)
                back = load_fine(argparse.Namespace(**vars(self.args)), p, "cpu")
                s = torch.full((5, 1, 1, 1), 0.5)
                with torch.no_grad():
                    self.assertTrue(torch.equal(self.fine.net(self.x, s, self.lab, cond=self.c),
                                                back.net(self.x, s, self.lab, cond=self.c)))
            finally:
                with torch.no_grad():
                    self.fine.net.model.cond_conv.weight.zero_()


if __name__ == "__main__":
    unittest.main()
