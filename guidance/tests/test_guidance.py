"""Tests for the guided sampler.

    python -m unittest discover -s guidance/tests -t .
"""
import unittest

import numpy as np
import torch

from guidance import pilot as P
from guidance.sampler import check_weights, guided_denoise, sample_piece

HAVE_V14 = (P.ROOT / "logs" / "ImagenFew" / "Rainfall_Regime" / "v14" / "best_regime_model.pt").exists()


class Weights(unittest.TestCase):
    def test_versions(self):
        x = torch.tensor([[[[0.5, -1.0], [-1.0, -1.0]]], [[[-1.0, -1.0], [-1.0, -1.0]]]], dtype=torch.float64)
        m = torch.ones_like(x)
        w, a = check_weights(x, m, z_wet=0.0, variant="plain", wet_weight=10)
        self.assertTrue(torch.equal(w, torch.ones_like(x)) and bool(a.all()))
        w, a = check_weights(x, m, z_wet=0.0, variant="wet", wet_weight=10)
        self.assertAlmostEqual(float(w[0, 0, 0, 0] / w[0, 0, 0, 1]), 10.0)      # wet counts 10x a dry step
        self.assertAlmostEqual(float(w[0].mean()), 1.0)                           # mean weight stays 1
        self.assertTrue(torch.allclose(w[1], torch.ones_like(w[1])))             # all-dry context: weight 1
        w, a = check_weights(x, m, z_wet=0.0, variant="rainy", wet_weight=10)
        self.assertEqual(a.tolist(), [True, False])          # the dry context is not guided

    def test_rules(self):
        ref = {"wet_share_pct": 9.0, "mm_per_year": 700.0, "p999_mm": 1.5}
        ok = {"continuation": 0.75, "wet_share_pct": 9.5, "mm_per_year": 720.0, "p999_mm": 1.6}
        self.assertTrue(all(P.judge(ok, ref).values()))
        self.assertFalse(P.judge(ok | {"mm_per_year": 600.0}, ref)["volume within 10%"])
        res = {"a": {"w": 0.1, "variant": "plain", "pass": {"x": True}},
               "b": {"w": 0.1, "variant": "rainy", "pass": {"x": True}},
               "c": {"w": 0.05, "variant": "wet", "pass": {"x": False}}}
        self.assertEqual(P.choose(res), "b")


@unittest.skipUnless(HAVE_V14, "v14 checkpoint not present")
class AgainstExistingSampler(unittest.TestCase):
    """With w = 0 the guided sampler is exactly regime_training/generate_context.sample_block (E2, no resampling)."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.scaler, cls.net, cls.t = P.load_v14("cpu")

    def test_w0_identical(self):
        import regime_training.generate_context as gc
        M = 3
        known = torch.zeros(M, 64, 1, dtype=torch.float64)
        known[:, :16, 0] = torch.linspace(-0.3, 1.5, 16)
        x_known = self.model.ts_to_img(known).to(torch.float64)
        mk = torch.zeros(M, 64, 1)
        mk[:, :16] = 1
        mask = self.model.ts_to_img(mk).to(torch.float64)
        labels = torch.eye(4)[[0, 1, 3]]
        with self.model.ema_scope():
            torch.manual_seed(5)
            a = gc.sample_block(self.net, x_known, mask, labels, self.t, [1] * (len(self.t) - 1))
            torch.manual_seed(5)
            b = sample_piece(self.net, x_known, mask, labels, self.t, w=0.0)
        self.assertTrue(torch.equal(a, b))

    def test_guidance_moves_only_the_unknown_part(self):
        M = 2
        x = torch.randn(M, 1, 8, 8, dtype=torch.float64)
        ctx = torch.randn_like(x)
        mask = torch.zeros_like(x)
        mask[..., :2] = 1
        labels = torch.eye(4)[[0, 0]]
        s = torch.tensor(2.0, dtype=torch.float64)
        with self.model.ema_scope():
            d0, _ = guided_denoise(self.net, x, s, labels, ctx, mask, torch.ones_like(x), 0.0, torch.ones(M, dtype=torch.bool))
            d1, rel = guided_denoise(self.net, x, s, labels, ctx, mask, torch.ones_like(x), 0.1, torch.ones(M, dtype=torch.bool))
            d2, _ = guided_denoise(self.net, x, s, labels, ctx, mask, torch.ones_like(x), 0.1, torch.tensor([True, False]))
        self.assertTrue(torch.allclose(d0 * mask, d1 * mask))          # known part untouched
        self.assertGreater(rel, 0)
        self.assertFalse(torch.allclose(d0, d1))
        self.assertTrue(torch.allclose(d0[1], d2[1]))                   # inactive chain untouched


if __name__ == "__main__":
    unittest.main()
