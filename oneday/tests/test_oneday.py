"""Tests for v19 (one-day window).

    python -m unittest discover -s oneday/tests -t .
"""
import argparse
import unittest

import numpy as np

from oneday import config as K
from sitetransfer.training import diff_keys

try:
    import torch
    HAVE_TORCH = True
except Exception:                       # login nodes cannot load libtorch
    HAVE_TORCH = False


class Config(unittest.TestCase):
    def test_only_allowed_keys_change(self):
        base, cfg = K.build(500, 2048)
        self.assertEqual(set(diff_keys(base, cfg)), set(K.WINDOW) | {"run_dir"})
        _, cfg2 = K.build(300, 1024)
        self.assertEqual(set(diff_keys(base, cfg2)), set(K.WINDOW) | {"run_dir", "epochs", "batch_size"})

    def test_window_is_one_day(self):
        _, cfg = K.build()
        self.assertEqual(cfg["seq_len"], 288)
        self.assertEqual((cfg["seq_len"] - cfg["embedding"]) // cfg["delay"] + 1, 12)    # 12 columns of 24 rows


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class Network(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.ImagenFew.ImagenFew import ImagenFew
        _, cfg = K.build()
        cls.args = argparse.Namespace(**cfg)
        cls.args.device = "cpu"
        cls.model = ImagenFew(cls.args, "cpu")

    def test_delay_embedding_round_trip_and_padding(self):
        x = torch.randn(3, 288, 1)
        img = self.model.ts_to_img(x)
        self.assertEqual(tuple(img.shape), (3, 1, 24, 24))
        self.assertTrue(torch.equal(self.model.img_to_ts(img)[:, :, :1], x))
        mask = self.model.ts_to_img(torch.zeros(3, 288, 1), pad_val=1)
        self.assertEqual(int(mask[0].sum()), 24 * 24 - 288)        # the padding is known, never scored
        self.assertEqual(float(mask[0, 0, :, :12].sum()), 0.0)     # the 12 data columns are all unknown

    @unittest.skipUnless(K.CHECKPOINT.exists(), "pretrained checkpoint not present")
    def test_checkpoint_matches_except_the_label_layer(self):
        ck = torch.load(K.CHECKPOINT, map_location="cpu", weights_only=False)["model"]
        cur = self.model.state_dict()
        mismatched = sorted(k for k, v in ck.items() if k not in cur or v.shape != cur[k].shape)
        missing = sorted(k for k in cur if k not in ck)
        self.assertEqual(mismatched, ["model_ema.modelmap_labelweight", "net.model.map_label.weight"])  # 32 -> 4
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
