"""
Regime-Conditional Synthetic Rainfall Generator
================================================
Generates synthetic rainfall data conditioned on gmm_regime labels
using a regime-trained ImagenFew checkpoint.

Usage
-----
  # Mixed mode — proportional regime sampling (default)
  python regime_training/generate_regime.py \\
      --model_ckpt logs/ImagenFew/Rainfall_Regime/<run_id>/best_regime_model.pt \\
      --scaler_path logs/ImagenFew/Rainfall_Regime/<run_id>/scaler.pkl \\
      --years 10

  # Single-regime mode — e.g. only heavy-storm (regime 3)
  python regime_training/generate_regime.py \\
      --model_ckpt ... --scaler_path ... --years 5 --regime 3
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.ImagenFew.ImagenFew import ImagenFew
from models.ImagenFew.sampler import DiffusionProcess

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Regime-Conditional Rainfall Generator")
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to the regime-trained checkpoint")
    p.add_argument("--scaler_path", type=str, required=True,
                   help="Path to scaler.pkl saved during training")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    p.add_argument("--years", type=float, default=10.0,
                   help="Years of synthetic data to generate")
    p.add_argument("--regime", type=int, default=None, choices=[0, 1, 2, 3],
                   help="Generate only this regime. If omitted, mix proportionally.")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "results", "generated_data"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    # Config → Namespace
    cfg = OmegaConf.to_container(OmegaConf.load(cli.config), resolve=True)
    args = argparse.Namespace(**cfg)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model_ckpt = cli.model_ckpt

    # ── Scaler ────────────────────────────────────────────────────────
    with open(cli.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logging.info("Loaded scaler from %s", cli.scaler_path)

    # ── Model ─────────────────────────────────────────────────────────
    logging.info("Building model (n_classes=%d) …", args.n_classes)
    model = ImagenFew(args, args.device).to(args.device)

    loaded = torch.load(cli.model_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(loaded["model"], strict=True)
    if "ema_model" in loaded and args.ema:
        model.model_ema.load_state_dict(loaded["ema_model"], strict=True)
    model.eval()
    logging.info("Model loaded from %s", cli.model_ckpt)

    # ── How many samples? ─────────────────────────────────────────────
    freq = "10min"
    steps_per_year = len(
        pd.date_range(start="2026-01-01", end="2027-01-01",
                      freq=freq, inclusive="left")
    )
    total_steps = int(cli.years * steps_per_year)
    num_samples = int(np.ceil(total_steps / args.seq_len))

    # ── Regime proportions (computed from training data) ────────────
    props_path = os.path.join(os.path.dirname(cli.scaler_path), "regime_proportions.pkl")
    if os.path.exists(props_path):
        with open(props_path, "rb") as f:
            regime_proportions = pickle.load(f)
        logging.info("Loaded regime proportions from %s: %s", props_path, regime_proportions)
    else:
        # Fallback to hardcoded values if pkl not found (legacy checkpoints)
        regime_proportions = {0: 0.255, 1: 0.370, 2: 0.279, 3: 0.096}
        logging.warning(
            "regime_proportions.pkl not found at %s — using hardcoded fallback: %s",
            props_path, regime_proportions,
        )

    if cli.regime is not None:
        regime_alloc = {cli.regime: num_samples}
        tag = f"regime{cli.regime}"
    else:
        regime_alloc = {
            r: max(1, int(p * num_samples))
            for r, p in regime_proportions.items()
        }
        # Reconcile rounding to hit the target total
        diff = num_samples - sum(regime_alloc.values())
        # add surplus to the most-common regime
        most_common = max(regime_proportions, key=regime_proportions.get)
        regime_alloc[most_common] += diff
        tag = "mixed"

    logging.info("Generating %.1f years  (%d steps, %d samples)",
                 cli.years, total_steps, num_samples)
    logging.info("Regime allocation: %s", regime_alloc)

    # ── Generate ──────────────────────────────────────────────────────
    channels = 1
    process = DiffusionProcess(
        args, model.net,
        (channels, args.img_resolution, args.img_resolution),
    )
    all_generated = []

    with model.ema_scope():
        for regime, n in regime_alloc.items():
            logging.info("  Regime %d  →  %d samples …", regime, n)
            for i in range(0, n, args.batch_size):
                b = min(args.batch_size, n - i)
                cls = torch.full((b,), regime, device=args.device, dtype=torch.long)
                oh = nn.functional.one_hot(cls, num_classes=args.n_classes).float()

                x_img = torch.zeros(
                    b, channels, args.img_resolution, args.img_resolution,
                    device=args.device,
                )
                mask = model.ts_to_img(
                    torch.zeros(b, args.seq_len, channels, device=args.device),
                    pad_val=1,
                )
                with torch.no_grad():
                    sampled = process.interpolate(x_img, mask, class_labels=oh)
                    ts = model.img_to_ts(sampled)[:, :, :channels]
                    all_generated.append(ts.cpu().numpy())

    generated = np.concatenate(all_generated, axis=0)        # [N, seq_len, 1]

    # Shuffle blocks if mixed mode (interleave regimes)
    if cli.regime is None:
        rng = np.random.default_rng(cli.seed)
        generated = generated[rng.permutation(len(generated))]

    # Flatten to a continuous 1-D series and trim to exact length
    continuous = generated.reshape(-1, channels)[:total_steps]

    # Inverse-scale
    unscaled = scaler.inverse_transform(continuous)

    # Hard zero threshold (matching generate_dataset.py)
    unscaled[unscaled < 0.005] = 0.0

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(cli.output_dir, exist_ok=True)
    run_id = os.path.splitext(os.path.basename(cli.model_ckpt))[0]
    out_path = os.path.join(
        cli.output_dir,
        f"rainfall_regime_{tag}_{cli.years}y_{run_id}.csv",
    )

    df = pd.DataFrame(unscaled, columns=["avg_rainfall"])
    df.insert(0, "date", pd.date_range(
        start="2026-01-01", periods=len(df), freq=freq,
    ))
    df.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────
    zero_frac = (df["avg_rainfall"] == 0).mean()
    nz = df.loc[df["avg_rainfall"] > 0, "avg_rainfall"]
    logging.info("═" * 50)
    logging.info("Saved → %s", out_path)
    logging.info("Shape : %s", df.shape)
    logging.info("Zero%% : %.2f%%", zero_frac * 100)
    logging.info("Mean  : %.4f  (non-zero: %.4f)", df["avg_rainfall"].mean(),
                 nz.mean() if len(nz) else 0.0)
    logging.info("Max   : %.4f", df["avg_rainfall"].max())
    logging.info("═" * 50)


if __name__ == "__main__":
    main()
