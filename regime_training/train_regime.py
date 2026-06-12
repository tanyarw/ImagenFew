"""
GMM Regime-Conditional Training Script
=======================================
Fine-tunes the ImagenFew diffusion model to condition on gmm_regime labels
(0-3) instead of dataset-level class indices.

Usage
-----
  python regime_training/train_regime.py \\
      --model_ckpt ./logs/ImagenFew/Rainfall/<run_id>/best_model.pt

Optional overrides:
  --epochs 500  --batch_size 1024  --learning_rate 5e-5
"""

import os
import sys

# ── project root on the import path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import logging
import pickle
import uuid

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from regime_training.regime_dataset import RegimeDataset
from models.ImagenFew.ImagenFew import ImagenFew
from models.ImagenFew.sampler import DiffusionProcess

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GMM Regime Conditional Training")
    p.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to the fine-tuned Rainfall checkpoint")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                   help="Path to regime config YAML")
    p.add_argument("--run_id", type=str, default=str(uuid.uuid4())[:8],
                   help="Unique run identifier (short UUID by default)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    return p.parse_args()


def load_config(config_path, cli):
    """Merge YAML config with CLI overrides into a single Namespace."""
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    for key in ("epochs", "batch_size", "learning_rate"):
        cli_val = getattr(cli, key, None)
        if cli_val is not None:
            cfg[key] = cli_val
    cfg["model_ckpt"] = cli.model_ckpt
    cfg["run_id"] = cli.run_id
    return argparse.Namespace(**cfg)

# ──────────────────────────────────────────────────────────────────────
# Checkpoint loading (shape-safe)
# ──────────────────────────────────────────────────────────────────────

def load_checkpoint_safe(model, ckpt_path, device):
    """
    Load weights from a checkpoint while gracefully skipping any
    parameters whose shapes don't match (e.g. the map_label layer
    changes from 43 → 4 classes).
    """
    if not os.path.exists(ckpt_path):
        logging.warning("No checkpoint at %s — using random init.", ckpt_path)
        return

    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_state = loaded.get("model", loaded)       # handle both dict styles
    cur_state = model.state_dict()

    matched, skipped = 0, 0
    filtered = {}
    for k, v in ckpt_state.items():
        if k in cur_state and v.shape == cur_state[k].shape:
            filtered[k] = v
            matched += 1
        else:
            shape_info = (
                f"ckpt={list(v.shape)} vs model={list(cur_state[k].shape)}"
                if k in cur_state else "key not in model"
            )
            logging.warning("  Skipping %-60s  (%s)", k, shape_info)
            skipped += 1

    model.load_state_dict(filtered, strict=False)
    logging.info("Loaded %d parameters, skipped %d", matched, skipped)

    # EMA weights
    if "ema_model" in loaded and hasattr(model, "model_ema"):
        ema_ckpt = loaded["ema_model"]
        ema_cur = model.model_ema.state_dict()
        ema_filtered = {
            k: v for k, v in ema_ckpt.items()
            if k in ema_cur and v.shape == ema_cur[k].shape
        }
        model.model_ema.load_state_dict(ema_filtered, strict=False)
        logging.info("Loaded EMA weights (%d params)", len(ema_filtered))

# ──────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_regime(model, args, regime, n_samples=100, channels=1):
    """Generate a small batch for a single regime and return basic stats."""
    model.eval()
    process = DiffusionProcess(
        args, model.net,
        (channels, args.img_resolution, args.img_resolution),
    )
    generated = []
    with model.ema_scope():
        for i in range(0, n_samples, args.batch_size):
            b = min(args.batch_size, n_samples - i)
            cls = torch.full((b,), regime, device=args.device, dtype=torch.long)
            oh = nn.functional.one_hot(cls, num_classes=args.n_classes).float()
            x_img = torch.zeros(b, channels, args.img_resolution,
                                args.img_resolution, device=args.device)
            mask = model.ts_to_img(
                torch.zeros(b, args.seq_len, channels, device=args.device),
                pad_val=1,
            )
            sampled = process.interpolate(x_img, mask, class_labels=oh)
            ts = model.img_to_ts(sampled)[:, :, :channels]
            generated.append(ts.cpu())
    return torch.cat(generated, dim=0).numpy()

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    args = load_config(cli.config, cli)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("═" * 60)
    logging.info("GMM Regime-Conditional Training")
    logging.info("═" * 60)
    logging.info("Run ID  : %s", args.run_id)
    logging.info("Device  : %s", args.device)
    logging.info("Epochs  : %d  |  Batch : %d  |  LR : %s",
                 args.epochs, args.batch_size, args.learning_rate)

    # ── Data ──────────────────────────────────────────────────────────
    train_csv = os.path.join(PROJECT_ROOT, args.train_csv)
    test_csv = os.path.join(PROJECT_ROOT, args.test_csv)

    train_ds = RegimeDataset(train_csv, seq_len=args.seq_len)
    test_ds = RegimeDataset(test_csv, seq_len=args.seq_len,
                            scaler=train_ds.scaler)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )

    regime_counts = train_ds.get_regime_counts()
    logging.info("Train samples : %d  |  Test samples : %d", len(train_ds), len(test_ds))
    logging.info("Regime dist   : %s", regime_counts)

    # ── Model ─────────────────────────────────────────────────────────
    logging.info("Building model (n_classes=%d) …", args.n_classes)
    model = ImagenFew(args, args.device).to(args.device)

    logging.info("Loading base checkpoint: %s", args.model_ckpt)
    load_checkpoint_safe(model, args.model_ckpt, args.device)

    # Re-initialise EMA from the (partially-loaded) model weights
    if args.ema:
        model.setup_finetune(args)

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ── Output dir ────────────────────────────────────────────────────
    ckpt_dir = os.path.join(
        PROJECT_ROOT, "logs", "ImagenFew", "Rainfall_Regime", args.run_id,
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    logging.info("Checkpoints → %s", ckpt_dir)

    # Persist the scaler so generation can inverse-transform later
    scaler_path = os.path.join(ckpt_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(train_ds.scaler, f)

    # Persist regime proportions so the generator doesn't need hardcoded values
    total = sum(regime_counts.values())
    regime_proportions = {r: c / total for r, c in regime_counts.items()}
    props_path = os.path.join(ckpt_dir, "regime_proportions.pkl")
    with open(props_path, "wb") as f:
        pickle.dump(regime_proportions, f)
    logging.info("Regime proportions: %s", regime_proportions)

    # ── Training loop ─────────────────────────────────────────────────
    best_loss = float("inf")
    channels = 1

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for x_ts, regime_labels in train_loader:
            x_ts = x_ts.to(args.device)                    # [B, seq_len, 1]
            regime_labels = regime_labels.to(args.device)   # [B]

            optimizer.zero_grad()

            # Convert to image representation (delay embedding)
            x_ts_mask = torch.zeros_like(x_ts)
            x_img = model.ts_to_img(x_ts)
            x_img_mask = model.ts_to_img(x_ts_mask, pad_val=1)

            # One-hot regime label → conditioning signal
            labels = nn.functional.one_hot(
                regime_labels, num_classes=args.n_classes,
            ).float()

            # Forward (EDM denoising)
            output, weight = model(x_img, x_img_mask, labels=labels)

            # Weighted MSE + FFT loss on the signal region only
            # (matches ImagenFew.loss_fn for consistency)
            fft_x = torch.fft.fft2(x_img, norm='forward')
            fft_output = torch.fft.fft2(output, norm='forward')
            fft_loss = (
                (torch.real(fft_output) - torch.real(fft_x)).square()
                + (torch.imag(fft_output) - torch.imag(fft_x)).square()
            )
            time_loss = (output - x_img).square()
            loss = (weight * (time_loss + fft_loss) * (1 - x_img_mask)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.on_train_batch_end()          # EMA update

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)

        # ── Periodic logging & evaluation ─────────────────────────────
        if epoch == 1 or epoch % args.logging_iter == 0:
            logging.info(
                "Epoch %4d/%d  |  loss = %.6f", epoch, args.epochs, avg_loss,
            )

            # Quick per-regime generation check
            for r in range(args.n_classes):
                gen = evaluate_regime(model, args, r, n_samples=50, channels=channels)
                logging.info(
                    "  Regime %d  →  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                    r, gen.mean(), gen.std(), gen.min(), gen.max(),
                )
            model.train()

        # ── Save best checkpoint ──────────────────────────────────────
        if epoch % args.logging_iter == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            _save_ckpt(model, args, epoch, avg_loss,
                       os.path.join(ckpt_dir, "best_regime_model.pt"))

    # ── Final save ────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final_regime_model.pt")
    _save_ckpt(model, args, args.epochs, avg_loss, final_path)
    logging.info("Training complete.  Final → %s", final_path)


def _save_ckpt(model, args, epoch, loss, path):
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "n_classes": args.n_classes,
        "seq_len": args.seq_len,
    }
    if args.ema:
        state["ema_model"] = model.model_ema.state_dict()
    torch.save(state, path)
    logging.info("  Saved checkpoint (epoch %d, loss %.6f) → %s", epoch, loss, path)


if __name__ == "__main__":
    main()
