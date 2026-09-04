"""
Regime-Conditional Training Script
====================================
Fine-tunes the ImagenFew diffusion model to condition on regime/state labels
(0-3) instead of dataset-level class indices.

Supports both:
  - GMM regimes: config.yaml  (regime_col=gmm_regime)
  - HMM states:  config_hmm.yaml  (regime_col=hmm_state)

Usage
-----
  # HMM states (recommended — use the run script for the full pipeline):
  bash scripts/run_hmm_training.sh              # 105120-fit (default)
  bash scripts/run_hmm_training.sh 365          # 365-fit variant

  # Or run directly:
  python regime_training/train_regime.py \\
      --model_ckpt models_ckpt/ImagenFew/ImagenFew_24.ckpt \\
      --config regime_training/config_hmm.yaml

  # GMM regimes (original):
  python regime_training/train_regime.py \\
      --model_ckpt ./logs/ImagenFew/Rainfall/<run_id>/best_model.pt \\
      --config regime_training/config.yaml

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
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Heavy-tail loss reweighting
# ──────────────────────────────────────────────────────────────────────

def intensity_weights(x_img, alpha, gamma, normalize=True):
    """
    Per-pixel weight w(x) = 1 + alpha * relu(x)^gamma, used to stop the plain
    L2 denoising objective from under-fitting extreme rainfall.

    `x_img` is StandardScaler-normalised, so a dry step sits at a small
    NEGATIVE z (about -0.15 for Astlingen) and only genuine above-mean
    rainfall is positive.  Taking relu() therefore leaves the ~91% dry mass at
    weight 1.0 and boosts only the wet tail, which is exactly the mass the
    unweighted objective spends almost no expected loss on.

    gamma < 1 is deliberate: raw z reaches ~120 at the 10-yr maximum, so a
    linear weight would let a single cloudburst dominate an entire batch.

    With `normalize`, weights are rescaled to mean 1.0 over the batch so the
    overall loss magnitude — and therefore the usable learning rate and the
    grad-clip threshold — stays comparable to the alpha=0 baseline.
    """
    if alpha <= 0.0:
        return torch.ones_like(x_img)
    w = 1.0 + alpha * torch.relu(x_img).pow(gamma)
    if normalize:
        w = w / w.mean().clamp_min(1e-8)
    return w


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

    train_ds = RegimeDataset(
        train_csv, seq_len=args.seq_len,
        regime_col=getattr(args, "regime_col", "gmm_regime"),
        data_col=getattr(args, "data_col", "avg_rainfall"),
    )
    test_ds = RegimeDataset(
        test_csv, seq_len=args.seq_len,
        scaler=train_ds.scaler,
        regime_col=getattr(args, "regime_col", "gmm_regime"),
        data_col=getattr(args, "data_col", "avg_rainfall"),
    )

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

    # ── Loss / optimisation knobs ─────────────────────────────────────
    # Defaults reproduce the v5–v7 runs exactly (alpha=0 → all weights 1.0,
    # grad_clip=1.0), so any change here is opt-in from the config.
    tail_alpha = float(getattr(args, "intensity_weight_alpha", 0.0))
    tail_gamma = float(getattr(args, "intensity_weight_gamma", 0.5))
    tail_normalize = bool(getattr(args, "intensity_weight_normalize", True))
    grad_clip = float(getattr(args, "grad_clip", 1.0))
    fft_weight = float(getattr(args, "fft_weight", 1.0))

    logging.info(
        "Loss    : EDM-weighted (time + %.3g*FFT)  |  tail w(x)=1+%.3g*relu(x)^%.3g "
        "(normalize=%s)  |  grad_clip=%.3g",
        fft_weight, tail_alpha, tail_gamma, tail_normalize, grad_clip,
    )
    if tail_alpha <= 0:
        logging.info("          tail reweighting DISABLED (alpha=0) — baseline objective")

    # ── Training loop ─────────────────────────────────────────────────
    best_loss = float("inf")
    channels = 1
    history = []
    log_interval = getattr(args, "logging_iter", 25)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_time_losses = []
        epoch_fft_losses = []
        epoch_grad_norms = []
        epoch_clip_frac = []

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
            fft_x = torch.fft.fft2(x_img, norm='forward')
            fft_output = torch.fft.fft2(output, norm='forward')
            fft_loss = (
                (torch.real(fft_output) - torch.real(fft_x)).square()
                + (torch.imag(fft_output) - torch.imag(fft_x)).square()
            )
            time_loss = (output - x_img).square()

            # Heavy-tail reweighting applies to the TIME term only. The FFT
            # term lives in frequency space, where a per-pixel intensity
            # weight has no meaningful counterpart.
            w_int = intensity_weights(x_img, tail_alpha, tail_gamma, tail_normalize)

            signal = 1 - x_img_mask
            loss = (weight * (w_int * time_loss + fft_weight * fft_loss) * signal).mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.on_train_batch_end()          # EMA update

            epoch_losses.append(loss.item())
            epoch_time_losses.append((weight * time_loss * signal).mean().item())
            epoch_fft_losses.append((weight * fft_loss * signal).mean().item())
            epoch_grad_norms.append(float(grad_norm))
            epoch_clip_frac.append(1.0 if float(grad_norm) > grad_clip else 0.0)

        avg_loss = float(np.mean(epoch_losses))
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Record epoch metrics
        epoch_record = {
            "epoch": epoch,
            "loss": avg_loss,
            "best_loss": best_loss,
            "time_loss": float(np.mean(epoch_time_losses)),
            "fft_loss": float(np.mean(epoch_fft_losses)),
            "grad_norm": float(np.mean(epoch_grad_norms)),
            # Fraction of steps whose gradient was truncated by the clip.
            # A high value means rare heavy-rain batches are being attenuated
            # relative to bulk dry batches — the suspected cause of the
            # flattened extremes in v5/v6.
            "clip_frac": float(np.mean(epoch_clip_frac)),
        }

        # ── Periodic logging & evaluation ─────────────────────────────
        if epoch == 1 or epoch % log_interval == 0 or epoch == args.epochs:
            logging.info(
                "Epoch %4d/%d  |  loss = %.6f  (best = %.6f)  |  time = %.6f  "
                "fft = %.6f  |  grad = %.3f  clipped = %.0f%%",
                epoch, args.epochs, avg_loss, best_loss,
                epoch_record["time_loss"], epoch_record["fft_loss"],
                epoch_record["grad_norm"], 100 * epoch_record["clip_frac"],
            )

            # Quick per-regime generation check
            for r in range(args.n_classes):
                gen = evaluate_regime(model, args, r, n_samples=50, channels=channels)
                p999 = float(np.percentile(gen, 99.9))
                epoch_record[f"regime_{r}_mean"] = float(gen.mean())
                epoch_record[f"regime_{r}_std"] = float(gen.std())
                epoch_record[f"regime_{r}_max"] = float(gen.max())
                # Tail health: the mean can look right while the tail is dead,
                # which is exactly how v6 passed epoch-level checks yet capped
                # out at 1.79 mm against a real maximum of 5.56 mm.
                epoch_record[f"regime_{r}_p999"] = p999
                logging.info(
                    "  Regime %d  →  mean=%.4f  std=%.4f  min=%.4f  p99.9=%.4f  max=%.4f",
                    r, gen.mean(), gen.std(), gen.min(), p999, gen.max(),
                )
            model.train()

            # Save best checkpoint
            if is_best or (epoch % log_interval == 0 and avg_loss <= best_loss):
                _save_ckpt(model, args, epoch, avg_loss,
                           os.path.join(ckpt_dir, "best_regime_model.pt"))

        history.append(epoch_record)

        # ── Save tracking CSV and plots periodically ──────────────────
        if epoch % log_interval == 0 or epoch == args.epochs:
            history_df = pd.DataFrame(history)
            history_path = os.path.join(ckpt_dir, "history.csv")
            history_df.to_csv(history_path, index=False)
            save_training_plots(history_df, ckpt_dir)

    # ── Final save ────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final_regime_model.pt")
    _save_ckpt(model, args, args.epochs, avg_loss, final_path)
    logging.info("Training complete.  Final → %s", final_path)


def save_training_plots(df_hist, ckpt_dir):
    """Saves loss curve and regime convergence monitoring plots."""
    try:
        # Plot 1: Loss curve
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_hist["epoch"], df_hist["loss"], color="#1f77b4", linewidth=2.0, label="Training Loss (EDM + FFT)")
        ax.plot(df_hist["epoch"], df_hist["best_loss"], color="#2ca02c", linestyle="--", linewidth=1.5, label="Best Loss")
        best_row = df_hist.loc[df_hist["loss"].idxmin()]
        ax.scatter([best_row["epoch"]], [best_row["loss"]], color="red", s=60, zorder=5, 
                   label=f"Best: {best_row['loss']:.5f} (Ep {int(best_row['epoch'])})")
        ax.set_title("Training Loss Curve", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

        # Plot 2: Per-regime generated intensity separation
        regime_cols = [c for c in df_hist.columns if c.startswith("regime_") and c.endswith("_mean")]
        if regime_cols:
            eval_hist = df_hist.dropna(subset=regime_cols)
            if len(eval_hist) > 1:
                colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
                fig, ax = plt.subplots(figsize=(10, 5))
                for idx, col in enumerate(regime_cols):
                    r_num = col.split("_")[1]
                    c = colors[idx % len(colors)]
                    ax.plot(eval_hist["epoch"], eval_hist[col], color=c, marker="o", markersize=4, linewidth=1.8, label=f"State {r_num} Generated Mean")
                ax.set_title("Per-Regime Generated Intensity Convergence", fontsize=13, fontweight="bold")
                ax.set_xlabel("Epoch", fontsize=11)
                ax.set_ylabel("Generated Mean (Scaled)", fontsize=11)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(loc="best")
                plt.tight_layout()
                fig.savefig(os.path.join(ckpt_dir, "regime_convergence.png"), dpi=150)
                plt.close(fig)

        # Plot 3: Tail health + gradient clipping
        # The generated p99.9 is the early-warning signal for a collapsed tail;
        # clip_frac says how often the extremes' gradient is being truncated.
        tail_cols = [c for c in df_hist.columns if c.startswith("regime_") and c.endswith("_p999")]
        if tail_cols and "clip_frac" in df_hist.columns:
            eval_hist = df_hist.dropna(subset=tail_cols)
            if len(eval_hist) > 1:
                colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                for idx, col in enumerate(tail_cols):
                    r_num = col.split("_")[1]
                    ax1.plot(eval_hist["epoch"], eval_hist[col],
                             color=colors[idx % len(colors)], marker="o",
                             markersize=4, linewidth=1.8, label=f"State {r_num} p99.9")
                ax1.set_title("Generated Tail Health (p99.9, scaled)",
                              fontsize=13, fontweight="bold")
                ax1.set_ylabel("p99.9 (scaled)", fontsize=11)
                ax1.grid(True, linestyle="--", alpha=0.5)
                ax1.legend(loc="best")

                ax2.plot(df_hist["epoch"], 100 * df_hist["clip_frac"],
                         color="#8e44ad", linewidth=1.8, label="% steps grad-clipped")
                ax2.plot(df_hist["epoch"], df_hist["grad_norm"],
                         color="#7f8c8d", linewidth=1.2, alpha=0.7, label="mean grad norm")
                ax2.set_title("Gradient Clipping Pressure", fontsize=13, fontweight="bold")
                ax2.set_xlabel("Epoch", fontsize=11)
                ax2.grid(True, linestyle="--", alpha=0.5)
                ax2.legend(loc="best")

                plt.tight_layout()
                fig.savefig(os.path.join(ckpt_dir, "tail_health.png"), dpi=150)
                plt.close(fig)
    except Exception as e:
        logging.warning("Could not save training plots: %s", e)


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
