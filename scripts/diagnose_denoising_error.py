#!/usr/bin/env python
"""
diagnose_denoising_error.py — WHICH rain is learnt worst, before vs after a transform change?

The training loss is one number per epoch, and it lives in model space, so a
lower loss under asinh does NOT mean a better model (the same mm error is a
different model-space error under each transform).  This script measures the
denoiser's error the fair way:

  1. take held-out windows (test_csv from the config, non-overlapping),
  2. add noise at a FIXED grid of noise levels sigma (the same noise draws for
     every run),
  3. denoise once with each trained model (EMA weights),
  4. map the prediction back to mm with that run's own scaler (negative
     values clipped to 0, as generation does),
  5. group every 5-min step by its TRUE rain band and report, per band and sigma:
        rmse_mm  — typical size of the error in mm
        bias_mm  — mean(pred - true); negative = the model makes rain too weak
        vol_err  — sum(pred) / sum(true) - 1 (wet bands only)

It also prints a single "training-weighted" summary per band: the average over
the sigma grid weighted by how often training samples each sigma
(sigma ~ LogNormal(-1.2, 1.2), as in ImagenFew.forward).

Needs torch + a GPU node (same env as training).  Outputs in --out_dir:
  denoise_error.csv, denoise_error_summary.csv,
  denoise_error_by_band.png, denoise_error_summary.png

    python scripts/diagnose_denoising_error.py \
        --run v12=logs/ImagenFew/Rainfall_Regime/v12 \
        --run v12_asinh=logs/ImagenFew/Rainfall_Regime/v12_asinh \
        --config regime_training/config_v12_asinh.yaml
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.ImagenFew.ImagenFew import ImagenFew  # noqa: E402

WET_THR = 0.005
# Same bands as visualize_transform.py (mm / 5 min)
BANDS = [("dry", 0.0, WET_THR), ("light", WET_THR, 0.1), ("moderate", 0.1, 0.5),
         ("heavy", 0.5, 2.0), ("extreme", 2.0, np.inf)]
SIGMAS = np.geomspace(0.02, 20.0, 13)
P_MEAN, P_STD = -1.2, 1.2                    # training sigma distribution (ImagenFew.py)
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated order, never cycled
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR",
                   help="run directory holding best_regime_model.pt and scaler.pkl (repeat per run)")
    p.add_argument("--config", required=True,
                   help="config for model architecture + test_csv (must match every run)")
    p.add_argument("--split", default="test", choices=["test", "train"],
                   help="test = held-out years (default). The held-out split has only a "
                        "handful of extreme steps, so 'train' is a secondary check on the tail "
                        "(in-sample: the model has seen these years).")
    p.add_argument("--ckpt_name", default="best_regime_model.pt")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="results/transform_diagnostics")
    return p.parse_args()


def load_model(args, ckpt_path):
    """Build ImagenFew and load weights the same way generate_hmm_v1.py does."""
    model = ImagenFew(args, args.device).to(args.device)
    loaded = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    state = loaded.get("model", loaded)
    cur = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in cur and v.shape == cur[k].shape}, strict=False)
    if "ema_model" in loaded and args.ema:
        cur_ema = model.model_ema.state_dict()
        model.model_ema.load_state_dict({k: v for k, v in loaded["ema_model"].items()
                                         if k in cur_ema and v.shape == cur_ema[k].shape}, strict=False)
    model.eval()
    return model


@torch.no_grad()
def denoise_all(model, scaler, windows_mm, args, seed):
    """Return {sigma: predicted mm, shape [n_windows, seq_len]} for one run."""
    n, L = windows_mm.shape
    z = scaler.transform(windows_mm.reshape(-1, 1)).reshape(n, L, 1)
    x_all = torch.as_tensor(z, dtype=torch.float32)
    preds = {}
    with model.ema_scope():
        for k, sigma in enumerate(SIGMAS):
            g = torch.Generator().manual_seed(seed + k)      # identical noise for every run
            out = []
            for i in range(0, n, args.batch):
                x_ts = x_all[i:i + args.batch].to(args.device)
                b = x_ts.shape[0]
                x_img = model.ts_to_img(x_ts)
                signal = 1 - model.ts_to_img(torch.zeros_like(x_ts), pad_val=1)
                eps = torch.randn(x_img.shape, generator=g).to(args.device)
                noisy = x_img + float(sigma) * eps * signal
                s = torch.full((b,), float(sigma), device=args.device)
                D = model.net(noisy, s, None)
                out.append(model.img_to_ts(D)[:, :, 0].cpu().numpy())
            pz = np.concatenate(out, axis=0).astype(np.float64)
            mm = scaler.inverse_transform(pz.reshape(-1, 1)).reshape(n, L)
            preds[float(sigma)] = np.clip(mm, 0.0, None)
    return preds


def band_metrics(true_mm, pred_mm):
    rows = []
    t, p = true_mm.ravel(), pred_mm.ravel()
    for name, lo, hi in BANDS:
        m = (t >= lo) & (t < hi)
        if not m.any():
            continue
        err = p[m] - t[m]
        rows.append(dict(
            band=name, n=int(m.sum()),
            rmse_mm=float(np.sqrt(np.mean(err ** 2))),
            bias_mm=float(err.mean()),
            # dry has no meaningful volume (true values are ~0), so no ratio
            vol_err=float(p[m].sum() / t[m].sum() - 1) if name != "dry" else np.nan,
        ))
    return rows


def sigma_weights():
    """Training-time density of each grid sigma (log-normal in sigma, on a log-spaced grid)."""
    w = np.exp(-0.5 * ((np.log(SIGMAS) - P_MEAN) / P_STD) ** 2)
    return w / w.sum()


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_by_band(df, runs, out, split_label):
    bands = [b[0] for b in BANDS if b[0] in df.band.unique()]
    fig, axes = plt.subplots(2, len(bands), figsize=(3.1 * len(bands), 6.2), facecolor="white")
    for j, band in enumerate(bands):
        for run, col in zip(runs, COLOURS):
            d = df[(df.run == run) & (df.band == band)].sort_values("sigma")
            axes[0, j].plot(d.sigma, d.rmse_mm, color=col, lw=2, marker="o", ms=4, label=run)
            axes[1, j].plot(d.sigma, d.bias_mm, color=col, lw=2, marker="o", ms=4, label=run)
        n = int(df[(df.band == band)].n.iloc[0])
        axes[0, j].set_title(f"{band}  (n = {n:,})", fontsize=10, fontweight="bold")
        axes[1, j].axhline(0, color=MUTED, lw=1)
        for i in (0, 1):
            axes[i, j].set_xscale("log")
            style(axes[i, j])
        axes[0, j].set_yscale("log")
        axes[1, j].set_xlabel("noise level σ", fontsize=9)
    axes[0, 0].set_ylabel("RMSE (mm / 5 min)")
    axes[1, 0].set_ylabel("bias (mm / 5 min)\nbelow 0 = too weak")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Denoising error by true rain band ({split_label}, measured in mm)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summ, runs, out):
    bands = [b[0] for b in BANDS if b[0] in summ.band.unique()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    width = 0.8 / len(runs)
    for k, (run, col) in enumerate(zip(runs, COLOURS)):
        d = summ[summ.run == run].set_index("band").loc[bands]
        pos = np.arange(len(bands)) + (k - (len(runs) - 1) / 2) * width
        ax1.bar(pos, d.rmse_mm, width=width - 0.03, color=col, label=run)
        ax2.bar(pos, 100 * d.vol_err, width=width - 0.03, color=col, label=run)
        for p_, v in zip(pos, d.rmse_mm):
            ax1.text(p_, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7, color=INK)
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(len(bands)))
        ax.set_xticklabels(bands)
        style(ax)
    ax1.set_yscale("log")
    ax1.set_ylabel("RMSE (mm / 5 min, log)")
    ax1.set_title("Typical error per band", fontsize=10, fontweight="bold")
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_ylabel("rain volume error (%)\nbelow 0 = too little rain")
    ax2.set_title("Volume error per band (dry has no volume)", fontsize=10, fontweight="bold")
    ax1.legend(frameon=False)
    fig.suptitle("Denoising error, averaged over noise levels as training samples them",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cli = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(cli.config), resolve=True)
    args = argparse.Namespace(**cfg)
    args.device = cli.device
    args.batch = cli.batch
    out_dir = os.path.join(ROOT, cli.out_dir, f"denoise_{cli.split}")
    os.makedirs(out_dir, exist_ok=True)

    # Held-out rainfall in mm, cut into non-overlapping windows (every step used once)
    csv = args.test_csv if cli.split == "test" else args.train_csv
    x = pd.read_csv(os.path.join(ROOT, csv))[args.data_col].values.astype(np.float64)
    n = len(x) // args.seq_len
    windows = x[: n * args.seq_len].reshape(n, args.seq_len)
    print(f"{cli.split} split: {csv}  →  {n} windows × {args.seq_len} steps")

    rows, runs = [], []
    for item in cli.run:
        name, run_dir = item.split("=", 1)
        run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
        runs.append(name)
        with open(os.path.join(run_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        model = load_model(args, os.path.join(run_dir, cli.ckpt_name))
        print(f"{name}: {scaler!r} from {run_dir}")
        preds = denoise_all(model, scaler, windows, args, cli.seed)
        for sigma, pred in preds.items():
            for r in band_metrics(windows, pred):
                rows.append(dict(run=name, sigma=sigma, **r))
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "denoise_error.csv"), index=False)

    # Training-weighted summary over sigma
    w = df.sigma.map(dict(zip(SIGMAS.astype(float), sigma_weights())))   # sums to 1 per (run, band)
    summ = (df.assign(mse=w * df.rmse_mm ** 2, bias_mm=w * df.bias_mm, vol_err=w * df.vol_err)
              .groupby(["run", "band"], sort=False)
              .agg(n=("n", "first"), mse=("mse", "sum"), bias_mm=("bias_mm", "sum"),
                   vol_err=("vol_err", lambda v: v.sum(min_count=1)))
              .reset_index())
    summ.insert(3, "rmse_mm", np.sqrt(summ.pop("mse")))
    summ.to_csv(os.path.join(out_dir, "denoise_error_summary.csv"), index=False)

    pd.set_option("display.width", 120)
    print("\nTraining-weighted denoising error (mm / 5 min):")
    print(summ.pivot(index="band", columns="run", values=["rmse_mm", "bias_mm", "vol_err"])
              .loc[[b[0] for b in BANDS if b[0] in summ.band.unique()]]
              .round(4).to_string())

    split_label = "held-out years" if cli.split == "test" else "training years, in-sample"
    plot_by_band(df, runs, os.path.join(out_dir, "denoise_error_by_band.png"), split_label)
    plot_summary(summ, runs, os.path.join(out_dir, "denoise_error_summary.png"))
    print(f"\nsaved CSVs + figures in {out_dir}")


if __name__ == "__main__":
    main()
