#!/usr/bin/env python
"""
visualize_transform.py — what does the model "see" under each rainfall transform,
and (after generation) did the change help?

Figure 1  transform_pixels.png      (needs only the training data, runs now)
  a) the mapping: rainfall in mm -> model value, standard vs asinh
  b) histogram of all model-space values (the pixels the diffusion model learns)
  c) how much of the value range each rain band gets
  d) the same four real 8x8 images under both transforms, light rain -> biggest storm

Figure 2  generated_vs_real.png      (only when --generated CSVs are given)
  a) exceedance curve of wet-step intensity: real vs each generated version
  b) quantile-quantile plot of wet steps against the real record

Pure numpy / pandas / matplotlib — runs in the local .venv, no torch.

    python scripts/visualize_transform.py
    python scripts/visualize_transform.py --generated \
        v12=results/generated_data/rainfall_synthetic_10y_v12.csv \
        v12_asinh=results/generated_data/rainfall_synthetic_10y_v12_asinh.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from regime_training.transforms import make_scaler  # noqa: E402

WET_THR = 0.005          # mm / 5 min, same as gate_a_scorecard.py
# Rain bands in mm / 5 min — shared with diagnose_denoising_error.py
BANDS = [("dry", 0.0, WET_THR), ("light", WET_THR, 0.1), ("moderate", 0.1, 0.5),
         ("heavy", 0.5, 2.0), ("extreme", 2.0, np.inf)]

# Colours: validated categorical pair (dataviz skill palette) + neutral ink for "real"
C = {"standard": "#2a78d6", "asinh": "#eb6834", "real": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f7f7f5", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def delay_embed(window, delay, embedding):
    """numpy copy of DelayEmbedder.ts_to_img (no padding): column i = window[i*delay : i*delay+embedding]."""
    cols = [window[i * delay: i * delay + embedding]
            for i in range((len(window) - embedding) // delay + 1)]
    return np.stack(cols, axis=1)


def pick_windows(x, seq_len):
    """Four non-overlapping windows whose peak is closest to light / moderate / heavy / record."""
    n = len(x) // seq_len
    W = x[: n * seq_len].reshape(n, seq_len)
    peaks = W.max(1)
    out = []
    for label, target in [("light", 0.05), ("moderate", 0.3), ("heavy", 1.5), ("biggest storm", None)]:
        i = int(np.argmax(peaks)) if target is None else int(np.argmin(np.abs(peaks - target)))
        out.append((label, W[i]))
    return out


def figure_pixels(x, scalers, args, out):
    fig = plt.figure(figsize=(12, 13), facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5], hspace=0.45, wspace=0.3)

    # a) mapping
    ax = fig.add_subplot(gs[0, 0:2])
    grid = np.concatenate([[0.0], np.geomspace(1e-3, x.max(), 400)])
    for name, sc in scalers.items():
        ax.plot(grid, sc.transform(grid.reshape(-1, 1)).ravel(), color=C[name], lw=2, label=name)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("rainfall (mm / 5 min)")
    ax.set_ylabel("model value")
    ax.set_title("a) Mapping: rainfall → model value")
    ax.legend(loc="upper left")

    # b) histogram of model-space values
    ax = fig.add_subplot(gs[0, 2:4])
    for name, sc in scalers.items():
        z = sc.transform(x.reshape(-1, 1)).ravel()
        ax.hist(z, bins=200, histtype="step", color=C[name], lw=1.5, label=f"{name} (max {z.max():.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("model value")
    ax.set_ylabel("count of 5-min steps (log)")
    ax.set_title("b) What the model learns: all pixel values")
    ax.legend(loc="upper right")

    # c) share of the value range each band occupies
    ax = fig.add_subplot(gs[1, 0:2])
    names = [b[0] for b in BANDS[1:]]
    width = 0.38
    rows = []
    for k, (name, sc) in enumerate(scalers.items()):
        z_lo = sc.transform([[0.0]])[0, 0]
        z_hi = sc.transform([[x.max()]])[0, 0]
        shares = []
        for bname, lo, hi in BANDS[1:]:
            a = sc.transform([[lo]])[0, 0]
            b = sc.transform([[min(hi, x.max())]])[0, 0]
            shares.append(100 * (b - a) / (z_hi - z_lo))
            rows.append(dict(transform=name, band=bname, pct_of_range=shares[-1]))
        pos = np.arange(len(names)) + (k - 0.5) * width
        ax.bar(pos, shares, width=width - 0.04, color=C[name], label=name)
        for p, s in zip(pos, shares):
            ax.text(p, s + 1, f"{s:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("% of the model's value range")
    ax.set_ylim(0, 105)
    ax.set_title("c) Room each rain band gets in model space")
    ax.legend(loc="upper left")
    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(out), "transform_range_shares.csv"), index=False)

    # d) the same real windows as 8x8 images under both transforms
    wins = pick_windows(x, args.seq_len)
    sub = gs[2, :].subgridspec(2, 4, wspace=0.05, hspace=0.25)
    for r, (name, sc) in enumerate(scalers.items()):
        vmin = sc.transform([[0.0]])[0, 0]
        vmax = sc.transform([[x.max()]])[0, 0]     # one colour scale per transform
        for c, (label, w) in enumerate(wins):
            ax = fig.add_subplot(sub[r, c])
            img = delay_embed(sc.transform(w.reshape(-1, 1)).ravel(), args.delay, args.embedding)
            ax.imshow(img, cmap=SEQ, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values(): sp.set_visible(False)
            if r == 0:
                ax.set_title(f"{label}\npeak {w.max():.2f} mm", fontsize=8, fontweight="normal")
            if c == 0:
                ax.set_ylabel(name, color=C[name], fontweight="bold")
    fig.text(0.5, 0.405, "d) Same real windows as the model's 8×8 images "
             "(colour scale = dry → record storm, per transform)",
             fontsize=10, fontweight="bold", ha="center")

    # caption in the empty bottom-left cell
    ax = fig.add_subplot(gs[1, 2:4]); ax.axis("off")
    ax.text(0, 0.9,
            "How to read this\n"
            "• b) and c): under 'standard' almost all of the value range is spent on\n"
            "  the rare extreme steps, and every other band is squeezed near the dry value.\n"
            "• d): under 'standard' the light and moderate images look almost blank —\n"
            "  the model has to learn them from tiny differences in value.\n"
            "• The dry spike (91% of steps) is one value under BOTH transforms:\n"
            "  asinh does not change the dry / wet problem.",
            fontsize=8.5, color=INK, va="top", linespacing=1.5)

    fig.suptitle(f"Rainfall → pixels: standard vs asinh (s = {args.asinh_scale} mm/5min), "
                 f"train split {args.train_csv_label}", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def load_mm(path):
    df = pd.read_csv(path)
    col = "avg_rainfall" if "avg_rainfall" in df.columns else df.columns[1]
    a = df[col].values.astype(float).clip(min=0)
    a[a < WET_THR] = 0.0
    return a


def figure_generated(real, gens, out):
    colours = [C["standard"], C["asinh"], "#1baf7a", "#eda100"]   # fixed order, never cycled
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    qs = np.linspace(0.5, 0.9999, 400)
    rw = real[real > 0]
    xs = np.geomspace(WET_THR, max(rw.max(), *[g.max() for g in gens.values()]), 300)

    ax1.plot(xs, [(rw > v).mean() for v in xs], color=C["real"], lw=2.5, label="real (train)")
    for (name, g), col in zip(gens.items(), colours):
        gw = g[g > 0]
        ax1.plot(xs, [(gw > v).mean() for v in xs], color=col, lw=2, label=name)
        ax2.plot(np.quantile(rw, qs), np.quantile(gw, qs), color=col, lw=2, label=name)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("rainfall (mm / 5 min)")
    ax1.set_ylabel("fraction of wet steps above x")
    ax1.set_title("a) Wet-step exceedance (the tail is on the right)")
    ax1.legend()

    lim = [WET_THR, max(np.quantile(rw, qs[-1]), *[np.quantile(g[g > 0], qs[-1]) for g in gens.values()])]
    ax2.plot(lim, lim, color=C["real"], lw=1, ls="--", label="perfect match")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("real quantile (mm / 5 min)")
    ax2.set_ylabel("generated quantile (mm / 5 min)")
    ax2.set_title("b) Q-Q of wet steps (below the line = too weak)")
    ax2.legend()

    fig.suptitle(f"Generated vs real rainfall: {' vs '.join(gens)}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_csv", default="data/rainfall/splits/train_years_labelled.csv")
    ap.add_argument("--data_col", default="avg_rainfall")
    ap.add_argument("--asinh_scale", type=float, default=0.035)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--delay", type=int, default=8)
    ap.add_argument("--embedding", type=int, default=8)
    ap.add_argument("--generated", nargs="*", default=[], metavar="NAME=CSV",
                    help="generated 10-year CSVs to compare against the real train split")
    ap.add_argument("--out_dir", default="results/transform_diagnostics")
    args = ap.parse_args()
    args.train_csv_label = "2000–2007"

    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    x = pd.read_csv(os.path.join(ROOT, args.train_csv))[args.data_col].values.astype(np.float32)

    # Fitted exactly as RegimeDataset fits them during training
    scalers = {name: make_scaler(name, args.asinh_scale).fit(x.reshape(-1, 1))
               for name in ("standard", "asinh")}
    figure_pixels(x, scalers, args, os.path.join(ROOT, args.out_dir, "transform_pixels.png"))

    if args.generated:
        real = x.astype(float).copy()
        real[real < WET_THR] = 0.0
        gens = {}
        for item in args.generated:
            name, path = item.split("=", 1)
            p = path if os.path.isabs(path) else os.path.join(ROOT, path)
            if os.path.exists(p):
                gens[name] = load_mm(p)
            else:
                print(f"MISSING: {p}")
        if gens:
            figure_generated(real, gens, os.path.join(ROOT, args.out_dir, "generated_vs_real.png"))


if __name__ == "__main__":
    main()
