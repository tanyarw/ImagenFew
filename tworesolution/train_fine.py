"""Fine-model training: v14 fine-tuned to take the half-hour totals as an extra input (plan sections 2, 4, 10).

Everything is v14's (tworesolution/configs/fine.yaml is config_v14.yaml with three keys added: loss with the FFT
term, AdamW, learning rate, batch, gradient clip, EMA, dropout, seasonal phases, 500 epochs). Only two things
differ from v14's own training:
  * the windows: 64 five-minute steps from 2000-2007 starting 2 steps before a half-hour (grid.train_starts),
    each with its condition (grid.condition_windows); both in v14's model units (v14's scaler, unchanged);
  * the starting point: v14's EMA weights plus the zero-initialised condition layer (network.fine_from_v14).
The rain target is the consistent 5-minute record (grid.two_level): the six values of every half-hour sum to its
total. Checkpoints go to logs/ImagenFew/Rainfall_Regime/<run_id>/ like every other run.
"""
import datetime as dt
import json
import logging
import os
import pickle
import shutil
import socket

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from regime_training.train_regime import denoising_loss, intensity_weights, seed_everything
from sitetransfer.provenance import ROOT, git_state, rel

from . import data as D
from .grid import (FACTOR, KEEP, N_CLASSES, PAD, condition_windows, train_starts, two_level, value_windows,
                   window_labels)
from .network import fine_from_v14, load_args, save_fine
from .sampling import heun, schedule

MONITOR_WINDOWS = 256


def windows(x5_raw, profile5, tau5, tau30, starts=None):
    """(rain [n, L] mm, condition [n, L] mm per 5 min, seasonal phase [n]) of the training layout."""
    x5c, x30 = two_level(x5_raw, tau5, tau30)
    starts = train_starts(len(x5c)) if starts is None else starts
    return value_windows(x5c, starts), condition_windows(x30, starts), window_labels(profile5, starts)


def to_model(scaler, a):
    z = scaler.transform(np.asarray(a, dtype=np.float64).reshape(-1, 1)).reshape(*np.shape(a), 1)
    return torch.as_tensor(z, dtype=torch.float32)


def edm_forward(model, x_img, c_img, labels):
    """ImagenFew.forward with nothing known, plus the condition: the same noise draws, in the same order."""
    rnd = torch.randn([x_img.shape[0], 1, 1, 1], device=x_img.device)
    sigma = (rnd * model.P_std + model.P_mean).exp()
    weight = (sigma ** 2 + model.sigma_data ** 2) / (sigma * model.sigma_data) ** 2
    n = torch.randn_like(x_img) * sigma
    return model.net(x_img + n, sigma, labels, cond=c_img), weight


@torch.no_grad()
def monitor(model, scaler, batch, tau5, num_steps, device):
    """Is the condition used? On fixed 2008 windows: generated rain / conditioned rain, and the share of the
    generated rain that falls in half-hours the condition says are wet (both 1.0 for a perfect disaggregator)."""
    c, lab, c_mm = batch
    model.eval()
    with model.ema_scope():
        cond = model.ts_to_img(c.to(device))
        img = heun(model.net, torch.randn(len(lab), 1, 8, 8, generator=torch.Generator().manual_seed(0)).to(device),
                   torch.eye(N_CLASSES, device=device)[lab.to(device)], schedule(model.net, num_steps, device), cond=cond)
    model.train()
    raw = scaler.inverse_transform(model.img_to_ts(img.float())[:, :, 0].cpu().numpy().reshape(-1, 1)).reshape(len(lab), -1)
    raw = np.where(raw >= tau5, raw, 0.0)[:, PAD:PAD + KEEP]
    cmm = c_mm[:, PAD:PAD + KEEP]
    wet_hh = np.repeat(cmm.reshape(len(lab), -1, FACTOR).sum(2) > 0, FACTOR, axis=1)
    total = raw.sum()
    return dict(volume_ratio=float(total / cmm.sum()),
                in_wet_half_hours=float(raw[wet_hh].sum() / max(total, 1e-12)))


def run(cli):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --device cpu only for a smoke test")
    args = load_args(cli.config, cli.device)
    for key in ("epochs", "batch_size"):
        if getattr(cli, key) is not None:
            setattr(args, key, getattr(cli, key))
    d = D.load_derived()
    tau5, tau30 = d["tau5_mm"], d["tau30_mm"]
    loader_gen = seed_everything(cli.seed)

    with open(ROOT / args.scaler_from, "rb") as f:
        scaler = pickle.load(f)
    profile5 = D.calendar5()
    x, c, lab = windows(D.train_record(), profile5, tau5, tau30)
    ds = TensorDataset(to_model(scaler, x), to_model(scaler, c), torch.as_tensor(lab))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,
                        generator=loader_gen)
    _, cv, lv = windows(D.load_split("val")[0], profile5, tau5, tau30)
    keep = np.flatnonzero(cv.max(1) > 0)
    pick = np.sort(np.random.default_rng(0).choice(keep, min(MONITOR_WINDOWS, len(keep)), replace=False))
    mon_batch = (to_model(scaler, cv[pick]), torch.as_tensor(lv[pick]), cv[pick])

    model = fine_from_v14(args, ROOT / args.init_from, cli.device)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    out = D.RUNS / cli.run_id
    if (out / "best_fine_model.pt").exists() and not cli.overwrite:
        raise FileExistsError(f"{rel(out)} already holds a fine model (--overwrite to replace it)")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cli.config, out / "config.yaml")
    shutil.copyfile(ROOT / args.scaler_from, out / "scaler.pkl")
    (out / "run_args.json").write_text(json.dumps({
        "run_id": cli.run_id, "config": rel(cli.config), "init_from": args.init_from, "seed": cli.seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "windows": len(ds), "steps_per_epoch": len(loader), "tau5_mm": tau5, "tau30_mm": tau30,
        "limit_batches": cli.limit_batches, "git": git_state(), "torch": torch.__version__,
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": dt.datetime.now().isoformat(timespec="seconds")}, indent=1))
    logging.info("fine model %s: %d windows, %d steps/epoch, %d epochs, from %s", cli.run_id, len(ds), len(loader),
                 args.epochs, args.init_from)

    alpha = float(getattr(args, "intensity_weight_alpha", 0.0))
    gamma = float(getattr(args, "intensity_weight_gamma", 0.5))
    norm = bool(getattr(args, "intensity_weight_normalize", True))
    fft_weight, grad_clip = float(args.fft_weight), float(args.grad_clip)
    log_every = int(getattr(args, "logging_iter", 25))
    eye = torch.eye(N_CLASSES, device=cli.device)
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        losses, norms = [], []
        for i, (xb, cb, lb) in enumerate(loader):
            if cli.limit_batches is not None and i >= cli.limit_batches:
                break
            xb, cb = xb.to(cli.device), cb.to(cli.device)
            x_img, c_img = model.ts_to_img(xb), model.ts_to_img(cb)
            x_mask = model.ts_to_img(torch.zeros_like(xb), pad_val=1)
            optimizer.zero_grad()
            out_img, weight = edm_forward(model, x_img, c_img, eye[lb.to(cli.device)])
            w_int = intensity_weights(x_img, alpha, gamma, norm)
            loss, _, _ = denoising_loss(out_img, x_img, x_mask, weight, w_int, fft_weight)
            loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)))
            optimizer.step()
            model.on_train_batch_end()
            losses.append(loss.item())
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "grad_norm": float(np.mean(norms)),
               "clip_frac": float(np.mean([g > grad_clip for g in norms])),
               "cond_weight_norm": float(model.net.model.cond_conv.weight.detach().norm())}
        is_best = rec["loss"] < best
        best = min(best, rec["loss"])
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            rec |= monitor(model, scaler, mon_batch, tau5, args.diffusion_steps, cli.device)
            logging.info("epoch %4d/%d  loss %.6f (best %.6f)  cond |w| %.4f  generated/conditioned rain %.3f  "
                         "in wet half-hours %.3f", epoch, args.epochs, rec["loss"], best, rec["cond_weight_norm"],
                         rec["volume_ratio"], rec["in_wet_half_hours"])
            if is_best:
                save_fine(model, out / "best_fine_model.pt", epoch=epoch, loss=rec["loss"])
            pd.DataFrame(history + [rec]).to_csv(out / "history.csv", index=False)
        history.append(rec)
    save_fine(model, out / "final_fine_model.pt", epoch=args.epochs, loss=history[-1]["loss"])
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    logging.info("done: %s", rel(out))
    return out
