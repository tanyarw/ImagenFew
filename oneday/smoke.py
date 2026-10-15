"""GPU smoke measurement before v19 is trained (plan section 3): speed and memory of the 24x24 network.

Builds v19 exactly as train_regime.py would (config_v19 window, the pretrained checkpoint loaded with the same
shape-safe loader, EMA), then times real training steps (forward, loss, backward, gradient clip, AdamW, EMA) on
random batches at batch 2048 and 1024, times production sampling (36 Heun steps), and reads the partition's time
limit. Writes results/oneday/smoke.json with the training hours that 500 epochs (v14's 205,000 steps at batch 2048)
would take. The epoch count is then fixed by the plan's rule, not chosen by hand.
"""
import argparse
import os
import subprocess
import time

import numpy as np
import torch

from sitetransfer.provenance import ROOT, write_json

from . import config as K

N_WINDOWS = 8 * 105_120 - 288 + 1           # 2000-2007, every start (RegimeDataset, continuous mode)
OUT = ROOT / "results" / "oneday" / "smoke.json"


def _model(batch_size, device):
    from models.ImagenFew.ImagenFew import ImagenFew
    from regime_training.train_regime import load_checkpoint_safe
    _, cfg = K.build(500, batch_size)
    args = argparse.Namespace(**cfg)
    args.device = device
    model = ImagenFew(args, device).to(device)
    load_checkpoint_safe(model, str(K.CHECKPOINT), device)
    model.setup_finetune(args)
    return model, args


def time_training(batch_size, device, steps=20, warmup=3):
    from regime_training.train_regime import denoising_loss
    model, args = _model(batch_size, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    x = torch.randn(batch_size, args.seq_len, 1, device=device)
    lab = torch.eye(args.n_classes, device=device)[torch.randint(0, args.n_classes, (batch_size,), device=device)]
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        mask = model.ts_to_img(torch.zeros_like(x), pad_val=1)
        x_img = model.ts_to_img(x)
        opt.zero_grad()
        out, w = model(x_img, mask, labels=lab)
        loss, _, _ = denoising_loss(out, x_img, mask, w, torch.ones_like(x_img), args.fft_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        model.on_train_batch_end()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps, torch.cuda.max_memory_allocated() / 2 ** 30


def time_sampling(device, n=512):
    from models.ImagenFew.sampler import DiffusionProcess
    model, args = _model(2048, device)
    model.eval()
    proc = DiffusionProcess(args, model.net, (1, args.img_resolution, args.img_resolution))
    lab = torch.eye(args.n_classes, device=device)[torch.zeros(n, dtype=torch.long, device=device)]
    mask = model.ts_to_img(torch.zeros(n, args.seq_len, 1, device=device), pad_val=1)
    x = torch.zeros(n, 1, args.img_resolution, args.img_resolution, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), model.ema_scope():
        proc.interpolate(x, mask, class_labels=lab)
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def partition_limit():
    part = os.environ.get("SLURM_JOB_PARTITION")
    if not part:
        return None
    try:
        return subprocess.run(["sinfo", "-h", "-p", part, "-o", "%l"], capture_output=True, text=True,
                              check=True).stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def run(device="cuda"):
    if not torch.cuda.is_available():
        raise RuntimeError("the smoke measurement needs a GPU")
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    res = {"gpu": torch.cuda.get_device_name(0), "gpu_memory_gb": gpu_gb, "partition": os.environ.get(
        "SLURM_JOB_PARTITION"), "partition_time_limit": partition_limit(), "batch": {}}
    for b in (2048, 1024):
        try:
            s, peak = time_training(b, device)
            steps = N_WINDOWS // b
            res["batch"][b] = {"s_per_step": s, "peak_memory_gb": peak, "steps_per_epoch": steps,
                               "hours_500_epochs": 500 * steps * s / 3600}
        except torch.cuda.OutOfMemoryError:
            res["batch"][b] = {"out_of_memory": True}
        torch.cuda.empty_cache()
    per_block = time_sampling(device)
    blocks = int(np.ceil(10 * 105_120 / (288 - 4)))           # 10 years, 4-step overlap, one member
    res["sampling"] = {"s_per_block": per_block, "minutes_4_members_10_years": 4 * blocks * per_block / 60,
                       # train_regime's check every 25 epochs: 4 phases x 50 samples
                       "minutes_training_checks_500_epochs": 20 * 200 * per_block / 60}
    write_json(OUT, res)
    return res
