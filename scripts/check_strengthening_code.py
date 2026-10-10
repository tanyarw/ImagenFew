#!/usr/bin/env python
"""
check_strengthening_code.py — fast correctness checks for the code behind the strengthening
runs (code_plan/STRENGTHENING_RUNS.md). Run before submitting, on the cluster or a laptop:

    .venv/bin/python scripts/check_strengthening_code.py            # ~1 min on CPU

Each check proves one claim the runs rely on, and exits non-zero on the first failure:

  1. make_scaler() without data_scale builds exactly the scaler it built before (v14 unchanged);
     data_scale = 0.5 halves model-space values, inverts exactly and survives pickling (v18).
  2. The known-prefix mask (E3) marks exactly the first k steps, and with delay = embedding = 8
     those steps are exactly the first k / 8 image columns; prob 0 gives an all-zero mask
     (every earlier run's training is unchanged).
  3. ImagenFew.forward with that mask feeds the network the KNOWN pixels clean and noises only
     the unknown ones (what "trained with a known prefix" means).
  4. generate_context.sample_block(clean_known=True) gives the same unknown pixels as the
     production DiffusionProcess.impute() from the same starting noise: E3 generation samples
     the way E3 training trained.
  5. train_regime.load_checkpoint_safe: 'none' keeps the random init; a mistyped path is an
     error, not a silent from-scratch run.
  6. Seeding: two seeded mask draws are identical.

Uses the v14 checkpoint for checks 3-4 if it is present, random weights otherwise (the
identities hold for any weights).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.ImagenFew.ImagenFew import ImagenFew  # noqa: E402
from models.ImagenFew.sampler import DiffusionProcess  # noqa: E402
from regime_training.generate_context import sample_block  # noqa: E402
from regime_training.train_regime import (known_prefix_mask, known_prefix_settings,  # noqa: E402
                                          load_checkpoint_safe, seed_everything)
from regime_training.transforms import AsinhScaler, ScaledScaler, make_scaler  # noqa: E402

V14 = os.path.join(ROOT, 'logs', 'ImagenFew', 'Rainfall_Regime', 'v14', 'best_regime_model.pt')


def ok(msg):
    print(f'  ok  {msg}', flush=True)


def build(config):
    cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(ROOT, 'regime_training', config)), resolve=True)
    args = argparse.Namespace(**cfg)
    args.device = 'cpu'
    return args, ImagenFew(args, 'cpu').eval()


def check_scalers():
    print('1. transforms', flush=True)
    assert type(make_scaler('asinh', 0.035)) is AsinhScaler
    assert type(make_scaler('standard')) is StandardScaler
    assert type(make_scaler('asinh', 0.035, 1.0)) is AsinhScaler
    x = pd.read_csv(os.path.join(ROOT, 'data/rainfall/splits/train_years_labelled.csv'),
                    usecols=['avg_rainfall'], nrows=200_000)['avg_rainfall'].to_numpy().reshape(-1, 1)
    base = make_scaler('asinh', 0.035).fit(x)
    scaled = make_scaler('asinh', 0.035, 0.5).fit(x)
    assert isinstance(scaled, ScaledScaler)
    np.testing.assert_allclose(scaled.transform(x), 0.5 * base.transform(x), rtol=1e-6, atol=1e-7)
    z = scaled.transform(x)
    assert abs(z.std() - 0.5) < 1e-3, z.std()
    np.testing.assert_allclose(scaled.inverse_transform(z), x, rtol=1e-4, atol=1e-6)
    again = pickle.loads(pickle.dumps(scaled))
    np.testing.assert_array_equal(again.transform(x), z)
    ok(f'default scalers unchanged; data_scale 0.5 -> std {z.std():.4f}, inverts, pickles')


def check_masks(model, args):
    print('2. known-prefix masks', flush=True)
    off = argparse.Namespace(seq_len=64)
    assert known_prefix_settings(off) == (0.0, None)
    x = torch.randn(512, 64, 1)
    assert torch.equal(known_prefix_mask(x, 0.0, None), torch.zeros_like(x))
    prob, steps = known_prefix_settings(argparse.Namespace(seq_len=64, known_prefix_prob=0.75,
                                                           known_prefix_steps=[8, 16, 24, 32]))
    torch.manual_seed(0)
    m = known_prefix_mask(x, prob, steps)
    k = m[:, :, 0].sum(1).long()
    assert set(k.tolist()) <= {0, 8, 16, 24, 32}
    for i in range(len(k)):        # a prefix: ones then zeros
        assert torch.equal(m[i, :, 0], (torch.arange(64) < k[i]).float())
    share = (k > 0).float().mean().item()
    assert 0.68 < share < 0.82, share
    img = model.ts_to_img(m, pad_val=1)
    for i in range(len(k)):
        cols = int(k[i]) // 8
        assert img[i, 0, :, :cols].eq(1).all() and img[i, 0, :, cols:].eq(0).all()
    for bad in ([0], [64], []):
        try:
            known_prefix_settings(argparse.Namespace(seq_len=64, known_prefix_prob=0.5, known_prefix_steps=bad))
        except ValueError:
            continue
        raise AssertionError(f'known_prefix_steps {bad} was accepted')
    ok(f'prefix = first k steps = first k/8 image columns; {100 * share:.0f}% of windows masked; prob 0 = no mask')


def check_forward(model):
    print('3. ImagenFew.forward with a known prefix', flush=True)
    seen = {}

    def spy(module, inputs):           # forward pre-hook: records what the network is given
        seen['x'] = inputs[0].detach().clone()

    handle = model.net.register_forward_pre_hook(spy)
    try:
        torch.manual_seed(1)
        x_ts = torch.randn(64, 64, 1)
        prob, steps = 1.0, torch.tensor([16])
        x_img = model.ts_to_img(x_ts)
        mask = model.ts_to_img(known_prefix_mask(x_ts, prob, steps), pad_val=1)
        labels = torch.nn.functional.one_hot(torch.arange(64) % 4, 4).float()
        with torch.no_grad():
            model(x_img, mask, labels=labels)
        net_in = seen['x']
        known, unknown = mask.bool(), ~mask.bool()
        assert torch.equal(net_in[known], x_img[known]), 'known pixels were changed'
        noise = (net_in - x_img)[unknown]
        assert noise.abs().min() > 0, 'some unknown pixels got no noise'
    finally:
        handle.remove()
    ok('known pixels reach the network clean; unknown pixels are noised')


def check_sampler(model, args):
    print('4. clean-known sampling = DiffusionProcess.impute', flush=True)
    net = model.net
    process = DiffusionProcess(args, net, (1, 8, 8))
    s_min, s_max = max(process.sigma_min, net.sigma_min), min(process.sigma_max, net.sigma_max)
    idx = torch.arange(process.num_steps, dtype=torch.float64)
    t_steps = (s_max ** (1 / process.rho) + idx / (process.num_steps - 1)
               * (s_min ** (1 / process.rho) - s_max ** (1 / process.rho))) ** process.rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    B = 4
    known_ts = torch.zeros(B, 64, 1, dtype=torch.float64)
    known_ts[:, :16, 0] = torch.linspace(-0.5, 2.0, 16, dtype=torch.float64)
    x_known = model.ts_to_img(known_ts).to(torch.float64)
    m = torch.zeros(B, 64, 1)
    m[:, :16] = 1
    mask = model.ts_to_img(m).to(torch.float64)
    labels = torch.eye(4)[[0, 1, 2, 3]]
    with torch.no_grad():
        torch.manual_seed(7)
        latents = torch.randn(B, 1, 8, 8, dtype=torch.float64)
        ref = process.impute(x_known, latents, mask, class_labels=labels)
        torch.manual_seed(7)
        mine = sample_block(net, x_known, mask, labels, t_steps, [1] * (len(t_steps) - 1), clean_known=True)
    unknown = mask == 0
    err = (mine[unknown] - ref[unknown]).abs().max().item()
    assert err < 1e-5, err
    assert torch.equal(mine[mask == 1], x_known[mask == 1])
    try:
        sample_block(net, x_known, mask, labels, t_steps, [2] * (len(t_steps) - 1), clean_known=True)
        raise AssertionError('resampling with clean known pixels was accepted')
    except ValueError:
        pass
    ok(f'unknown pixels match impute() to {err:.1e}; known pixels returned exactly')


def check_checkpoint_loading(model):
    print('5. checkpoint loading', flush=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    load_checkpoint_safe(model, 'none', 'cpu')
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    try:
        load_checkpoint_safe(model, '/no/such/ImagenFew_64.ckpt', 'cpu')
        raise AssertionError('a missing checkpoint was accepted')
    except FileNotFoundError:
        pass
    ok("'none' keeps random weights; a missing path is an error")


def check_seeding():
    print('6. seeding', flush=True)
    x = torch.randn(256, 64, 1)
    steps = torch.tensor([8, 16, 24, 32])
    g1 = seed_everything(3)
    a = known_prefix_mask(x, 0.75, steps)
    g2 = seed_everything(3)
    b = known_prefix_mask(x, 0.75, steps)
    assert torch.equal(a, b) and torch.equal(torch.randperm(100, generator=g1), torch.randperm(100, generator=g2))
    ok('same seed, same masks and same data order')


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    check_scalers()
    args, model = build('config_v17.yaml')
    if os.path.exists(V14):
        state = torch.load(V14, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=False)
        print(f'   (checks 3-4 use the v14 checkpoint {os.path.relpath(V14, ROOT)})')
    check_masks(model, args)
    check_forward(model)
    check_sampler(model, args)
    check_checkpoint_loading(build('config_v14.yaml')[1])
    check_seeding()
    print('all checks passed')


if __name__ == '__main__':
    main()
