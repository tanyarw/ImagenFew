#!/usr/bin/env python
"""
base_paper_metrics.py — the ImagenFew paper's own metrics (discriminative score, predictive score,
context-FID) on every generated version, for the metric-validity analysis
(scripts/metric_validity.py: which metrics predict the sewer error?).

Two steps, because the 10-year series are not in git:

  pack   (laptop)  64-step windows, non-overlapping, 3,000 drawn per series with seed 0, in mm:
                   real 2000-2007 (the reference), real 2008-2009 (the real-vs-real floor) and
                   every version in results/evaluation/eval_cache.pkl that has a 10-year CSV.
                   -> results/evaluation/base_paper_windows.npz (~25 MB; copy it to the cluster)
  score  (cluster) min-max scaled by the reference (min 0, max of the reference windows), then
                   * discriminative score: metrics/discriminative_torch.py unchanged
                     (|0.5 - accuracy| of a post-hoc GRU; 0 = can't tell), mean of --iters runs;
                   * predictive score: TimeGAN's protocol (train a post-hoc GRU on generated windows,
                     MAE on real ones). The TimeGAN code predicts the last channel from the others and
                     is undefined for one channel, so here the GRU predicts x[t+1] from x[<=t]. Hidden
                     size 1, as both TimeGAN networks use for one channel (max(1, dim // 2));
                   * context-FID: metrics/context_fid.py's TS2Vec (fitted once on the reference
                     windows) and Fréchet distance of the full-window embeddings.
                   -> results/evaluation/base_paper_metrics.json

    python scripts/base_paper_metrics.py pack
    python scripts/base_paper_metrics.py score --device cuda --iters 10      # sbatch scripts/submit_base_paper_metrics.sh
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GEN = os.path.join(ROOT, 'results', 'generated_data')
PACK = os.path.join(ROOT, 'results', 'evaluation', 'base_paper_windows.npz')
OUT = os.path.join(ROOT, 'results', 'evaluation', 'base_paper_metrics.json')
L, N = 64, 3000
REF = 'real 2000-2007'
TS2VEC = os.path.join(ROOT, 'logs', 'TS2VEC', f'Rainfall_{N}_{L}_1_ref2000-2007.ckpt')


def windows(a, rng):
    w = np.asarray(a, np.float32)[:len(a) // L * L].reshape(-1, L)
    return w[np.sort(rng.choice(len(w), min(N, len(w)), replace=False))]


def pack():
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from eval_suite import load_real
    train, held = load_real()
    cache = pickle.load(open(os.path.join(ROOT, 'results', 'evaluation', 'eval_cache.pkl'), 'rb'))
    out = {REF: windows(train, np.random.default_rng(0)), 'real 2008-2009': windows(held, np.random.default_rng(0))}
    for nm in cache['windows']:
        f = os.path.join(GEN, f'rainfall_synthetic_10y_{nm}.csv')
        if nm.startswith('real') or not os.path.exists(f):
            continue
        out[nm] = windows(pd.read_csv(f, usecols=['avg_rainfall']).avg_rainfall.to_numpy(), np.random.default_rng(0))
    np.savez_compressed(PACK, **out)
    print(f'{len(out)} series, {N} windows of {L} steps each -> {os.path.relpath(PACK, ROOT)}')


def predictive_score(real, fake, device, iters=5000, batch=128):
    """TimeGAN's post-hoc predictor, one channel: train on generated windows, MAE on real ones."""
    import torch
    import torch.nn as nn

    class P(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn, self.out = nn.GRU(1, 1, batch_first=True), nn.Linear(1, 1)

        def forward(self, x):
            return torch.sigmoid(self.out(self.rnn(x)[0]))

    net = P().to(device)
    opt = torch.optim.Adam(net.parameters())
    f = torch.as_tensor(fake, device=device)
    for _ in range(iters):
        x = f[torch.randint(len(f), (batch,), device=device)]
        loss = (net(x[:, :-1]) - x[:, 1:]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        r = torch.as_tensor(real, device=device)
        return float((net(r[:, :-1]) - r[:, 1:]).abs().mean())


def score(device, iters, only):
    import torch
    from metrics.context_fid import calculate_fid
    from metrics.discriminative_torch import discriminative_score_metrics
    from metrics.models.ts2vec.ts2vec import TS2Vec

    W = dict(np.load(PACK))
    hi = float(W[REF].max())
    X = {k: np.clip(v / hi, 0, None)[..., None].astype(np.float32) for k, v in W.items()}   # [n, L, 1]
    real = X[REF]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}

    torch.manual_seed(0)
    np.random.seed(0)
    t0 = time.time()
    enc = TS2Vec(input_dims=1, device=device, batch_size=8, lr=0.001, output_dims=320, max_train_length=3000)
    if os.path.exists(TS2VEC):                        # fitted once, so a restarted job scores on the same encoder
        enc.load(TS2VEC)
    else:
        os.makedirs(os.path.dirname(TS2VEC), exist_ok=True)
        enc.fit(real, verbose=False)
        enc.save(TS2VEC)
    real_emb = enc.encode(real, encoding_window='full_series')
    print(f'TS2Vec ready ({len(real)} reference windows) in {time.time() - t0:.0f} s', flush=True)

    for nm, x in X.items():
        if nm == REF or (only and nm not in only) or res.get(nm, {}).get('iters') == iters:
            continue
        t0 = time.time()
        disc, pred = [], []
        for i in range(iters):
            torch.manual_seed(i)
            np.random.seed(i)
            disc.append(float(discriminative_score_metrics(real, x, device)))
            pred.append(predictive_score(real, x, device))
        fid = float(calculate_fid(real_emb, enc.encode(x, encoding_window='full_series')))
        res[nm] = dict(disc_mean=float(np.mean(disc)), disc_std=float(np.std(disc)), pred_mean=float(np.mean(pred)),
                       pred_std=float(np.std(pred)), context_fid=fid, n_windows=len(x), iters=iters)
        print(f"{nm:32s} disc {res[nm]['disc_mean']:.4f}±{res[nm]['disc_std']:.4f}  "
              f"pred {res[nm]['pred_mean']:.5f}  cFID {fid:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)      # after every series, so a stopped job keeps its work

    # predictive score of a GRU trained on real windows (TRTR, the ceiling the generated ones are compared to)
    res['_reference'] = dict(scale_max_mm=hi, window=L, pred_trained_on_reference=float(np.mean(
        [predictive_score(real, real, device) for _ in range(iters)])))
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'-> {os.path.relpath(OUT, ROOT)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['pack', 'score'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--iters', type=int, default=10, help='repeats of the discriminative and predictive scores')
    ap.add_argument('--only', nargs='*', help='score only these series')
    a = ap.parse_args()
    pack() if a.step == 'pack' else score(a.device, a.iters, a.only)


if __name__ == '__main__':
    main()
