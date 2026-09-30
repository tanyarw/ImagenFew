#!/usr/bin/env python
"""
ablate_block_assembly.py — does HOW the generated blocks are joined matter?

Generation makes every 64-step block independently (walkthrough:
notebooks/bridging_and_stitching_tutorial.ipynb) and then joins them. Two parts of the
joining are hand-made choices:

  bridge : at each state change, insert one extra block whose label is a 50/50 blend of the
           two states (e.g. [0, .5, 0, .5]); the model never saw blended labels in training
  xfade  : overlap neighbouring blocks by 4 steps (8 at state changes) and crossfade them

This script generates ONE set of blocks (Markov plan, same flags as the production jobs) and
joins the SAME blocks four ways, so the variants differ only in the joining:

  A  bridge + xfade      (production)
  B  no bridge + xfade
  C  bridge + hard join  (blocks placed end to end, no overlap)
  D  no bridge + hard join

"No bridge" drops the bridge blocks; every normal block is identical across variants.

Stage 1 (torch; GPU or CPU): generate the blocks, save them in mm  -> <out>/blocks_<run>.npz
Stage 2 (numpy only):        join four ways and score              -> <out>/assembly_<run>.csv
Stage 1 is skipped when the .npz already exists, and resumes from <out>/*.partial.npz if an
earlier run was interrupted (blocks are saved after every chunk).

    python scripts/ablate_block_assembly.py --run v14 --years 2 --device cpu    # ~20 min on a laptop
    python scripts/ablate_block_assembly.py --run v14 --years 10                # GPU (cluster)
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import gate_a_scorecard as sc  # noqa: E402  (core_stats, acf, hourly: the Gate A definitions)

WET_THR = 0.005
STEPS_PER_YEAR = 105120
RUN_DIRS = {'v10': 'aebe363f'}          # everything else lives under its own name
VARIANTS = [('A bridge + xfade', True, True), ('B no bridge + xfade', False, True),
            ('C bridge + hard join', True, False), ('D no bridge + hard join', False, False)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='v14', help='version name, e.g. v10, v13, v14')
    p.add_argument('--years', type=float, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--chunk', type=int, default=1024, help='blocks per sampler batch')
    p.add_argument('--overlap', type=int, default=4)
    p.add_argument('--transition_overlap', type=int, default=8)
    p.add_argument('--out_dir', default='results/assembly_ablation')
    return p.parse_args()


def generate_blocks(cli, plan, npz):
    """Stage 1: sample every block in the plan and save them in mm (before thresholding)."""
    import torch
    from omegaconf import OmegaConf
    import regime_training.generate_hmm_v1 as gen
    from models.ImagenFew.ImagenFew import ImagenFew

    run_dir = os.path.join(ROOT, 'logs', 'ImagenFew', 'Rainfall_Regime', RUN_DIRS.get(cli.run, cli.run))
    args = argparse.Namespace(**OmegaConf.to_container(
        OmegaConf.load(os.path.join(ROOT, 'regime_training', f'config_{cli.run}.yaml')), resolve=True))
    args.device = cli.device
    with open(os.path.join(run_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Load weights as gen.main() does: shape-matched weights, then EMA weights
    torch.manual_seed(cli.seed)
    model = ImagenFew(args, args.device).to(args.device)
    ckpt = torch.load(os.path.join(run_dir, 'best_regime_model.pt'), map_location=args.device,
                      weights_only=False)
    cur = model.state_dict()
    model.load_state_dict({k: v for k, v in ckpt.get('model', ckpt).items()
                           if k in cur and v.shape == cur[k].shape}, strict=False)
    cur_ema = model.model_ema.state_dict()
    model.model_ema.load_state_dict({k: v for k, v in ckpt['ema_model'].items()
                                     if k in cur_ema and v.shape == cur_ema[k].shape}, strict=False)
    model.eval()
    process = gen.DiffusionProcess(args, model.net, (1, args.img_resolution, args.img_resolution))

    # Blocks are saved after every chunk, so an interrupted run resumes where it stopped
    labels = [p['label_weights'] for p in plan]
    part = npz.replace('.npz', '.partial.npz')
    blocks = list(np.load(part)['blocks_mm']) if os.path.exists(part) else []
    if blocks:
        print(f'  resuming after {len(blocks)} saved blocks', flush=True)
        torch.manual_seed(cli.seed + len(blocks))
    t0 = time.time()
    with model.ema_scope():
        for i in range(len(blocks), len(labels), cli.chunk):
            new = gen.batch_generate_blocks(model, process, labels[i:i + cli.chunk], args,
                                            batch_size=cli.chunk)
            blocks += [scaler.inverse_transform(b.reshape(-1, 1)).ravel().astype(np.float32) for b in new]
            np.savez_compressed(part, blocks_mm=np.stack(blocks))
            print(f'  {len(blocks)}/{len(labels)} blocks  ({time.time() - t0:.0f}s)', flush=True)
    mm = np.stack(blocks)
    np.savez_compressed(npz, blocks_mm=mm,
                        types=np.array([p['type'] for p in plan]),
                        states=np.array([-1 if p['state'] is None else p['state'] for p in plan]))
    os.remove(part)
    return mm


def assemble(blocks, plan, bridge, xfade, cli, total):
    """Join blocks the way generate_hmm_v1.main() does (mm first, stitch, trim, threshold)."""
    import regime_training.generate_hmm_v1 as gen
    keep = [i for i, p in enumerate(plan) if bridge or p['type'] != 'bridge']
    ov, tov = (cli.overlap, cli.transition_overlap) if xfade else (0, 0)
    a = gen.stitch_blocks_transition_aware([blocks[i] for i in keep], [plan[i] for i in keep],
                                           base_overlap=ov, transition_overlap=tov)
    assert len(a) >= total, f'plan too short for this variant ({len(a)} < {total})'
    a = a[:total].astype(np.float64)
    a[a < WET_THR] = 0.0
    return a


def score(a, real_hourly_acf):
    s = sc.core_stats(a)
    runs = sc.storms(a)
    ny = len(a) / STEPS_PER_YEAR
    long_ = [r for r in runs if len(r) > 64]
    return {
        'volume mm/yr': s['volume'],
        'zero %': s['zero_pct'],
        'wet spell min': s['wet_spell_min'],
        'storms/yr': s['storm_count_yr'],
        'storm dur min': s['storm_dur_min'],
        'P(storm > 240 min) %': 100 * np.mean([len(r) > 48 for r in runs]),
        'storms > 320 min /yr': len(long_) / ny,
        'rain in storms > 320 min %': 100 * sum(r.sum() for r in long_) / a.sum(),
        'P99 wet': s['p99_wet'],
        'P99.9 wet': s['p999_wet'],
        'max 5-min': s['max_burst'],
        'lag-1 ACF': s['acf1_native'],
        'hourly ACF RMSE 1-24h': float(np.sqrt(np.mean((sc.acf(sc.hourly(a), 24)[1:] - real_hourly_acf) ** 2))),
    }


def main():
    cli = parse_args()
    import regime_training.generate_hmm_v1 as gen
    out_dir = os.path.join(ROOT, cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Markov plan exactly as the production jobs build it (same seed, same transition matrix)
    L = 64
    total = int(cli.years * STEPS_PER_YEAR)
    n_base = int(np.ceil(total / (L - cli.overlap))) + 10
    with open(os.path.join(ROOT, 'data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl'), 'rb') as f:
        tm = pickle.load(f)
    states = gen.sample_markov_state_sequence(tm['P_block'], tm['pi_block'], n_base, seed=cli.seed)
    plan = gen.build_generation_plan(states, n_bridge=1)
    n_br = sum(p['type'] == 'bridge' for p in plan)
    print(f'{cli.run}: {cli.years} y, {len(plan)} blocks ({n_br} bridges)')

    npz = os.path.join(out_dir, f'blocks_{cli.run}_{cli.years:g}y_seed{cli.seed}.npz')
    if os.path.exists(npz):
        blocks = np.load(npz)['blocks_mm']
        print(f'loaded {len(blocks)} blocks from {os.path.relpath(npz, ROOT)}')
    else:
        blocks = generate_blocks(cli, plan, npz)
    assert len(blocks) == len(plan)

    real = sc.load(os.path.join(ROOT, 'data/rainfall/splits/train_years_labelled.csv'))['avg_rainfall'].to_numpy()
    real_acf = sc.acf(sc.hourly(real), 24)[1:]
    rows = {'real (2000-2007)': score(real, real_acf)}
    for name, bridge, xfade in VARIANTS:
        rows[name] = score(assemble(blocks, plan, bridge, xfade, cli, total), real_acf)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, f'assembly_{cli.run}_{cli.years:g}y_seed{cli.seed}.csv')
    df.to_csv(out_csv)
    pd.set_option('display.width', 160)
    print('\n' + df.round(3).to_string())
    print(f'\nwrote {os.path.relpath(out_csv, ROOT)}')


if __name__ == '__main__':
    main()
