#!/usr/bin/env python
"""
generate_context.py — E2: generate each block with the end of the previous block as context.

Production generation (generate_hmm_v1.py) makes every 64-step block independently and joins
the blocks afterwards, so a storm can only continue across a block boundary by coincidence
(the storm-duration cliff; code_plan/NEXT_STEPS_AFTER_v14.md, section E2). Here every block
after the first is an in-fill problem:

  * the first K columns of the 8x8 image are KNOWN: they hold the last K*8 steps (K*40 min)
    that the previous block generated;
  * the model fills in the other 8-K columns, which become the next 64 - 8K steps.

With delay == embedding == 8, image column j is exactly steps 8j..8j+7, so "the first K
columns" means "the first K*8 steps". This is checked at start-up.

Sampler: the production EDM Heun sampler (36 steps, sigma 80 -> 0.002, rho 7), with the known
region handled as in RePaint (Lugmayr et al., CVPR 2022). The fine-tuned model never saw
known pixels in training, so:
  * the known pixels are re-noised to the current noise level at every step
    (x_known + sigma * eps), so the network always sees one noise level, as in training;
  * resampling: steps with sigma >= --resample_min_sigma are repeated --resample times,
    jumping back up to that step's noise level in between, so the filled-in part harmonises
    with the known part. Intermediate passes take an Euler step; the final pass the usual
    Heun step.
Noisy history (Diffusion Forcing, Chen et al., NeurIPS 2024): the known values are the
previous block's output plus a small fixed noise (--history_noise, in model units), so the
model is never handed its own output exactly, ~22,000 times in a row.

Labels: one-hot HMM state per block, looked up by TIME on the same state timeline as the
production run with the same seed. Markov: the seed's production plan, including its bridge
offsets, so year y here lines up with year y of v14 (the drift test compares them year by
year). Calendar: the climatology profile indexed by the step's time of year, so there is no
calendar drift. No bridge blocks: the context does their job.

--members independent chains run in parallel as one batch; each is a full-length chain.
Member 0 -> <name>.csv, member m -> <name>_m<m>.csv. Progress is saved every --save_every
blocks and rerunning the same command resumes.

    python regime_training/generate_context.py --run v14 --mode markov   --members 4
    python regime_training/generate_context.py --run v14 --mode calendar --members 4
    python regime_training/generate_context.py --run v14 --years 0.01 --members 2 --device cpu   # smoke test
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import regime_training.generate_hmm_v1 as gen  # noqa: E402  (plan builders; DiffusionProcess settings)
from models.ImagenFew.ImagenFew import ImagenFew  # noqa: E402

STEPS_PER_YEAR = 105120
WET_THR = 0.005
RUN_DIRS = {'v10': 'aebe363f'}          # everything else lives under its own name


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='v14', help='trained version to sample from')
    p.add_argument('--mode', default='markov', choices=['markov', 'calendar'])
    p.add_argument('--years', type=float, default=10)
    p.add_argument('--seed', type=int, default=42, help='plan seed (as production) and noise seed')
    p.add_argument('--members', type=int, default=4, help='independent chains, run as one batch')
    p.add_argument('--context_cols', type=int, default=2,
                   help='K known columns (K*40 min of history); 0 = independent blocks, the no-context control')
    p.add_argument('--resample', type=int, default=10, help='RePaint passes per step (1 = no resampling)')
    p.add_argument('--resample_min_sigma', type=float, default=0.1,
                   help='only steps with sigma >= this are resampled')
    p.add_argument('--history_noise', type=float, default=0.05,
                   help='std of the fixed noise added to the known history (model units)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--save_every', type=int, default=500, help='blocks between progress saves')
    p.add_argument('--output_dir', default='results/generated_data')
    p.add_argument('--output_name', default=None,
                   help='default: rainfall_synthetic_10y_<run>_ctx[_cal]')
    p.add_argument('--transition_matrix_path',
                   default='data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl')
    # Production plan settings, only used to rebuild the Markov state timeline
    p.add_argument('--overlap', type=int, default=4)
    p.add_argument('--transition_overlap', type=int, default=8)
    p.add_argument('--bridge_blocks', type=int, default=1)
    return p.parse_args()


def load_model(cli, args):
    """Build ImagenFew and load weights the way generate_hmm_v1.main() does."""
    run_dir = os.path.join(PROJECT_ROOT, 'logs', 'ImagenFew', 'Rainfall_Regime', RUN_DIRS.get(cli.run, cli.run))
    with open(os.path.join(run_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    model = ImagenFew(args, args.device).to(args.device)
    ckpt = torch.load(os.path.join(run_dir, 'best_regime_model.pt'), map_location=args.device, weights_only=False)
    cur = model.state_dict()
    model.load_state_dict({k: v for k, v in ckpt.get('model', ckpt).items()
                           if k in cur and v.shape == cur[k].shape}, strict=False)
    if 'ema_model' in ckpt and args.ema:
        cur_ema = model.model_ema.state_dict()
        model.model_ema.load_state_dict({k: v for k, v in ckpt['ema_model'].items()
                                         if k in cur_ema and v.shape == cur_ema[k].shape}, strict=False)
    model.eval()
    return model, scaler


def state_timeline(cli, total, L):
    """HMM state of every output step, on the same timeline as the production run."""
    with open(os.path.join(PROJECT_ROOT, cli.transition_matrix_path), 'rb') as f:
        tm = pickle.load(f)
    if cli.mode == 'calendar':
        clim = np.asarray(tm['climatology_profile_1yr'], dtype=int)
        return clim[np.arange(total) % len(clim)]

    # Markov: rebuild the production plan for this seed and lay it out exactly as
    # stitch_blocks_transition_aware does (including the time the bridge blocks take)
    stride = L - cli.overlap
    n_base = int(np.ceil(total / stride)) + 10
    states = gen.sample_markov_state_sequence(tm['P_block'], tm['pi_block'], n_base, seed=cli.seed)
    plan = gen.build_generation_plan(states, n_bridge=cli.bridge_blocks)
    timeline = np.zeros(total, dtype=int)
    pos = 0
    for i, p in enumerate(plan):
        if pos >= total:
            break
        # a bridge block's time goes to the state it leads into
        s = p['state'] if p['state'] is not None else next(q['state'] for q in plan[i:] if q['state'] is not None)
        timeline[pos:pos + L] = s
        if i < len(plan) - 1:
            q = plan[i + 1]
            change = p['state'] != q['state'] or 'bridge' in (p['type'], q['type'])
            pos += L - min(cli.transition_overlap if change else cli.overlap, L - 1)
    return timeline


@torch.no_grad()
def sample_block(net, x_known, mask, labels, t_steps, passes_per_step):
    """EDM Heun sampling with RePaint conditioning on the known pixels (mask == 1)."""
    x = torch.randn_like(x_known) * t_steps[0]
    n = len(t_steps) - 1
    for i in range(n):
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        passes = passes_per_step[i]
        for u in range(passes):
            x = mask * (x_known + t_cur * torch.randn_like(x)) + (1 - mask) * x
            d = (x - net(x, t_cur, labels).to(torch.float64)) / t_cur
            x_next = x + (t_next - t_cur) * d
            last = u == passes - 1
            if last and i < n - 1:                      # Heun correction, as in production
                d2 = (x_next - net(x_next, t_next, labels).to(torch.float64)) / t_next
                x_next = x + (t_next - t_cur) * (0.5 * d + 0.5 * d2)
            x = x_next if last else x_next + (t_cur ** 2 - t_next ** 2).sqrt() * torch.randn_like(x_next)
    return mask * x_known + (1 - mask) * x


def main():
    cli = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(
        PROJECT_ROOT, 'regime_training', f'config_{cli.run}.yaml')), resolve=True)
    args = argparse.Namespace(**cfg)
    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; pass --device cpu only for a small smoke test')
    args.device = cli.device
    L, M, K = args.seq_len, cli.members, cli.context_cols
    known_steps = 8 * K
    total = int(cli.years * STEPS_PER_YEAR)
    name = cli.output_name or f"rainfall_synthetic_10y_{cli.run}_ctx{'_cal' if cli.mode == 'calendar' else ''}"
    out_dir = os.path.join(PROJECT_ROOT, cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, f'{name}.ctx_state.npz')

    model, scaler = load_model(cli, args)
    net = model.net
    process = gen.DiffusionProcess(args, net, (1, args.img_resolution, args.img_resolution))

    # Start-up check: the first K image columns must hold exactly the first 8K steps
    tmask = torch.zeros(1, L, 1, device=args.device)
    tmask[:, :known_steps] = 1
    img_mask = model.ts_to_img(tmask)
    probe = torch.arange(L, dtype=torch.float32, device=args.device).view(1, L, 1)
    if not (args.delay == args.embedding == 8 and L == 64 and 0 <= K < 8
            and int(img_mask.sum()) == known_steps and int(img_mask[..., :K].sum()) == known_steps
            and torch.equal(model.img_to_ts(model.ts_to_img(probe)).cpu(), probe.cpu())):
        raise RuntimeError('context generation assumes seq_len 64, delay = embedding = 8 and 0 <= K < 8')
    mask = img_mask.to(torch.float64).expand(M, -1, -1, -1).contiguous()

    # Noise schedule exactly as DiffusionProcess.impute()
    s_min, s_max = max(process.sigma_min, net.sigma_min), min(process.sigma_max, net.sigma_max)
    idx = torch.arange(process.num_steps, dtype=torch.float64, device=args.device)
    t_steps = (s_max ** (1 / process.rho) + idx / (process.num_steps - 1)
               * (s_min ** (1 / process.rho) - s_max ** (1 / process.rho))) ** process.rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    n_res = int((t_steps[:-2] >= cli.resample_min_sigma).sum())
    # RePaint passes per solver step (the last step, to sigma = 0, is never resampled)
    passes = [cli.resample if (cli.resample > 1 and i < len(t_steps) - 2
                               and float(t_steps[i]) >= cli.resample_min_sigma) else 1
              for i in range(len(t_steps) - 1)]
    single = [1] * (len(t_steps) - 1)                 # first block: nothing known, no resampling
    evals = n_res * (cli.resample - 1 + 2) + (len(t_steps) - 1 - n_res) * 2 - 1
    new_per_block = L - known_steps
    n_blocks = 1 + int(np.ceil(max(total - L, 0) / new_per_block))

    timeline = state_timeline(cli, total, L)
    n_cls = getattr(args, 'n_classes', 0)
    eye = torch.eye(max(n_cls, 1), device=args.device)

    print(f'{cli.run} | {cli.mode} | {cli.years} y = {total} steps | {M} members | K = {K} '
          f'({known_steps} known + {new_per_block} new steps per block) | {n_blocks} blocks x '
          f'~{evals} network evals | resample {cli.resample} for sigma >= {cli.resample_min_sigma} '
          f'({n_res} steps) | history noise {cli.history_noise}', flush=True)

    # Resume if a matching progress file exists
    key = np.array([cli.run, cli.mode, str(cli.years), str(cli.seed), str(M), str(K), str(cli.resample),
                    str(cli.resample_min_sigma), str(cli.history_noise)])
    out = np.zeros((M, total), dtype=np.float32)
    ctx, pos, b = None, 0, 0
    if os.path.exists(state_path):
        st = np.load(state_path)
        if not np.array_equal(st['key'], key):
            raise RuntimeError(f'{state_path} was made with different settings; delete it to start over')
        pos, b = int(st['pos']), int(st['b'])
        out[:, :pos] = st['out']
        ctx = torch.as_tensor(st['ctx'], dtype=torch.float64, device=args.device)
        print(f'resuming at block {b}, step {pos}', flush=True)

    torch.manual_seed(cli.seed + b)
    t0, b0 = time.time(), b
    with model.ema_scope():
        while pos < total:
            known = torch.zeros(M, L, 1, dtype=torch.float64, device=args.device)
            if ctx is None or K == 0:                         # first block (or K = 0): nothing known
                m, first_new = torch.zeros_like(mask), 0
            else:
                known[:, :known_steps, 0] = ctx + cli.history_noise * torch.randn_like(ctx)
                m, first_new = mask, known_steps
            x_known = model.ts_to_img(known).to(torch.float64)
            n_new = L - first_new
            t_mid = min(pos + n_new // 2, total - 1)
            labels = eye[timeline[t_mid]].expand(M, -1) if n_cls > 0 else None

            img = sample_block(net, x_known, m, labels, t_steps, single if (ctx is None or K == 0) else passes)
            ts = model.img_to_ts(img.to(torch.float32))[:, :, 0].to(args.device, torch.float64)   # [M, L]

            new = ts[:, first_new:].cpu().numpy()
            take = min(n_new, total - pos)
            out[:, pos:pos + take] = scaler.inverse_transform(new[:, :take].reshape(-1, 1)).reshape(M, take)
            pos += take
            ctx = ts[:, L - known_steps:]
            b += 1

            if b % 100 == 0 or pos >= total:
                rate = (time.time() - t0) / (b - b0)
                done = out[:, :pos]
                # running check against the real record (9.2% wet, 710 mm/yr) and v14 (9.0%, 716)
                print(f'  block {b}/{n_blocks}  step {pos}/{total}  {rate:.2f} s/block  '
                      f'ETA {(n_blocks - b) * rate / 3600:.1f} h  |  so far: wet '
                      f'{100 * (done >= WET_THR).mean():.1f}%, '
                      f'{done[done >= WET_THR].sum() / M / (pos / STEPS_PER_YEAR):.0f} mm/yr', flush=True)
            if b % cli.save_every == 0 and pos < total:
                np.savez(state_path, key=key, pos=pos, b=b, out=out[:, :pos], ctx=ctx.cpu().numpy())

    out[out < WET_THR] = 0.0
    dates = pd.date_range(start='2026-01-01', periods=total, freq='5min')
    for mbr in range(M):
        path = os.path.join(out_dir, f'{name}{"" if mbr == 0 else f"_m{mbr}"}.csv')
        pd.DataFrame({'date': dates, 'avg_rainfall': out[mbr]}).to_csv(path, index=False)
        a = out[mbr]
        print(f'wrote {os.path.relpath(path, PROJECT_ROOT)}  | zero {100 * (a == 0).mean():.2f}%  '
              f'| {a.sum() / (total / STEPS_PER_YEAR):.1f} mm/yr  | max {a.max():.3f} mm', flush=True)
    if os.path.exists(state_path):
        os.remove(state_path)


if __name__ == '__main__':
    main()
