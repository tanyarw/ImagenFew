#!/usr/bin/env python
"""
diagnose_storm_duration_cliff.py — show that the storm-duration survival curve of every
generated version collapses at exactly that version's block length (`seq_len`).

Motivation and full write-up: `docs/ATTENTION_AND_RECEPTIVE_FIELD_AUDIT.md` §3.3.

Each 64-step (or 24-, or 36-step) block is generated independently
(`regime_training/generate_hmm_v1.py` passes an all-zero known-pixel mask), so a storm
longer than one block can only occur if two consecutive blocks happen to be heavy at the
same time. The prediction is a discontinuity in P(duration > k) at k = seq_len, and a
different discontinuity per version. This script measures it.

Storm definition is the canonical one (`day_1.md` §3.2 def. 6, as in
`scripts/gate_a_scorecard.py`): a contiguous run of wet steps (>= 0.005 mm) of length
>= 3 steps. No gap tolerance.

Pure numpy / pandas / matplotlib — runs in the local .venv, no torch.

    python scripts/diagnose_storm_duration_cliff.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, 'results', 'generated_data')
WET_THR = 0.005            # day_1.md §2.1, frozen
STEPS_PER_YEAR = 365 * 288

# dataviz categorical slots 1-3 (validated: worst adjacent CVD dE 9.2, normal 27.6) plus
# the neutral text-secondary grey for the observed record.
REAL_C = '#52514e'
SERIES_C = ['#2a78d6', '#eb6834', '#1baf7a']
GRID_C = '#d8d7d2'

# (label, csv basename, seq_len). seq_len is the generated block length in 5-min steps.
DEFAULT_RUNS = [
    ('v8  (L=24)', 'rainfall_synthetic_10y_v8.csv', 24),
    ('v9  (L=36)', 'rainfall_synthetic_10y_v9.csv', 36),
    ('v10 (L=64)', 'rainfall_synthetic_10y_v10.csv', 64),
]
REAL_CSV = os.path.join('data', 'rainfall', 'splits', 'train_years_labelled.csv')


def load(path):
    """Read a rainfall series and apply the frozen wet threshold."""
    df = pd.read_csv(path)
    col = 'avg_rainfall' if 'avg_rainfall' in df.columns else df.columns[1]
    a = df[col].clip(lower=0).to_numpy(dtype=float)
    a[a < WET_THR] = 0.0
    return a


def storm_durations(a, min_steps=3):
    """Lengths (in steps) of contiguous wet runs of >= min_steps. Canonical definition."""
    wet = (a > 0).astype(int)
    ch = np.diff(wet, prepend=-1)
    starts = np.where(ch != 0)[0]
    lengths = np.diff(np.append(starts, len(a)))
    types = wet[starts]
    return np.array([l for l, t in zip(lengths, types) if t == 1 and l >= min_steps])


def survival(d, ks):
    """P(duration > k) for each k in ks."""
    return np.array([(d > k).mean() for k in ks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default=REAL_CSV, help='observed record (default: train split)')
    ap.add_argument('--out', default=os.path.join('results', 'transform_diagnostics',
                                                  'storm_duration_cliff.png'))
    args = ap.parse_args()

    ks = np.array([3, 6, 9, 12, 18, 24, 30, 36, 48, 64, 80, 96, 128])

    real = load(os.path.join(ROOT, args.real))
    d_real = storm_durations(real)
    s_real = survival(d_real, ks)

    runs = []
    for label, base, L in DEFAULT_RUNS:
        path = os.path.join(GEN, base)
        if not os.path.exists(path):
            print(f'  skip (missing): {base}')
            continue
        a = load(path)
        d = storm_durations(a)
        runs.append((label, L, d, survival(d, ks)))

    # ---- console report (the numbers quoted in the audit) --------------------------
    print('\nCANONICAL storm def (contiguous wet run >= 3 steps, no gap tolerance)')
    print('Survival P(duration > k), and the generated/real ratio\n')
    head = '  %-13s' % 'k (steps)' + ''.join('%8d' % k for k in ks)
    print(head)
    print('  %-13s' % '(minutes)' + ''.join('%8d' % (k * 5) for k in ks))
    print('  ' + '-' * (len(head) - 2))
    print('  %-13s' % 'real' + ''.join('%7.2f%%' % (100 * v) for v in s_real))
    for label, L, d, s in runs:
        print('  %-13s' % label + ''.join('%8.2f' % (a / b if b > 0 else np.nan)
                                          for a, b in zip(s, s_real))
              + '   <- block L=%d' % L)

    print('\nStorms per year and long-storm volume share:')
    yy_r = len(real) / STEPS_PER_YEAR
    print('  %-13s storms/yr %6.1f   mean dur %5.1f min' % ('real', len(d_real) / yy_r,
                                                            d_real.mean() * 5))
    for label, L, d, s in runs:
        print('  %-13s storms/yr %6.1f   mean dur %5.1f min   P(>L) ratio %.3f'
              % (label, len(d) / 10.0, d.mean() * 5,
                 (d > L).mean() / max((d_real > L).mean(), 1e-12)))

    # ---- figure -------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    fig.patch.set_facecolor('#fcfcfb')
    mins = ks * 5

    # Panel A — absolute survival curves.
    ax0.plot(mins, 100 * s_real, color=REAL_C, lw=2.4, marker='o', ms=5,
             label='observed (2000–2007)', zorder=5)
    for (label, L, d, s), c in zip(runs, SERIES_C):
        ax0.plot(mins, 100 * s, color=c, lw=2, marker='o', ms=5, label=label)
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel('storm duration threshold  (minutes)')
    ax0.set_ylabel('P(duration > threshold)   [%]')
    ax0.set_title('A · Storm-duration survival curve', loc='left', fontsize=11,
                  color='#0b0b0b')

    # Panel B — the ratio, where the cliff is unmistakable.
    ax1.axhline(1.0, color=REAL_C, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax1.text(mins[0] * 1.04, 1.03, 'matches observed', fontsize=8.5, color=REAL_C,
             va='bottom')
    # Stagger the block-length annotations: L=24 and L=36 are close together on a log
    # axis and their labels collide if placed at a common height.
    label_y = [1.30, 1.13, 1.30]
    for i, ((label, L, d, s), c) in enumerate(zip(runs, SERIES_C)):
        ratio = np.divide(s, s_real, out=np.full_like(s, np.nan), where=s_real > 0)
        ax1.plot(mins, ratio, color=c, lw=2, marker='o', ms=5, label=label, zorder=5)
        # vertical marker at this version's block length — the predicted cliff location
        ax1.axvline(L * 5, color=c, lw=1.2, ls=(0, (2, 3)), alpha=0.75, zorder=1)
        ax1.annotate('L=%d  (%d min)' % (L, L * 5), xy=(L * 5, label_y[i % len(label_y)]),
                     ha='center', va='bottom', fontsize=8.5, color=c)
    ax1.set_xscale('log')
    ax1.set_ylim(-0.04, 1.46)
    ax1.set_xlabel('storm duration threshold  (minutes)')
    ax1.set_ylabel('generated / observed')
    ax1.set_title('B · The cliff sits at each version\'s own block length', loc='left',
                  fontsize=11, color='#0b0b0b')

    for ax in (ax0, ax1):
        ax.grid(True, which='major', color=GRID_C, lw=0.7, alpha=0.9)
        ax.grid(True, which='minor', color=GRID_C, lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID_C)
        ax.tick_params(colors='#52514e', labelsize=9)
        ax.set_facecolor('#fcfcfb')
        ax.legend(frameon=False, fontsize=9, labelcolor='#52514e', loc='lower left')

    fig.suptitle('Every version reproduces storm durations up to its block length, then collapses',
                 x=0.008, ha='left', fontsize=12.5, color='#0b0b0b')
    fig.text(0.008, 0.005,
             'Blocks are generated independently, so a storm cannot span a block boundary '
             'except by coincidence. Raising seq_len moved the wall; it did not remove it.',
             ha='left', fontsize=8.5, color='#52514e')
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    print('\nwrote %s' % os.path.relpath(out, ROOT))


if __name__ == '__main__':
    main()
