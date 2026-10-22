#!/usr/bin/env python
"""
export_deliverable.py — the synthetic rain handed to flood-control: one npz per member in the format of
flood-control's data/synthetic/v14_nobridge_cal.npz (analysis/synthetic_rain/vendor_v14.py), plus a manifest.

    python scripts/export_deliverable.py      # -> results/deliverable/v19_scratch_ep50/

Each npz holds `rain` (mean of the four gauges, mm per 5 min, float32, as the sewer check used it), `phase` (season
phase 0-3 of every step: the calendar profile the series was generated with, exact because both are laid out in
365-day years), `start`, the sha256 of the source CSV and profile, and what v14's file lacked: `member`,
`generation_seed`, `steps_per_year` and `calendar`. The steps are 365-day years (no 29 February; ImagenFew's series
since 2026-10-18), so dates rebuilt as a continuous 5-min range from `start` run one day ahead after 28 Feb 2028 and
two after 28 Feb 2032; the hour of the day, which is all the sewer model takes from a date, is unchanged.

manifest.json: the weights (path, epoch, sha256), the files' hashes, the sewer check per member and the 2009 test
verdict, each with the hash of the file it was read from. DATA_CARD.md next to it is written by hand.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from gate_b_sewer import check_steps  # noqa: E402  (365-day calendar check)

SERIES = 'v19_scratch_ep50_nobridge_cal'
MEMBERS = 4
SEED0 = 42                      # oneday/slurm/train.sbatch: member K is generated with seed 42 + K
PROFILE = ROOT / 'data/rainfall/splits/seasonal_transition_matrix_train_len288.pkl'
WEIGHTS = ROOT / 'logs/ImagenFew/Rainfall_Regime/v19_scratch/epoch_0050_regime_model.pt'
STEPS_PER_YEAR = 105120
SEWER = ROOT / 'results/reference/gate_b_sewer.json'
TEST = ROOT / 'results/testyear/test_2009.json'
PREREG = ROOT / 'testyear/preregistration.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def member_name(k):
    return SERIES if k == 0 else f'{SERIES}_m{k}'


def export_member(k, profile, out_dir):
    name = member_name(k)
    csv = ROOT / f'results/generated_data/rainfall_synthetic_10y_{name}.csv'
    df = pd.read_csv(csv, parse_dates=['date'])
    check_steps(df.date, name)
    rain = df.avg_rainfall.to_numpy(np.float32)
    if len(rain) % STEPS_PER_YEAR or (rain < 0).any() or not np.isfinite(rain).all():
        raise ValueError(f'{name}: {len(rain)} steps, or negative / non-finite rain')
    phase = profile[np.arange(len(rain)) % STEPS_PER_YEAR]
    path = out_dir / f'{name}.npz'
    np.savez_compressed(path, rain=rain, phase=phase, start=str(df.date.iloc[0]), csv_sha256=sha256(csv),
                        profile_sha256=sha256(PROFILE), member=k, generation_seed=SEED0 + k,
                        steps_per_year=STEPS_PER_YEAR, calendar='365-day years, no 29 February')
    z = np.load(path)                                     # read back: what is shipped is what was generated
    if not (np.array_equal(z['rain'], rain) and np.array_equal(z['phase'], phase)):
        raise RuntimeError(f'{path.name} does not read back')
    return name, path, rain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'results/deliverable/v19_scratch_ep50'))
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = np.asarray(pickle.loads(PROFILE.read_bytes())['climatology_profile_1yr'], dtype=np.int8)
    if len(profile) != STEPS_PER_YEAR or not set(np.unique(profile)) <= {0, 1, 2, 3}:
        raise ValueError(f'{PROFILE.name}: not one 365-day year of phases 0-3')

    import torch
    epoch = int(torch.load(WEIGHTS, map_location='cpu', weights_only=False)['epoch'])
    if epoch != 50:
        raise ValueError(f'{WEIGHTS.name} holds epoch {epoch}, not 50')

    sewer = json.loads(SEWER.read_text())['summary']
    test = json.loads(TEST.read_text())
    files, sewer_rows = {}, {}
    for k in range(MEMBERS):
        name, path, rain = export_member(k, profile, out_dir)
        files[path.name] = {'sha256': sha256(path), 'steps': len(rain), 'years': len(rain) // STEPS_PER_YEAR,
                            'mm_per_year': float(rain.sum() / (len(rain) / STEPS_PER_YEAR)),
                            'generation_seed': SEED0 + k}
        sewer_rows[name] = {r['controller']: {'total_m3_vs_real': r['total_m3_vs_real'], 'total_m3': r['total_m3']}
                            for r in sewer if r['source'] == name}
        if set(sewer_rows[name]) != {'BC', 'EFD'}:
            raise ValueError(f'{name}: sewer check missing for {sorted({"BC", "EFD"} - set(sewer_rows[name]))}')
        print(f'{path.name}: {len(rain)} steps, {files[path.name]["mm_per_year"]:.0f} mm/yr, '
              f'BC {sewer_rows[name]["BC"]["total_m3_vs_real"]:.2f} of real')
    manifest = {
        'deliverable': 'v19 scratch, epoch 50: mean rain of the Astlingen catchment, 5 min, 4 members x 10 years',
        'data_card': 'DATA_CARD.md',
        'weights': {'path': str(WEIGHTS.relative_to(ROOT)), 'epoch': epoch, 'sha256': sha256(WEIGHTS),
                    'run_args': json.loads((WEIGHTS.parent / 'run_args.json').read_text())},
        'generation': {'script': 'regime_training/generate_hmm_v1.py via oneday/slurm/train.sbatch (job 994673)',
                       'assembly': 'calendar, no bridge blocks, overlap 4, transition overlap 8',
                       'profile': str(PROFILE.relative_to(ROOT)), 'profile_sha256': sha256(PROFILE)},
        'files': files,
        'sewer_check': {'source': str(SEWER.relative_to(ROOT)), 'sha256': sha256(SEWER), 'members': sewer_rows},
        'test_2009': {'source': str(TEST.relative_to(ROOT)), 'sha256': sha256(TEST), 'verdict': test['verdict'],
                      'tests': test['tests'], 'rain_outside': test['rain_outside'],
                      'preregistration_sha256': sha256(PREREG)},
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'-> {out_dir.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
