#!/usr/bin/env python
"""
summarise_strengthening.py — one verdict per strengthening run (code_plan/STRENGTHENING_RUNS.md),
using the decision rules written in that plan BEFORE the runs (pre-registered). Reads the
scores of scripts/eval_suite.py (computing them for any new CSV first, ~1 min per series) and,
for runs 4 and 7, the result files those runs write. Runs whose outputs do not exist yet are
reported as "not run yet".

    python scripts/summarise_strengthening.py

Yardstick for "is this difference real": the spread between training seeds of the same
config (run 2). A single run is never called better or worse than v14 unless it falls
outside v14's seed range.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import eval_suite as ev  # noqa: E402

REAL = 'real 2000-2007'
METRICS = [  # (key in eval_suite hydro, label, how to read it)
    ('volume_mm_yr', 'rain mm/yr', 'ratio'),
    ('wet_spell_min', 'wet spell min', 'ratio'),
    ('storm_dur_min', 'storm min', 'ratio'),
    ('daily_max', 'wettest day mm', 'ratio'),
    ('idf_cells_in_band', 'IDF cells /15', 'count'),
    ('long_storms_yr', 'runs > 5h20 /yr', 'count'),
    ('long_storm_rain_pct', '% rain in them', 'count'),
    ('storms_ge10mm_yr', 'storms >= 10 mm /yr', 'count'),
]


def row(c, name):
    if name not in c['hydro']:
        return None
    h = c['hydro'][name]
    r = {'version': name, 'Gate A /18': sum(c['gate'][name].values()) if c['gate'].get(name) else np.nan,
         'Tier 6 /3': sum(c['tier6'][name].values()) if c['tier6'].get(name) else np.nan}
    for k, lab, _ in METRICS:
        r[lab] = h.get(k, np.nan)
    return r


def table(c, names, title):
    rows = [row(c, n) for n in [REAL] + names]
    have = [r for r in rows if r is not None]
    missing = [n for n, r in zip([REAL] + names, rows) if r is None]
    print(f'\n### {title}')
    if len(have) > 1:
        df = pd.DataFrame(have).set_index('version')
        print(df.round(2).to_string())
    if missing:
        print(f'not run yet: {", ".join(missing)}')
    return {r['version']: r for r in have}


def seed_range(rows, names, key='Gate A /18'):
    v = [rows[n][key] for n in names if n in rows]
    return (min(v), max(v), len(v)) if v else None


def verdict(text):
    print(f'VERDICT: {text}')


def main():
    c = ev.compute()

    print('=' * 100)
    print('RUN 2: training seeds. Does "asinh beats standard scaling" survive retraining?')
    r = table(c, ['v10', 'v10_s1', 'v14', 'v14_s1', 'v14_s2'], 'Markov plan, generation seed 42 (as the scored v10 / v14)')
    table(c, ['v14_nobridge_cal', 'v14_s1_nobridge_cal', 'v14_s2_nobridge_cal', 'v10_s1_nobridge_cal'],
          'calendar plan, no bridges')
    v10, v14 = seed_range(r, ['v10', 'v10_s1']), seed_range(r, ['v14', 'v14_s1', 'v14_s2'])
    if v10 and v14 and v10[2] == 2 and v14[2] == 3:
        print(f'Gate A: v10 seeds {v10[0]}-{v10[1]}, v14 seeds {v14[0]}-{v14[1]}')
        vol = {n: abs(1 - r[n]['rain mm/yr'] / r[REAL]['rain mm/yr']) for n in ['v10', 'v10_s1', 'v14', 'v14_s1', 'v14_s2']}
        vol_sep = max(vol[n] for n in ('v14', 'v14_s1', 'v14_s2')) < min(vol[n] for n in ('v10', 'v10_s1'))
        if v14[0] > v10[1] and vol_sep:
            verdict('HOLDS. Every v14 seed passes more Gate A checks than every v10 seed, and every v14 seed '
                    'is closer to the real rain volume.')
        elif v14[0] > v10[1] or vol_sep:
            verdict('PARTLY. One of the two pre-registered criteria separates the transforms, the other overlaps.')
        else:
            verdict('DOES NOT HOLD. The seed ranges overlap: the v10 -> v14 gain is within training noise.')
    else:
        verdict('not decidable yet (needs v10, v10_s1, v14, v14_s1, v14_s2)')

    print('\n' + '=' * 100)
    print('RUN 3: no pretraining. Does starting from the ImagenFew checkpoint help?')
    r3 = table(c, ['v14', 'v14_s1', 'v14_s2', 'v14_scratch'], 'Markov plan')
    table(c, ['v14_nobridge_cal', 'v14_scratch_nobridge_cal'], 'calendar plan, no bridges')
    if 'v14_scratch' in r3 and v14 and v14[2] == 3:
        s = r3['v14_scratch']['Gate A /18']
        if s < v14[0]:
            verdict(f'PRETRAINING HELPS: from scratch {s}/18, below every pretrained seed ({v14[0]}-{v14[1]}).')
        elif s > v14[1]:
            verdict(f'PRETRAINING HURTS: from scratch {s}/18, above every pretrained seed ({v14[0]}-{v14[1]}).')
        else:
            verdict(f'NO MEASURABLE EFFECT: from scratch {s}/18 lies inside the pretrained seed range '
                    f'({v14[0]}-{v14[1]}). Claim the checkpoint as an initialisation only.')
    else:
        verdict('not decidable yet (needs v14_scratch and the three v14 seeds)')

    print('\n' + '=' * 100)
    print('RUN 5: E3. Does training with known prefixes let the diffusion model make long storms itself?')
    members = ['v17_ctx_cal', 'v17_ctx_cal_m1', 'v17_ctx_cal_m2', 'v17_ctx_cal_m3']
    r5 = table(c, ['v14_nobridge_cal', 'v17_nobridge_cal', 'v17_ctx0_cal'] + members, 'calendar timeline')
    have = [m for m in members if m in r5]
    if have:
        real = r5[REAL]
        long_ = np.mean([r5[m]['runs > 5h20 /yr'] for m in have])
        share = np.mean([r5[m]['% rain in them'] for m in have])
        vol = np.mean([r5[m]['rain mm/yr'] for m in have]) / real['rain mm/yr']
        spell = np.mean([r5[m]['wet spell min'] for m in have]) / real['wet spell min']
        print(f'chained, mean of {len(have)} chains: {long_:.1f} long runs/yr (real {real["runs > 5h20 /yr"]:.2f}), '
              f'{share:.1f}% of rain in them (real {real["% rain in them"]:.1f}), volume ratio {vol:.3f}, '
              f'wet-spell ratio {spell:.3f}')
        checks = {'>= 10 long runs/yr': long_ >= 10, '>= 15% of rain in them': share >= 15,
                  'volume 0.95-1.05': 0.95 <= vol <= 1.05, 'wet spell 0.90-1.10': 0.90 <= spell <= 1.10}
        print('pre-registered targets: ' + ', '.join(f'{k} {"PASS" if v else "FAIL"}' for k, v in checks.items()))
        ctrl = r5.get('v17_ctx0_cal')
        if ctrl:
            print(f'control (same model, no context): {ctrl["runs > 5h20 /yr"]:.1f} long runs/yr; the gain is '
                  f'{"from the context" if ctrl["runs > 5h20 /yr"] < 0.5 * long_ else "NOT clearly from the context"}')
        verdict('SUCCESS: the diffusion model continues storms across blocks.' if all(checks.values()) else
                'PARTIAL / FAIL on the targets above (report which).')
    else:
        verdict('not run yet')
    if 'v17_nobridge_cal' in r5 and 'v14_nobridge_cal' in r5:
        print(f'side check, did mask training harm ordinary generation? Gate A v17_nobridge_cal '
              f'{r5["v17_nobridge_cal"]["Gate A /18"]} vs v14_nobridge_cal {r5["v14_nobridge_cal"]["Gate A /18"]} '
              f'(compare with the run-2 seed spread)')

    print('\n' + '=' * 100)
    print('RUN 4: Gate B KPIs (results/reference/gate_b_kpis.json)')
    path = os.path.join(ROOT, 'results', 'reference', 'gate_b_kpis.json')
    if os.path.exists(path):
        s = pd.DataFrame(json.load(open(path))['summary'])
        cols = ['controller', 'source', 'years', 'total_ratio', 'events_ratio', 'peak_ks', 'tank_ks_worst',
                'annual_max_event_ratio', 'stressed_diff_pp', 'kpis_passed']
        print(s[[k for k in cols if k in s]].round(3).to_string(index=False))
    else:
        verdict('not run yet')

    print('\n' + '=' * 100)
    print('RUN 7: denoising error on the training years (results/transform_diagnostics/denoise_train/)')
    path = os.path.join(ROOT, 'results', 'transform_diagnostics', 'denoise_train', 'denoise_error_summary.csv')
    if os.path.exists(path):
        s = pd.read_csv(path)
        print(s.pivot(index='band', columns='run', values='vol_err').round(4).to_string())
        print('(training-weighted volume error by true rain band; compare the heavy/extreme rows with the '
              'held-out result in denoise_test/)')
    else:
        verdict('not run yet')


if __name__ == '__main__':
    main()
