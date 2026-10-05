"""compare_bartlett_lewis.py: storm-and-cell fits side by side, 10 generation seeds each (42-51; make
them with run_bartlett_lewis.py --generate_only --member k), scored with eval_suite's own hydro /
gate_a / tier6. Prints median and range over seeds and writes a JSON of every series' scores.

    python scripts/baselines/compare_bartlett_lewis.py           # old vs relaxed (Oct 5)
    python scripts/baselines/compare_bartlett_lewis.py --models bartlett_lewis_relaxed \
        bartlett_lewis_relaxed_wet --out results/reference/bartlett_lewis_objective_comparison.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import eval_suite as es  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROWS = [('volume_mm_yr', 'volume (mm/yr)'), ('max_5min', 'strongest 5-min burst (mm)'),
        ('p99_wet', 'P99 wet 5-min (mm)'), ('p999_wet', 'P99.9 wet 5-min (mm)'),
        ('single_step_showers_pct', 'single-step showers (%)'), ('wet_spell_min', 'mean wet spell (min)'),
        ('zero_pct', 'dry 5-min steps (%)'), ('long_storms_yr', 'storms > 5h20 per yr'),
        ('storms_ge10mm_yr', 'storms >= 10 mm per yr (2-h gap)'), ('rain_in_320min_pct', 'rain in storms > 320 min (%)'),
        ('max_1h', 'max 1-h (mm)'), ('daily_max', 'max daily (mm)'), ('idf_cells_in_band', 'IDF cells in band (/15)'),
        ('w1_daily', 'daily totals W1'), ('hourly_acf_rmse', 'hourly ACF RMSE'), ('monthly_r', 'monthly cycle r')]

train, held = es.load_real()
R = es.reference(train)


def score(name):
    a, res = es.load_series(name)
    m, _, _ = es.hydro(a, R, res)
    g, t = es.gate_a(m, R), es.tier6(m, R)
    return name, {k: float(v) for k, v in m.items()}, sum(g.values()), sum(t.values())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=['bartlett_lewis', 'bartlett_lewis_relaxed'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'reference', 'bartlett_lewis_relaxed_comparison.json'))
    args = ap.parse_args()
    members = {mdl: [f'{mdl}{"" if k == 0 else f"_m{k}"}' for k in range(10)] for mdl in args.models}
    with Pool(8) as pool:
        out = pool.map(score, [n for names in members.values() for n in names])
    res = {n: dict(m=m, gate=g, tier6=t) for n, m, g, t in out}
    real, _, _ = es.hydro(train, R)
    held_m, _, _ = es.hydro(held, R)
    print(f"{'metric':36s} {'real 00-07':>10s} {'held 08-09':>10s} | " + ' | '.join(f'{m[-24:]:>24s}' for m in args.models))
    for k, label in ROWS + [('gate', 'Gate A (/18)'), ('tier6', 'Tier 6 (/3)')]:
        cells = []
        for mdl in args.models:
            v = np.array([res[n]['m'][k] if k in res[n]['m'] else res[n][k] for n in members[mdl]])
            cells.append(f"{np.median(v):8.3g} [{v.min():.3g}-{v.max():.3g}]")
        r = real.get(k, np.nan) if k not in ('gate', 'tier6') else np.nan
        h = held_m.get(k, np.nan) if k not in ('gate', 'tier6') else np.nan
        print(f"{label:36s} {r:10.3g} {h:10.3g} | " + ' | '.join(f'{c:>24s}' for c in cells))
    json.dump(res, open(args.out, 'w'), indent=1)
