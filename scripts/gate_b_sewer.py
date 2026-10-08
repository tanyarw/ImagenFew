#!/usr/bin/env python
"""
gate_b_sewer.py — Gate B (code_plan/ACCEPTANCE_CRITERIA.md): route a generated series through
SWMM-Astlingen under a fixed controller and compare the overflow with real rain.

Runs the sibling repository's own sewer check (flood-control, analysis/synthetic_rain/
bc_efd_check.py) on any ImagenFew version without changing that repository:

  * the catchment-mean series is split into the four Astlingen gauges by flood-control's
    src/rain/split.py with seed 0, exactly as its `v14` rain source is;
  * the gauge files and a copy of the control-free model that reads them go into a temporary
    directory (about 100 MB per version), deleted afterwards;
  * every calendar year is simulated from empty tanks under BC (and optionally EFD), and
    overflow totals are scaled to 365 days, as in bc_efd_check.py.

Reproduces flood-control's saved v14 result exactly (2026 under BC: 188,327 m3).
Results: results/reference/gate_b_sewer.json (one row per version, year and controller, plus
flood-control's saved real and v14 rows for comparison). Needs flood-control's .venv (pyswmm):

    cd ../flood-control && PYTHONPATH=. .venv/bin/python ../ImagenFew/scripts/gate_b_sewer.py v16 v16_m1 \\
        --controllers BC EFD

A version can be given as `csvname=label` to store it under another name: the ImagenFew `v14`
series must be run as `v14=v14_markov`, because `v14` here means flood-control's saved run
(`v14_nobridge_cal`).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'results' / 'generated_data'
OUT = ROOT / 'results' / 'reference' / 'gate_b_sewer.json'
STEP = pd.Timedelta(minutes=5)


def flood_control(path):
    sys.path.insert(0, str(path))
    from src.astlingen.controllers import BCController, EFDController   # noqa: F401  (imported for workers)
    return path


def events(x):
    """Totals of runs of x > 0 with no gap of 2 h or more (flood-control's storm definition)."""
    from src.rain.split import storm_ids
    ids = storm_ids(np.where(x > 0, x, 0.0))
    wet = ids >= 0
    return np.bincount(ids[wet], x[wet]) if wet.any() else np.zeros(0)


def build(name, fc, tmp):
    """Four gauge files and a model copy reading them; returns (directory, first date, steps)."""
    from src.astlingen.model import control_free_inp
    from src.rain.split import load_params, split
    csv, _, name = name.partition('=') if '=' in name else (name, '', name)
    df = pd.read_csv(GEN / f'rainfall_synthetic_10y_{csv}.csv', parse_dates=['date'])
    if not (df.date.diff().dropna() == STEP).all():
        raise ValueError(f'{name} is not contiguous at 5 min')
    d = Path(tmp) / name
    d.mkdir()
    g = split(df.avg_rainfall.to_numpy(np.float32).astype(np.float64), load_params(), 0)   # float32 as in vendor_v14.py
    days, times = df.date.dt.strftime('%m/%d/%Y').to_numpy(), df.date.dt.strftime('%H:%M:%S').to_numpy()
    for i in range(4):
        pd.DataFrame({'d': days, 't': times, 'v': g[:, i]}).to_csv(d / f'g{i + 1}.dat', sep=' ', header=False,
                                                                  index=False, float_format='%.4f')
    np.save(d / 'rain_mean.npy', g.mean(axis=1))
    text = Path(control_free_inp()).read_text()
    for i in range(1, 5):
        real = str((fc / f'data/SWMM-Astlingen/{i}Astlingen_Erft{i}.txt').resolve())
        if real not in text:
            raise RuntimeError(f'{real} not found in the control-free model')
        text = text.replace(real, str((d / f'g{i}.dat').resolve()))
    (d / 'model.inp').write_text(text)
    return d, df.date.iloc[0], len(df)


def one(job):
    name, d, start0, n_steps, year, ctrl, fc = job
    flood_control(fc)
    from src.astlingen.controllers import BCController, EFDController
    from src.astlingen.simulate import run
    start = pd.Timestamp(f'{year}-01-01')
    end = min(pd.Timestamp(f'{year}-12-31 23:55'), start0 + n_steps * STEP)
    res = run({'BC': BCController, 'EFD': EFDController}[ctrl](), start, end, inp_path=str(d / 'model.inp'),
              record_trace=True)
    i0 = int((start - start0) / STEP)
    rain = np.load(d / 'rain_mean.npy')[i0:i0 + res.steps]
    storms, overflow = events(rain), events(res.trace.overflow_step_m3.to_numpy())
    return dict(source=name, year=year, controller=ctrl, days=res.steps / 288, rain_mm=float(rain.sum()),
                storms_ge10mm=int((storms >= 10).sum()), max_storm_mm=float(storms.max()),
                total_m3=res.summary['total'], creek_m3=res.summary['creek'], river_m3=res.summary['river'],
                overflow_events=len(overflow), overflow_events_ge1000m3=int((overflow >= 1000).sum()))


def summary(rows):
    """Per source and controller: means per year (scaled to 365 days) and the ratio to real."""
    df = pd.DataFrame(rows)
    k = 365 / df.days
    for c in ('rain_mm', 'storms_ge10mm', 'total_m3', 'creek_m3', 'overflow_events', 'overflow_events_ge1000m3'):
        df[c] = df[c] * k
    t = df.groupby(['controller', 'source'])[['rain_mm', 'storms_ge10mm', 'total_m3', 'creek_m3', 'overflow_events',
                                              'overflow_events_ge1000m3']].mean()
    t['years'] = df.groupby(['controller', 'source']).size()
    if 'real' in t.index.get_level_values(1):
        for c in ('total_m3', 'creek_m3', 'overflow_events'):
            t[f'{c}_vs_real'] = [t.loc[(ctl, s), c] / t.loc[(ctl, 'real'), c] if (ctl, 'real') in t.index else np.nan
                                 for ctl, s in t.index]
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('versions', nargs='+')
    ap.add_argument('--controllers', nargs='+', default=['BC'], choices=['BC', 'EFD'])
    ap.add_argument('--flood_control', default=str(ROOT.parent / 'flood-control'))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    fc = flood_control(Path(args.flood_control).resolve())

    tmp = tempfile.mkdtemp(prefix='gate_b_')
    try:
        jobs = []
        for spec in args.versions:
            name = spec.split('=')[-1]
            if name in ('real', 'v14'):
                raise SystemExit(f"'{name}' is flood-control's saved run; store this series under another label")
            d, start0, n = build(spec, fc, tmp)
            years = range(start0.year, (start0 + (n - 1) * STEP).year + 1)
            jobs += [(name, d, start0, n, y, c, fc) for y in years for c in args.controllers]
        rows = []
        with ProcessPoolExecutor(args.workers) as pool:
            for r in pool.map(one, jobs):
                rows.append(r)
                print(f"{r['source']:12s} {r['year']} {r['controller']:3s} {r['total_m3']:10.0f} m3", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    saved = fc / 'analysis' / 'synthetic_rain' / 'results' / 'bc_efd.csv'
    ref = pd.read_csv(saved) if saved.exists() else pd.DataFrame()
    ref = ref[ref.source.isin(['real', 'v14'])] if len(ref) else ref     # both controllers: older runs may need them
    old = json.loads(OUT.read_text())['rows'] if OUT.exists() else []
    keep = [r for r in old if r['source'] not in {v.split('=')[-1] for v in args.versions} | {'real', 'v14'}]
    all_rows = keep + (json.loads(ref.to_json(orient='records')) if len(ref) else []) + rows
    t = summary(all_rows)
    print(t.round(3).to_string())
    OUT.write_text(json.dumps({'protocol': 'flood-control analysis/synthetic_rain/bc_efd_check.py: each calendar '
                                           'year from empty tanks, four gauges split with seed 0, totals per 365 days',
                               'real_and_v14_rows': str(saved), 'rows': all_rows,
                               'summary': json.loads(t.reset_index().to_json(orient='records'))}, indent=1) + '\n')
    print(f'-> {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
