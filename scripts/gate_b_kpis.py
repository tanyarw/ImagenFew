#!/usr/bin/env python
"""
gate_b_kpis.py — the Gate B KPIs that gate_b_sewer.py does not compute (strengthening run 4,
code_plan/STRENGTHENING_RUNS.md; bands from code_plan/ACCEPTANCE_CRITERIA.md, Gate B).

gate_b_sewer.py reports overflow volume and overflow-event counts per year. This script routes
the same rain through the same sewer (SWMM-Astlingen, flood-control's code, unchanged) and
records what the sewer did step by step, so the rest of the Gate B table can be scored:

| KPI                                   | Here                                                    | Pass band   |
| total overflow per year               | sum of all node overflow, per 365 days                  | 0.85 - 1.15 |
| overflow events per year              | runs of overflow, a 2-h gap ends one (as gate_b_sewer)  | 0.85 - 1.15 |
| peak spill rate distribution          | largest 5-min overflow of every event, in m3/s; KS      | KS <= 0.10  |
| tank filling-level distribution       | depth of each of the six tanks at every step; KS per    | KS <= 0.10  |
|                                       | tank, the worst tank is scored                          |  (worst)    |
| flooding volume, annual maximum       | the largest single overflow event of each year (all     | 0.80 - 1.20 |
|                                       | overflow in Astlingen is node flooding), mean over years|             |
| time in "system stressed" state       | share of steps with at least one tank >= 90% full       | |diff| <= 5 pp |
|                                       | (4.5 m of 5.0 m; every tank is 5.0 m deep)              |             |

Same protocol as gate_b_sewer.py and flood-control's bc_efd_check.py: every calendar year from
empty tanks, under BC (and optionally EFD); the mean series is split into the four gauges by
flood-control's src/rain/split.py with seed 0 (gate_b_sewer.build, reused, so the `v14` series
here is exactly flood-control's v14 source). Real = the benchmark's four gauge files, years
2000-2008 (the test year 2009 is not simulated, flood-control D13), simulated here so that its
distributions exist (flood-control's saved table has totals only).

Distributions are stored compactly so later invocations can add versions without rerunning the
others: event peaks as a list per year (a few dozen numbers), tank depths as histograms on fixed
0.02 m bins (KS from binned CDFs is exact to the bin width).

Run from flood-control with its .venv (pyswmm), ~40 s per simulated year, in parallel:

    cd ../flood-control && PYTHONPATH=. .venv/bin/python ../ImagenFew/scripts/gate_b_kpis.py \\
        v14_nobridge_cal=v14 bartlett_lewis_relaxed_wet arima_copula_occ_len64 --controllers BC EFD

    ... gate_b_kpis.py --summary          # print the table from the saved file, no simulation

`csvname=label` stores a series under another name. Output: results/reference/gate_b_kpis.json.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_b_sewer as gb  # noqa: E402  (build: four gauge files + model copy; flood_control: path setup)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'reference' / 'gate_b_kpis.json'
STEP = gb.STEP
TANK_DEPTH_M = 5.0                       # all six Astlingen tanks (plant full_depth, checked 2026-10-08)
STRESS_FILL = 0.9                        # "stressed": at least one tank this full
DEPTH_EDGES = np.round(np.arange(0.0, TANK_DEPTH_M + 0.02 + 1e-9, 0.02), 4)   # 0 .. 5.02 m, last bin catches 5.0
REAL_YEARS = list(range(2000, 2009))     # train + validation; 2009 untouched (flood-control D13)
BANDS = {
    'total_ratio': (0.85, 1.15),
    'events_ratio': (0.85, 1.15),
    'peak_ks': (None, 0.10),
    'tank_ks_worst': (None, 0.10),
    'annual_max_event_ratio': (0.80, 1.20),
    'stressed_diff_pp': (-5.0, 5.0),
}


def event_volumes_and_peaks(x):
    """Volume (m3) and largest 5-min value (m3) of every overflow event (2-h gap, as gate_b_sewer)."""
    from src.rain.split import storm_ids
    ids = storm_ids(np.where(x > 0, x, 0.0))
    wet = ids >= 0
    if not wet.any():
        return np.zeros(0), np.zeros(0)
    vol = np.bincount(ids[wet], x[wet])
    peak = np.zeros(len(vol))
    np.maximum.at(peak, ids[wet], x[wet])
    return vol, peak


def one(job):
    """Simulate one calendar year of one source under one controller; return its KPI row."""
    name, inp, rain_path, start0, n_steps, year, ctrl, fc = job
    gb.flood_control(fc)
    from src.astlingen.controllers import BCController, EFDController
    from src.astlingen.model import TANKS
    from src.astlingen.simulate import run
    start = pd.Timestamp(f'{year}-01-01')
    end = min(pd.Timestamp(f'{year}-12-31 23:55'), start0 + n_steps * STEP)
    res = run({'BC': BCController, 'EFD': EFDController}[ctrl](), start, end, inp_path=inp, record_trace=True)
    i0 = int((start - start0) / STEP)
    rain = np.load(rain_path)[i0:i0 + res.steps]
    tr = res.trace
    overflow = tr.overflow_step_m3.to_numpy()
    vol, peak = event_volumes_and_peaks(overflow)
    depths = np.column_stack([tr[f'depth_{t}'].to_numpy() for t in TANKS])
    hist = {t: np.histogram(np.clip(depths[:, i], 0.0, TANK_DEPTH_M), bins=DEPTH_EDGES)[0].tolist()
            for i, t in enumerate(TANKS)}
    stressed = (depths >= STRESS_FILL * TANK_DEPTH_M).any(axis=1)
    return dict(source=name, year=int(year), controller=ctrl, days=res.steps / 288, rain_mm=float(rain.sum()),
                total_m3=float(res.summary['total']), trace_total_m3=float(overflow.sum()),
                overflow_events=int(len(vol)), event_peaks_m3s=(peak / 300.0).round(6).tolist(),
                annual_max_event_m3=float(vol.max()) if len(vol) else 0.0,
                stressed_share=float(stressed.mean()), tank_depth_hist=hist, max_depth_m=float(depths.max()))


def ks_samples(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (largest gap between the empirical CDFs)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side='right') / len(a)
                                - np.searchsorted(b, grid, side='right') / len(b))))


def ks_hist(h1, h2):
    """KS statistic between two distributions given as counts on the same bins."""
    c1, c2 = np.cumsum(h1) / max(np.sum(h1), 1), np.cumsum(h2) / max(np.sum(h2), 1)
    return float(np.max(np.abs(c1 - c2)))


def summary(rows):
    """Per (controller, source): KPIs and their comparison with real under the same controller."""
    df = pd.DataFrame(rows)
    out = []
    for (ctrl, src), g in df.groupby(['controller', 'source']):
        k = 365 / g.days
        out.append(dict(controller=ctrl, source=src, years=len(g), rain_mm=float((g.rain_mm * k).mean()),
                        total_m3=float((g.total_m3 * k).mean()), overflow_events=float((g.overflow_events * k).mean()),
                        annual_max_event_m3=float(g.annual_max_event_m3.mean()),
                        stressed_pct=float(100 * np.average(g.stressed_share, weights=g.days)),
                        peak_list=np.concatenate([np.asarray(p, float) for p in g.event_peaks_m3s]),
                        depth_hist={t: np.sum([h[t] for h in g.tank_depth_hist], axis=0)
                                    for t in g.tank_depth_hist.iloc[0]}))
    t = pd.DataFrame(out).set_index(['controller', 'source'])
    res = []
    for (ctrl, src), r in t.iterrows():        # r[...] throughout: r.hist would be pandas' plot method
        peaks, hist = r['peak_list'], r['depth_hist']
        row = dict(controller=ctrl, source=src, years=int(r['years']), rain_mm=r['rain_mm'], total_m3=r['total_m3'],
                   overflow_events=r['overflow_events'], annual_max_event_m3=r['annual_max_event_m3'],
                   stressed_pct=r['stressed_pct'], median_peak_m3s=float(np.median(peaks)) if len(peaks) else np.nan)
        if (ctrl, 'real') in t.index and src != 'real':
            real = t.loc[(ctrl, 'real')]
            tank_ks = {tk: ks_hist(hist[tk], real['depth_hist'][tk]) for tk in hist}
            row.update(total_ratio=r['total_m3'] / real['total_m3'],
                       events_ratio=r['overflow_events'] / real['overflow_events'],
                       peak_ks=ks_samples(peaks, real['peak_list']), tank_ks_worst=max(tank_ks.values()),
                       tank_ks_worst_tank=max(tank_ks, key=tank_ks.get),
                       annual_max_event_ratio=r['annual_max_event_m3'] / real['annual_max_event_m3'],
                       stressed_diff_pp=r['stressed_pct'] - real['stressed_pct'])
            passed = []
            for kpi, (lo, hi) in BANDS.items():
                v = row[kpi]
                passed.append(bool(np.isfinite(v) and (lo is None or v >= lo) and (hi is None or v <= hi)))
            row['kpis_passed'] = f'{sum(passed)}/{len(passed)}'
        res.append(row)
    return pd.DataFrame(res)


def real_job_inputs(fc, tmp):
    """The benchmark model and the real mean rain on the 5-min grid of 2000-2009 (29 Feb = dry, as SWMM)."""
    from src.astlingen.model import control_free_inp
    from src.astlingen.references import T0, rain_series
    path = Path(tmp) / 'real_rain_mean.npy'
    np.save(path, rain_series('real'))
    return control_free_inp(), path, T0, len(rain_series('real'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('versions', nargs='*', help='ImagenFew series (csv name, or csvname=label)')
    ap.add_argument('--controllers', nargs='+', default=['BC'], choices=['BC', 'EFD'])
    ap.add_argument('--flood_control', default=os.environ.get('FLOOD_CONTROL', str(ROOT.parent / 'flood-control')))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--refresh_real', action='store_true', help='simulate the real years again')
    ap.add_argument('--summary', action='store_true', help='print the table from the saved file and exit')
    ap.add_argument('--out', type=Path, default=OUT, help='results file (default %(default)s)')
    ap.add_argument('--max_years', type=int, default=None, help='first N years per source only (smoke tests)')
    args = ap.parse_args()
    out = args.out

    old = json.loads(out.read_text())['rows'] if out.exists() else []
    if args.summary:
        print(summary(old).round(3).to_string())
        return
    if not args.versions and not args.refresh_real:
        ap.error('name at least one version (or --refresh_real, or --summary)')
    fc = gb.flood_control(Path(args.flood_control).resolve())
    labels = [v.split('=')[-1] for v in args.versions]
    if 'real' in labels or len(set(labels)) != len(labels):
        raise SystemExit("labels must be unique and not 'real'")

    tmp = tempfile.mkdtemp(prefix='gate_b_kpis_')
    try:
        jobs = []
        have_real = {(r['controller'], r['year']) for r in old if r['source'] == 'real'}
        inp, rain, start0, n = real_job_inputs(fc, tmp)
        for c in args.controllers:
            for y in REAL_YEARS[:args.max_years]:
                if args.refresh_real or (c, y) not in have_real:
                    jobs.append(('real', inp, str(rain), start0, n, y, c, fc))
        for spec in args.versions:
            d, start0, n = gb.build(spec, fc, tmp)
            years = list(range(start0.year, (start0 + (n - 1) * STEP).year + 1))[:args.max_years]
            jobs += [(spec.split('=')[-1], str(d / 'model.inp'), str(d / 'rain_mean.npy'), start0, n, y, c, fc)
                     for y in years for c in args.controllers]
        print(f'{len(jobs)} simulated years on {args.workers} workers', flush=True)
        rows = []
        with ProcessPoolExecutor(args.workers) as pool:
            for r in pool.map(one, jobs):
                rows.append(r)
                print(f"{r['source']:28s} {r['year']} {r['controller']:3s} {r['total_m3']:10.0f} m3  "
                      f"{r['overflow_events']:3d} events  stressed {100 * r['stressed_share']:5.2f}%", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # SWMM's total and the step trace must agree (the trace is what the distributions use)
    for r in rows:
        if not np.isclose(r['total_m3'], r['trace_total_m3'], rtol=1e-6, atol=1e-3):
            raise RuntimeError(f"{r['source']} {r['year']}: summary {r['total_m3']} != trace {r['trace_total_m3']}")
    new_keys = {(r['source'], r['controller'], r['year']) for r in rows}
    all_rows = [r for r in old if (r['source'], r['controller'], r['year']) not in new_keys] + rows
    t = summary(all_rows)
    pd.set_option('display.width', 200)
    print(t.round(3).to_string())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': 'each calendar year from empty tanks; four gauges split from the mean with seed 0 '
                    '(gate_b_sewer.build); real = benchmark gauges 2000-2008; events: 2-h gap; '
                    f'stressed: any tank >= {STRESS_FILL:.0%} of {TANK_DEPTH_M} m; tank depth bins 0.02 m',
        'bands': {k: list(v) for k, v in BANDS.items()},
        'rows': all_rows,
        'summary': json.loads(t.to_json(orient='records')),
    }) + '\n')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
