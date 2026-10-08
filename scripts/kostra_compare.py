#!/usr/bin/env python
"""
kostra_compare.py — heavy rain of the real and generated series against the official German
design-rainfall table KOSTRA-DWD-2020 for the Erft basin (where the Astlingen rain comes from).

The Astlingen benchmark rain is four 5-minute gauge series from the Erftverband
(flood-control/data/SWMM-Astlingen/*Astlingen_Erft*.txt); this project's "real" series is their
exact average. KOSTRA gives point rainfall depths (mm) for durations from 5 minutes to 7 days
and return periods of 1 to 100 years on a 5 km grid (DWD open data, fitted to 1951-2020).

Which KOSTRA cells: the exact gauge sites are not published, so the Erft basin is represented
by the grid cells under the Erftverband's own KOSTRA stations (station_list_KOSTRA-DWD-2020.csv,
provider = Erftverband); the report gives their median and range.

Comparison at three levels, because KOSTRA is for single points:
  1. KOSTRA (point, long record)        vs  the four real gauges (point, 2000-2007)
  2. real single gauges                 vs  their average (what the models learn): the "areal" drop
  3. real average                       vs  generated series (same 2000-2007 reference as elsewhere)
Return levels for our series: Gumbel (method of moments) on annual maxima of rolling totals, as in
gate_a_scorecard.py. KOSTRA itself uses a different, longer-record method, so read level 1 as a
plausibility check, not a test.

Downloads are cached in data/external/kostra/ (git-ignored). Pure numpy / pandas / matplotlib.

    python scripts/kostra_compare.py
"""
import io
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import gate_a_scorecard as sc  # noqa: E402  (Gumbel quantile, AMS definitions)

BASE = 'https://opendata.dwd.de/climate_environment/CDC/grids_germany/return_periods/precipitation/KOSTRA/KOSTRA_DWD_2020'
CACHE = os.path.join(ROOT, 'data', 'external', 'kostra')
GAUGES = os.path.join(os.path.dirname(ROOT), 'flood-control', 'data', 'SWMM-Astlingen')
GEN = os.path.join(ROOT, 'results', 'generated_data')
OUT = os.path.join(ROOT, 'results', 'kostra')
DURATIONS = [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440]       # minutes
RETURN = [2, 5, 10]                                             # years
Y = 105120
SERIES = {'v10': 'v10 (previous best)', 'v14': 'v14 (current best)', 'bartlett_lewis': 'storm-and-cell model',
          'bartlett_lewis_classic': 'storm-and-cell (classic)', 'arima_copula_occ_len64': 'latent Gaussian AR(64)'}


# ──────────────────────────────────────────────────────────────────────
# Lambert azimuthal equal-area, ETRS89 (EPSG:3035), ellipsoidal (Snyder 1987, eqs 3-12, 24-27)
# ──────────────────────────────────────────────────────────────────────

A, F = 6378137.0, 1 / 298.257222101
E2 = 2 * F - F * F
E = np.sqrt(E2)
LAT0, LON0, FE, FN = np.radians(52.0), np.radians(10.0), 4321000.0, 3210000.0


def _q(phi):
    s = np.sin(phi)
    return (1 - E2) * (s / (1 - E2 * s * s) - np.log((1 - E * s) / (1 + E * s)) / (2 * E))


def laea(lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    qp, q, q1 = _q(np.pi / 2), _q(phi), _q(LAT0)
    beta, beta1 = np.arcsin(q / qp), np.arcsin(q1 / qp)
    rq = A * np.sqrt(qp / 2)
    m1 = np.cos(LAT0) / np.sqrt(1 - E2 * np.sin(LAT0) ** 2)
    d = A * m1 / (rq * np.cos(beta1))
    b = rq * np.sqrt(2 / (1 + np.sin(beta1) * np.sin(beta) + np.cos(beta1) * np.cos(beta) * np.cos(lam - LON0)))
    x = FE + b * d * np.cos(beta) * np.sin(lam - LON0)
    y = FN + (b / d) * (np.cos(beta1) * np.sin(beta) - np.sin(beta1) * np.cos(beta) * np.cos(lam - LON0))
    return x, y


# EPSG Guidance Note 7-2 worked example for EPSG:3035: 50 N, 5 E -> E 3962799.45, N 2999718.85
_x, _y = laea(50.0, 5.0)
assert abs(_x - 3962799.45) < 0.05 and abs(_y - 2999718.85) < 0.05, (_x, _y)


# ──────────────────────────────────────────────────────────────────────
# KOSTRA data
# ──────────────────────────────────────────────────────────────────────

def fetch(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name.split('/')[-1])
    if not os.path.exists(path):
        urllib.request.urlretrieve(f'{BASE}/{name}', path)
    return path


def read_grid(zpath, member):
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(member))
        lines = z.read(name).decode('latin1').splitlines()
    head = {l.split()[0].lower(): float(l.split()[1]) for l in lines[:6]}
    grid = np.loadtxt(io.StringIO('\n'.join(lines[6:])))
    grid[grid == head['nodata_value']] = np.nan
    return head, grid


def erft_cells():
    st = pd.read_csv(fetch('tab/station_list_KOSTRA-DWD-2020.csv'), sep=';', encoding='latin1')
    st.columns = [c.strip().lstrip('﻿').lstrip('ï»¿') for c in st.columns]
    erft = st[st['PROVIDER'].astype(str).str.strip() == 'Erftverband'].copy()
    erft['x'], erft['y'] = laea(erft['LAT'].astype(float).to_numpy(), erft['LON'].astype(float).to_numpy())
    return erft


def kostra_values(erft):
    rows = []
    for d in DURATIONS:
        z = fetch(f'asc/StatRR_KOSTRA-DWD-2020_D{d:05d}_ASC.zip')
        for t in RETURN:
            head, g = read_grid(z, f'Hn_KOSTRA-DWD-2020_D{d:05d}_T{t:03d}.asc')
            col = ((erft['x'] - head['xllcorner']) // head['cellsize']).astype(int)
            row = (int(head['nrows']) - 1 - (erft['y'] - head['yllcorner']) // head['cellsize']).astype(int)
            cells = sorted(set(zip(row, col)))
            vals = np.array([g[r, c] for r, c in cells])
            vals = vals[~np.isnan(vals)]
            rows.append(dict(duration_min=d, T=t, kostra_median=np.median(vals), kostra_min=vals.min(),
                             kostra_max=vals.max(), cells=len(vals)))
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Our series
# ──────────────────────────────────────────────────────────────────────

def clean(a):
    a = np.clip(np.asarray(a, dtype=float), 0, None)
    a[a < 0.005] = 0.0
    return a


def gauges():
    out = {}
    for i in range(1, 5):
        f = os.path.join(GAUGES, f'{i}Astlingen_Erft{i}.txt')
        v = np.array([float(l.split()[-1]) for l in open(f) if l.strip() and not l.startswith(';')])
        out[f'gauge {i}'] = clean(v[:8 * Y])                    # 2000-2007, as the reference
    return out


def levels(a):
    ny = len(a) // Y
    res = {}
    for d in DURATIONS:
        w = d // 5
        ams = sc.ams(a, w, ny)
        for t in RETURN:
            res[(d, t)] = sc.gumbel_quantile(ams, t)
    return res


def compare():
    """Table indexed by (duration_min, T): KOSTRA Erft median / range and every series' return level."""
    erft = erft_cells()
    print(f'{len(erft)} Erftverband KOSTRA stations; lat {erft.LAT.min():.2f}-{erft.LAT.max():.2f}, '
          f'lon {erft.LON.min():.2f}-{erft.LON.max():.2f}')
    K = kostra_values(erft)

    series = dict(gauges())
    real = pd.read_csv(os.path.join(ROOT, 'data', 'rainfall', 'splits', 'train_years_labelled.csv'),
                       usecols=['avg_rainfall'])['avg_rainfall'].to_numpy()
    series['real average of 4 gauges'] = clean(real)
    for k, lab in SERIES.items():
        p = os.path.join(GEN, f'rainfall_synthetic_10y_{k}.csv')
        if os.path.exists(p):
            series[lab] = clean(pd.read_csv(p, usecols=['avg_rainfall'])['avg_rainfall'].to_numpy())
    L = pd.DataFrame({k: levels(a) for k, a in series.items()})
    L.index = pd.MultiIndex.from_tuples(L.index, names=['duration_min', 'T'])
    table = K.set_index(['duration_min', 'T']).join(L)
    gauge_cols = [c for c in table.columns if c.startswith('gauge ')]
    table['gauges mean'] = table[gauge_cols].mean(1)
    table['areal factor (average / gauges)'] = table['real average of 4 gauges'] / table['gauges mean']
    return table


def main():
    os.makedirs(OUT, exist_ok=True)
    table = compare()
    table.to_csv(os.path.join(OUT, 'kostra_comparison.csv'))
    table.reset_index().to_json(os.path.join(OUT, 'kostra_comparison.json'), orient='records', indent=1)
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
    show = ['kostra_median', 'kostra_min', 'kostra_max', 'gauges mean', 'real average of 4 gauges'] + \
           [SERIES[k] for k in SERIES if SERIES[k] in table.columns] + ['areal factor (average / gauges)']
    print(table[show].round(2).to_string())
    print(f'wrote {os.path.relpath(OUT, ROOT)}/kostra_comparison.csv and .json')


if __name__ == '__main__':
    main()
