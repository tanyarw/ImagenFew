"""Singapore raw files -> the four-gauge target series with a fill flag on every step.

Steps (code_plan/SINGAPORE_TRANSFER_PLAN.md, section 4):
  1. read      one row per station per 5 minutes, local wall-clock time; round to the 5-minute grid;
               relabel each 5-minute total by the START of its interval; drop 29 February; keep the
               latest-updated reading when two land in one slot
  2. stations  usable = complete enough every year (except exempt years) and on the gauge's 0.2 mm grid
  3. cluster   the most compact set of `cluster_size` usable stations
  4. fill      a missing cluster gauge from nearby usable stations reporting at that moment (inverse
               distance weighting, rounded to the gauge grid); network gaps up to `max_time_gap_steps`
               linearly in time; longer network gaps are outages (zero placeholder, flagged)

Flags: 0 observed (all gauges observed), 1 filled_space, 2 filled_time, 3 outage.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import timegrid as tg

FLAG_NAMES = {0: "observed", 1: "filled_space", 2: "filled_time", 3: "outage"}
RAW_COLS = ["timestamp", "reading_update_timestamp", "station_id", "station_name",
            "location_longitude", "location_latitude", "reading_value"]


# ── 1. Reading ───────────────────────────────────────────────────────────────────────────────────

@dataclass
class RawYear:
    year: int
    readings: pd.DataFrame          # columns: pos, station_id, value (one row per station and slot)
    stations: pd.DataFrame          # station_id -> name, lon, lat
    counts: dict                    # rows read, duplicates removed, dropped (29 Feb, outside range, off grid)


def read_raw_year(path, year, first_year, n_steps):
    df = pd.read_csv(path, usecols=RAW_COLS, dtype={"station_id": str, "station_name": str})
    n_rows = len(df)
    stamp = pd.to_datetime(df["timestamp"].str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S")   # local wall clock
    end = stamp.dt.round("5min")
    off_grid = (stamp - end).abs() > pd.Timedelta(seconds=60)
    start = end - pd.Timedelta(minutes=tg.STEP_MINUTES)                                     # label = interval start
    pos = tg.slot_index(start, first_year)
    keep = (~off_grid.to_numpy()) & (pos >= 0) & (pos < n_steps)
    is_feb29 = ((start.dt.month == 2) & (start.dt.day == 29)).to_numpy()
    d = pd.DataFrame({"pos": pos, "station_id": df["station_id"].to_numpy(),
                      "value": df["reading_value"].to_numpy(dtype=np.float64),
                      "update": df["reading_update_timestamp"].to_numpy()})[keep]
    d = d.sort_values(["station_id", "pos", "update"], kind="mergesort")
    before = len(d)
    d = d.drop_duplicates(["station_id", "pos"], keep="last")
    stations = (df.groupby("station_id")
                  .agg(name=("station_name", "first"), lon=("location_longitude", "median"),
                       lat=("location_latitude", "median")))
    counts = dict(rows=int(n_rows), off_grid=int(off_grid.sum()), feb29=int(is_feb29.sum()),
                  outside_range=int(((pos < 0) | (pos >= n_steps)).sum() - is_feb29.sum()),
                  duplicates_removed=int(before - len(d)), kept=int(len(d)))
    return RawYear(year, d[["pos", "station_id", "value"]].reset_index(drop=True), stations, counts)


def assemble(raw_years, n_steps):
    """Station -> float32 array over the whole calendar (NaN where the station did not report)."""
    series = {}
    for ry in raw_years:
        for sid, g in ry.readings.groupby("station_id"):
            arr = series.setdefault(sid, np.full(n_steps, np.nan, dtype=np.float32))
            arr[g["pos"].to_numpy()] = g["value"].to_numpy(dtype=np.float32)
    meta = pd.concat([ry.stations for ry in raw_years]).groupby(level=0).agg(
        name=("name", "first"), lon=("lon", "median"), lat=("lat", "median"))
    return series, meta


# ── 2. Usable stations ───────────────────────────────────────────────────────────────────────────

def station_year_stats(series, years, first_year, gauge_resolution):
    rows = []
    for sid, arr in series.items():
        for y in years:
            seg = arr[tg.year_slice(y - first_year)]
            have = ~np.isnan(seg)
            pos = seg[have & (seg > 0)].astype(np.float64)
            on_grid = np.isclose(pos / gauge_resolution, np.round(pos / gauge_resolution), atol=1e-6)
            rows.append(dict(station_id=sid, year=y, completeness=float(have.mean()), n_wet=int(len(pos)),
                             resolution_share=float(on_grid.mean()) if len(pos) else np.nan))
    return pd.DataFrame(rows)


def usable_stations(stats, years, min_completeness, exempt_years, min_resolution_share):
    """Stations complete enough in every non-exempt year and on the gauge grid in every year."""
    out = []
    for sid, g in stats.groupby("station_id"):
        g = g.set_index("year")
        if any(y not in g.index for y in years):
            continue
        comp_ok = all(g.loc[y, "completeness"] >= min_completeness for y in years if y not in exempt_years)
        res = g["resolution_share"].dropna()
        res_ok = len(res) == len(years) and bool((res >= min_resolution_share).all())
        if comp_ok and res_ok:
            out.append(sid)
    return sorted(out)


# ── 3. Cluster ───────────────────────────────────────────────────────────────────────────────────

def km_matrix(lon, lat):
    """Equirectangular distances (km); exact enough over a 50 km island."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    kx = 111.32 * np.cos(np.radians(lat.mean()))
    dx = (lon[:, None] - lon[None, :]) * kx
    dy = (lat[:, None] - lat[None, :]) * 110.57
    return np.hypot(dx, dy)


def compact_cluster(station_ids, dist, mean_completeness, k):
    """The k stations with the smallest largest pairwise distance (ties: higher mean completeness)."""
    best = None
    for c in itertools.combinations(range(len(station_ids)), k):
        spread = max(dist[a, b] for a, b in itertools.combinations(c, 2))
        key = (round(spread, 6), -np.mean([mean_completeness[station_ids[i]] for i in c]))
        if best is None or key < best[0]:
            best = (key, c)
    idx = list(best[1])
    return [station_ids[i] for i in idx], float(best[0][0])


# ── 4. Filling ───────────────────────────────────────────────────────────────────────────────────

def fill_gauge(target, neighbours, dist_km, gauge_resolution, power, max_neighbours, radius_km):
    """Fill NaNs of one gauge from neighbours reporting at the same step.

    target: [N] values; neighbours: [N, M] values ordered by distance; dist_km: [M].
    Returns (filled values, filled-here mask). Steps where no neighbour reports stay NaN.
    """
    miss = np.isnan(target)
    have = ~np.isnan(neighbours)
    near = dist_km <= radius_km
    sel_near = have & near[None, :]
    sel_near &= np.cumsum(sel_near, axis=1) <= max_neighbours
    sel_any = have & (np.cumsum(have, axis=1) <= max_neighbours)
    sel = np.where(sel_near.any(axis=1, keepdims=True), sel_near, sel_any)
    w = sel / np.maximum(dist_km, 1e-6)[None, :] ** power
    wsum = w.sum(axis=1)
    est = np.divide((w * np.nan_to_num(neighbours)).sum(axis=1), wsum,
                    out=np.full(len(target), np.nan), where=wsum > 0)
    est = np.round(est / gauge_resolution) * gauge_resolution
    filled = np.where(miss, est, target)
    return filled, miss & ~np.isnan(est)


def fill_network_gaps(cluster, max_gap, grid):
    """Short network gaps: linear in time (zeros if both sides dry); longer ones: outage placeholder 0.

    Returns (values, flags) where flags are 2 (filled_time) or 3 (outage) on gap steps, -1 elsewhere.
    """
    x = cluster.copy()
    flag = np.full(len(x), -1, dtype=np.int8)
    nan = np.isnan(x)
    if not nan.any():
        return x, flag
    edges = np.diff(np.r_[0, nan.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for a, b in zip(starts, ends):
        L = b - a
        if L <= max_gap and a > 0 and b < len(x):
            left, right = x[a - 1], x[b]
            if left == 0 and right == 0:
                x[a:b] = 0.0
            else:
                t = np.arange(1, L + 1) / (L + 1)
                x[a:b] = np.round((left + (right - left) * t) / grid) * grid
            flag[a:b] = 2
        else:
            x[a:b] = 0.0
            flag[a:b] = 3
    return x, flag


def build_cluster_series(series, meta, cluster, usable, cfg):
    """The cluster average with flags. Returns (values, flags, per-gauge filled matrix, fill counts)."""
    res = float(cfg["gauge_resolution_mm"])
    f = cfg["fill"]
    ids = list(usable)
    d = km_matrix(meta.loc[ids, "lon"], meta.loc[ids, "lat"])
    V = np.stack([series[s].astype(np.float64) for s in ids], axis=1)
    gauges, filled_any, counts = [], np.zeros(V.shape[0], bool), {}
    for s in cluster:
        i = ids.index(s)
        others = [j for j in np.argsort(d[i]) if j != i]
        g, filled = fill_gauge(V[:, i], V[:, others], d[i, others], res, f["idw_power"], f["max_neighbours"],
                               f["radius_km"])
        gauges.append(g)
        filled_any |= filled
        counts[s] = dict(observed=int((~np.isnan(V[:, i])).sum()), filled_space=int(filled.sum()))
    G = np.stack(gauges, axis=1)
    cluster_mean = G.mean(axis=1)                                   # NaN where any gauge is still missing
    grid = res / len(cluster)
    values, gap_flag = fill_network_gaps(cluster_mean, f["max_time_gap_steps"], grid)
    flags = np.where(gap_flag >= 0, gap_flag, np.where(filled_any, 1, 0)).astype(np.int8)
    values = np.round(values, 6)
    return values, flags, G, counts


def outage_runs(flags):
    o = (flags == 3).astype(np.int8)
    edges = np.diff(np.r_[0, o, 0])
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
