#!/usr/bin/env python
"""
generate_two_level.py — v16: a storm model decides when it rains, for how long and how much;
v14's diffusion output supplies the 5-minute rain inside each storm.

Why: at a given storm length v14 already has the right storm depth (325-720 min storms: 4.52 mm
vs 4.53 real) and the right texture inside (wet fraction 0.64 vs 0.62), but its storms are too
short (longer than 12 h: 1.7/yr vs 19.7 real) because every 5h20 block is generated on its own
(diary, Sept 28 architecture audit). The sewer fills its tanks over whole storms, so storm depth
and duration are the targets here (Gate A Tier 6 in code_plan/ACCEPTANCE_CRITERIA.md).

Storm: wet steps with no dry run of 24 steps (2 h) or more between them, the definition of
flood-control's src/rain/split.py (tanks take ~45 h to drain, so 2 h of dry weather does not
reset them).

Level 1 (outer): an alternating renewal process, dry gap -> storm -> dry gap -> ..., fitted to
  the 2000-2007 training years. Each draw is a smoothed bootstrap (a kernel density estimate):
  a real storm or dry gap G that started within +-15 days of the current day of the year and
  +-2 h of the current time of day, picked at random and perturbed by a log-normal kernel of
  width --bandwidth (mean 1). A storm is summarised by its duration D, depth P, number of wet
  steps W, number of wet runs and largest 5-min value (peak). D, W and the wet runs are scaled
  together (the storm keeps its wet fraction), P with them (it keeps its mean intensity), and P
  and the peak get one more shared noise term; G - 23 steps is scaled. Draws are independent
  given the season and hour because the real record shows almost no memory between storms:
  successive gaps r = 0.05, successive depths r = 0.06, duration and the following gap r = -0.09.

Level 2 (inner): the rain inside each storm is v14's (v14_nobridge_cal, generated on the same
  calendar, so the season of the texture matches). v14's storms are chained in their generated
  order, starting from one that began within +-30 days of the current day of the year; the
  >= 2 h gaps between them are closed, each either to nothing (the two v14 storms run on into
  each other) or to a dry run drawn from the real within-storm dry runs (1-23 steps), with a
  probability of joining directly that differs from chain to chain. Over --candidates chains
  and every v14 storm end a chain could stop at, the one closest to the target in duration, wet
  steps, wet runs and peak (after scaling to the target depth) is kept, with half the penalty on
  the scale factor itself, and scaled to the target depth. Real storms longer than 12 h are
  steady rain (wet runs ~70 min, wet fraction ~0.7, gentle peaks) while v14's storms are
  showers; matching on wet steps, wet runs and peak, and allowing direct joins, is what lets
  level 2 build the former out of the latter. Showers, gaps inside v14 storms and the shape of
  the 5-minute intensities are v14's; how they cluster into storms, and each storm's summary,
  come from level 1.

Output: the format and calendar of v14_nobridge_cal (date, avg_rainfall; 5 min from 2026-01-01;
  1,050,924 steps; season = step of a 365-day year), so it replaces v14 in flood-control's sewer
  check unchanged. Plus <name>.storms.csv: every storm's target and realised duration, wet
  steps, depth and peak, the scale factor and the first v14 storm used.

    python regime_training/generate_two_level.py                       # v16, seed 0 (~10 s)
    python regime_training/generate_two_level.py --seed 1 --name v16_m1
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, 'results', 'generated_data')
REAL = os.path.join(ROOT, 'data', 'rainfall', 'real_rainfall_data.csv')
WET = 0.005            # mm per 5 min, frozen wet threshold
Y = 105120             # 5-min steps per 365-day year (the real record has no 29 February)
DAY = 288
GAP = 24               # a dry run this long (2 h) ends a storm
TRAIN_YEARS = 8        # 2000-2007


def clean(a):
    a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
    a[a < WET] = 0.0
    return a


def storms(a):
    """(first step, last step, depth) of every storm."""
    wet = np.flatnonzero(a > 0)
    br = np.flatnonzero(np.diff(wet) > GAP)
    s, e = np.r_[wet[0], wet[br + 1]], np.r_[wet[br], wet[-1]]
    c = np.r_[0.0, np.cumsum(a)]
    return s, e, c[e + 1] - c[s]


def storm_summary(a, s, e):
    """Wet steps, wet runs and largest 5-min value of every storm."""
    w = a > 0
    c = np.r_[0, np.cumsum(w)]
    r = np.r_[0, np.cumsum(w & ~np.r_[False, w[:-1]])]  # wet runs start where a dry step precedes
    return c[e + 1] - c[s], r[e + 1] - r[s], np.maximum.reduceat(a, s)  # segments are dry after e


def day_of_year(step):
    return (np.asarray(step) // DAY) % 365


def hour_of_day(step):
    return (np.asarray(step) % DAY) // 12


def near(values, period, window):
    """[period, n] mask: value within +-window of each position on a circle of length `period`."""
    dist = np.abs(np.arange(period)[:, None] - values[None, :])
    return np.minimum(dist, period - dist) <= window


class SeasonalPool:
    """Indices of items tagged with a step, grouped by those within +-window days of the year and,
    if hours is given, within +-hours hours of the day as well."""
    def __init__(self, steps, window, hours=None):
        by_day = near(day_of_year(steps), 365, window)
        by_hour = near(hour_of_day(steps), 24, hours) if hours is not None else np.ones((1, len(steps)), bool)
        self.pools = [[np.flatnonzero(d & h) for h in by_hour] for d in by_day]

    def draw(self, rng, step, size=None):
        row = self.pools[int(day_of_year(step))]
        pool = row[int(hour_of_day(step))] if len(row) > 1 else row[0]
        return pool[rng.integers(len(pool), size=size)]


class Outer:
    """Level 1: seasonal smoothed bootstrap of real storms and the dry gaps between them."""
    def __init__(self, real, window, hours, bandwidth):
        s, e, P = storms(real)
        self.D, self.P = e - s + 1, P
        self.W, self.R, self.peak = storm_summary(real, s, e)
        self.G = s[1:] - e[:-1] - 1                      # dry steps between storms, >= GAP
        self.storm_pool = SeasonalPool(s, window, hours)
        self.gap_pool = SeasonalPool(e[:-1] + 1, window, hours)
        self.bw = bandwidth

    def kernel(self, rng):
        """Log-normal factor with mean 1."""
        return np.exp(self.bw * rng.standard_normal() - self.bw ** 2 / 2)

    def gap(self, rng, t):
        k = self.gap_pool.draw(rng, t)
        return GAP - 1 + max(1, int(round((self.G[k] - GAP + 1) * self.kernel(rng))))

    def storm(self, rng, t):
        """Target (duration, wet steps, wet runs, depth, peak) of a storm starting at step t."""
        k = self.storm_pool.draw(rng, t)
        d = max(1, int(round(self.D[k] * self.kernel(rng))))
        w = int(np.clip(round(self.W[k] * d / self.D[k]), min(d, 2), d))
        n_runs = int(np.clip(round(self.R[k] * d / self.D[k]), 1, min(w, d - w + 1)))
        q = self.kernel(rng)
        p = max(self.P[k] * d / self.D[k] * q, WET * min(d, 2))
        return d, w, n_runs, p, max(self.peak[k] * q, WET)


class Inner:
    """Level 2: v14's storms chained in generated order, long gaps closed, scaled to the target depth."""
    def __init__(self, texture, real, window, candidates, max_chain):
        self.a = texture
        self.s, self.e, self.P = storms(texture)
        self.D = self.e - self.s + 1
        self.W, self.R, self.peak = storm_summary(texture, self.s, self.e)
        self.pool = SeasonalPool(self.s, window)
        wet = (real > 0).astype(np.int8)                  # real dry runs shorter than GAP: inside storms
        ch = np.flatnonzero(np.diff(np.r_[-1, wet, -1]) != 0)
        runs, kind = np.diff(ch), wet[ch[:-1]]
        self.inner_gaps = runs[(kind == 0) & (runs < GAP)][1:-1]
        self.K, self.J = candidates, max_chain

    def storm(self, rng, t, d, w, n_runs, p, peak):
        """5-min rain of one storm: target duration d, wet steps w, wet runs n_runs, depth p and peak (mm)."""
        n = len(self.D)
        J = min(self.J, 8 + d // 8)
        start = self.pool.draw(rng, t, self.K)
        idx = start[:, None] + np.arange(J)[None, :]
        ok = idx < n
        idx = np.minimum(idx, n - 1)
        g = rng.choice(self.inner_gaps, size=idx.shape)
        g[rng.random(idx.shape) < rng.random((self.K, 1))] = 0     # direct joins, a different share per chain
        g[:, 0] = 0
        L = np.cumsum(self.D[idx] + g, axis=1)
        W = np.cumsum(self.W[idx], axis=1)
        R = np.cumsum(self.R[idx], axis=1) - np.cumsum(g == 0, axis=1) + 1   # a direct join merges two runs
        f = p / np.cumsum(self.P[idx], axis=1)                     # scale factor to the target depth
        top = np.maximum.accumulate(self.peak[idx], axis=1) * f
        score = (np.abs(np.log(L / d)) + np.abs(np.log(W / w)) + np.abs(np.log(R / n_runs))
                 + np.abs(np.log(top / peak)) + 0.5 * np.abs(np.log(f)))
        k, j = np.unravel_index(np.where(ok, score, np.inf).argmin(), score.shape)
        parts = []
        for m in range(j + 1):
            if g[k, m]:
                parts.append(np.zeros(g[k, m]))
            i = idx[k, m]
            parts.append(self.a[self.s[i]:self.e[i] + 1])
        x = np.concatenate(parts) * f[k, j]
        x[x < WET] = 0.0                                  # keep the wet threshold; restore the depth
        if x.sum() > 0:
            x *= p / x.sum()
            x[x < WET] = 0.0
        return x, idx[k, 0], f[k, j]


def generate(n_steps, seed, window, hours, texture_window, bandwidth, candidates, max_chain):
    real = clean(pd.read_csv(REAL, usecols=['avg_rainfall'])['avg_rainfall'].to_numpy())[:TRAIN_YEARS * Y]
    texture = clean(pd.read_csv(os.path.join(GEN, 'rainfall_synthetic_10y_v14_nobridge_cal.csv'),
                                usecols=['avg_rainfall'])['avg_rainfall'].to_numpy())
    outer = Outer(real, window, hours, bandwidth)
    inner = Inner(texture, real, texture_window, candidates, max_chain)
    rng = np.random.default_rng(seed)
    out = np.zeros(n_steps)
    rows = []
    t = 0
    while True:
        t += outer.gap(rng, t)
        if t >= n_steps:
            break
        d, w, n_runs, p, peak = outer.storm(rng, t)
        x, src, f = inner.storm(rng, t, d, w, n_runs, p, peak)
        x = x[:n_steps - t]
        out[t:t + len(x)] = x
        rows.append(dict(start=t, target_steps=d, steps=len(x), target_wet=w, wet=int((x > 0).sum()),
                         target_runs=n_runs,
                         target_mm=p, mm=x.sum(), target_peak=peak, peak=x.max(), scale=f, v14_storm=src))
        t += len(x)
    return out, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='v16')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--window', type=int, default=15, help='season window, +- days')
    ap.add_argument('--hours', type=int, default=2, help='level 1 time-of-day window, +- hours (-1: off)')
    ap.add_argument('--texture_window', type=int, default=30, help='level 2 season window, +- days')
    ap.add_argument('--bandwidth', type=float, default=0.15, help='log-normal kernel width of level 1')
    ap.add_argument('--candidates', type=int, default=600, help='v14 chains tried per storm')
    ap.add_argument('--max_chain', type=int, default=120, help='most v14 storms joined into one storm')
    args = ap.parse_args()

    dates = pd.read_csv(os.path.join(GEN, 'rainfall_synthetic_10y_v14_nobridge_cal.csv'), usecols=['date'])['date']
    rain, log = generate(len(dates), args.seed, args.window, None if args.hours < 0 else args.hours,
                         args.texture_window, args.bandwidth, args.candidates, args.max_chain)
    path = os.path.join(GEN, f'rainfall_synthetic_10y_{args.name}.csv')
    pd.DataFrame({'date': dates, 'avg_rainfall': rain}).to_csv(path, index=False, float_format='%.10g')
    log.to_csv(path.replace('.csv', '.storms.csv'), index=False)
    ny = len(rain) / Y
    print(f'{len(log)} storms, {rain.sum() / ny:.0f} mm/yr, {(rain == 0).mean():.2%} dry -> {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
