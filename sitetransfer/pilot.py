"""The perfect window generator: real windows of the reference years, drawn independently and joined.

It is what any generator that makes each `window` independently could at best achieve, so it measures,
before training, how much structure a block of that length can carry at a site.

Each output block at calendar position p takes a real window that starts on a day within `season_days`
of p's day of year (any reference year) and, when `match_time_of_day`, at the same time of day as p.
Windows that touch an outage step or cross a gap between non-consecutive reference years are never used.
Blocks are joined end to end without smoothing.
"""
import numpy as np

from .timegrid import DAYS_PER_YEAR, STEPS_PER_DAY, STEPS_PER_YEAR


def valid_starts(n, mask, window):
    """Starts whose window lies inside one reference year and contains no outage step."""
    m = np.zeros(n, bool) if mask is None else np.asarray(mask, bool)
    c = np.r_[0, np.cumsum(m)]
    s = np.arange(n - window + 1)
    clean = (c[s + window] - c[s]) == 0
    same_year = (s // STEPS_PER_YEAR) == ((s + window - 1) // STEPS_PER_YEAR)
    return s[clean & same_year]


def perfect_window(x, mask, window, n_years, seed, match_time_of_day=True, season_days=15):
    rng = np.random.default_rng(seed)
    starts = valid_starts(len(x), mask, window)
    doy = (starts // STEPS_PER_DAY) % DAYS_PER_YEAR
    slot = starts % STEPS_PER_DAY
    groups = {}
    if match_time_of_day:
        for k in np.unique(slot):
            sel = slot == k
            groups[int(k)] = (starts[sel], doy[sel])
    else:
        groups[None] = (starts, doy)
    n_out = n_years * STEPS_PER_YEAR
    out = np.empty(n_out + window, dtype=np.float64)
    fallbacks = 0
    for p in range(0, n_out, window):
        key = int(p % STEPS_PER_DAY) if match_time_of_day else None
        st, dy = groups[key]
        d = np.abs(dy - (p // STEPS_PER_DAY) % DAYS_PER_YEAR)
        d = np.minimum(d, DAYS_PER_YEAR - d)
        cand = st[d <= season_days]
        if not len(cand):                      # no clean window this close in season: widen once
            cand = st[d <= 2 * season_days]
            fallbacks += 1
        s = rng.choice(cand)
        out[p:p + window] = x[s:s + window]
    return out[:n_out], {"fallbacks": fallbacks, "candidates": int(len(starts))}


VARIANTS = {
    "w64": dict(window=64, match_time_of_day=True),          # a perfect 5h20 generator
    "w64_anytime": dict(window=64, match_time_of_day=False),  # ... that is not told the time of day
    "w288": dict(window=288, match_time_of_day=True),         # a perfect one-day generator
}
