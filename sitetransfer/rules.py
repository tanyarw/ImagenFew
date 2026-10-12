"""Rules that re-measure data-derived parameters. Pure functions of arrays: no files, no globals.

Each rule is written so that, applied to Astlingen's training years, it returns the value the existing
pipeline already uses (invariant 1). `python -m sitetransfer derive --site astlingen` checks that.
"""
import numpy as np

from .timegrid import STEPS_PER_YEAR

STORM_GAP_STEPS = 24            # a dry run this long (2 h) ends a storm (eval_suite.STORM_GAP)
LONG_STEPS = 64                 # one generator block (5 h 20 min)


def _positive_values(x, decimals=6):
    x = np.asarray(x, dtype=np.float64)
    return np.round(x[x > 0], decimals)


def resolution(x):
    """Smallest non-zero value of the series (its measurement step)."""
    pos = _positive_values(x)
    if not len(pos):
        raise ValueError("series has no rain")
    return float(pos.min())


def wet_threshold(x, min_share=0.01):
    """The smallest value that makes up at least `min_share` of the non-zero steps.

    Values below it count as dry, in real and generated rain alike. Astlingen: 0.0025 mm is only 0.09%
    of wet steps, 0.005 mm is 2.6%, so the rule gives 0.005 mm, the project's frozen threshold.
    """
    pos = _positive_values(x)
    vals, counts = np.unique(pos, return_counts=True)
    ok = counts / len(pos) >= min_share
    if not ok.any():
        raise ValueError(f"no value reaches {min_share:.0%} of wet steps")
    return float(vals[np.argmax(ok)])


def asinh_scale(x, tau):
    """Median of the wet steps (>= tau): the s in asinh(x / s). Astlingen: 0.035 mm."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(x[x >= tau]))


def storms_2h(x, gap_steps=STORM_GAP_STEPS):
    """(duration in steps, depth) of every storm: wet steps with no dry run of `gap_steps` or more."""
    x = np.asarray(x, dtype=np.float64)
    wet = np.flatnonzero(x > 0)
    if not len(wet):
        return np.zeros(0, int), np.zeros(0)
    br = np.flatnonzero(np.diff(wet) > gap_steps)
    s, e = np.r_[wet[0], wet[br + 1]], np.r_[wet[br], wet[-1]]
    c = np.r_[0.0, np.cumsum(x)]
    return e - s + 1, c[e + 1] - c[s]


def depth_share(depths, threshold_mm):
    """Share of storms at least `threshold_mm` deep (Astlingen, 10 mm: 4.1%)."""
    return float(np.mean(np.asarray(depths) >= threshold_mm))


def storm_depth_threshold(depths, share):
    """The depth of the k-th deepest storm, k = round(share * number of storms).

    So exactly k storms are at least this deep. Applied to Astlingen with Astlingen's own share for 10 mm,
    it returns the shallowest storm of at least 10 mm, which classifies every storm exactly as 10 mm does.
    """
    d = np.sort(np.asarray(depths, dtype=np.float64))[::-1]
    k = int(round(share * len(d)))
    if k < 1:
        raise ValueError("share too small for this many storms")
    return float(d[k - 1])


def outage_share_by_year(mask, first_year):
    m = np.asarray(mask, dtype=bool)
    n = len(m) // STEPS_PER_YEAR
    return {first_year + k: float(m[k * STEPS_PER_YEAR:(k + 1) * STEPS_PER_YEAR].mean()) for k in range(n)}


def reference_years(outage_share, candidate_years, max_share):
    """Training years clean enough to serve as the evaluation reference (outage share below max_share)."""
    return [y for y in candidate_years if outage_share[y] < max_share]


def matched_epochs(n_windows, batch_size, reference_steps):
    """Epochs giving at least `reference_steps` optimiser steps (DataLoader drop_last=True)."""
    steps_per_epoch = n_windows // batch_size
    if steps_per_epoch < 1:
        raise ValueError("fewer windows than one batch")
    return int(np.ceil(reference_steps / steps_per_epoch))


def usable_windows(mask, seq_len):
    """Number of seq_len windows that contain no masked step."""
    m = np.asarray(mask, dtype=np.int64)
    if len(m) < seq_len:
        return 0
    c = np.r_[0, np.cumsum(m)]
    return int(((c[seq_len:] - c[:-seq_len]) == 0).sum())
