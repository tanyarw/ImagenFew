"""Microcanonical random cascade, 30 -> 15 -> 5 minutes: the classical comparator for the fine model (plan section 4).

Each wet box is split into b children by weights that sum to 1, so rain is conserved exactly (microcanonical;
Olsson 1998; Mueller & Haberlandt 2018). As in Mueller & Haberlandt, the weights depend on the box's position
(isolated, starting, enclosed, ending: whether the neighbouring boxes at the same level are wet) and on two volume
classes (at or below / above the median wet volume of that position class). The weights are not fitted to a
distribution: each wet box takes one of the splits observed in 2000-2007 in its class, drawn at random ("fitted
by counting"). Branching 2 (30 -> 15 min), then 3 (15 -> 5 min). No season split, no fitting of parameters.
"""
from dataclasses import dataclass

import numpy as np

POSITIONS = ("isolated", "starting", "enclosed", "ending")
BRANCHING = (2, 3)                        # 30 -> 15 min, then 15 -> 5 min


def position_class(v):
    """0 isolated, 1 starting, 2 enclosed, 3 ending (neighbours outside the series count as dry)."""
    wet = np.asarray(v) > 0
    prev, nxt = np.r_[False, wet[:-1]], np.r_[wet[1:], False]
    return np.select([~prev & ~nxt, ~prev & nxt, prev & nxt], [0, 1, 2], 3)


@dataclass
class Level:
    b: int                                # children per box
    median: np.ndarray                    # [4] median wet box volume per position class
    weights: dict                         # (position, volume class) -> [n, b] observed splits


def _classes(parent, median):
    pos = position_class(parent)
    return pos, (parent > median[pos]).astype(int)


def fit_level(parent, children, b):
    parent = np.asarray(parent, dtype=np.float64)
    wet = parent > 0
    W = np.asarray(children, dtype=np.float64).reshape(-1, b)[wet] / parent[wet, None]
    pos = position_class(parent)[wet]
    v = parent[wet]
    median = np.array([np.median(v[pos == k]) if (pos == k).any() else np.inf for k in range(4)])
    vol = (v > median[pos]).astype(int)
    return Level(b, median, {(k, c): W[(pos == k) & (vol == c)] for k in range(4) for c in range(2)})


def fit(x5c):
    """Both levels from a consistent 5-minute record (grid.two_level), length a whole number of half-hours."""
    x5c = np.asarray(x5c, dtype=np.float64)
    x15 = x5c.reshape(-1, 3).sum(1)
    x30 = x15.reshape(-1, 2).sum(1)
    return [fit_level(x30, x15, 2), fit_level(x15, x5c, 3)]


def split(level, parent, rng):
    parent = np.asarray(parent, dtype=np.float64)
    out = np.zeros((len(parent), level.b))
    wet = parent > 0
    pos, vol = _classes(parent, level.median)
    for (k, c), W in level.weights.items():
        sel = np.flatnonzero(wet & (pos == k) & (vol == c))
        if not len(sel):
            continue
        if not len(W):
            raise ValueError(f"no observed split for position {POSITIONS[k]}, volume class {c}")
        out[sel] = parent[sel, None] * W[rng.integers(0, len(W), len(sel))]
    return out.reshape(-1)


def disaggregate(levels, x30, seed):
    """5-minute values [n30, 6] whose half-hours sum to x30 (up to rounding)."""
    rng = np.random.default_rng(seed)
    x15 = split(levels[0], x30, rng)
    return split(levels[1], x15, rng).reshape(-1, 6)


def summary(levels):
    """What was fitted, for the results file: class sizes, medians, and the share of all-in-one-child splits."""
    out = {}
    for name, lv in zip(("30->15", "15->5"), levels):
        out[name] = {f"{POSITIONS[k]}/{'high' if c else 'low'}": {
            "n": int(len(W)), "all_in_one_child_pct": float(100 * (W.max(1) > 1 - 1e-9).mean()) if len(W) else None}
            for (k, c), W in lv.weights.items()}
        out[name]["median_mm"] = dict(zip(POSITIONS, map(float, lv.median)))
    return out
