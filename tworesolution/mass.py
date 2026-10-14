"""Mass matching (plan section 2): make each half-hour's six fine values sum to its coarse total.

Applied in the same way to the fine model's output and to the cascade's, so the two are compared on structure
and not on this finishing step. For every half-hour with total T:
  1. values below the 5-minute wet threshold tau are set to zero;
  2. if nothing is left but T > 0, T goes to the step where the raw value was largest (counted: `empty`);
  3. the remaining values are scaled to sum to T;
  4. a scaled value still below tau is moved onto the half-hour's largest value (counted: `moved`).
Dry half-hours (T = 0) get six zeros. Every wet total must be >= tau, which holds because coarse totals are
cleaned with the coarse threshold (0.0075 mm >= 0.005 mm); the result then has no value in (0, tau).
"""
import numpy as np

from .grid import FACTOR


def finish(raw, totals, tau):
    """(fine [n, 6] in mm, counts) for raw fine values [n, 6] (any sign) and coarse totals [n]."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, FACTOR)
    t = np.asarray(totals, dtype=np.float64)
    if raw.shape[0] != len(t):
        raise ValueError(f"{raw.shape[0]} half-hours of fine values for {len(t)} totals")
    if (t < 0).any() or ((t > 0) & (t < tau)).any():
        raise ValueError("totals must be 0 or at least the 5-minute wet threshold")
    wet = t > 0
    r = np.where(raw >= tau, raw, 0.0)
    empty = wet & (r.sum(1) == 0)
    r[np.flatnonzero(empty), np.argmax(raw[empty], axis=1)] = 1.0
    f = np.zeros_like(r)
    f[wet] = r[wet] * (t[wet] / r[wet].sum(1))[:, None]

    small = (f > 0) & (f < tau)
    rows = np.flatnonzero(small.any(1))
    moved_mm = 0.0
    if len(rows):
        sub = f[rows]
        j = np.argmax(sub, axis=1)
        moved = np.where(small[rows], sub, 0.0).sum(1)
        sub = np.where(small[rows], 0.0, sub)
        sub[np.arange(len(rows)), j] += moved
        f[rows] = sub
        moved_mm = float(moved.sum())
    counts = dict(half_hours=int(len(t)), wet_half_hours=int(wet.sum()), empty=int(empty.sum()),
                  moved_values=int(small.sum()), moved_mm=moved_mm, total_mm=float(t.sum()))
    return f, counts
