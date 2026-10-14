"""The two resolutions and the conversions between them (numpy only).

Coarse = 30-minute totals, fine = 5-minute rain: half-hour k is fine steps 6k..6k+5. Every series starts at 00:00
on 1 January of the project's 365-day calendar (sitetransfer/timegrid.py), so the coarse calendar is the fine one
taken at every 6th step.

Fine windows (plan section 10): L = 64 fine steps starting PAD = 2 steps before a half-hour boundary. The middle
KEEP = 60 steps are 10 whole half-hours; the 2 steps at each end belong to the neighbouring half-hours, so the
model sees the totals on both sides of what it keeps. Training windows and generation windows have this same
layout. At generation, window k keeps half-hours 10k..10k+9, which tile the series exactly once.
"""
import numpy as np

from sitetransfer.site import clean_series as clean      # non-negative, zero below the threshold

FACTOR = 6                       # 5-minute steps per half-hour
L = 64                           # window length at both levels (8 x 8 image)
PAD = 2                          # fine steps of each neighbouring half-hour at the window's ends
KEEP = L - 2 * PAD               # 60 fine steps = 10 half-hours kept per fine window
HALF_HOURS_KEPT = KEEP // FACTOR
STEPS_PER_YEAR_5 = 105_120
STEPS_PER_YEAR_30 = STEPS_PER_YEAR_5 // FACTOR
N_CLASSES = 4                    # seasonal phases


def aggregate(x5):
    """Half-hour totals of a 5-minute series."""
    x5 = np.asarray(x5, dtype=np.float64)
    if len(x5) % FACTOR:
        raise ValueError(f"{len(x5)} five-minute steps is not a whole number of half-hours")
    return x5.reshape(-1, FACTOR).sum(1)


def spread(x30):
    """Each half-hour's total spread evenly over its six 5-minute steps.

    This is both the fine model's condition and the "30-minute view" used to compare series at 30 minutes with
    the existing 5-minute metric code."""
    return np.repeat(np.asarray(x30, dtype=np.float64) / FACTOR, FACTOR)


def two_level(x5, tau5, tau30):
    """(x5c, x30): a 5-minute series at both resolutions, consistent by construction.

    x30 = half-hour totals of the cleaned 5-minute series, cleaned with the coarse threshold tau30;
    x5c = the cleaned 5-minute series with every half-hour whose total fell below tau30 set dry, so the six values
    of every half-hour sum to its total exactly."""
    x5 = clean(x5, tau5)
    x30 = clean(aggregate(x5), tau30)
    return x5 * np.repeat(x30 > 0, FACTOR), x30


def view30(x5, tau5, tau30):
    """The 30-minute view of a 5-minute series: its cleaned half-hour totals, spread evenly."""
    return spread(two_level(x5, tau5, tau30)[1])


# ── fine windows ────────────────────────────────────────────────────────────────────────────────────

def train_starts(n5):
    """Training window starts: every start 2 steps before a half-hour boundary, window inside the series."""
    return np.arange(FACTOR - PAD, n5 - L + 1, FACTOR)


def generation_starts(n30):
    """Generation window starts (fine steps; the first is -2, before the series): window k keeps 10k..10k+9."""
    return KEEP * np.arange(-(-n30 // HALF_HOURS_KEPT)) - PAD


def condition_windows(x30, starts):
    """[n, L] condition of each window in mm per 5 minutes (half-hour total / 6); dry outside the series."""
    c5 = spread(x30)
    cpad = np.r_[np.zeros(PAD), c5, np.zeros(L)]
    return cpad[np.asarray(starts)[:, None] + PAD + np.arange(L)]


def value_windows(x5, starts):
    """[n, L] values of a 5-minute series in each window (windows must lie inside the series)."""
    starts = np.asarray(starts)
    if starts.min() < 0 or starts.max() + L > len(x5):
        raise ValueError("a window reaches outside the series")
    return np.asarray(x5)[starts[:, None] + np.arange(L)]


def kept(windows, n30):
    """Join the kept middle of each generation window into one [n30, 6] array of half-hours."""
    w = np.asarray(windows)[:, PAD:PAD + KEEP].reshape(-1)
    return w[:n30 * FACTOR].reshape(n30, FACTOR)


# ── seasonal phase of a window ──────────────────────────────────────────────────────────────────────

def window_labels(profile, starts, length=L):
    """Majority seasonal phase over each window; ties go to the lowest phase (as RegimeDataset).

    `profile` is the phase of every step of one calendar year at the series' resolution; a window's step at
    position p (which may be negative, before the series) has phase profile[p mod len(profile)]."""
    profile = np.asarray(profile)
    idx = (np.asarray(starts)[:, None] + np.arange(length)) % len(profile)
    states = profile[idx]
    counts = np.stack([(states == c).sum(1) for c in range(N_CLASSES)], axis=1)
    return counts.argmax(1)
