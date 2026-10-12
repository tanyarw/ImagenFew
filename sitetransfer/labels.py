"""Seasonal phases: a site-independent port of scripts/fit_seasonal_labels.py.

The procedure (unchanged): average the training years slot by slot into one 365-day climatology, smooth
it with a 14-day centred rolling mean, fit a 4-state Gaussian HMM, order the states by mean rain, take the
most common state of each day, then merge runs shorter than the median run length. The phase of every
step is a function of its calendar slot only. Outage steps are left out of the slot averages.

Regression test (tests/test_labels_regression.py): on Astlingen's training years this reproduces the
existing labels and calendar profile exactly.
"""
from math import floor

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from .timegrid import STEPS_PER_DAY, STEPS_PER_YEAR

N_STATES = 4
SMOOTH_WINDOW = 4032            # 14 days of 5-minute steps
HMM_SEED = 42
BLOCK_SIZES = [24, 36, 48, 64, 72, 96, 144, 288]


def slot_climatology(x, mask=None):
    """Mean rain of each of the 105,120 calendar slots over the years given (outage steps excluded)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) % STEPS_PER_YEAR:
        raise ValueError("whole years expected")
    X = x.reshape(-1, STEPS_PER_YEAR)
    if mask is None:
        return X.mean(axis=0)
    M = np.asarray(mask, bool).reshape(-1, STEPS_PER_YEAR)
    X = np.where(M, np.nan, X)
    if np.isnan(X).all(axis=0).any():
        raise ValueError("a calendar slot has no observed value in any year")
    return np.nanmean(X, axis=0)


def smooth_min_duration(state_seq, min_days, steps_per_day=STEPS_PER_DAY):
    """Merge runs shorter than min_days into a neighbour (verbatim from fit_seasonal_labels.py)."""
    min_steps = int(min_days * steps_per_day)
    s = state_seq.copy()
    changed = True
    while changed:
        changed = False
        runs = []
        cur_st = s[0]
        st_idx = 0
        for i in range(1, len(s)):
            if s[i] != cur_st or i == len(s) - 1:
                en_idx = i if s[i] != cur_st else i + 1
                runs.append((cur_st, st_idx, en_idx, en_idx - st_idx))
                cur_st = s[i]
                st_idx = i
        for idx, (st_val, start, end, dur) in enumerate(runs):
            if dur < min_steps:
                prev_st = runs[idx - 1][0] if idx > 0 else (runs[idx + 1][0] if idx < len(runs) - 1 else st_val)
                next_st = runs[idx + 1][0] if idx < len(runs) - 1 else prev_st
                if prev_st == next_st:
                    s[start:end] = prev_st
                else:
                    prev_dur = runs[idx - 1][3] if idx > 0 else 0
                    next_dur = runs[idx + 1][3] if idx < len(runs) - 1 else 0
                    s[start:end] = prev_st if prev_dur >= next_dur else next_st
                changed = True
                break
    return s


def fit_phases(climatology):
    """Calendar profile of seasonal phases (one state per slot) from a slot climatology."""
    clim = pd.DataFrame({"annual_interval_idx": np.arange(STEPS_PER_YEAR), "mean_intensity": climatology})
    clim["mean_intensity_smooth"] = clim["mean_intensity"].rolling(window=SMOOTH_WINDOW, min_periods=1,
                                                                   center=True).mean()
    X = StandardScaler().fit_transform(clim[["mean_intensity_smooth"]])
    model = hmm.GaussianHMM(n_components=N_STATES, covariance_type="full", n_iter=200, random_state=HMM_SEED)
    model.fit(X)
    clim["raw_state"] = model.predict(X)
    order = clim.groupby("raw_state")["mean_intensity_smooth"].mean().sort_values()
    clim["hmm_state"] = clim["raw_state"].map({old: new for new, old in enumerate(order.index)})
    clim["day_idx"] = clim["annual_interval_idx"] // STEPS_PER_DAY
    modes = clim.groupby("day_idx")["hmm_state"].agg(lambda s: s.value_counts().index[0])
    clim["hmm_state"] = clim["day_idx"].map(modes).astype(int)

    states = clim["hmm_state"].values
    runs, cur, start = [], states[0], 0
    for i in range(1, len(states)):
        if states[i] != cur or i == len(states) - 1:
            end = i if states[i] != cur else i + 1
            runs.append((end - start) / STEPS_PER_DAY)
            cur, start = states[i], i
    min_days = max(1, floor(float(np.median(runs))))
    profile = smooth_min_duration(states.copy(), min_days=min_days)
    return dict(profile=profile.astype(int), min_days=int(min_days),
                n_transitions=int(np.sum(profile[:-1] != profile[1:])),
                state_means=[float(v) for v in order.values])


def transition_bundle(train_states, profile, min_days, block_size=64, smooth=1e-5):
    """The calendar/transition file the generators read (keys as fit_seasonal_labels.py writes them)."""
    n_blocks = len(train_states) // block_size
    seq = np.asarray(train_states)[:n_blocks * block_size].reshape(n_blocks, block_size)[:, 0]
    C = np.zeros((N_STATES, N_STATES)) + smooth
    for a, b in zip(seq[:-1], seq[1:]):
        C[a, b] += 1.0
    P = C / C.sum(axis=1, keepdims=True)
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    pi = np.abs(pi / pi.sum())
    u, c = np.unique(train_states, return_counts=True)
    return {"P_block": P, "P": P, "C_block": C, "pi_block": pi, "pi": pi, "block_size": block_size,
            "duration_hours": block_size * 5 / 60.0, "block_sequence": seq, "n_blocks": n_blocks,
            "n_transitions": int(np.sum(seq[:-1] != seq[1:])),
            "regime_proportions": {int(k): float(v / len(train_states)) for k, v in zip(u, c)},
            "n_states": N_STATES, "post_processing_min_days": min_days,
            "calendar_sequence_1yr": profile[::block_size], "climatology_profile_1yr": profile}


def label_steps(profile, n_steps):
    """Phase of every step of an n-step series that starts on 1 January."""
    return np.asarray(profile)[np.arange(n_steps) % STEPS_PER_YEAR]
