"""The 365-day, 5-minute calendar every series in the project uses.

A series starts at 00:00 on 1 January of its first year; step i covers [i*5 min, (i+1)*5 min) in local
time; 29 February does not exist (the Astlingen record has none, and Singapore's leap days are dropped).
"""
import numpy as np
import pandas as pd

STEP_MINUTES = 5
STEPS_PER_HOUR = 12
STEPS_PER_DAY = 288
DAYS_PER_YEAR = 365
STEPS_PER_YEAR = STEPS_PER_DAY * DAYS_PER_YEAR            # 105,120
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DOY_MONTH = np.repeat(np.arange(1, 13), MONTH_DAYS)          # day of year (0-based) -> month 1..12


def month_hour(n):
    """Month (1-12) and hour (0-23) of every step of an n-step series (same as eval_suite.month_hour)."""
    p = np.arange(n)
    return DOY_MONTH[(p // STEPS_PER_DAY) % DAYS_PER_YEAR], (p % STEPS_PER_DAY) // STEPS_PER_HOUR


def slot_index(start_times, first_year):
    """Position on the 365-day calendar of each interval start (naive local datetimes).

    Returns int64 positions counted from 00:00 on 1 January of `first_year`; -1 for times on 29 February
    (dropped by the calendar) and for times that are not on the 5-minute grid.
    """
    t = pd.DatetimeIndex(start_times)
    year = t.year.to_numpy()
    doy = t.dayofyear.to_numpy() - 1                          # 0-based, counts 29 Feb in leap years
    leap = t.is_leap_year
    feb29 = leap & (t.month == 2) & (t.day == 29)
    doy = np.where(leap & (doy >= 59), doy - 1, doy)          # days after 28 Feb shift back by one
    minute = t.hour.to_numpy() * 60 + t.minute.to_numpy()
    on_grid = (minute % STEP_MINUTES == 0) & (t.second.to_numpy() == 0)
    pos = (year - first_year) * STEPS_PER_YEAR + doy * STEPS_PER_DAY + minute // STEP_MINUTES
    return np.where(feb29 | ~on_grid, -1, pos).astype(np.int64)


def slot_starts(n, first_year):
    """Naive local start time of each of n calendar steps (29 February skipped)."""
    years = np.arange(n) // STEPS_PER_YEAR + first_year
    within = np.arange(n) % STEPS_PER_YEAR
    days = within // STEPS_PER_DAY
    minutes = (within % STEPS_PER_DAY) * STEP_MINUTES
    out = []
    for y in np.unique(years):
        sel = years == y
        jan1 = pd.Timestamp(f"{y}-01-01")
        d = days[sel]
        if pd.Timestamp(f"{y}-12-31").dayofyear == 366:     # leap year: skip 29 Feb
            d = np.where(d >= 59, d + 1, d)
        out.append(jan1 + pd.to_timedelta(d, unit="D") + pd.to_timedelta(minutes[sel], unit="min"))
    return pd.DatetimeIndex(np.concatenate([o.to_numpy() for o in out]))


def year_slice(year_offset):
    """Slice of calendar positions for the year `year_offset` years after the first year."""
    return slice(year_offset * STEPS_PER_YEAR, (year_offset + 1) * STEPS_PER_YEAR)
