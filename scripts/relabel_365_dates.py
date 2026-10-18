#!/usr/bin/env python
"""
relabel_365_dates.py — rewrite the date column of generated series to the 365-day calendar (29 February
left out; user's rule, 2026-10-18). Rain values are not touched.

Every generator lays rain out by position in 365-day years; before 2026-10-18 the date column came from
pd.date_range, which adds 29 February 2028 / 2032 and puts every later date one or two days ahead of the
season the rain was generated for. Relabelling is exact: only the labels change. Files already on the
365-day calendar are left alone. Only for 5-min series that start on 2026-01-01 00:00.

    python scripts/relabel_365_dates.py results/generated_data/rainfall_synthetic_10y_v17_ctx_cal_final*.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sitetransfer.timegrid import slot_starts  # noqa: E402  (the calendar generate_hmm_v1.calendar365_dates uses)


def relabel(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df['date'])
    new = slot_starts(len(df), 2026)
    if dates.equals(pd.Series(new)):
        return 'already 365-day'
    if dates.iloc[0] != pd.Timestamp('2026-01-01') or not (dates.diff().iloc[1:] == pd.Timedelta('5min')).all():
        raise ValueError(f'{path}: not a continuous 5-min series from 2026-01-01; not relabelled')
    df['date'] = new.strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(path, index=False)
    return f'relabelled ({len(df)} rows, now ends {new[-1]})'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='+')
    for p in ap.parse_args().csv:
        print(f'{p}: {relabel(p)}')


if __name__ == '__main__':
    main()
