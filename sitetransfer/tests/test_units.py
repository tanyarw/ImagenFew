"""Fast unit tests on small synthetic inputs (no project data needed).

    python -m unittest discover -s sitetransfer/tests -t .
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sitetransfer import ingest_singapore as ing
from sitetransfer import pilot, rules
from sitetransfer import timegrid as tg


class TimeGrid(unittest.TestCase):
    def test_feb29_dropped_and_march_shifted(self):
        pos = tg.slot_index(pd.to_datetime(["2020-02-29 12:00", "2020-03-01 00:00", "2019-03-01 00:00"]), 2017)
        self.assertEqual(pos[0], -1)
        self.assertEqual(pos[1], 3 * tg.STEPS_PER_YEAR + 59 * tg.STEPS_PER_DAY)
        self.assertEqual(pos[2], 2 * tg.STEPS_PER_YEAR + 59 * tg.STEPS_PER_DAY)

    def test_round_trip(self):
        n = 4 * tg.STEPS_PER_YEAR                       # 2019-2022 contains the leap year 2020
        starts = tg.slot_starts(n, 2019)
        np.testing.assert_array_equal(tg.slot_index(starts, 2019), np.arange(n))
        self.assertFalse(((starts.month == 2) & (starts.day == 29)).any())

    def test_off_grid_rejected(self):
        self.assertEqual(tg.slot_index(pd.to_datetime(["2018-01-01 00:03"]), 2018)[0], -1)


class Rules(unittest.TestCase):
    def test_wet_threshold_skips_rare_values(self):
        x = np.r_[np.zeros(1000), [0.0025] * 5, [0.005] * 100, [0.1] * 400]
        self.assertEqual(rules.wet_threshold(x), 0.005)          # 0.0025 is 5 of 505 wet steps (< 1%)

    def test_storm_depth_threshold_classifies_like_reference(self):
        d = np.array([1, 2, 5, 9.9, 10, 10.5, 30])
        thr = rules.storm_depth_threshold(d, rules.depth_share(d, 10))
        self.assertEqual(int((d >= thr).sum()), int((d >= 10).sum()))

    def test_storms_2h(self):
        x = np.zeros(200)
        x[[10, 20, 60, 61]] = 1.0                            # gap 10-20 < 24 -> one storm; 20-60 >= 24 -> new
        dur, dep = rules.storms_2h(x)
        np.testing.assert_array_equal(dur, [11, 2])
        np.testing.assert_array_equal(dep, [2.0, 2.0])

    def test_usable_windows_and_epochs(self):
        m = np.zeros(100, bool)
        m[50] = True
        self.assertEqual(rules.usable_windows(m, 10), 91 - 10)
        self.assertEqual(rules.matched_epochs(4096, 2048, 10), 5)


class Ingest(unittest.TestCase):
    def test_read_raw_year(self):
        rows = [  # (timestamp, update, station, value)
            ("2020-02-28T23:59:59+08:00", "2020-02-29T00:01:00+08:00", "S1", 0.2),   # end 00:00 -> start 23:55 (28 Feb)
            ("2020-03-01T00:05:00+08:00", "2020-03-01T00:06:00+08:00", "S1", 0.4),   # start 00:00 1 Mar
            ("2020-03-01T00:04:59+08:00", "2020-03-01T00:09:00+08:00", "S1", 0.6),   # same slot, updated later: kept
            ("2020-02-29T12:00:00+08:00", "2020-02-29T12:01:00+08:00", "S1", 1.0),   # 29 Feb: dropped
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.csv"
            pd.DataFrame([{"timestamp": t, "reading_update_timestamp": u, "station_id": s, "station_name": "x",
                           "location_longitude": 103.8, "location_latitude": 1.3, "reading_value": v}
                          for t, u, s, v in rows]).to_csv(p, index=False)
            r = ing.read_raw_year(p, 2020, 2020, tg.STEPS_PER_YEAR)
        self.assertEqual(r.counts["duplicates_removed"], 1)
        self.assertEqual(r.counts["feb29"], 1)
        got = dict(zip(r.readings["pos"], r.readings["value"]))
        mar1 = 59 * tg.STEPS_PER_DAY
        self.assertEqual(got[mar1], 0.6)
        self.assertEqual(got[mar1 - 1], 0.2)                 # 28 Feb 23:55 is the step before 1 Mar 00:00

    def test_fill_gauge_idw_and_grid(self):
        target = np.array([0.2, np.nan, np.nan])
        nb = np.array([[0.0, 0.0], [0.4, 0.2], [np.nan, np.nan]])
        filled, flag = ing.fill_gauge(target, nb, np.array([1.0, 2.0]), 0.2, 2, 4, 10.0)
        self.assertAlmostEqual(filled[1], 0.4)               # (0.4*1 + 0.2*0.25)/1.25 = 0.36 -> 0.4 on the grid
        self.assertTrue(np.isnan(filled[2]))
        np.testing.assert_array_equal(flag, [False, True, False])

    def test_network_gaps(self):
        x = np.array([0.0, np.nan, 0.0, 0.1, np.nan, np.nan, 0.4] + [np.nan] * 20 + [0.0])
        v, f = ing.fill_network_gaps(x, max_gap=12, grid=0.05)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(f[1], 2)
        np.testing.assert_allclose(v[4:6], [0.2, 0.3])
        self.assertTrue((f[7:27] == 3).all() and (v[7:27] == 0).all())


class Pilot(unittest.TestCase):
    def test_windows_are_real_and_clean(self):
        rng = np.random.default_rng(0)
        x = rng.random(2 * tg.STEPS_PER_YEAR)
        mask = np.zeros(len(x), bool)
        mask[1000:2000] = True
        y, _ = pilot.perfect_window(x, mask, window=64, n_years=1, seed=1)
        starts = pilot.valid_starts(len(x), mask, 64)
        self.assertFalse(((starts < 2000) & (starts + 64 > 1000)).any())
        self.assertFalse(((starts // tg.STEPS_PER_YEAR) != ((starts + 63) // tg.STEPS_PER_YEAR)).any())
        block = y[640:704]
        self.assertTrue(any(np.array_equal(block, x[s:s + 64]) for s in starts[(starts % tg.STEPS_PER_DAY) == 640 % 288]))


if __name__ == "__main__":
    unittest.main()
