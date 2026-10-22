"""Tests for the test year: the scoring and, above all, that no path opens 2009 without every check.

    python -m unittest discover -s testyear/tests -t .
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sitetransfer.timegrid import STEPS_PER_YEAR
from testyear import config as K
from testyear import guard
from testyear import score as S


def _year_with_storm(depth_mm=12.0, steps=84):
    """A dry year with one storm of `steps` 5-min steps (84 = 7 h, longer than 5 h 20) holding `depth_mm`."""
    x = np.zeros(STEPS_PER_YEAR)
    x[1000:1000 + steps] = depth_mm / steps
    return x


class Score(unittest.TestCase):
    def test_percentile_counts_ties_half(self):
        self.assertEqual(S.percentile(5, [1, 2, 5, 9]), 0.625)
        self.assertEqual(S.percentile(0, [1, 2]), 0.0)
        self.assertEqual(S.percentile(10, [1, 2]), 1.0)
        with self.assertRaises(ValueError):
            S.percentile(1, [])

    def test_inside_is_the_closed_band(self):
        self.assertTrue(S.inside(K.BAND[0]) and S.inside(K.BAND[1]))
        self.assertFalse(S.inside(0.0) or S.inside(1.0))

    def test_rain_quantities_of_one_storm(self):
        q = S.rain_quantities(_year_with_storm(), 10.0)
        self.assertAlmostEqual(q["annual_mm"], 12.0)
        self.assertEqual((q["storms_ge_10mm"], q["long_storms"]), (1, 1))
        self.assertAlmostEqual(q["rain_in_long_pct"], 100.0)
        self.assertAlmostEqual(q["max_storm_mm"], 12.0)
        self.assertEqual(set(q), set(K.RAIN_QUANTITIES))
        short = S.rain_quantities(_year_with_storm(steps=60), 10.0)     # 5 h: not a long storm
        self.assertEqual(short["long_storms"], 0)

    def test_years_and_one_year_lengths(self):
        self.assertEqual(S.years(np.zeros(2 * STEPS_PER_YEAR)).shape, (2, STEPS_PER_YEAR))
        with self.assertRaises(ValueError):
            S.years(np.zeros(STEPS_PER_YEAR + 1))
        with self.assertRaises(ValueError):
            S.rain_quantities(np.zeros(STEPS_PER_YEAR), 10.0)          # no rain at all

    def test_sewer_scaled_to_365_days(self):
        rows = [{"controller": "BC", "days": 366.0, "total_m3": 366.0, "creek_m3": 0.0, "overflow_events_ge1000m3": 0}]
        self.assertAlmostEqual(S.sewer_quantities(rows)[("BC", "total_m3")], 365.0)

    def test_verdict(self):
        gen = {q: list(range(40)) for q in K.RAIN_QUANTITIES}
        gen.update({S.sewer_key(*q): list(range(40)) for q in K.SEWER_QUANTITIES})
        mid = {q: 20 for q in K.RAIN_QUANTITIES}
        sewer = S.place({S.sewer_key(*q): 20 for q in K.SEWER_QUANTITIES}, gen)
        self.assertTrue(S.verdict(S.place(mid, gen), sewer, 5.0)["pass"])
        three_out = dict(mid, annual_mm=-1, wet_pct=-1, p999_wet=99)
        v = S.verdict(S.place(three_out, gen), sewer, 5.0)
        self.assertFalse(v["pass"])
        self.assertEqual(v["rain_outside"], ["annual_mm", "p999_wet", "wet_pct"])
        self.assertFalse(S.verdict(S.place(mid, gen), sewer, 10.5)["pass"])                 # copying
        low = S.place({S.sewer_key(*q): -1 for q in K.SEWER_QUANTITIES}, gen)
        self.assertFalse(S.verdict(S.place(mid, gen), low, 5.0)["pass"])                    # sewer


class Guard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(K, "OUT_DIR", self.tmp / "out"),
                        mock.patch.object(K, "PREREG", self.tmp / "preregistration.json")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _nothing_written(self):
        self.assertFalse(K.claim_path(K.TEST_YEAR).exists())
        self.assertFalse(K.result_path(K.TEST_YEAR).exists())

    def test_key_only_from_unlock(self):
        with self.assertRaises(guard.Locked):
            guard.Key("sha", object())

    def test_test_year_needs_a_key(self):
        with self.assertRaises(guard.Locked):
            guard.require(K.TEST_YEAR, None)
        guard.require(K.REHEARSAL_YEAR, None)

    def test_loader_refuses_before_reading(self):
        from testyear import run as R
        with mock.patch.object(R.TD, "load_split", side_effect=AssertionError("read 2009")):
            with self.assertRaises(guard.Locked):
                R.real_year(K.TEST_YEAR)

    def test_command_line_without_flag(self):
        from testyear.__main__ import main
        with self.assertRaises(guard.Locked):
            main(["test"])
        self._nothing_written()

    def test_unlock_without_flag_or_preregistration(self):
        with self.assertRaises(guard.Locked):
            guard.unlock(False)
        with self.assertRaises(guard.Locked):
            guard.unlock(True)                                      # no preregistration
        self._nothing_written()

    def test_unlock_refuses_uncommitted_preregistration(self):
        K.PREREG.write_text(json.dumps({"series_sha256": {}}))     # outside the repository: never committed
        with self.assertRaises(guard.Locked) as e:
            guard.unlock(True)
        self.assertIn("not committed", str(e.exception))
        self._nothing_written()

    def test_unlock_refuses_after_a_result(self):
        K.result_path(K.TEST_YEAR).parent.mkdir(parents=True)
        K.result_path(K.TEST_YEAR).write_text("{}")
        with self.assertRaises(guard.Locked):
            guard.unlock(True)

    def test_claim_is_exclusive(self):
        path = K.claim_path(K.TEST_YEAR)
        guard.claim(path, "sha")
        with self.assertRaises(guard.Locked):
            guard.claim(path, "sha")
        self.assertEqual(json.loads(path.read_text())["prereg_sha256"], "sha")

    def test_series_changes_are_found(self):
        f = self.tmp / "rainfall_synthetic_10y_x.csv"
        f.write_text("date,avg_rainfall\n")
        with mock.patch.object(K, "GEN", self.tmp):
            self.assertEqual(guard.series_unchanged({"series_sha256": {"x": guard.sha256_file(f)}}), [])
            self.assertEqual(len(guard.series_unchanged({"series_sha256": {"x": "0" * 64}})), 1)
            self.assertEqual(len(guard.series_unchanged({"series_sha256": {"y": "0" * 64}})), 1)

    def test_rehearsal_must_match_and_reproduce(self):
        self.assertTrue(guard.rehearsal_ok("sha"))                  # missing
        p = K.result_path(K.REHEARSAL_YEAR)
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"prereg_sha256": "other", "checks": {"sewer_reproduces_flood_control": True}}))
        self.assertTrue(guard.rehearsal_ok("sha"))
        p.write_text(json.dumps({"prereg_sha256": "sha", "checks": {"sewer_reproduces_flood_control": False}}))
        self.assertTrue(guard.rehearsal_ok("sha"))
        p.write_text(json.dumps({"prereg_sha256": "sha", "checks": {"sewer_reproduces_flood_control": True}}))
        self.assertEqual(guard.rehearsal_ok("sha"), [])

    def test_evaluation_refuses_a_changed_rule(self):
        from testyear import run as R
        stored = dict(R.rule(), max_rain_outside=K.MAX_RAIN_OUTSIDE + 1)
        K.PREREG.write_text(json.dumps({"series_sha256": {}, "rule": stored}))
        with mock.patch.object(R.TD, "load_split", side_effect=AssertionError("read a real year")):
            with self.assertRaises(SystemExit) as e:
                R.evaluate_year(K.REHEARSAL_YEAR)
        self.assertIn("rule", str(e.exception))

    def test_rule_round_trips_through_json(self):
        from testyear import run as R
        self.assertEqual(json.loads(json.dumps(R.rule())), R.rule())

    def test_sewer_module_cannot_open_the_test_year(self):
        from testyear import sewer_real
        with self.assertRaises(guard.Locked):
            sewer_real.main(["--year", str(K.TEST_YEAR), "--prereg-sha256", "sha", "--out", str(self.tmp / "s.json")])
        guard.claim(K.claim_path(K.TEST_YEAR), "other")             # a claim for another preregistration
        with self.assertRaises(guard.Locked):
            sewer_real.main(["--year", str(K.TEST_YEAR), "--prereg-sha256", "sha", "--out", str(self.tmp / "s.json")])
        self.assertFalse((self.tmp / "s.json").exists())


if __name__ == "__main__":
    unittest.main()
