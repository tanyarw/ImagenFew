"""The test year's fixed settings (code_plan/TEST_YEAR_PLAN.md section 4). Standard library only: the guard and the
real-year sewer run (flood-control's environment) import this file too."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "testyear"
PREREG = PACKAGE / "preregistration.json"
OUT_DIR = ROOT / "results" / "testyear"
GEN = ROOT / "results" / "generated_data"
SEWER_GENERATED = ROOT / "results" / "reference" / "gate_b_sewer.json"
FLOOD_CONTROL = ROOT.parent / "flood-control"
FC_REAL_ROWS = "analysis/synthetic_rain/results/bc_efd.csv"       # flood-control's saved real years (2000-2008)

SERIES = "v19_scratch_ep50_nobridge_cal"                          # members: SERIES, SERIES_m1 ... _m3
MEMBERS = 4
TEST_YEAR = 2009
REHEARSAL_YEAR = 2008
TRAIN_YEARS = tuple(range(2000, 2008))
SPLIT_FILE = {2008: "val", 2009: "test"}                          # data/rainfall/splits/<name>_years_labelled.csv

# The rule
BAND = (0.05, 0.95)                         # a real year is "inside" when its percentile among the generated years is here
PRIMARY = ("BC", "total_m3")                # the sewer quantity that must be inside
MAX_RAIN_OUTSIDE = 2                        # of the rain quantities
TOO_CLOSE_MAX_PCT = 10.0                    # one-day windows nearer a training day than the nearest 5% of real ones

CONTROLLERS = ("BC", "EFD")
SEWER_FIELDS = ("total_m3", "creek_m3", "overflow_events_ge1000m3")
SEWER_QUANTITIES = (("BC", "total_m3"), ("EFD", "total_m3"), ("BC", "creek_m3"), ("BC", "overflow_events_ge1000m3"))
RAIN_QUANTITIES = ("annual_mm", "wet_pct", "wet_spell_min", "p999_wet", "max_1h_mm", "max_day_mm",
                   "storms_ge_10mm", "long_storms", "rain_in_long_pct", "max_storm_mm")
MEMORISATION = {"length": 288, "agg": 3, "min_total": 1.0}       # oneday/evaluate.py's one-day setting


def member_names(series=SERIES, members=MEMBERS):
    return [series if k == 0 else f"{series}_m{k}" for k in range(members)]


def series_csv(name):
    return GEN / f"rainfall_synthetic_10y_{name}.csv"


def result_path(year):
    return OUT_DIR / (f"test_{year}.json" if year == TEST_YEAR else f"rehearsal_{year}.json")


def claim_path(year):
    return OUT_DIR / f"test_{year}.claim.json"


def sewer_real_path(year):
    return OUT_DIR / f"sewer_real_{year}.json"
