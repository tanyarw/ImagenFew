# Singapore Transfer Plan

**Written:** 2026-10-12, before any Singapore code or model exists. **Status:** implemented on 2026-10-12
in the package `sitetransfer/` (decisions in section 10 taken as recommended); the trained model is
pending. **Changes made while implementing** are listed in section 13.
**Question:** does our rainfall generator recipe, built and tuned on Astlingen (Germany), work on a site
with very different rain, Singapore, when it is applied with no retuning?

Every number about the Singapore data below was measured on 2026-10-12 from
`data/singapore/raw/HistoricalRainfallacrossSingapore{2017..2024}.csv` (scratch scripts; the build
script in step 2 will reproduce them).

---

## 1. The plan in one page

* **What "transfer" means here.** We take the v14 recipe exactly as it is (the asinh transform,
  64-step blocks, four seasonal phases, the same network, checkpoint, settings and assembly), re-estimate
  only the quantities that are *measured from data* (using written rules), train once on Singapore, and
  score it with the same evaluation code. Nothing is tuned by looking at Singapore results.
* **Two questions, kept apart.**
  1. *Does the recipe transfer?* Does the Singapore model get right what the Astlingen model got right,
     judged against Singapore's own year-to-year variation?
  2. *Do the findings transfer?* Do the paper's claims (for example "a fixed-window generator loses
     the structure longer than its window") hold at a second site?
* **The target series.** The average of four nearby Singapore gauges (Marine Parade Road, Nicoll
  Highway, Kim Chuan Road, Poole Road; all within 4.9 km, about 95% complete), mirroring Astlingen's
  average of four gauges. The same four gauges at every time step.
* **Missing data.** Interpolated in space where any nearby gauge reports (most gaps); interpolated in
  time for network-wide gaps up to 1 hour; longer network outages are flagged, left out of training
  windows and handled explicitly in evaluation. Every filled value is flagged.
* **Splits.** Train 2017–2022, validation 2023, test 2024. Nothing is fitted on 2023–2024.
* **Code.** One site description file per site; the existing scripts read it instead of hard-coded
  Astlingen values. A regression test proves the Astlingen results are unchanged to the last digit.
* **Evaluation.** The same metrics, with pass bands recalculated from Singapore's own real-vs-real
  variation; the same comparison baselines refitted on Singapore; a "perfect window" pilot that tells us,
  before training, how much a 5h20 window can capture at Singapore.
* **Expected outcome (to be written down properly before training).** Amounts and extremes transfer;
  the daily cycle fails, because the model has no time-of-day input and Singapore's rain has a strong
  afternoon peak; long-storm losses are smaller than at Astlingen because Singapore's rain is shorter.
* **Effort.** About 7–9 working days of code and checks, plus one cluster training run (~9 hours).

---

## 2. What the Singapore data look like

### 2.1 The raw files

| Property | What we found | Consequence |
| :--- | :--- | :--- |
| Source | National network, one row per station per 5 minutes, 2017–2024, 49 million rows, ~8 GB | Read one year at a time (8 s per year) |
| Measurement | Tipping-bucket gauges: every reading is a multiple of **0.2 mm** at the stations we use | 80 times coarser than Astlingen's averaged series (steps of 0.0025 mm) |
| Instruments | From 2020, some stations report finer values (e.g. 17.622 mm); 67–78% of non-zero values are still multiples of 0.2 | Use only stations that stay at 0.2 mm in every year |
| Stations | 54 to 80 per year; the network changes | Never average over a changing set of stations |
| Time zone | Raw timestamps are Singapore time (UTC+8) | Keep local time; the daily cycle depends on it |
| Timestamps | Two styles: `hh:mm:00` and `hh:mm:59` (one second early); 86 readings collide after rounding (2017) | Round to the 5-minute grid; one value per station and slot |
| Label | A reading is the total of the 5 minutes *ending* at its timestamp | Shift to Astlingen's convention: label = start of the interval |
| Leap years | 2020 and 2024 have 29 February | Drop 29 February, as the Astlingen record does (365-day years) |
| Start | The first 8 hours of 2017 are missing for every station | Network outage (section 4.4) |
| 2018 | 16% of all 5-minute slots have no data from any station; August almost entirely missing; February–April patchy | 2018 is used for training windows only where observed; not part of the evaluation reference |

**The existing file `data/singapore/interim/singapore_rainfall_raw_mean_2017_2024.csv` must not be
used.** Its timestamps are in UTC (its rain peaks at "07:00", which is 15:00 in Singapore), and it
averages over whichever stations reported at each step (its smallest value is 0.2 mm ÷ 78), so its
statistics drift as the network changes.

### 2.2 The four-gauge average next to Astlingen

Computed over steps where all four gauges report (2017–2024 unless stated).

| | Astlingen (train 2000–2007) | Singapore cluster |
| :--- | :---: | :---: |
| Gauges in the average | 4 | 4 (max spacing 4.9 km) |
| Smallest step (resolution) | 0.0025 mm | 0.05 mm |
| Rain per year | 710 mm | 1,600–2,850 mm (2019 driest) |
| Dry 5-minute steps | 90.8% | 94.7% |
| Median wet step | 0.035 mm | 0.15 mm |
| 99.9th percentile of wet steps | 1.60 mm | 5.81 mm |
| Largest 5-minute value | 5.55 mm | 8.3 mm |
| Single-step showers | 38% | 52% |
| Mean wet run | 32 min | 21 min |
| Unbroken wet runs > 5h20 per year | 14.3 | about 2.5 |
| Storms per year (a 2-hour dry gap ends one) | 384 | 340 |
| Storms longer than 5h20 (2-hour gap) per year | 74.8 | 32 |
| Share of rain in those storms | 65% | 38% |
| Daily cycle (mean rain, peak hour ÷ lowest hour) | weak | strong: peak 15:00, lowest 01:00, ratio 8 |
| Wettest months | summer | November–January; driest February and July |

What this means for the model, in short:

* **Heavier and drier.** More zeros and much larger wet values. The asinh transform with a re-measured
  scale handles the larger range (section 5).
* **Coarser.** A third of all wet steps are a single tip at one of the four gauges (0.05 mm). Short-scale
  "texture" statistics at Singapore mostly describe the gauges, as they partly did at Astlingen.
* **Shorter rain, but long storms still exist.** Unbroken rain rarely outlasts a block, but storms with
  short breaks often do (32 a year carrying 38% of the rain). The window question is still live, at the
  storm scale.
* **A strong daily cycle.** v14 has no input that tells it the time of day. At Astlingen this did not
  matter much (no version passed the diurnal check, and nor did real held-out years). At Singapore it
  will be a visible failure.

---

## 3. The guiding rule: same procedure, re-measured inputs, no tuning

Every quantity in the pipeline falls into exactly one of three classes. This table is the contract;
the code enforces it (section 7).

| Class | Meaning | Examples |
| :--- | :--- | :--- |
| **Fixed** | Copied from Astlingen unchanged | network, checkpoint, training settings, block length (64 steps), delay embedding, number of seasonal phases (4), assembly mode, sampler, metric definitions, random seeds |
| **Re-measured by a rule** | Computed from Singapore *training years only* by a written rule; the same rule must give Astlingen's existing value | wet threshold, asinh scale, standardisation, seasonal phases, transition matrix, reference statistics, pass bands, storm-depth threshold |
| **Site-specific, replaced or dropped** | Exists only for Astlingen | the German design-rain table (KOSTRA), the sewer check, the four-gauge split for SWMM |

**Invariant 1:** every rule in the second class, run on the Astlingen data, must return the value
Astlingen already uses. If it doesn't, the rule is wrong, not the old value.

---

## 4. Building the Singapore series (data rules)

### 4.1 Reading and cleaning

1. Read only `timestamp, station_id, reading_value` (plus coordinates once).
2. Parse the local wall-clock time (the first 19 characters; do not convert to UTC).
3. Round to the nearest 5 minutes. `08:04:59` and `08:05:00` both become `08:05`.
4. If a station has two readings in one slot after rounding, keep the later-updated one and count it
   (expected: about 86 cases, all in 2017).
5. Shift labels by −5 minutes so that a value is labelled by the start of its interval, as in
   Astlingen (`2000-01-01 00:00` is the first Astlingen step).
6. Drop 29 February. Each year is then exactly 105,120 steps, starting at 00:00 on 1 January.

**Check:** in the built series, mean rain by hour must peak between 14:00 and 16:00. A peak near
07:00 means a UTC slip.

### 4.2 Choosing the stations

* **Rule for usable stations:** at least 85% complete in every year except 2018, and every non-zero
  reading a multiple of 0.2 mm in every year. Result: **20 stations**.
* **Rule for the target cluster:** among those 20, the four with the smallest largest pairwise
  distance. Result: S113 Marine Parade Road, S119 Nicoll Highway, S43 Kim Chuan Road, S78 Poole Road
  (4.9 km, mean completeness 0.951).
* The other 16 usable stations are used only to fill gaps (section 4.3).

**Invariant 2:** the target is the average of the *same four gauges* at every step.

Why not one gauge, or the whole island? A single gauge is too coarse (0.2 mm steps turn steady
rain into isolated tips: 63% single-step showers, 0.26 unbroken runs > 5h20 per year). An island-wide
average of ~60 gauges is a different, much smoother quantity than Astlingen's four-gauge average. The
four-gauge cluster is the closest match to what the method was built on. A single gauge can be run
later as a robustness check.

### 4.3 Filling missing values

Gaps fall into three kinds. Measured for the cluster:

| Year | All four report | Some of the four missing | All four missing, network reports | Whole network down |
| :--- | :---: | :---: | :---: | :---: |
| 2017 | 89.9% | 8.6% | 0.1% | 1.4% |
| 2018 | 76.7% | 7.3% | 0.2% | **15.7%** |
| 2019 | 93.4% | 2.9% | 0.0% | 3.7% |
| 2020 | 94.3% | 2.0% | 0.0% | 3.7% |
| 2021 | 92.9% | 5.8% | 0.0% | 1.3% |
| 2022 | 97.2% | 2.6% | 0.0% | 0.1% |
| 2023 | 96.2% | 0.8% | 0.0% | 3.0% |
| 2024 | 94.5% | 4.9% | 0.4% | 0.3% |

The rules, in order:

1. **A gauge is missing but other usable stations report (spatial fill).** Fill that gauge from the
   nearest usable stations that report at that step: inverse-distance weighting (weights 1/d²) of
   up to 4 stations within 10 km, or the nearest 4 if none are that close. Round the filled value to
   the 0.2 mm grid. Flag `filled_space`.
   *Why space and not time:* a storm reaches nearby gauges within the same 5 minutes, so neighbours at
   the same moment carry the timing of the storm. Interpolating along time would invent drizzle
   between two readings and smear storm edges.
2. **The whole network is down for at most 1 hour (12 steps) (short time fill).** If the cluster value
   is zero on both sides, fill zeros. Otherwise interpolate the cluster value linearly in time and
   round to the 0.05 mm grid. Flag `filled_time`.
3. **The whole network is down for more than 1 hour (outage).** No honest interpolation exists. Write
   a zero placeholder and flag `outage`. Outage steps are never used to train or to judge the model
   (rules below).

After filling, the cluster value is the average of four values on the 0.2 mm grid, so every value is a
multiple of 0.05 mm.

**Invariant 3:** every filled value carries a flag; the build writes the share of each flag per year
to an audit file, and the evaluation reports it.

### 4.4 How filled data are used

* **Training:** a 64-step window is used only if it contains no `outage` step. Spatially and
  temporally filled steps are allowed (they are the "interpolate for now" part).
  Measured: 452,861 of 630,657 training windows have no missing value at all before filling; after
  filling, only windows touching outages drop out (estimate: about 590,000 windows remain).
* **Evaluation reference:** a year enters the reference only if fewer than 5% of its steps are
  `outage`. That keeps 2017, 2019, 2020, 2021, 2022 (training reference) and 2023, 2024 (held out),
  and leaves out 2018.
* **Volume statistics** (annual, monthly and mean rain) are computed over non-outage steps and scaled
  to a full year, so the 1–4% outage placeholders do not make the reference look drier than it was.
  Other statistics treat placeholders as dry; a sensitivity check (section 8.6) repeats the main
  scores on the cleanest years only (2021, 2022, 2024).

### 4.5 Output files

Same format as the Astlingen splits, so the training code reads them unchanged:
`data/singapore/splits/{train,val,test}_years_labelled.csv` with the columns
`datetime, date, avg_rainfall, rainfall_intensity, hmm_state, day_of_year, annual_interval_idx`, plus a
`fill_flag` column, an `audit.json` (station list, distances, flag shares per year, duplicates removed)
and a `MANIFEST.json` with SHA-256 hashes of the raw files and outputs, as for Astlingen.

---

## 5. Parameters re-measured by rule

Every rule uses the Singapore training years (2017–2022) only.

| Quantity | Rule | Astlingen (must reproduce) | Singapore |
| :--- | :--- | :---: | :---: |
| Resolution *q* | smallest non-zero value of the target | 0.0025 mm | 0.05 mm |
| Wet threshold τ | the smallest value that makes up at least 1% of real wet steps; generated values below τ are set to zero | **0.005 mm** (0.0025 is only 0.09% of wet steps; 0.005 is 2.6%) | 0.05 mm (30.6% of wet steps) |
| asinh scale *s* | median of real wet steps (≥ τ) | **0.035 mm** | 0.15 mm |
| Standardisation | mean and standard deviation after asinh, training years | as in v14 | re-fitted |
| Seasonal phases | the existing procedure (`fit_seasonal_labels.py`): 4-state model on the 14-day smoothed average rain for each 5-minute slot of the year, daily pooling, minimum phase length = the median phase length | 4 phases, 21 changes a year | re-fitted (expected to follow the monsoons) |
| Transition matrix, calendar profile | same script | as in v14 | re-fitted |
| Storm-depth threshold (Tier 6 S1) | the depth reached by the same share of storms as 10 mm at Astlingen (the top 4.1%) | **10 mm** | about 33 mm |
| Heavy-rain forecast task (TSTR) | the 6-hour total exceeded as often as 2 mm is at Astlingen | **2 mm** | computed at build time |
| Pass bands for every check | 95% range of the check when a 10-year sample and a reference-sized sample, both resampled from the real reference years, are compared (the method used for Tier 6) | existing bands kept for Astlingen tallies; resampled bands also computed | computed at build time |
| Training length | the same number of optimiser steps as v14: 500 epochs × 410 batches = 205,000 | 500 epochs | about 700 epochs (computed from the window count) |

**Fixed, not re-measured:** block length 64 steps, delay and embedding 8, network and checkpoint
(`ImagenFew_64.ckpt`), batch size 2048, learning rate, weight decay, dropout 0.10, EMA, the loss, the
sampler, 4 seasonal phases, generation in calendar mode without bridge blocks, 10 generated years,
plan seed 42, training seed 0.

Expected model-space range: with *s* = 0.15 the largest Singapore value maps to asinh 4.7; Astlingen's
maps to 5.8. The median wet step maps to 0.88 at both sites (by construction). So the transform
puts both sites on a similar scale, which is what we want.

---

## 6. Which recipes to transfer

* **v14 recipe: always.** It is the deliverable.
* **v17 recipe (trained continuation): only if v17 passes its written success rule at Astlingen.**
  If it does, run it at Singapore with the same settings (80 minutes of known context).
* **Optional:** a v10-style run (standard scaling instead of asinh) to test whether "asinh helps"
  holds at Singapore. One more training run; decide in section 10.

The Singapore runs are named `sg_v14` (and `sg_v17`, `sg_v10`), trained into
`logs/ImagenFew/Rainfall_Regime/sg_v14/` etc., and generated as
`results/generated_data/singapore/rainfall_synthetic_10y_sg_v14_nobridge_cal.csv`.

---

## 7. The code to write

### 7.1 A site description file

`configs/sites/astlingen.yaml` and `configs/sites/singapore.yaml`, each holding everything that is
site-specific: data paths, the target definition, time zone, years per split, steps per year, the
leap-day rule, and the re-measured parameters (τ, *s*, storm-depth threshold, forecast threshold,
pass bands). Rule-derived values are *written by the build script*, not typed by hand, and stored with
the rule's name and inputs so they can be re-derived.

`scripts/site.py`: loads a site file into one `Site` object, and holds the rule functions
(`resolution`, `wet_threshold`, `asinh_scale`, `storm_depth_threshold`, `forecast_threshold`,
`resampled_bands`), each a small pure function with a unit test.

### 7.2 New scripts

| Script | Does |
| :--- | :--- |
| `scripts/singapore/build_dataset.py` | Sections 4.1–4.5: raw files → the cluster series with flags, splits, audit, manifest. One year at a time; saves per year so a stopped run resumes |
| `scripts/site_rules.py --site X` | Computes every rule-based value and writes it into the site file |
| `scripts/window_horizon_pilot.py --site X` | The "perfect window generator": joins real 64-step windows drawn independently (matching season) into 10 years and scores them. Shows how much a perfect 5h20 model can capture. The 2026-10-10 Astlingen version was a scratch script; this makes it a real script and reproduces its Astlingen numbers first |
| `scripts/check_site_invariants.py` | Runs every invariant in section 7.4 and stops on the first failure |
| `regime_training/config_sg_v14.yaml` | v14's config with only the data paths, asinh scale and epoch count changed |
| `scripts/submit_sg_v14.sh` | Cluster job: train, then generate 10 years in calendar mode without bridges |

### 7.3 Changes to existing scripts (smallest possible)

Each gets a `--site` argument (default `astlingen`, so every existing command behaves exactly as
today), and reads from the `Site` object what it now hard-codes:

| Script | What it hard-codes today |
| :--- | :--- |
| `scripts/eval_suite.py`, `scripts/gate_a_scorecard.py` | wet threshold 0.005, 105,120 steps per year, the real data file, training years 2000–2007 and held-out 2008–2009, "8 years" in the annual-maximum fits, storm gap, the 10 mm depth, the 2 mm forecast task, the pass bands. Singapore gets its own registry and its own cache (`results/evaluation/singapore/`) |
| `scripts/fit_seasonal_labels.py` | file paths and the year ranges of the splits |
| `regime_training/generate_hmm_v1.py`, `generate_context.py` | the 0.005 cut-off applied to generated values (must become τ = 0.05 for Singapore) and the output folder |
| `scripts/baselines/run_copula_arima_occ.py`, `run_bartlett_lewis.py` | the training file and the wet threshold |
| `scripts/kostra_compare.py`, `scripts/gate_b_sewer.py`, `scripts/metric_validity.py` | Astlingen only: refuse to run for another site |

`train_regime.py` and `regime_dataset.py` need no change: they already take the data files and the
asinh scale from the config. The window filter for outage steps is applied when the split files are
written (outage steps are written with a flag, and `RegimeDataset` gets one optional argument to skip
windows containing them).

### 7.4 Invariants checked by code

| # | Invariant | How it is checked |
| :---: | :--- | :--- |
| 1 | Every rule reproduces Astlingen's existing value | `site_rules.py --site astlingen` must return τ = 0.005, *s* = 0.035, 10 mm, 2 mm |
| 2 | Astlingen results are unchanged by the refactor | `eval_suite.py --site astlingen --refresh` gives a cache identical to the current one (every number compared) |
| 3 | No Singapore value is fitted on 2023–2024 | every rule function receives only the training-year array, and asserts the years it was given |
| 4 | The target uses the same four gauges at every step | the build asserts the station set per step |
| 5 | Local time, 365-day years, index 0 = 00:00 on 1 January | the daily-cycle peak of the built series lies between 14:00 and 16:00; every year has 105,120 steps |
| 6 | Values are non-negative and on the 0.05 mm grid | asserted after building |
| 7 | Every filled value is flagged; shares are reported | audit file; the evaluation prints the shares |
| 8 | Training windows contain no outage step | asserted when the dataset is built |
| 9 | The Singapore model differs from v14 only in data-derived fields | the two configs are compared key by key; only a short allowed list may differ |
| 10 | The same evaluation code scores both sites | the site object is the only input that differs |

---

## 8. How we evaluate

### 8.1 The reference and the noise floor

* **Reference:** the real Singapore training years kept by section 4.4 (2017, 2019–2022).
* **Held-out real years (2023–2024), scored as if they were a generator.** This calibrates every check:
  at Astlingen, real held-out years passed only 10 of 18, which showed the bands were stricter than real
  year-to-year variation.
* **Noise floor:** each check's 95% range when real years are resampled (section 5). Nothing is called
  a difference unless it is outside this range. Note: 5 reference years give wider ranges than
  Astlingen's 8.

### 8.2 Metric families

The 18 checks and the other metrics are grouped into families. Some are comparable across sites; some
mainly describe the gauges and must only be read within a site.

| Family | Metrics | Compare across sites? |
| :--- | :--- | :--- |
| A. Amount and occurrence | volume, wet fraction, dry spells | volume yes (as a ratio); wet fraction within site only |
| B. Intensity and extremes | P99, P99.9, largest value, wettest day, design-rain (IDF) cells for 15 min–24 h at 2, 5, 10 years | yes, as ratios to real |
| C. Short-scale timing | wet-spell length, single-step showers, storm duration and count, lag-1 and hourly autocorrelation | within site only: strongly shaped by gauge resolution |
| D. Long-scale structure | variance across time scales (5 min–1 week), daily totals (Wasserstein distance), unbroken runs > 5h20, storm-scale checks S1–S3 | yes, as ratios to real |
| E. Cycles | monthly correlation, daily-cycle correlation and peak/trough ratio | yes |
| F. Machine-learning checks | classifier tests on 5-hour and 1-day windows, coverage and recall, memorisation, train-on-synthetic forecasting | yes (scale-free) |

At Astlingen, the metrics that best predicted sewer overflow were variance across scales and daily
totals (family D). Singapore has no sewer model, so these two stand in as the "downstream" metrics,
with the explicit caveat that their link to the sewer was shown at Astlingen only.

**Note on design rain at Singapore:** 10-year return levels from 5 years of annual maxima are an
extrapolation. They are still computed, but their resampled bands will be wide, and that is reported.

### 8.3 Comparators

Each is built with the same rules, on the same Singapore training years:

1. **Real held-out years** (2023–2024).
2. **Perfect window generator** (the pilot): real 64-step windows joined independently. It answers
   "how much could any 5h20 model capture here?"
3. **Occurrence-matched latent Gaussian AR**: the main classical baseline (fits in seconds).
4. **Storm-and-cell model** (Bartlett–Lewis, wet-fraction objective): the standard hydrology model
   (about 20 minutes to fit; 10 seeds).

### 8.4 Decision rules for "does the recipe transfer?" (fixed before any result)

For each family A–E, a family **passes** if at least half its checks lie inside the Singapore bands.
The checks belong to families as follows, with v14's Astlingen result (calendar, no bridges; 14/18 and
Tier 6 0/3, from the evaluation cache):

| Family | Checks | v14 at Astlingen |
| :--- | :--- | :--- |
| A | volume, zero fraction, dry spell | 3 of 3: **pass** |
| B | P99 wet, P99.9 wet, largest value, hourly intensity distribution (JSD, KS), wettest day, all IDF cells | 5 of 7 (fails wettest day and IDF): **pass** |
| C | wet spell, lag-1 autocorrelation, hourly autocorrelation, storm duration, storm count, storm volume | 5 of 6 (fails hourly autocorrelation): **pass** |
| D | Tier 6 S1, S2, S3; unbroken runs > 5h20 within the noise floor | 0 of 4: **fail** |
| E | monthly cycle; daily cycle (scored separately) | monthly **pass**, daily **fail** |

* **TRANSFERS** if `sg_v14` shows the same pattern: A, B, C pass, and the monthly cycle passes.
* **PARTLY** if one of A, B, C fails.
* **DOES NOT TRANSFER** if two or more of A, B, C fail, or training does not converge (best loss not
  improving over the last 100 epochs and generated volume off by more than 20%).
* Family D and the daily cycle are *predicted*, not required (section 9): D by the pilot, the daily
  cycle by the missing time-of-day input.
* **Relative standing:** diffusion should still beat both classical baselines where it beat them at
  Astlingen: extremes (family B) and 5-minute texture (family C). Reported as "keeps / loses its
  advantage".

### 8.5 Decision rules for "do the findings transfer?"

| Finding at Astlingen | Test at Singapore | Holds if |
| :--- | :--- | :--- |
| A fixed-window generator loses structure longer than its window | compare `sg_v14` with the perfect window generator on family D | `sg_v14` is inside or close to the pilot's range (at Astlingen: v14 1.9 vs pilot 2.0–3.0 long runs a year) |
| The window, not the network, sets the long-scale numbers | the pilot at 64 steps versus 1 day | the 1-day pilot recovers family D, the 64-step pilot does not (only if Singapore's 64-step pilot shows a real loss) |
| Diffusion beats the latent Gaussian AR on extremes; the latent Gaussian AR wins long-scale timing | family B and family D, both models | same direction, outside the noise floor |
| Short-window metrics miss long-scale failures | classifier test on 5-hour windows versus family D | the 5-hour classifier rates `sg_v14` close to real while family D fails |
| asinh helps (optional run) | `sg_v10` versus `sg_v14` | `sg_v14` better on families A and B |

### 8.6 Sensitivity checks

* The main scores repeated on the cleanest years only (2021, 2022, 2024).
* The reference with 2018 included (outages as placeholders) versus without.
* A single gauge (S78) instead of the cluster, for the v14 recipe and the latent Gaussian AR only.

---

## 9. What we expect (to be finalised as written predictions before training)

The diary practice applies: predictions go into the research diary with numbers, **after** the pilot
(step 5) and **before** the Singapore model is trained.

| | Expectation | Reason |
| :--- | :--- | :--- |
| Volume, wet fraction | pass | they passed at Astlingen and the threshold rule is consistent |
| Extremes (P99.9, IDF) | pass at 15 min–6 h; 24 h short | same pattern as Astlingen; 24-hour totals depend on storms longer than a block |
| Short-scale timing | pass | v14's strongest family at Astlingen |
| Unbroken runs > 5h20 | close to real (real is only ~2.5 a year) | Singapore's unbroken rain rarely outlasts a block |
| Storm-scale structure (2-hour gap) | short of real, by about what the pilot shows | storms with breaks still outlast a block (32 a year) |
| Monthly cycle | pass in calendar mode | the seasonal phases follow the monsoons |
| **Daily cycle** | **fail: nearly flat** (real peak/trough ratio 8) | the model is never told the time of day; blocks are placed on the timeline without it |
| Classifier on 1-day windows | easier to tell apart than at Astlingen | a flat daily cycle is visible in whole days |

If the daily cycle fails as expected, that is a finding about the method, not about Singapore: the
recipe has no time-of-day input, and Astlingen's weak daily cycle hid that. The natural fix (adding
time of day to the conditioning) is a change to the method and is not part of this transfer test.

---

## 10. Decisions for you

| # | Decision | Options | Recommendation |
| :---: | :--- | :--- | :--- |
| 1 | Target series | four-gauge cluster / one gauge / island average | **Four-gauge cluster**; one gauge as a sensitivity check |
| 2 | Missing data | the three rules of section 4.3 / interpolate everything in time | **Section 4.3**: interpolating month-long outages would invent rain |
| 3 | 2018 | keep its observed windows for training, leave it out of the reference / drop the whole year | **Keep for training, leave out of the reference** |
| 4 | Training length | same number of optimiser steps as v14 / same number of epochs | **Same steps** (about 700 epochs); same epochs would train Singapore 25% less |
| 5 | Recipes | v14 only / v14 + v17 if it passes / also v10 | **v14, plus v17 only if it passes at Astlingen**; v10 optional |
| 6 | Splits | train 2017–2022, val 2023, test 2024 / other | **As proposed**, mirroring Astlingen's 8 + 1 + 1 at a smaller scale |

---

## 11. Order of work

| Step | What | Output | Effort |
| :---: | :--- | :--- | :--- |
| 1 | `site.py`, site files, rule functions with unit tests; Invariant 1 on Astlingen | rules reproduce 0.005 / 0.035 / 10 mm / 2 mm | 1 day |
| 2 | `build_dataset.py`; audit | Singapore splits, `audit.json`, Invariants 3–8 | 1–1.5 days |
| 3 | Seasonal phases and transition matrix for Singapore | label files; a figure of the phases against the monsoon calendar | 0.5 day |
| 4 | `--site` in the evaluation; Invariant 2 (Astlingen cache identical) | Singapore real reference, bands, held-out calibration | 1.5–2 days |
| 5 | Window pilot at both sites; **write the predictions into the diary** | pilot numbers; dated predictions | 0.5 day |
| 6 | Baselines at Singapore | latent Gaussian AR; storm-and-cell (10 seeds) | 0.5–1 day |
| 7 | `sg_v14` on the cluster | trained model; 10 generated years | ~9 h of cluster time |
| 8 | Score, apply sections 8.4–8.6, write the diary entry | verdicts | 1 day |

Total: about 7–9 working days plus one training run. **Stop rule:** if step 2 shows the cluster
series cannot pass its own checks (for example the daily peak is not in the afternoon, or more than
10% of training windows are lost), stop and revisit decisions 1–3 before writing any model code.

---

## 12. What this test cannot show

* **Usefulness for drainage control.** There is no Singapore drainage model here; the sewer check and
  the agent test stay Astlingen-only.
* **Multi-site or spatial realism.** We generate one averaged series, as at Astlingen.
* **Climate change or years beyond the record.** Both sites are modelled on their own records.
* **Training-run variation at Singapore.** One training run; differences between Singapore and
  Astlingen smaller than the Astlingen seed spread (strengthening run 2) are not interpreted.

---

## Glossary

| Term | Meaning |
| :--- | :--- |
| Tipping-bucket gauge | A rain gauge that records rain in fixed amounts (here 0.2 mm per tip) |
| Resolution *q* | The smallest step the series can take |
| Wet threshold τ | Values below it count as dry, in both real and generated rain |
| Inverse-distance weighting | Filling a missing gauge with a weighted average of nearby gauges, closer gauges weighted more (1/distance²) |
| Outage | A period when no station in the network reports |
| Noise floor / pass band | How much a statistic varies between real years; a model is only called wrong outside this range |
| Perfect window generator | Real 5h20 windows stitched together independently: the best any 5h20 model could do |
| IDF / design rain | Rain totals over 15 min to 24 h expected once every 2, 5 or 10 years |
| Classifier test | A classifier tries to tell real windows from generated ones; 0.5 means it cannot |
| Train-on-synthetic forecasting (TSTR) | Train a forecaster on generated rain, test it on real rain |

---

## 13. Changes made while implementing (2026-10-12)

* **Code layout.** Instead of adding `--site` to the existing scripts (section 7.3), everything new is
  in one package, `sitetransfer/`, which imports the existing code and reproduces it exactly on
  Astlingen (regression tests). The only edit to existing code is the opt-in `mask_col` in
  `regime_training/regime_dataset.py` (and its pass-through in `train_regime.py`).
* **Bands at both sites.** The verdict (section 8.4) uses *resampled* bands at Astlingen as well as at
  Singapore, so the two sites are judged the same way. The fixed Gate A bands are still reported.
  Under resampled bands v14 at Astlingen passes families A, B, C and the monthly cycle, and fails D,
  the same pattern as with the fixed bands.
* **Resampling is two-sample.** A first version compared resampled 10-year sets with the fixed
  reference; that ignored the reference's own sampling noise and made real held-out years fail almost
  every check. Each draw now resamples both sides, as the Tier 6 bands did.
* **The verdict is relative.** "Transfers" means every core family that passes at Astlingen also
  passes at Singapore, rather than a fixed list.
* **Storm-and-cell baseline deferred.** Its cut-off and parameter search box are inside
  `run_bartlett_lewis.py`; adapting it means editing that script. The latent Gaussian AR is the
  classical comparator.
* **Family F (classifier, coverage, forecasting) not computed** in this first version.
* **Measured values** (build and derive, 2026-10-12): 20 usable stations; cluster S113, S119, S43, S78
  (4.91 km); wet threshold 0.05 mm; asinh scale 0.15 mm; deep-storm threshold 36.4 mm; reference years
  2017, 2019–2022 (2018: 12.7% outage); 603,641 usable training windows, so 698 epochs give 205,212
  optimiser steps (v14: 205,000); seasonal phases change 31 times a year (Astlingen 21).
