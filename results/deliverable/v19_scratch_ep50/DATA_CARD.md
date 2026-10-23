# Synthetic rain for Astlingen: v19 scratch, epoch 50

ImagenFew's chosen generator, handed to flood-control on 2026-10-22. This folder is rebuilt by
`python scripts/export_deliverable.py`. `manifest.json` holds the hashes of every file and of the results quoted
here.

**Update 2026-10-23:** a second set of four members from the same recipe (training seed 1) is in
`../v19_scratch_s1_ep50/`. It is closer to real (BC overflow 0.95 of real) but was not tested on 2009. The files in
this folder are unchanged.

## What it is

- **Four members × 10 years** of 5-minute rain: the mean over the catchment's four gauges, in mm per 5 min.
- **Format:** one `.npz` per member, the same as flood-control's `data/synthetic/v14_nobridge_cal.npz`.
  - `rain` (float32), `phase` (season phase 0–3 for each step), `start`
  - source hashes, plus `member`, `generation_seed` (42–45), `steps_per_year` and `calendar`
- **Gauges:** split the mean into four gauges with flood-control's `src/rain/split.py` (seed 0), as for v14. The
  split keeps the mean exactly (checked on member 0).
- **Calendar:** **365-day years, with no 29 February.** Each year is exactly 105,120 steps, so year *k* is
  `rain[k*105120:(k+1)*105120]`. If dates are rebuilt as a continuous 5-minute range from `start`, they run one day
  ahead after 28 Feb 2028 and two after 28 Feb 2032. The sewer model only uses the hour of the day, which is
  unaffected. The dates 2026–2035 are labels, not a forecast.

## How it was made

| | |
| :--- | :--- |
| **Model** | ImagenFew diffusion, one-day windows (288 steps), asinh transform, conditioned on 4 season phases |
| **Training** | Random initial weights (no pretraining), real 2000–2007 only |
| **Weights** | Epoch 50 of 325: `logs/ImagenFew/Rainfall_Regime/v19_scratch/epoch_0050_regime_model.pt` |
| **Generation** | Calendar assembly, no bridge blocks; one generation seed per member (42–45) |

**Why epoch 50:** loss on unseen real years is lowest at epoch 50. After that the model fits its training days
better and unseen days worse: the gap reaches +73% by epoch 325. The choice was made on 2008, before 2009 was
opened. Epoch 50 keeps the final weights' sewer level (0.76 against 0.75), so the sewer realism was not a product
of overfitting.

## How it compares with real rain

Real is 2000–2007, except storms ≥ 10 mm and overflow, which use flood-control's sewer check over 2000–2008. The
sewer runs each year from empty tanks, as in `bc_efd_check`.

| Member mean | Generated | Real |
| :--- | :---: | :---: |
| Rain per year | 630 mm (0.89 of real) | 710 mm |
| Storms ≥ 10 mm per year | 11.8 | 15.7 |
| Unbroken rain runs longer than 5 h 20 per year | 9.75 | 14.25 |
| Overflow under BC, as a share of real | **0.76** (members 0.79, 0.74, 0.96, 0.57) | 1 |
| Overflow under EFD, as a share of real | 0.77 | 1 |

### Test on 2009, a year the generator never saw

The test was registered before it ran, and run once (`results/testyear/test_2009.json`). Result: **PASS.**

- **Sewer:** 2009's BC overflow (228,143 m³) sits at the **65th percentile** of the 40 generated years. All four
  sewer quantities are inside the generated range.
- **Rain:** 2 of 10 quantities are outside, the most the rule allows. 2009 was wet more often and had softer
  5-minute peaks than any generated year.
- **Copying:** 3.8% of generated days are suspiciously close to a training day, below the 5% a real unseen year
  scores.

## Known limits (please read before using)

1. **It does not reach the record's extremes.** 2008's largest storm (102 mm) is far beyond the generated 95th
   percentile (43 mm). 2008's overflow (413,768 m³) falls above 97% of generated years. An agent trained only on
   this rain will rarely meet a storm of that size.
2. **It is about 11% too dry,** with too few storms of 10 mm or more. Overflow is about a quarter below real on
   average.
3. **The year-to-year range in rain is too narrow.** Only 65% of the training years' rain quantities fall inside it
   (90% would be expected). Gentle, drizzly years like 2009 are rare.
4. **Members differ:** overflow ranges from 0.57 to 0.96 of real. Use all four together, not one.
5. **Single catchment:** fitted to one 10-year record at one site.

## Use

- **Pool all four members** (40 years). Keep the member label so results can be checked for dependence on a member.
- **2009 is spent for generator choices.** No further generator should be chosen on it. Controllers can still be
  evaluated on real 2008 and 2009 under flood-control's own rules (D13).
- **Report the limits above** next to any result trained or tested on this rain. On the largest real storms, check
  any controller trained on it against the fixed controller (BC).
