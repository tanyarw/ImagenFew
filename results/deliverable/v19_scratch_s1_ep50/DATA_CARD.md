# Synthetic rain for Astlingen: v19 scratch, training seed 1, epoch 50

A second set of four members from ImagenFew's chosen recipe, trained again from a different random start. Made on
2026-10-23 for the equal-sampling comparison (diary 2026-10-23, "Equal Sampling: Result"). **It was not tested on
2009:** that test was spent on training seed 0 (`../v19_scratch_ep50/`). This folder is rebuilt by
`python scripts/export_deliverable.py --run s1`. `manifest.json` holds the hashes of every file.

## What it is

The same format as `../v19_scratch_ep50/` (see its DATA_CARD for the details):

- **Four members × 10 years** of 5-minute rain: the mean over the catchment's four gauges, in mm per 5 min.
- **One `.npz` per member,** with `rain`, `phase`, `start`, source hashes, `member`, `generation_seed` (42–45),
  `steps_per_year` and `calendar`.
- **Calendar:** 365-day years, with no 29 February.
- **Gauges:** split the mean with flood-control's `src/rain/split.py`, as for seed 0.

## How it was made

| | |
| :--- | :--- |
| **Recipe** | Identical to seed 0: one-day windows, asinh transform, 4 season phases, random initial weights, real 2000–2007 only |
| **Difference** | Training seed 1 instead of 0 (a different random start and a different order of training batches) |
| **Weights** | Epoch 50 (training stopped there): `logs/ImagenFew/Rainfall_Regime/v19_scratch_s1/final_regime_model.pt` |
| **Generation** | Calendar assembly, no bridge blocks; one generation seed per member (42–45); cluster job 995711 |

**Epoch 50 checked again:** at epoch 50 the loss on unseen 2008 (0.098) is no worse than on the training years
(0.101), as for seed 0. Between epochs 25 and 50 it rose by about 1%, so memorising starts after epoch 25 but is
negligible at 50.

## How it compares

Real is 2000–2007, except storms ≥ 10 mm and overflow, which use flood-control's sewer check over 2000–2008.

| Member mean | Seed 1 (this folder) | Seed 0 | Real |
| :--- | :---: | :---: | :---: |
| Rain per year | 700 mm (0.99 of real) | 630 mm (0.89) | 710 mm |
| Storms ≥ 10 mm per year | 15.0 | 11.8 | 15.7 |
| Unbroken rain runs longer than 5 h 20 per year | 12.65 | 9.75 | 14.25 |
| Overflow under BC, as a share of real | **0.95** (members 0.99, 0.96, 1.12, 0.71) | 0.76 | 1 |
| Overflow under EFD, as a share of real | 0.95 | 0.77 | 1 |
| Too-close one-day windows (2008 threshold) | 9.3% | 9.7% | 5% |

Seed 1 is closer to real than seed 0 on every quantity above. The two seeds differ more than four members of one
seed do, which is why both are shipped.

## Known limits (please read before using)

1. **Not tested on 2009.** The 2009 PASS belongs to seed 0's weights. This set has passed only the checks above, all
   on training years and 2008.
2. **It does not reach the record's extremes either.** The largest generated storm in these 40 years is 48 mm.
   2008's largest is 102 mm.
3. **Members differ:** overflow ranges from 0.71 to 1.12 of real. Use all four together.
4. **Copying:** 9.3% of generated days are close to a training day, under the 10% limit but above the 5% an unseen
   real year scores. Seed 0 scores the same (9.7%) on this yardstick.
5. The year-to-year range of rain quantities was not measured for this seed (seed 0: too narrow).

## Use

- **On its own or pooled with seed 0** (80 years from two sets of weights). Keep the seed and member labels so
  results can be checked for dependence on either.
- Anything reported on this set must say it was not tested on 2009.
