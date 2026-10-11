# Strengthening Runs

**Written:** 2026-10-11, before any of these runs started (the decision rules below are
pre-registered). **Send:** `scripts/submit_strengthening.sh` (`--dry-run` first).
**Verdicts:** `python scripts/summarise_strengthening.py` applies the rules below to whatever
has finished. **Code checks:** `scripts/check_strengthening_code.py` (seven checks; all passed
on 2026-10-12; check 7, the loss fix, added that day).

## Why these runs

The window-horizon pilot (2026-10-10; a scratch script, not yet in the repo) built a *perfect* 64-step generator:
real 2000–2007 windows drawn independently and joined end to end. It reproduces v14's
long-scale failures almost exactly:

| | v14 | perfect 64-step generator (3 seeds) |
| :--- | :---: | :---: |
| Unbroken wet runs > 5h20 /yr (real 14.25) | 1.9 | 2.0–3.0 |
| Variance-scaling RMSE | 0.171 | 0.15–0.21 |
| Sewer overflow (BC), share of real | 0.62–0.66 | 0.57–0.71 |

The perfect 24-step generator gives 0.50 of real overflow (v8: 0.48). Lengthening the window
raises it: 12 h 0.81, 1 day 0.94. A continuation version (each window matched to the last
80–160 min generated) brings long storms back (16.5–21 /yr). Two consequences frame the runs:

1. **Changes inside the 64-step window cannot move the long-scale or sewer numbers.** Runs 2
   and 3 change only how a window is learned, so they test this prediction.
2. **A model trained to continue from 80 min of context can, in principle, reach the real
   storm structure.** Run 5 (E3) tests whether the diffusion model learns to.

Paper goal: *the window is the horizon*. A fixed-window generator cannot carry dependence beyond
its window; per-window metrics cannot see this; the horizon the downstream user needs (about
one day for the sewer) can be measured before training; and trained continuation is the way to
reach it.

## The runs

| Run | What | Jobs | Single change vs v14 |
| :--- | :--- | :--- | :--- |
| 1 | Agent test rerun (flood-control) | sent from flood-control | only if run 5 succeeds (D24: rerun when the generator gains large storms) |
| 2 | Training seeds | `v14_s1`, `v14_s2`, `v10_s1` | training seed |
| 3 | No pretraining | `v14_scratch` | random initial weights (`--model_ckpt none`) |
| 4 | Gate B KPIs | `submit_gate_b_kpis.sh` (CPU) | none: sewer KPIs for real, v14 and the baselines |
| 5 | E3, trained continuation | `v17`, then chained generation K = 2 and K = 0 | 75% of windows get a clean known prefix of 8/16/24/32 steps |
| 6 | ~~sigma_data~~ **dropped 2026-10-12, before running** | ~~`v18`~~ | data × 0.5, so its std matches EDM's sigma_data = 0.5 |
| 7 | Denoising check | `submit_denoise_diagnostic.sh train` | none: denoising error on the training years |

Every retrain also generates the two standard 10-year series with plan seed 42 (Markov with
bridges, and calendar without), so only the trained weights differ from v14's rows.

## Decision rules (as coded in `summarise_strengthening.py`)

* **Run 2.** "asinh beats standard scaling" HOLDS if every v14 seed passes more Gate A checks
  than every v10 seed *and* every v14 seed is closer to the real volume; PARTLY if one
  criterion separates; otherwise DOES NOT HOLD. The v14 seed range is the yardstick for every
  other run: nothing is called better or worse than v14 inside it.
* **Run 3.** Pretraining HELPS / HURTS if `v14_scratch` is below / above every pretrained seed
  on Gate A; otherwise NO MEASURABLE EFFECT (claim the checkpoint as an initialisation only).
* **Run 5.** SUCCESS if the mean of the four chained members (`v17_ctx_cal`, `_m1`–`_m3`)
  meets all four targets: ≥ 10 unbroken wet runs > 5h20 /yr, ≥ 15% of rain in them, volume
  0.95–1.05 of real, wet spell 0.90–1.10. The gain counts as coming from the context only if
  the K = 0 control (`v17_ctx0_cal`) has under half the chained long-run rate. Side check:
  `v17_nobridge_cal` against the run-2 seed spread (did mask training harm ordinary blocks?).
* **Run 6 (dropped 2026-10-12, before running; research diary "v18 Dropped").** Written rule, kept
  for the record: v18 is BETTER / WORSE only outside the v14 seed range on Gate A; otherwise the
  mismatch does not matter at this budget. Dropped because by point 1 it cannot move the
  long-scale or sewer numbers, its likely effect is small enough to need several seeds of each
  model, and its result would be confounded: the ImagenFew pretraining data were also
  standardised to spread 1, so v18 matches EDM but departs from the pretrained weights.

## Pre-registered sewer predictions (from the pilot)

Run the sewer check (`scripts/gate_b_sewer.py`, BC) on each new calendar series once it exists:

* **Runs 2, 3** (`v14_s1/_s2`, `v14_scratch`, calendar no-bridge series): overflow
  **0.55–0.70** of real, inside the perfect 64-step generator's range. A run outside it
  would falsify "the window sets the sewer number".
* **Run 5 chained** (`v17_ctx_cal` and members): if the model learns to continue, overflow
  **≥ 0.85** of real and 14–20 long runs /yr. Falling short while the continuation pilot shows
  the information is there would be a learning failure on zero-inflated data, reported as such.
* **Run 5 control** (`v17_ctx0_cal`): **0.55–0.70**, like v14.

Compare against the 2000–2007 real mean (277k m³/yr), the pool the pilot drew from; the
usual 2000–2008 reference (292k) includes 2008, a 414k year.
