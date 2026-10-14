# Guided Continuation of v14

**Written:** 2026-10-14, before any code or run. **Status:** proposed.
**Question:** can v14, unchanged, carry storms across its 5h20 pieces if each new piece is *steered* to agree
with the end of the previous one while it is being generated?

---

## 1. Summary

* **No new model and no training.** v14 makes every guess; a few lines of arithmetic, using gradients through
  v14's own network, steer those guesses. v14's weights never change, so its calibration (amounts, extremes,
  14/18 checks) is kept. This is what v17 could not do: retraining for continuation cost it its calibration.
* **The method is established** (reconstruction guidance, Ho et al. 2022, "Video Diffusion Models"; the same idea
  as Diffusion Posterior Sampling, Chung et al. 2023; for time series, TSDiff, Kollovieh et al. 2023). The new
  part is its use on zero-heavy rainfall and its test against the window-horizon diagnostic and the sewer.
* **Two cluster jobs, each gated.** A pilot (13 settings, one short GPU array job) picks a setting by rules
  written here; only if one passes, a full 10-year run (a second array job), then the usual evaluation.
* **What success buys the paper:** calibrated synthetic rain with long storms, no retraining, nothing copied from
  the record: the fix that v16 attempted by hand and v17 attempted by retraining.

---

## 2. How it works

A piece is generated in 35 steps (EDM sampler, noise level σ from 80 down to 0.002). At each step v14 looks at
the noisy piece and returns its **guess** of the clean piece. The first 80 minutes of a new piece should repeat
the last 80 minutes the previous piece generated (the **context**; the model's own output, not real data).

At every step:

1. **Guess.** `x̂ = D(x, σ)`: v14's guess of the clean piece from the noisy piece `x`.
2. **Check.** `E = Σ m · v · (x̂ − c)²` over the first 80 minutes (`m` marks them, `c` is the context, `v` are
   per-step weights, section 3).
3. **Direction.** `g = ∂E/∂x`: one backward pass through v14. Because v14 looks at the whole piece, this says how
   the *unknown* part must change for v14's guess of the start to agree with the context.
4. **Correct.** `x̂_rest ← x̂_rest − w · σ² · g_rest`, with strength `w` (section 3).
5. **Step.** The usual Heun step from the corrected guess; the known 80 minutes are re-noised to the current σ,
   as in E2 (v14 was trained on fully noised pieces only, so it must not be shown clean known values).

Steps 1–4 run at both network calls of each Heun step. Cost: one backward pass per call, 3.4 times plain
chaining (timed on the laptop, 2026-10-14).

**Why this can work where pasting (E2) did not.** Pasting forced the known values in, but nothing asked v14's
guess of the *rest* to agree with them; it drew its own piece next to them (rain continued across 58–61% of
seams, long storms 2.3–2.9 a year). Guidance asks what the rest must look like for v14 itself to reproduce the
start, so a storm in the context pulls the new piece towards a storm v14 finds plausible.

---

## 3. Settings

**Fixed** (as v14_nobridge_cal, E2 and v17, so results are comparable): v14 checkpoint and EMA weights; 35-step
Heun sampler; context 80 minutes (2 image columns, 48 new steps per piece); calendar timeline, no bridges;
plan seed 42; known values re-noised, no RePaint resampling (E2 showed resampling drives the series dry).

**Swept in the pilot** (3 versions × 4 strengths, plus strength 0):

| Version | Weights `v` in the check | Guidance applied | Why |
| :--- | :--- | :--- | :--- |
| plain | all context steps 1 | every seam | the textbook method |
| wet-weighted | wet context steps ~10, dry 1 (10 ≈ dry : wet share) | every seam | stop the 91% dry steps outvoting a storm in progress |
| rainy seams | as wet-weighted | only when the context holds rain | dry seams are v14's default anyway, so nothing there can push towards dry |

**Strength `w`.** `σ² · g` can be far larger or smaller than the guess itself at high noise, so the grid is set
in the smoke test (section 6, step 2): measure the size of `σ² · g` relative to `x̂` across σ, then choose four
values of `w` spaced by factors of 4 so that the correction runs from about 1% to about 100% of the guess. The
grid is written into the diary before the pilot is sent. `w = 0` is pasting only (E2 without resampling, here
on the calendar timeline): the built-in comparison.

---

## 4. Pilot (cluster, one GPU array job)

* **Size.** 32 chains × 20 days per setting: about 1.75 chain-years, enough seams with rain to measure
  continuation. On a GPU a piece takes about the same time for 8 or 32 chains (the network is tiny), so the
  extra chains cost almost nothing.
* **Days.** The same 20 days and labels for every setting; chains start on 8 dates spread across the year
  (4 chains each), so every season is represented.
* **Job.** 13 array tasks, one per setting, about 10–20 minutes each on an A100 (estimate: 121 pieces at
  ~5–10 s per guided piece); each task saves its series and scores itself. A final step collects the 13 results
  into `results/guidance/pilot.json` and applies the rules below.
* **Also generated, once:** the same days with nothing known (independent pieces), the reference for amounts.

**Pass rules** (all four, pooled over the 32 chains):

| Measure | Pass if | For reference |
| :--- | :--- | :--- |
| Rain carries across a seam: P(wet at first new step \| wet at last known step) | **≥ 0.70** | inside a piece ~0.83; E2 without resampling 0.58–0.61; v17 0.68 |
| Share of wet steps | within **±1 pp** of the independent-piece run | v17 chained was −1.9 pp |
| Total rain | within **±10%** | v17 chained −12% |
| 99.9th percentile of wet steps | within **±15%** | v17 +25% |

**Choice:** the smallest strength that passes, preferring the rainy-seams version on ties (fewest
interventions). **If none passes: stop.** The negative result is reported (inference-time guidance cannot
carry storms on zero-heavy rain without drying it out), and section 7 is the only next step.

**Measured but not a rule:** how long rain continues after a wet seam (survival to 1, 2, 4 hours), rain at
seams versus inside pieces, and the correction size by σ.

---

## 5. Full run (cluster, only if the pilot passes)

* **10 years, 4 chains, the chosen setting** (`v14_guided_cal`, `_m1`–`_m3`).
* **Split by year to fit the time limit.** A guided 10-year chain needs ~30 GPU-hours. Instead, 10 array tasks
  each generate one calendar year (all 4 chains); each year starts with one unguided piece. This adds 10 seams
  without context in 10 years (out of ~21,900), which is negligible and is recorded. Wall time: a few hours.
* **Judged by v17's pre-registered rule**, so the two are directly comparable: ≥ 10 unbroken runs > 5h20 a year,
  ≥ 15% of rain in them, volume 0.95–1.05, wet spells 0.90–1.10 (mean of 4 chains).
* **Plus:** Gate A (fixed bands; the hope is v14's 14/18 kept), storm-scale share of rain, and the sewer check with
  both controllers (`--controllers BC EFD`; one controller alone deletes the other's rows).

**What would count as a real gain over v17:** long storms at or above v17's (8+ a year, ~50% of rain in storms
over 5h20) with v14's calibration kept (about 14/18, volume within 5%).

---

## 6. Order of work

| Step | Where | Time | Output |
| :---: | :--- | :--- | :--- |
| 1 | laptop | ~1 h of writing | this plan and the pass rules in the diary |
| 2 | laptop | < 10 min of running | `regime_training/generate_guided.py` (new file; nothing existing changes). Checks: `w = 0` reproduces `generate_context.py` exactly for the same seed (a few pieces); a guided smoke run of a few pieces; the gradient sizes that set the `w` grid |
| 3 | cluster | ~20 min wall (array) | pilot, `results/guidance/pilot.json`, verdict by the rules |
| 4 | cluster | a few hours wall (array) | full run, only if step 3 passes |
| 5 | laptop | minutes | scoring with `sitetransfer.metrics`, sewer check, diary |

---

## 7. Fallback: a small helper model (only if the pilot fails for the right reason)

If the pilot shows the corrections are too weak at high noise (v14's guess of the first 80 minutes is too blurry
to steer when the piece's layout is being decided), a separate small network can supply the direction instead.
It is trained on real 2000–2007 windows to answer one question from a *noisy* piece: "given this context, how
likely is rain right after it?". Its gradient then steers v14 (classifier guidance, Dhariwal & Nichol 2021). It is
trained once on the cluster (small, about an hour), and v14 is still untouched. This is a separate experiment with
its own plan; it is not started unless step 3 fails this way.

---

## 8. Predictions (written before any run)

1. Plain guidance raises continuation over pasting but dries the series (wet share −1 to −2 pp).
2. The rainy-seams version passes all four rules at some strength.
3. In the full run, long storms reach 6–10 a year (between pasting's 2.3–2.9 and real 14.3), with Gate A
   12–14/18 and volume within 5%.

## 9. Not covered

A second training seed of v14, the context length (kept at 80 minutes for comparability with E2 and v17), and
Singapore (guidance is tested on the home site only).

---

## 10. Changes before the pilot was sent (2026-10-14)

* **Concrete check on one real storm** (17 April 2008, 32 samples of one piece): without guidance v14 lets the
  storm die (38% still raining after 30 min; mean 0.020 mm against a real 0.118 mm); w = 0.1 continues it at
  about the real intensity (56%; 0.108 mm); w = 0.5 overshoots (94%; 0.361 mm). Corrections act at high noise
  and vanish at the end, as the sigma^2 scaling intends.
* **Weights rescaled.** A tiny smoke run (1 day, 2 chains) showed the wet-weighted versions 10-35 times too wet:
  weighting wet steps 10x multiplied the whole correction. The weights are now rescaled to a mean of 1 over each
  chain's known steps, so they change the wet/dry balance, not the size; a strength means the same in all
  versions. After the fix (8 chains x 3 days) totals were 262-508 mm/yr against 411 for independent pieces.
* **Grid** widened downwards to six strengths, 0.003-0.1 (doubling); 20 array tasks in all.
* **Context length.** The chosen setting is rerun with 2 hours of context (3 columns) in a follow-up job.
  Rule written before: 2 h is adopted for the full run if it passes all four rules and its continuation beats
  80 minutes by at least 0.03; otherwise 80 minutes stays (and the paper reports that 2 h was tested).
* **Known history noise** is 0: the context is the previous piece's output exactly (E2 used 0.05).
* Code: package `guidance/` (README there); with w = 0 the sampler reproduces `generate_context.sample_block`
  bit for bit (unit test).
