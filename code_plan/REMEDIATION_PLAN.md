# Remediation Plan — Responding to the Panel Review

Companion to [`PANEL_REVIEW.md`](PANEL_REVIEW.md). Written 2026-09-04.

This is the chronological work plan that takes the project from "promising" to "defensible".
Every task cites the panel section(s) it closes. Nothing here is optional for the thesis;
the ordering is what is negotiable, and it is chosen so that each phase produces the inputs
the next one needs.

---

## Finding → task crosswalk

| Panel § | Problem in one line | Closed by | Status |
|---|---|---|---|
| §0, §2 | Mechanism mis-described as HMM/Markov weather state | T0.2, T1.2, T8.3, (Phase 10 for Path B) | in progress |
| §1 | Every number is one draw vs one draw, no uncertainty | T2.1, T2.2, T3.1 |
| §2 | Degenerate transition matrix, K not established, emission-independence violated, interannual variability suppressed | T0.2, T1.3, T3.3, T8.1, Phase 10 |
| §3 | Independent-block generation, 100-min seam artefact, temporal eval stops where method is weakest | T2.3, T4.2, T4.3, T4.4 |
| §4 | Loss audit aimed at a failure v7 does not have | T0.5, T6.1, T6.2, T6.3, T8.2 |
| §5 | Delay-embedding "image" is 62.5% structural zero, never justified | T5.1, T5.2 |
| §6 | Acceptance thresholds not derived; v7 fails its own gates; KS misused; diurnal tier unmeetable; 0.005 mm knob; storm defs differ by resolution | T2.4, T2.5, T7.3, T7.4, T8.1 |
| §7 | "Ground truth" moved between June and Aug; auto-report contradicts itself; radar weights unmotivated | T0.4, T2.6, T8.5 |
| §8 | Base checkpoint untraceable; >1 variable per version step; notebook inconsistencies; holdout removed; model selection on training loss | T0.3, T1.1, T1.2, T3.3, T6.5 |
| §9 | Dead schedule code; v3 mislabelled "unconditional"; EDM determinism; sigma_data; tail truncation & leap days; diurnal phase | T6.4, T2.5, T8.2 |
| §10 | The consolidated "defensible" checklist | All phases; exit criteria at the bottom of this doc |

---

## Five rules that hold for the whole plan

1. **One canonical observed-statistics table.** After T0.4, no script recomputes an observed
   statistic. Everything reads `results/reference/observed_stats.json`. If a number is not in
   that file, it does not go in the thesis.
2. **One canonical eval script.** After T2.2, `scripts/evaluate_rainfall.py` is the only
   thing that produces pass/fail. `compare_all_versions.py` is demoted to plots labelled
   "descriptive, not a test".
3. **Ensembles, never point estimates.** Every synthetic metric is reported as
   `mean ± sd` over ≥ 30 realisations, against a block-bootstrap band on the observed record.
   A ratio quoted to three decimals from a single draw is not a result.
4. **Honest numbers come from the holdout model.** After Phase 1 there are two models: the
   *production* generator (trained on all 10 years, used only to make training data for RL)
   and the *holdout* generator (trained on 2000–2007, used for every reported evaluation
   number). Never report a metric from the production model.
5. **One variable per experiment.** v5→v6→v7 changed five things at once. From now on, one
   change, same seed-ensemble, same eval, or it does not go in the ablation table.

---

# Phase 0 — Triage and framing (no GPU, ~3–5 days)

**Goal:** stop the narrative from drifting any further, and lay the fixed reference points
every later phase reads from. All of this is writing and small local scripts — no cluster.


### T0.2 — Rename the mechanism everywhere  *(1 day)*  **[load-bearing]**
- **Addresses:** §0, §2.
- **Decision to make first:** Path A or Path B (see advisor's note). If unsure, choose
  **Path A now** — you can still do Path B as Phase 10, and the rename is required either way
  for the historical versions.
- **Steps:**
  1. Global terminology change, in `my notes/changelogs/research_progress_diary.md`,
     `code_plan/LOSS_FUNCTION.md` §1, `code_plan/ACCEPTANCE_CRITERIA.md`,
     `code_plan/CONDITIONAL_GEN_GUIDE.md`, and the memory notes:
     - "HMM state" / "HMM state conditioning" → **"seasonal-phase index"** /
       **"seasonal-phase conditioning"**.
     - "transition-aware Markov assembly" → **"calendar-ordered block assembly"**.
     - "1st-order Markov chain" / "stationary distribution" → delete, or keep only with the
       explicit caveat below.
  2. Do **not** silently rewrite dated diary entries. Add a dated **`Correction (2026-09-04)`**
     block at the top of the Aug 19–24, Aug 25, Aug 26 and Sep 1 entries, each 3–5 lines,
     stating what the entry called an HMM/Markov process and what it actually was.
  3. Add one new subsection to `LOSS_FUNCTION.md` and `ACCEPTANCE_CRITERIA.md`:
     *"Why we no longer call this an HMM."* Cite the three reasons from panel §2: the label
     is a periodic function of day-of-year identical in all 10 years; the HMM is fitted on a
     14-day-smoothed series so consecutive samples are ~100% autocorrelated and the
     emission-independence assumption is void; the 4×4 block transition matrix is estimated
     from a few hundred near-deterministic seasonal crossings with `smooth=1e-5` Laplace fill,
     so its off-diagonals are "which season follows which".
  4. In `c` = "one-hot HMM state label (4 classes)" in `LOSS_FUNCTION.md` §1 table, change to
     "one-hot seasonal-phase index (4 levels, ascending mean intensity)".
- **Done when:** `grep -ri "markov\|hmm state\|weather state" code_plan/ "my notes/"` returns
  only lines that are explicitly flagged as historical or as the caveat.
- **Write-up:** new diary entry `2026-09-04 — Renaming the conditioning mechanism`
  (Overview / Findings / Decision), verdict in bold.

### T0.3 — Pin the base-checkpoint provenance  *(1 day)*
- **Addresses:** §8 bullet 1, §8 bullet 5.
- **Files:** `configs/pretrain/pretrain.yaml` (currently `train_on_datasets: [stock]` only),
  `models_ckpt/ImagenFew/ImagenFew_{12,24,36,64}.ckpt`, `run.py`, `logs/ImagenFew/`,
  `scripts/run_hmm_training.sh` (fine-tunes `ImagenFew_24.ckpt`), the data-scarcity scripts
  that reference the missing `models_ckpt/ImagenFew/dyConv_Basic_24.ckpt`.
- **Steps:**
  1. Determine what `ImagenFew_24.ckpt` was actually trained on. Read `run.py` and the
     pretrain logs; check the checkpoint's own metadata (`python -c "import torch;
     print(torch.load(p, map_location='cpu').keys())"` on the cluster — locally you have no
     torch, so do this in a SLURM interactive shell or a one-line batch job).
  2. Record, in a new `code_plan/PROVENANCE.md`:
     - exact pretraining corpus (if it really is TimeGAN `stock` only, **say so plainly** and
       drop the word "foundation" — reframe as "transfer from a single stock-price series");
     - `pretrain.yaml` as used, `sha256` of the `.ckpt`, epoch count, EMA on/off;
     - the mapping `ImagenFew_{12,24,36,64}` → which `seq_len` each corresponds to;
     - what `dyConv_Basic_24.ckpt` was, whether it still exists anywhere (cluster scratch,
       old SLURM job dirs), and if not, that the data-scarcity scripts are **not runnable as
       written** and will be pointed at `ImagenFew_24.ckpt` in T1.
  3. Add a "Provenance" line to `results/generated_data/generated_data_log.json` for every
     version: base ckpt hash, training script, config file, SLURM job id.
- **Done when:** you can state, in one sentence with a hash, what weights v5–v7 started from.

### T0.4 — Freeze the canonical observed-statistics table  *(1.5 days)*  **[parallel]**
- **Addresses:** §1, §6 (0.005 mm), §7 bullet 1, §9 (leap days, truncation).
- **New file:** `scripts/observed_reference_stats.py`.
- **Steps:**
  1. Single input: `data/rainfall/real_rainfall_data.csv`. Print and record its `sha256`,
     row count, first/last timestamp, and sampling interval at the top of the output.
  2. Resolve the leap-day / truncation question from §9: the CSV is 3650 days = exactly
     10×365, so ~3 calendar days were dropped upstream. Find where (in the Astlingen
     preprocessing under `preprocessing/` or `data/rainfall/astlingen/`), document it, and
     decide one convention (drop Feb 29 everywhere, or carry it). Apply the same convention
     to synthetic series in T2.
  3. Compute **every** statistic used anywhere in the project, each with an explicit
     definition string:
     - water balance: annual volume (mean & per-year vector), zero fraction;
     - spells: mean/median dry-spell and wet-spell duration, full duration distributions;
     - intensity marginal: mean wet, P90/P95/P99/P99.9, max, wet-intensity histogram
       (native and hourly);
     - storms: **one** storm definition — contiguous run with depth > `thr`, minimum
       duration ≥ 15 physical minutes, inter-storm gap ≥ 15 min — count, mean duration, mean
       volume, peak-timing fraction. State `thr` and apply it identically to observed and
       synthetic (see T2.5 on the 0.005 mm knob);
     - temporal: ACF at native resolution lags 1–288, hourly ACF lags 1–24 h, daily ACF
       lags 1–30 d, wet-only ACF, lag-1 for reference;
     - extremes: daily maximum, annual-maximum series for durations D ∈ {15 min, 1 h, 3 h,
       6 h, 24 h}, and the empirical IDF cells at return periods T ∈ {1, 2, 5, 10} yr
       (Gumbel or GEV fit on the 10 annual maxima per D — 10 points is thin, so report the
       fit CI);
     - seasonality: monthly volume profile (12-vector), diurnal profile (24-vector or
       48 half-hours), aligned to the real record's calendar phase.
  4. Write `results/reference/observed_stats.json` (machine-readable, every value keyed with
     its definition) and `results/reference/observed_stats.md` (human table). Header of both:
     git commit, input hash, script version, leap-day convention.
  5. Add a `--bootstrap` mode: moving-block bootstrap of the observed record (block length
     5–10 days, ≥ 1000 resamples) → 2.5/97.5 percentile band for every scalar statistic.
     Store as `observed_stats_bootstrap.json`. This is the band synthetic ensembles are
     judged against in T2.2.
- **Local validation:** pure numpy/pandas/scipy — runs in `.venv`. No torch. Add a
  `pytest` in `scripts/tests/test_observed_reference_stats.py` asserting volume, zero
  fraction and P99 against the values already in the Aug 26 diary table (709.4 mm, 90.95%,
  0.160) so drift is caught.
- **Done when:** `observed_stats.json` exists, is committed, and the June-era numbers
  (storm count 4852, mean storm duration 100 min, daily max 52.1 mm) and the Aug-era numbers
  (6800, 62.3 min) are both reproduced from it by toggling only the documented definition
  knobs — proving the "ground truth moved" was a definition change, and pinning which
  definition the thesis uses.
- **Write-up:** diary entry `2026-09-05 — Canonical observed reference`; a short table in
  `PANEL_RESPONSE.md` §7 showing June vs Aug vs canonical.

### T0.5 — Reconcile the two docs on extremes, in writing  *(0.5 day)*
- **Addresses:** §4 (the loss work targets a failure v7 does not have).
- **Steps:**
  1. Add the same paragraph to `LOSS_FUNCTION.md` (top of §2) and `ACCEPTANCE_CRITERIA.md`
     (under "Current verdict on v7"): *"v7 passes Tier 2 outright (P99 1.006×, P99.9 1.006×,
     max 1.016×). Heavy-tail extremes are therefore **not** the open problem after v7. The
     open problem is storm geometry (Tier 3 / panel §3), which loss reweighting does not
     touch. The `intensity_weight_alpha/gamma`, `fft_weight` and `grad_clip` apparatus is
     retained as instrumentation and as insurance for the holdout-trained model (Phase 1),
     which will have weaker tails, not as a next step for the production model."*
  2. In the diary's Sep 1 "Next actions" list, demote items 1–2 (seq_len, clip_frac) from
     "loss tuning" framing to "generation-continuity / diagnostic" framing.
- **Done when:** neither doc reads as if heavy-tail loss weighting is a required next step.

---

# Phase 1 — The honest data split (no GPU for the split itself; retrain is Phase 3)

**Goal:** create a real holdout so that every later evaluation number is reportable. This is
panel §8 bullet 4 and the Gate C blocking prerequisite, pulled forward because Gate A needs
it too (§10 point 7).

### T1.1 — Re-split by year  *(1 day)*
- **Addresses:** §8 bullet 4, §10 point 7.
- **New file:** `scripts/split_by_year.py`.
- **Steps:**
  1. Train = 2000–2007, validation = 2008, test = 2009. (Two held-out years; 2008 tunes,
     2009 is touched only for the final number.)
  2. Output `data/rainfall/splits/{train,val,test}_years.csv` plus a
     `data/rainfall/splits/MANIFEST.json` with row counts, date ranges, and the input hash.
  3. Deprecate the old `data/rainfall/{train,val,test}_dataset.csv` (row-based 80/10/10 from
     the v1–v4 era) — move to `data/rainfall/_deprecated/` with a README explaining they
     leaked across years.
- **Done when:** no year appears in more than one split; manifest committed.

### T1.2 — Re-fit the seasonal-phase labels on training years only  *(2 days)*
- **Addresses:** §2, §8 bullet 4, §8 bullet 3 (notebook inconsistencies).
- **Why:** the phase labels are fitted quantities (`GaussianHMM` on the smoothed climatology).
  They leak the held-out years exactly as the diffusion weights would.
- **Steps:**
  1. **De-notebook it.** Port the labelling logic out of
     `data_analysis/Rainfall_10Yr_105120_Interval_HMM_v2_Mean_smoothed.ipynb` into
     `scripts/fit_seasonal_labels.py` with CLI args `--years 2000-2007 --grid 105120
     --smooth-days 14 --n-states 4 --min-duration-steps N`. One script, deterministic,
     seeded, version-controlled.
  2. Fit only on 2000–2007. Produce: the state label map (interval index → state), the
     ascending-intensity relabelling, the min-duration merge, and — if kept at all — the 4×4
     block transition matrix, all written to `results/reference/seasonal_labels_train.pkl`
     and a diagnostic PNG.
  3. Broadcast labels to every row of all splits by calendar position, as before. The 2008–09
     rows get labels from the 2000–07 fit — that is the point.
  4. Resolve the §8 bullet-3 inconsistencies while you are in here: header says "median
     profile" but code uses `mean_intensity_smooth`; `min_days=3` in the signature vs "≥ 7
     Days" in a plot title. Pick one, name it in the script, regenerate
     `dataset_with_105120_v2_fit_hmm.csv` → `dataset_105120_labels_train.csv`.
- **Local validation:** `hmmlearn` + numpy, runs in `.venv`, no torch. `ast.parse` the new
  script; add a test that label proportions on train years match the retained
  `regime_proportions.pkl` within a tolerance you document.
- **Done when:** labels for 2008–09 are produced without those years touching the fit;
  old notebook marked "superseded by `scripts/fit_seasonal_labels.py`".

### T1.3 — Establish the state count properly  *(1 day)*  **[parallel]**
- **Addresses:** §2 ("K = 4 is not established for the model that shipped").
- **Steps:**
  1. In `scripts/fit_seasonal_labels.py`, add `--select-k`: fit for K ∈ 2…10 on the training
     years, report BIC, AIC, and held-out (2008) log-likelihood per K. Also report a
     segmentation-quality metric that does not assume emission independence (e.g. mean
     within-state variance of the smoothed feature vs between-state).
  2. Write `results/reference/k_selection.{json,png}`.
  3. If K = 4 is not clearly best: either switch to the supported K, or keep 4 and state
     explicitly *"4 is a pragmatic choice matching the four-season narrative; BIC prefers K;
     the states are magnitude quartiles of the smoothed climatology, not meteorological
     regimes"* — the honest version of the July 27 "4 states = 4 seasons" claim.
  4. Note clearly: because the feature is 14-day-smoothed, BIC/log-likelihood here are
     descriptive, not inferential (emission independence is violated). Say so next to the
     numbers.
- **Done when:** `k_selection.json` exists and the thesis text on K points to it, not to the
  Aug 5 "ELBO metrics" (which were a *GMM* `lower_bound_`, not an HMM quantity — correct that
  sentence in the diary too).

### T1.4 — Point the training config at the split  *(0.5 day)*
- **Addresses:** §8 bullet 4, Gate C prerequisite.
- **Files:** `regime_training/config_hmm.yaml`.
- **Steps:**
  1. `train_csv:` → `data/rainfall/splits/train_years_labelled.csv`.
  2. `test_csv:` → `data/rainfall/splits/val_years_labelled.csv` (2008) — a **real** holdout,
     not `== train_csv`.
  3. Add a commented `production_csv:` pointing at the all-years labelled file, used only by
     the production-generator job.
  4. Update the header comment: "No separate test split" is now false — delete it.
- **Done when:** `grep test_csv regime_training/config_hmm.yaml` no longer equals `train_csv`.

---

# Phase 2 — The measurement layer: ensembles + one canonical eval

**Goal:** replace "one draw vs one draw, point estimates, KS as a test" with an ensembled,
uncertainty-aware, single-source evaluation. No cluster needed for the eval code; ensemble
*generation* needs GPU (Phase 3 runs it).

### T2.1 — Ensemble generation harness  *(1.5 days)*
- **Addresses:** §1.
- **Files:** `regime_training/generate_hmm_v1.py` (currently hard-codes `--seed 42` and seeds
  torch/numpy/the block sampler all from it → one 10-year realisation per version).
- **Steps:**
  1. Add `--n-realisations N` (default 30) and `--seed-base S`. Loop N times, seed
     `{torch, numpy, block-sampler}` from `S + i` for realisation `i`, write
     `results/generated_data/ensembles/<version>/real_<i>.csv`.
  2. Write `results/generated_data/ensembles/<version>/MANIFEST.json`: checkpoint hash, config,
     seed list, generation args (`--overlap`, `--transition_overlap`, `diffusion_steps`,
     `S_churn`), git commit.
  3. Keep single-draw mode for debugging but make N ≥ 30 the default path.
  4. Note in the manifest: with `S_churn = 0` and Heun ODE, per-realisation diversity comes
     **only** from the initial noise `x_T` and the sampled label sequence (panel §9). If
     inter-realisation spread turns out tiny, raise `S_churn` or switch the sampler to
     stochastic and document it.
- **Local validation:** `ast.parse`; dry-run the argument plumbing with a stub that writes
  random arrays instead of calling the model, to confirm file layout and manifest.
- **Done when:** one command produces 30 CSVs + a manifest for a given checkpoint.

### T2.2 — The canonical eval script  *(3 days)*
- **Addresses:** §1, §6, §7.
- **New file:** `scripts/evaluate_rainfall.py`. This becomes the only pass/fail authority.
- **Steps:**
  1. Inputs: `--ensemble-dir results/generated_data/ensembles/<version>/`,
     `--reference results/reference/observed_stats.json` (+ its bootstrap file). Never
     recompute an observed statistic here — read it.
  2. For every metric in T0.4's list: compute it on each of the 30 realisations → report
     `syn_mean, syn_sd, syn_p5, syn_p95`. Compute the ratio to the observed value and its
     spread.
  3. **Pass rule:** a metric passes if the synthetic 5–95% inter-realisation interval
     overlaps the observed moving-block-bootstrap 2.5–97.5% band. A point ratio to three
     decimals never passes or fails on its own. Emit `results/eval/<version>/metrics.json`
     with columns `{metric, definition, obs, obs_lo, obs_hi, syn_mean, syn_sd, syn_lo,
     syn_hi, pass}` and a machine-readable per-tier `pass` boolean.
  4. **Resolution guard:** the script refuses to compare a 5-min series against a 10-min
     reference. Cross-version claims are computed on hourly and daily aggregates only.
  5. **Storm definition:** import the single definition from `observed_stats.json`; apply the
     identical `thr`, min-duration-in-minutes and gap to synthetic. No
     `min_dur_steps = max(1, 15//res)` (panel §6 bullet 6 — that gave 3 steps at 5-min but
     1 step at 10-min).
  6. Emit `results/eval/<version>/report.md` from a template with **no** unfilled sentences
     and **no** composite score.
- **Local validation:** pure numpy/pandas/scipy; test against a synthetic fixture with known
  statistics; test that feeding the observed record to itself yields all-pass with ratios ≈ 1.
- **Done when:** `python scripts/evaluate_rainfall.py --ensemble-dir ... ` prints a tier-by-tier
  table with `mean ± sd` and pass/fail from band overlap, for any version.

### T2.3 — Native-resolution temporal diagnostics  *(1.5 days)*
- **Addresses:** §3 (the eval stops exactly where the method is weakest).
- **Steps (add to `evaluate_rainfall.py`):**
  1. **Native ACF RMSE over lags 1–288** (not just hourly, not just lag-1). Report the curve
     and the scalar RMSE vs observed, with the observed bootstrap band on the curve.
  2. **Wet-only ACF** and an **indicator-vs-intensity ACF split** (ACF of the 0/1 wet mask
     vs ACF of positive depths) — a 91%-zero series' lag-1 is dominated by zero–zero pairs,
     so these separate "does it rain in bursts" from "how hard".
  3. **Intensity periodogram** of the native series; flag any line at **100 min** (the
     same-state seam stride = `seq_len − overlap` = 20 steps) and its harmonics.
  4. Explicit table of ACF value at **lags 20, 40, 60** with the bootstrap band — the seam
     lags. "Seam artefact present" = a spike at lag 20/40/60 outside the band, or a
     periodogram line at 100 min above noise.
  5. Emit `results/eval/<version>/temporal.{json,png}`.
- **Done when:** for v7 you can state, with a number and a band, whether the 100-min seam
  artefact is real. (The panel predicts it is; confirm or refute.)

### T2.4 — Fix the KS misuse  *(0.5 day)*
- **Addresses:** §6 bullet 3, §9.
- **Steps:**
  1. In `evaluate_rainfall.py`, report the KS **statistic** as a descriptive distance only —
     no critical value, no "PASS". At n ≈ 10⁶ every version rejects at p ≈ 0; that is a
     category error, not a result.
  2. If a hypothesis test is wanted: subsample to n ≈ 2000, repeat 500×, report the
     distribution of the KS statistic and the fraction of subsamples that reject at α = 0.05.
  3. Same treatment anywhere `ks_2samp` appears in `compare_all_versions.py`.
- **Done when:** no document reports "KS ... PASS" against the 0.05 small-sample critical
  value.

### T2.5 — Retire the per-version tuned knobs from the metrics  *(1 day)*
- **Addresses:** §6 bullet 5 (0.005 mm), §6 bullet 6 (storm defs), §9 (diurnal phase).
- **Steps:**
  1. **0.005 mm threshold.** `generate_hmm_v1.py` and `compare_all_versions.py` both zero
     synthetic values < 0.005 mm while the real series is only `clip(lower=0)`. Two options,
     pick one and apply identically to both series:
     - (a) Drop the threshold entirely; report zero-fraction as it falls out.
     - (b) Keep 0.005 mm **as the gauge quantisation floor** (defensible rationale), apply it
       to the observed record too, and report it per version in the results table.
     Either way, "zero % absolute difference ≤ 1.0 pp → 0.02 pp PASS" stops being circular.
  2. **Storm definition:** already unified in T0.4/T2.2 — here, add a row to every results
     table stating the definition used, and re-flag every historical v3/v4-vs-v7 storm
     comparison (and the radar) as "not comparable — different min-duration".
  3. **Diurnal phase:** align the synthetic index to the observed record's calendar phase
     before computing the diurnal profile. Then add the honesty note from §6 bullet 4: the
     model has **no time-of-day input** and generates in 2-h blocks with a 100-min seam
     stride, so any diurnal structure is an artefact; Tier 5's diurnal `r ≥ 0.80` is
     "informational, expected to pass only if both profiles are near-flat".
- **Done when:** every metric that depended on a hidden per-version constant now names that
  constant in the output.

### T2.6 — Regenerate the comparison report; kill the radar composite  *(1 day)*
- **Addresses:** §7 bullet 2 (report contradicts itself), §7 bullet 3 (radar weights).
- **Files:** `scripts/compare_all_versions.py`, `results/comparison_all_versions/comparison_report.md`.
- **Steps:**
  1. Regenerate `comparison_report.md` from `evaluate_rainfall.py`. Delete the §4 sentence
     "HMM-conditioned models generate longer, more persistent wet spells matching real storm
     durations" (contradicts §2 of the same file: v7 wet spell 27.4 vs 32.1). Delete every
     unfilled template sentence ("The diurnal cycle plot reveals whether ...").
  2. Remove the radar scorecard, or replace it with small multiples: one panel per metric,
     synthetic `mean ± sd` bar against the observed bootstrap band. No `1 − |1 − VolRatio|·3`,
     no `1 − Wasserstein·20` — the multipliers 3/5/20/2 are an unmotivated weighted composite,
     which is the exact thing `ACCEPTANCE_CRITERIA.md` was written to abandon.
  3. Keep `compare_all_versions.py` only for descriptive time-series plots; put
     "DESCRIPTIVE — NOT A TEST" in its output header.
- **Done when:** the regenerated report's narrative matches its own tables, and contains no
  composite score.

---

# Phase 3 — Re-run the version history honestly (GPU: generation only)

**Goal:** produce the first ensembled, holdout-aware version table, and replace the
"smoothed 1-D mean feature was the change that mattered" narrative with a controlled
ablation.

### T3.1 — Ensemble + eval the existing checkpoints  *(2 days, mostly cluster wait)*
- **Addresses:** §1.
- **Steps:**
  1. With T2.1, generate 30-realisation ensembles for **v5, v6, v7** from the existing
     retrained checkpoints (SLURM 760225/760226/760227). These are still the *production*
     (all-years) checkpoints — label the output accordingly.
  2. Run `evaluate_rainfall.py` on each. This is the first `mean ± sd` table.
  3. Diary entry `2026-09-NN — v5/v6/v7 re-evaluated with ensembles`: the Aug 26 table, now
     with error bars and band-overlap pass/fail. Expect several Aug 26 "PASS"es to become
     "within noise" and the three-decimal ratios to widen.
- **Done when:** `results/eval/v{5,6,7}/metrics.json` exist with ensemble spread.

### T3.2 — Explain v6's dead tail with evidence  *(1 day)*
- **Addresses:** §2 bullet 2 (v6-vs-v7 causal story is inconsistent).
- **Why:** the diary says v6's dead tail (max 1.79 mm) is because "365-day DoY-broadcast
  labels [are] too coarse to carry storm intensity" — but v7 is *also* DoY-broadcast and
  matches the tail. So that cannot be the reason.
- **Steps:**
  1. Dump for v6 and v7: per-state occupancy (fraction of intervals in each of the 4
     states), per-state generated P99.9 over training epochs (`tail_health.png` already
     exists — use it), and the state-separation of the underlying fit (365-point vs
     105120-point).
  2. Candidate causes to test: near-degenerate 365-point HMM fit (two states collapse);
     state-occupancy skew (the "Peak" state gets < 2% of intervals so the model rarely sees
     it); the min-duration merge in v7 lengthening Peak runs enough to matter.
  3. Write the real cause in the diary, or write "cause undetermined; ruled out X; candidates
     Y, Z" — either is defensible, the current sentence is not.
- **Done when:** the diary's v6 explanation is consistent with v7 also being DoY-broadcast.

### T3.3 — Controlled v5→v7 ablation, one variable at a time  *(3–4 days, cluster)*
- **Addresses:** §8 bullet 2, §10 point 6.
- **Steps:**
  1. Fix a base: v5 config, one seed-ensemble spec, `evaluate_rainfall.py`.
  2. Change exactly one thing per run, from the same base:
     - (a) feature: 3 daily features → smoothed 1-D mean;
     - (b) grid: 105120-interval → 365-day;
     - (c) smoothing: 14-day rolling mean on/off;
     - (d) min-duration merge: on/off;
     - (e) transition-matrix variant (if retained at all).
  3. Same 30-seed ensemble, same eval, for every run. Build the table: rows = the five
     changes, columns = the tier metrics, cells = Δ(metric) vs base with ensemble spread.
  4. Verdict in bold replacing "the smoothed 1-D mean feature is the change that mattered":
     name the change(s) that actually moved the intensity tail, with numbers.
- **Cluster note:** this is 5 fine-tunes × (train + generate). Budget accordingly; these can
  be the reduced-epoch runs if you state the epoch count.
- **Done when:** `results/eval/ablation_v5_v7/table.md` exists; diary entry with the verdict;
  memory note updated (it currently asserts the unproven version).

---

# Phase 4 — Fix generation continuity (GPU: retrain + generate)

**Goal:** close, or characterise, the storm-geometry failure that is the real open problem
after v7. This is panel §3 and "Experiment 2" owed since July 27.

### T4.1 — One diagnostic run: read `clip_frac`  *(1 day, cluster)*
- **Addresses:** §4 bullet "mechanisms (b),(c) asserted not measured".
- **Steps:**
  1. Fine-tune once with current defaults on the **holdout** split (2000–2007). Read
     `clip_frac` and `grad_norm` from `history.csv`.
  2. If `clip_frac` > ~20%, raise `grad_clip` to 5.0 and re-run before anything else — that
     removes a distortion rather than adding the `alpha` counter-distortion.
  3. Diary note with the actual number. This retires the "most concrete and most fixable of
     the three" claim in `LOSS_FUNCTION.md` §2(b) — replace it with the measurement.
- **Done when:** `clip_frac` is a logged number in the diary, not a hypothesis.

### T4.2 — `seq_len` 24 → 288  *(4–5 days, cluster)*  **[the headline method experiment]**
- **Addresses:** §3, §6 (context window), diary Aug 26 root-cause.
- **Files:** `regime_training/config_hmm.yaml` (`seq_len: 24`), `regime_dataset.py`,
  `generate_hmm_v1.py`, the `DelayEmbedder` call, `configs/finetune/Rainfall.yaml`.
- **Steps:**
  1. Check the architecture takes `seq_len` 288 without a shape break: `img_resolution: 8`,
     `delay: 8`, `embedding: 8` currently fill 3 of 8 columns for `seq_len 24`. For 288 you
     need a different embedding geometry (e.g. `img_resolution` up, or a 1-D model per
     Phase 5). Decide and record the geometry.
  2. Fine-tune from `ImagenFew_24.ckpt` (or the matching `ImagenFew_36/64` if one is closer)
     on the holdout split, `seq_len 288` (24 h at 5-min).
  3. Generate a 30-realisation ensemble; run `evaluate_rainfall.py` with emphasis on T2.3:
     native ACF RMSE 1–288, periodogram, storm duration/count/volume, wet-spell.
  4. Compare head-to-head with v7 (holdout-trained, same eval). Table + diary entry.
- **Done when:** you can say whether widening the context window closes storm geometry, with
  ensembled numbers — not a guess.

### T4.3 — Autoregressive block generation (if T4.2 doesn't close it)  *(3–4 days)*
- **Addresses:** §3 (blocks still i.i.d. given the label; 20-min crossfade is the whole
  continuity mechanism).
- **New file:** `regime_training/generate_hmm_v2.py`.
- **Steps:**
  1. Instead of generating each 2-h block from fresh noise and raised-cosine stitching,
     condition each block on the **tail of the previous block** via the model's own
     `impute` / `forecast` path (`models/ImagenFew/`). The overlap region is then *modelled*,
     not averaged.
  2. Keep `generate_hmm_v1.py` runnable for comparison.
  3. Ensemble + eval both; the acceptance target is native ACF RMSE and periodogram, not
     lag-1.
- **Done when:** a generation method exists whose seam check (T2.3 step 4) passes, or you have
  documented that both methods fail it and why.

### T4.4 — Seam-artefact check on every generation method  *(0.5 day)*
- **Addresses:** §3 ✓ (seam periodicity).
- **Steps:** for v1-stitch, v2-autoregressive, and any `seq_len`/overlap combo, report the
  T2.3 lag-20/40/60 ACF table and the 100-min periodogram line. Put them in one figure in the
  thesis.
- **Done when:** the figure exists and each method has a one-line verdict.

---

# Phase 5 — The delay-embedding confound (GPU: small ablation)

**Goal:** justify or drop the 8×8 "image", which is currently 62.5% structural zero and never
motivated (panel §5).

### T5.1 — Quantify and document the padding  *(0.5 day, no GPU)*
- **Addresses:** §5, §4 (3.1 — the FFT term is padding housekeeping).
- **Steps:**
  1. Add a numeric check to a notebook or script: for `seq_len=24, delay=8, embedding=8` the
     `DelayEmbedder` while-loop fills exactly 3 of 8 columns (steps 0–7, 8–15, 16–23);
     `delay == embedding` so there is **no delay-coordinate structure** — it is `reshape(3,8)`
     in a zero frame; 5/8 = 62.5% is structural zero.
  2. Add a one-paragraph caveat to `LOSS_FUNCTION.md` §3.1 and anywhere an "image-domain" or
     "frequency-domain" argument is made: 5/8 of the modelled tensor is padding, a hard
     vertical edge sits at column 3, and square conv kernels treat Δt = 1 (row) and Δt = 8
     (column) neighbours as equivalent — a strange prior for a 1-D signal.
- **Done when:** no "frequency domain" sentence in the docs stands without the padding caveat.

### T5.2 — Ablate: 1-D vs image vs real Takens embedding  *(3 days, cluster)*
- **Addresses:** §5.
- **Steps:**
  1. Three fine-tunes, same holdout data, same seed-ensemble, same eval:
     - (a) **1-D model** — no delay embedding, a 1-D conv/attention U-Net on the length-24
       (or 288) series;
     - (b) **`delay < embedding`** — an actual Takens delay-coordinate embedding with no
       zero padding;
     - (c) **current** `delay == embedding == 8` image.
  2. Keep whichever wins on the Phase 2 eval. If (c) wins, you owe a justification of the
     square-kernel isotropy; if (a) or (b) wins, switch and note the simplification.
- **Done when:** `results/eval/ablation_embedding/table.md` exists and the thesis states which
  representation the final model uses and why.

---

# Phase 6 — Confirmed loss/code defects → v8 (GPU: retrain)

**Goal:** fix the things the panel confirmed are defects, and re-scope the loss narrative.
Most of this is small edits; the retrain at the end folds them into the reported model.

### T6.1 — Fix the FFT term  *(1 day + retrain)*
- **Addresses:** §4 (3.1).
- **Files:** `regime_training/train_regime.py`.
- **Steps:**
  1. `norm='forward'` → `norm='ortho'` (Parseval-exact, ratio 1.0 not 1/64).
  2. Take the FFT of the **length-`seq_len` time series**, not of the zero-padded 2-D image.
  3. Drop the spatial `(1 - x_img_mask)` multiply on a frequency-domain quantity — it is a
     genuine bug (keeps an arbitrary 3 of 8 frequency-column bins).
  4. Default `fft_weight: 0.0` for now (it was doing nothing useful); make it a deliberate
     knob for v8 spectral experiments.
- **Local validation:** `ast.parse`; a numpy check that `‖ortho-FFT(r)‖² == ‖r‖²`.
- **Done when:** the term is either off or a real spectral prior, and `LOSS_FUNCTION.md` §3.1
  says which.

### T6.2 — Fix `sigma_data`  *(0.5 day + retrain)*
- **Addresses:** §4 (third ✓), §9 bullet 4.
- **Files:** `models/ImagenFew/ImagenFew.py` (`sigma_data = 0.5` hard-coded),
  `regime_training/regime_dataset.py` (`StandardScaler` → std 1.0).
- **Steps:** set `sigma_data` to match the scaler output (1.0), or switch the scaler; either
  way `c_skip/c_out/c_in` and `λ(σ)` emphasis shift — re-derive the one line in
  `LOSS_FUNCTION.md` §1 and note that every shipped version including v7 was 2× mis-tuned, so
  v8 is the first calibrated one.
- **Done when:** scaler std and `sigma_data` agree, documented.

### T6.3 — Demote intensity weighting in the docs  *(0.5 day)*
- **Addresses:** §4 ("Demote the intensity-weighting to 'available, off by default, not
  currently motivated by evidence'").
- **Steps:** in `LOSS_FUNCTION.md` §2 "The fix now implemented" and the change log, and in the
  diary Sep 1 "Code changes": relabel `intensity_weight_alpha/gamma` as *available
  instrumentation, off by default, only revisited if the holdout-trained model (weaker tails)
  needs it*. Remove it from any "suggested v8 sweep" ordered list that implies it is a next
  step. Keep the measured weight profile table (dry 1.00×, P99 1.54×, ...) as a reference.
- **Done when:** the docs no longer imply heavy-tail loss weighting is planned work.

### T6.4 — Clear the dead code and the mislabels  *(1 day)*
- **Addresses:** §9 bullets 1, 2, 3; §4 (3.2).
- **Steps:**
  1. `models/ImagenFew/sampler.py`: `self.betas / alphas / alpha_bars` computed in
     `__init__` and never used (pure EDM/Heun). Delete or move behind a comment
     `# unused: sampler is EDM/Karras, no DDPM schedule`.
  2. `regime_training/config_hmm.yaml`: `beta1`, `betaT`, `deterministic` are dead — remove or
     comment as unused.
  3. `models/ImagenFew/ImagenFew.py`: `ImagenFew.loss_fn` is dead code (nothing calls it;
     `handler.py:44` is the real path for v3/v4, `train_regime.py` for v5–v7). Delete it or
     mark `# DEAD — real losses live in handler.py and train_regime.py`.
  4. Relabel **v3**: it is "conditioned on class 0", not "unconditional" — the network keeps a
     class embedding (`label_dim = n_classes`) and inference feeds a concrete label. Fix in
     `generated_data_log.json`, the diary (July 27 "v3 stays locked in State 2"
     interpretation), and `ACCEPTANCE_CRITERIA.md`.
  5. Add one sentence somewhere central: EDM with `diffusion_steps = 36`, `S_churn = 0` is a
     deterministic Heun ODE from `x_T`, so per-block diversity comes only from the initial
     noise.
- **Done when:** `grep -n "betas\|alpha_bars\|loss_fn" models/ImagenFew/` shows only
  annotated/removed lines; "unconditional" no longer describes v3.

### T6.5 — Model selection on held-out loss  *(1 day + retrain)*
- **Addresses:** §8 bullet 5.
- **Files:** `regime_training/train_regime.py` (`best_loss = avg_loss` where `avg_loss` is
  mean *training* loss; `test_ds` built but no `test_loader`).
- **Steps:**
  1. Build `test_loader` from the 2008 val split; compute val loss each `logging_iter`.
  2. `best_model` = argmin val loss, not training loss. Log both curves; mark the selected
     epoch on `loss_curve.png`.
  3. Add a simple early-stopping / patience readout (no need to actually stop, just report
     where it would have).
- **Done when:** `best_regime_model.pt` is selected on 2008 loss and the diary says so.

### T6.6 — Retrain the reported model with all Phase 4–6 fixes  *(4–5 days, cluster)*
- **Steps:** one clean fine-tune on the holdout split folding in: the chosen context window
  (T4.2/T4.3), the chosen embedding (T5.2), FFT fix (T6.1), `sigma_data` fix (T6.2),
  held-out selection (T6.5). Generate a 30-realisation ensemble. This is the model whose
  numbers go in the thesis.
- **Done when:** `results/eval/v8_holdout/metrics.json` exists.

---

# Phase 7 — Gate B: hydraulic response in SWMM (repo: flood-control; no RL)

**Goal:** the panel's highest-value missing piece (§10 point 3). It is where the Gate A
thresholds actually come from, and it tells you whether the storm-geometry failure matters
hydraulically. Can run **in parallel with Phase 4** — different repo, different compute.

### T7.1 — Rainfall → SWMM routing harness  *(3 days)*
- **Addresses:** §10 point 3, `ACCEPTANCE_CRITERIA.md` Gate B.
- **Files (flood-control):** `data/SWMM-Astlingen/Astlingen_SWMM.inp`,
  `src/simulation/`, `src/controllers/` (use the existing rule-based controller — **no RL**),
  `src/metrics/`.
- **Steps:**
  1. New script `flood-control/src/rainfall_datagen/route_rainfall.py`: take a rainfall CSV,
     write it as the `.inp` rain time series (or via the SWMM API rain interface), run SWMM
     under a fixed controller, extract per-year KPIs:
     - CSO event count / yr, total CSO spill volume / yr;
     - peak spill rate distribution, tank filling-level distribution;
     - flooding volume annual maximum;
     - fraction of simulated time in "system stressed" state (define the threshold once).
  2. Output `flood-control/results/gate_b/<series>/kpis.json`.
- **Done when:** routing the observed record reproduces the KPIs the flood-control repo
  already reports for real rain.

### T7.2 — Route observed + every synthetic ensemble member  *(2 days, mostly compute)*
- **Steps:**
  1. Route the observed record and all 30 realisations of each candidate version
     (v7-holdout, v8-holdout, and any Phase 4 variant).
  2. Report each KPI as synthetic `mean ± sd` vs observed, with an observed block-bootstrap
     band (route bootstrapped observed years too, or bootstrap the KPI series).
  3. Pass bands from `ACCEPTANCE_CRITERIA.md` Gate B (CSO count ratio 0.85–1.15, etc.) —
     but now judged by band overlap, not point ratio.
- **Done when:** `flood-control/results/gate_b/summary.md` compares every candidate.

### T7.3 — Derive Gate A thresholds from Gate B sensitivity  *(2 days)*
- **Addresses:** §6 bullet 1 (thresholds not grounded in a tolerance), §10 point 3.
- **Steps:**
  1. Perturb the observed rainfall along each Gate A axis independently: volume ±5/10/20%,
     storm duration ±10/20/30%, P99 ±10/20%, ACF (via block-reshuffle) etc.
  2. Route each perturbed series through SWMM; measure the CSO-volume / flooding response.
  3. Back out: how tight must each Gate A band be so that a series passing Gate A cannot
     produce more than (say) 10% CSO-volume error? Rewrite the `ACCEPTANCE_CRITERIA.md` Tier
     bands with this justification — each band gets a sentence *"a X% error here maps to a
     Y% CSO-volume error, measured in T7.3"*.
- **Done when:** every Gate A pass band cites a hydraulic sensitivity number.

### T7.4 — Resolve v7's status against its own gates  *(0.5 day)*
- **Addresses:** §6 bullet 2 ("a version must pass every tier to proceed" rejects v7).
- **Steps:** with rederived thresholds + ensembles + Gate B, state explicitly in
  `ACCEPTANCE_CRITERIA.md`:
  - whether any shipped version passes Gate A end-to-end (likely none do — say so);
  - that the gates are **screening, advisory, and reported in full** (every tier shown, pass
    or fail), not a single gate you must clear — which is *how* they are less gameable than a
    weighted score: nothing is hidden inside a sum;
  - which model is the **production** generator (fails some tiers, used only to make RL
    training data) and which is the **reported** model (holdout-trained, all tiers shown).
- **Done when:** there is no contradiction between "v7 is the best" and "v7 fails the
  framework" — both are stated, with their scopes.

---

# Phase 8 — Reconcile every document (no GPU, ~4 days)

**Goal:** the thesis, the diary, and the three design docs all tell the same story. Panel §7,
§10 points 5 and 8.

### T8.1 — Rewrite `ACCEPTANCE_CRITERIA.md`  *(1.5 days)*
- Thresholds justified by T7.3. KS demoted (T2.4). Diurnal tier marked artefact-limited
  (T2.5). 0.005 mm stated as gauge floor, applied to both series, reported per version
  (T2.5). Storm definition unified and named (T0.4). Add the **interannual-variability
  limitation** the panel says is unnamed: *"the synthetic series cannot contain an unusually
  wet winter or a drought summer; between-year variation is per-block sampling noise only.
  For flood-control RL this removes exactly the rare years that are the training material —
  Path B (Phase 10) is the fix; Path A accepts this as a stated scope limit."*
- Add the machine-readable per-tier pass column that Gate A "still needs".

### T8.2 — Rewrite `LOSS_FUNCTION.md`  *(1 day)*
- FFT-bug and `sigma_data` finding move to "confirmed defects, fixed in v8" (T6.1, T6.2).
  Intensity weighting demoted (T6.3). The `clip_frac` number from T4.1 replaces the
  "asserted not measured" mechanism (b). Keep the strong parts verbatim: the "L2 regresses
  to the mean is folklore" debunk, the Parseval derivation, the `sigma_data` mismatch. Add
  the delay-embedding padding caveat (T5.1). One sentence stating the post-v7 open problem is
  storm geometry, not extremes.

### T8.3 — Rewrite the diary  *(1 day)*
- Keep reverse-chronological, one dated entry per experiment, emoji subheadings
  (Overview / Findings / Decision), tables, bold verdicts — the established style.
- Add the `Correction (2026-09-04)` blocks from T0.2 to the Aug/Sep entries (do not delete
  history).
- New entries for: the rename (T0.2), canonical reference (T0.4), year split (T1),
  ensembled re-eval (T3.1), v6 tail diagnosis (T3.2), the v5→v7 ablation (T3.3), `clip_frac`
  (T4.1), `seq_len` 288 (T4.2), autoregressive generation (T4.3), embedding ablation (T5.2),
  v8 retrain (T6.6), Gate B (T7).
- Fix the Aug 5 "ELBO metrics" sentence: it was a GMM `lower_bound_`, not an HMM quantity;
  K was never selected for the shipped model (point at T1.3).

### T8.4 — Update memory + skills notes  *(0.5 day)*
- `memory.md`: v7 is "best on intensity, fails storm geometry and has no interannual
  variability"; the mechanism is seasonal-phase not HMM; "smoothed 1-D mean feature is the
  change that mattered" → replace with the T3.3 verdict; add the holdout-vs-production model
  distinction; add "one canonical observed table / one canonical eval script" as project
  invariants. `skills.md`: log which skills actually helped (`dataviz` for the eval plots,
  `run` if used for SWMM).

### T8.5 — Regenerate the auto-report; final consistency sweep  *(0.5 day)*
- Re-run `evaluate_rainfall.py` → `comparison_report.md`; diff its narrative against its
  tables; grep the whole repo for "Markov", "HMM state", "unconditional" (for v3),
  "regresses to the mean", "P99 1.006" (point estimates without error bars). Each hit is
  either fixed or explicitly historical.

---

# Phase 9 — Gate C: TSTR (spec now, run only after Phase 7 passes)

**Goal:** the terminal claim. Per your staged-evaluation preference and the panel's, this is
specified now and executed last. Panel `ACCEPTANCE_CRITERIA.md` Gate C.

### T9.1 — Finalise the protocol  *(0.5 day, now)*
- Four PPO arms — R-full / R-scarce (1 real year) / S-only (synthetic, volume-matched) /
  R-scarce + S — identical hyperparameters and ≥ 5 seeds, all evaluated on the **held-out
  real 2008–2009** rainfall in SWMM-Astlingen (`flood-control/src/rl/train_ppo.py`,
  `evaluate.py`).
- Report episode return **and** domain KPIs (annual CSO volume, flooding volume), mean ± sd
  over seeds.
- **Non-negotiable safety check:** on the largest held-out storm, S-only must not do worse
  than the fixed baseline controller. Reported separately; never averaged into a mean return.

### T9.2 — Run it  *(2 weeks, only when unblocked)*
- Preconditions: Gate A Tier 3 (temporal) passes for the reported model **and** Gate B
  passes. Until both hold, Gate C numbers are not reportable.
- Success ladder: necessary = S-only beats R-scarce; headline = R-scarce + S within 10% of
  R-full; safety = the storm check above.

---

# Phase 10 — Path B: a real per-year weather state (optional upgrade)

Run this only if Phase 3's honest numbers plus the Path A scope limits are not enough for the
examiner, and time allows. It is the substantive answer to §2 rather than the rename.

### T10.1 — Fit the state model on real per-year statistics
- Fit the HMM (or an HSMM, which does not assume geometric dwell times) on the **unsmoothed**
  rolling statistics (e.g. 6-h and 24-h rolling mean, wet fraction, max) of **each individual
  year**, so the label carries genuine weather variability and a dry August and a wet August
  get different labels.
- The transition matrix is then estimated from thousands of real 5-min or hourly transitions,
  not a few hundred seasonal crossings — the Markov framing becomes legitimate.

### T10.2 — Re-condition, retrain, re-evaluate
- Feed the per-window state (or its posterior probabilities) into the U-Net conditioning.
- Retrain on the holdout split; ensemble; full Phase 2 eval; Gate B.
- The headline metric is **interannual variability**: does the synthetic record now contain
  wet winters and drought summers, measured as the spread of per-year volume / per-year
  annual maximum against the observed 10-year spread?

### T10.3 — Report Path A vs Path B side by side
- One table. If Path B closes interannual variability and storm geometry without losing the
  intensity match, it is the thesis result and Path A becomes the ablation.

---

# Critical path and parallelism

```
Phase 0  ──►  Phase 1  ──►  Phase 2  ──►  Phase 3  ──►  Phase 6 (T6.6 retrain)  ──►  Phase 8  ──►  Phase 9
                                    │                                              ▲
                                    ├──►  Phase 4 (seq_len / autoregressive) ──────┤
                                    ├──►  Phase 5 (embedding ablation) ────────────┤
                                    └──►  Phase 7 (Gate B, flood-control) ─────────┘   (also feeds T7.3 → Phase 8)
```

- **Serial spine:** 0 → 1 → 2 → 3 → 6 → 8 → 9. Nothing downstream is trustworthy until
  Phase 1 (holdout) and Phase 2 (ensembled eval) exist.
- **Parallel once Phase 2 is done:** Phase 4, Phase 5, Phase 7 are independent experiments
  that all feed the Phase 6 retrain and the Phase 8 rewrite.
- **Phase 7 can start as soon as Phase 1 gives it a holdout** — it only needs rainfall CSVs
  and SWMM, not the new checkpoints. Start the harness (T7.1) early.
- **Do not start Phase 9** until Phase 7 passes and Gate A Tier 3 passes for the reported
  model.

Rough calendar if one person, one cluster allocation: Phase 0 ≈ 1 week, Phase 1 ≈ 1 week,
Phase 2 ≈ 1.5 weeks, Phase 3 ≈ 1.5 weeks, Phases 4/5/7 in parallel ≈ 3–4 weeks, Phase 6 ≈ 1
week, Phase 8 ≈ 1 week, Phase 9 ≈ 2–3 weeks. Path B (Phase 10) adds ≈ 3–4 weeks.

---

# Exit criteria — when the thesis is "defensible"

Mapped to panel §10. Tick when the evidence exists, not when the task is "done".

- [ ] **§10.1** The mechanism is named correctly everywhere (seasonal-phase conditioning),
  with the "why not an HMM" caveat — **or** Path B makes the weather-state claim real.
- [ ] **§10.2** Every reported metric is `mean ± sd` over ≥ 30 realisations against an
  observed block-bootstrap band. No three-decimal single-draw ratios remain.
- [ ] **§10.3** Gate B exists: real vs synthetic routed through SWMM under a fixed
  controller, KPIs compared, and the Gate A thresholds cite the sensitivities it produced.
- [ ] **§10.4** Generation continuity is fixed or characterised: native ACF RMSE over lags
  1–288 and the periodogram are reported; the 100-min seam is confirmed or refuted with a
  band.
- [ ] **§10.5** One observed-statistics table, one eval script; the comparison report's
  narrative matches its tables; the radar composite is gone or motivated.
- [ ] **§10.6** The v5→v7 ablation changed one variable at a time; the "what mattered"
  verdict is backed by that table.
- [ ] **§10.7** The reported model is trained on 2000–2007 with labels re-fit on train years
  only; the degraded holdout numbers are the ones in the thesis.
- [ ] **§10.8** `LOSS_FUNCTION.md` and `ACCEPTANCE_CRITERIA.md` agree that extremes are
  solved for v7 and the open problem is storm geometry; the loss work is scoped accordingly.
- [ ] Bonus (not in §10, but a reviewer will ask): interannual variability is either
  reproduced (Path B) or named as a scope limit (Path A) in `ACCEPTANCE_CRITERIA.md`.
