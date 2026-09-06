# Acceptance Criteria for Synthetic Rainfall

When is generation "done"? Defined 2026-09-01. Supersedes the informal weighted
"RL Fitness Score" used for v3/v4, which was shown in the July audit to be gameable —
it ranked v3 (0.733) above v4 (0.622) while v3 halved the flood peaks.

**Governing principle.** The purpose of this data is to train valve-control agents in
SWMM-Astlingen. So the question is never "does this look like rain?" but **"does this
drive the hydraulic system through the same distribution of states the real rain does?"**
Metrics are ordered by how directly they bear on that.

---

## Gate A — Statistical screening (cheap, run on every version)

Runs from `scripts/compare_all_versions.py` against the 10-yr Astlingen gauge average.
All ratios are synthetic / real. **A version must pass every tier to proceed to Gate B.**

### Tier 1 — Water balance and intermittency
*Does the agent see the right amount of water, arriving as often?*

| Metric | Pass band | v7 | |
|---|---|---|---|
| Annual volume ratio | 0.95 – 1.05 | **1.000** | PASS |
| Zero fraction, absolute difference | <= 1.0 pp | **0.02 pp** | PASS |
| Mean dry-spell duration ratio | 0.90 – 1.10 | not computed | — |
| Mean wet-spell duration ratio | 0.90 – 1.10 | **0.854** | **FAIL** |

### Tier 2 — Intensity marginal
*Are the individual bursts the right size, including the rare ones?*

| Metric | Pass band | v7 | |
|---|---|---|---|
| P99 ratio | 0.90 – 1.10 | **1.006** | PASS |
| P99.9 ratio | 0.85 – 1.15 | **1.006** | PASS |
| Maximum ratio | 0.80 – 1.25 | **1.016** | PASS |
| JSD of wet-intensity histogram (hourly) | <= 0.05 | **0.0507** | MARGINAL |
| KS statistic (hourly) | <= 0.05 | **0.0381** | PASS |

### Tier 3 — Temporal structure
*This is the tier that valve control actually depends on. A controller decides when to
open based on how rain is expected to persist, not on any single 5-min value.*

| Metric | Pass band | v7 | |
|---|---|---|---|
| Lag-1 ACF, absolute difference (native) | <= 0.02 | **0.0059** | PASS |
| ACF RMSE over lags 1–24 h (hourly) | <= 0.05 | **0.0906** | **FAIL** |
| Mean storm duration ratio | 0.90 – 1.10 | **0.764** | **FAIL** |
| Storm count ratio | 0.90 – 1.10 | **1.298** | **FAIL** |
| Mean storm volume ratio | 0.90 – 1.10 | not computed | — |

### Tier 4 — Multi-scale extremes (flood relevance)
*Not yet computed. This is the tier that decides whether the data is safe for flood
training at all, and its absence is the biggest hole in the current benchmark.*

| Metric | Pass band |
|---|---|
| Daily maximum ratio | 0.85 – 1.15 |
| Annual maximum series, durations D in {15 min, 1 h, 3 h, 6 h, 24 h}, at return periods 1/2/5/10 yr | ratio within 0.80 – 1.20 at every (D, T) cell |
| Number of exceedances of the real 10-yr 1 h maximum | within a factor of 2 |

An IDF-style table over (duration, return period) is the standard hydrological
acceptance object and is what a reviewer will ask for. It also catches the specific
failure the June diary named the "Daily Max Paradox": matching the peak 10-min burst
while failing the 24-hour total.

### Tier 5 — Seasonality and diurnal cycle
| Metric | Pass band |
|---|---|
| Monthly volume profile, Pearson r vs real | >= 0.90 |
| Per-month volume ratio | 0.75 – 1.25 for all 12 |
| Diurnal profile, Pearson r vs real | >= 0.80 |

### Current verdict on v7

> **v7 delivers the right amount of water, in the right sized drops, arriving in the
> wrong shaped storms.** It passes Tiers 1 and 2 outright — the first version to match
> the intensity distribution across its full range including the 10-yr maximum — and
> fails Tier 3 on all three storm-geometry metrics. Tier 4 is uncomputed.

This is the same failure the June diary recorded for v4, unchanged in character but
smaller in magnitude. It is not a loss-function problem. `seq_len` is still **24**, and
because v5–v7 moved to 5-min data, the physical context window **shrank from 4 hours to
2 hours** — v3/v4 saw 24 x 10 min, v5–v7 see 24 x 5 min. Real mean storm duration is
62 minutes, so the model is being asked to learn storm persistence through a window barely
twice the length of a storm. "Experiment 2: expand the context horizon" has been on the
plan since July and has never been run. It should be run before any further loss tuning.

### What `compare_all_versions.py` still needs
Dry-spell durations, mean storm volume, daily maxima, the annual-maximum-series / IDF
table, monthly and diurnal correlation coefficients, and a machine-readable pass/fail
column per gate rather than a rendered table only.

---

## Gate B — Hydraulic response (SWMM, no learning)

**The highest-value thing not currently being done, and much cheaper than Gate C.**

Route real and synthetic rainfall through SWMM-Astlingen under a **fixed baseline
controller** (static valve settings, or the existing rule-based controller — no RL, no
training). Then compare the distributions of what the network actually did:

| KPI | Comparison |
|---|---|
| CSO event count per year | ratio 0.85 – 1.15 |
| Total CSO spill volume per year | ratio 0.85 – 1.15 |
| Peak spill rate distribution | KS <= 0.10 |
| Tank filling-level distribution | KS <= 0.10 |
| Flooding volume, annual maximum | ratio 0.80 – 1.20 |
| Fraction of simulated time in "system stressed" state | absolute difference <= 5 pp |

Why this earns its place: it is the first metric that is **non-linear in the rainfall**.
Every Gate A metric is a summary statistic of the rain itself. Sewer response is
threshold-driven — a storm either overtops a weir or it does not — so two rainfall series
with near-identical marginals and ACF can produce very different spill distributions. Gate B
answers "is this rain hydraulically equivalent?" directly, needs no RL training, and is
cheap enough to run on every candidate version.

Expect Gate B to fail for v7 on CSO counts specifically, since 30% too many storms of 76%
the correct duration should produce too many small spills and too few large ones. If it
does, that is a clean, physically interpretable confirmation of the Tier 3 failure and a
much stronger result to report than the statistics alone.

---

## Gate C — Train-Synthetic-Test-Real (terminal test)

Spec only; do not run until Gate A Tier 3 and Gate B pass. Uses
`flood-control/src/rl/train_ppo.py` and `evaluate.py`.

### Blocking prerequisite: there is currently no held-out real data

`regime_training/config_hmm.yaml` sets `train_csv == test_csv`, and
`hmm_105120_train.csv` covers **2000-01-01 to 2009-12-31** — the whole record. The file
named `hmm_105120_test.csv` covers 2009, which is a **subset of the training range**, not a
holdout. The generator has therefore seen every year of real rainfall in existence for this
site.

Any TSTR number computed against 2009 today is contaminated and cannot be reported. Before
Gate C:

1. Re-split by **year**: fit the seasonal-phase index and train the diffusion model on 2000–2007, hold out
   2008–2009 untouched.
2. Re-fit the seasonal-phase index on the training years only — the phase labels are themselves fitted
   quantities and leak just as the diffusion weights do.
3. Retrain the best configuration on the reduced set. Expect metrics to degrade; that
   degradation is the honest number.
4. Keep the full-record model as the *production* generator, clearly labelled as such,
   and use the holdout-trained model for *all reported evaluation*.

### Protocol

Four agents, identical PPO hyperparameters and seeds, each evaluated on the **held-out real
2008–2009 rainfall** in SWMM-Astlingen:

| Arm | Training rainfall | Role |
|---|---|---|
| **R-full** | all real training years | upper bound |
| **R-scarce** | 1 real year | the data-scarce baseline this whole project exists to beat |
| **S-only** | synthetic only, matched volume | tests whether synthetic data alone suffices |
| **R-scarce + S** | 1 real year + synthetic | **the headline claim** |

Report mean and standard deviation over >= 5 seeds, on both episode return and the domain
KPIs (annual CSO volume, flooding volume), since return is reward-shaping-dependent and
the KPIs are not.

### Success criteria

- **Necessary:** `S-only` beats `R-scarce` on held-out real rainfall. If synthetic data is
  not worth more than one real year, the generator has not earned its place.
- **Headline:** `R-scarce + S` reaches within 10% of `R-full`. That is the data-scarcity
  claim, and it is the result worth publishing.
- **Safety, and non-negotiable:** on the largest held-out storm, the `S-only` agent must not
  perform *worse than* a fixed baseline controller. An agent trained on rain whose extremes
  are too weak will have learned that storms are survivable and can act catastrophically when
  a real one arrives. Report this separately; it must never be averaged into a mean return.

---

## Why we no longer call this an HMM

Previous versions framed the conditioning model as an HMM and sequence generation as Markovian weather-state sampling. We have abandoned this framing for three concrete reasons (panel §2):
1. **The label is a periodic function of day-of-year identical in all 10 years:** The state assignment is a deterministic seasonal calendar pattern rather than a dynamic meteorological weather state.
2. **Autocorrelation and violation of emission independence:** The model is fitted on a 14-day moving-average smoothed series, making consecutive 5-minute intervals ~100% autocorrelated and voiding the HMM conditional emission-independence assumption ($P(X_t \mid S_t, X_{<t}) = P(X_t \mid S_t)$).
3. **The transition matrix is a calendar lookup, not a Markov chain:** The 4×4 block transition matrix is estimated from a few hundred near-deterministic seasonal boundary crossings with `smooth=1e-5` Laplace fill. Its off-diagonal entries merely reflect "which season follows which" in the annual calendar rather than true transitions of a 1st-order Markov chain.

All conditioning is therefore referred to as **seasonal-phase conditioning** using a discrete **seasonal-phase index**, assembled via **calendar-ordered block assembly**.

---

## Reporting rule

Two cross-resolution traps in the current benchmark table, both of which will mislead a
reader:

1. **Per-step metrics are not resolution-invariant.** `Mean_Wet`, `P99`, `P99_9` and `Max`
   are per-interval depths. v3/v4 are 10-min and v1/v2/v5/v6/v7 are 5-min, so v3's
   `Mean_Wet` of 0.130 mm/10-min is roughly equal to real, not double it — reading down that
   column as if it were comparable is wrong.
2. **Spell and storm counts scale with sampling interval too**, in the opposite direction.

Every cross-version claim must be made on **hourly or daily aggregates**, or restricted to
versions at a single resolution. Native-resolution columns should be kept in the table but
visibly grouped by resolution and never compared across the groups.
