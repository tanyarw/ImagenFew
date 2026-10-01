# Research Progress Diary & Changelog: Synthetic Rainfall Generation via Time-Series Diffusion

**Project:** Adapting ImagenFew and Time-Series Diffusion Models for Sparse Rainfall Data  
**Goal:** Create realistic synthetic precipitation datasets to train Reinforcement Learning (RL) agents for stormwater management, reservoir control, and flood regulation.

## October 1, 2026 — Heavy Rain Against the Official German Table (KOSTRA-DWD-2020)

### 🔍 Overview
KOSTRA is the German Weather Service's design-rainfall table (5 km grid; 5 minutes to 7 days;
1- to 100-year return periods; fitted to 1951–2020), the standard for sewer design. The Astlingen
rain is four 5-minute Erftverband gauge series (Astlingen benchmark; confirmed in
`flood-control/data/SWMM-Astlingen/*Astlingen_Erft*.txt`), and **this project's real series is
their exact average** (checked to 4e-16 mm). Script: `scripts/kostra_compare.py` (downloads cached
in `data/external/kostra/`, git-ignored; ETRS89-LAEA projection checked against the EPSG worked
example). Cells: those under the 36 Erftverband KOSTRA stations (exact gauge sites unpublished).
Return levels for our series: Gumbel on annual maxima, as in Gate A.

### 📊 Findings
| Duration | gauges / KOSTRA | 4-gauge average / single gauge | v14 / real average | storm-and-cell / real average |
| :--- | :---: | :---: | :---: | :---: |
| 5 min | 0.81–0.83 | **0.57–0.59** | 0.87–0.97 | 0.55–0.57 |
| 15 min | 0.95–1.05 | 0.67–0.68 | 0.93–0.96 | 0.61–0.66 |
| 1 h | 0.94–0.97 | 0.75–0.80 | 0.90–0.95 | 0.89–0.94 |
| 6 h | 0.94–0.98 | 0.79–0.89 | 0.85–0.86 | 0.88–0.92 |
| 24 h | 0.95–0.96 | 0.93–0.94 | **0.73–0.78** | 0.72–0.86 |

(ranges over the 2- and 10-year return periods)

* The real gauges match KOSTRA within about 5% from 15 minutes up: the record is consistent with
  German design practice.
* **Averaging four gauges removes about 40% of the 5-minute peak.** The models learn the average,
  so they are judged against it; v14 is within ~10% of it up to 1 hour and ~25% short at a day.
* **Consequence for the sewer test:** SWMM-Astlingen reads the four gauges separately. One
  generated average fed to all four gauges would have 5-minute peaks ~40% weaker than any real
  gauge. The generator needs to produce four gauges (jointly, or by spreading the average back out)
  before the agent-in-the-loop test.

**Verdict:** **data validated against the official table; a multi-site gap identified.**

---

## October 1, 2026 — Storm-and-Cell (Bartlett-Lewis) Baseline

### 🔍 Overview
The literature review (`docs/LITERATURE_RAINFALL_GENERATORS.md`) named the randomised
Bartlett-Lewis rectangular-pulse model as the standard competitor for 5-minute point rain. Built
from scratch on numpy/scipy (`scripts/baselines/run_bartlett_lewis.py`; no rainfall package):
storms arrive at random, each sets off rain cells; the variant used (Kaczmarska, Isham & Onof 2014)
makes short cells more intense. Fitted per calendar month to the 2000–2007 training years by
simulated method of moments (differential evolution, 100 simulated years per step; mean, CV, lag-1
autocorrelation, dry fraction at 5 min / 1 h / 6 h / 24 h, skewness at 5 min / 1 h), then each
month's long-run mean matched exactly. A first "classic" fit (fixed cell intensity, 30 simulated
years per step) over-fitted the random seed and is kept as `bartlett_lewis_classic`.

### 📊 Findings
| | real | v14 (diffusion) | storm-and-cell |
| :--- | :---: | :---: | :---: |
| Storms > 5h20 per year | 14.25 | 1.9 | **13.8** |
| Single-step showers | 38% | **38%** | 7% |
| Mean wet spell (min) | 32 | **30** | 67 |
| Strongest 5-min burst (mm) | 5.6 | **6.4** | 2.8 |
| Daily totals (Wasserstein; real 2-year floor 0.15–0.25) | — | 0.50 | **0.26** |
| Hourly ACF RMSE (real floor 0.02–0.04) | — | 0.05 | **0.034** |
| Monthly cycle r | — | −0.11 (Markov) / 0.97 (calendar, no bridges) | **0.88** |
| Classifier AUC, 5-h windows (real 0.48) | — | **0.78** | 0.83 |
| Gate A | 10/18 (held-out years) | **14/18** | 2/18 |

The seed-42 draw (635 mm/yr) is the driest of 22 seeds (others 676–754; the fitted monthly means
match the real ones exactly), so its volume result is pessimistic.

**Verdict:** **complementary failures.** Diffusion wins the 5-minute texture and short extremes;
the storm-and-cell model wins long storms, daily totals, persistence and seasons. Neither reaches
the real 24-hour extremes. This motivates a two-level design (storm-scale model for *when and how
long*, diffusion for the 5-minute detail), noted in `notebooks/research_evaluation.ipynb` §7.

---

## October 1, 2026 — Full Evaluation of Every Version (`notebooks/research_evaluation.ipynb`)

### 🔍 Overview
One notebook scores every series generated so far: v1–v15, the checkpoint control, the
no-bridge calendar run, the partial E2 runs and eight classical baselines (36 series). It uses
hydrology metrics (water balance, occurrence, intensity, storms, IDF, structure across scales,
seasons) and generative-model metrics (noise floor, precision / recall / density / coverage,
classifier two-sample tests, memorisation, train-on-synthetic-test-on-real, conditioning
fidelity, denoising error). Metrics live in `scripts/eval_suite.py` (reuses the Gate A
definitions; `python scripts/eval_suite.py --refresh`, ~4 min); the notebook reads its cache.
Versions sharing the seed-42 plan are compared year by year (10 paired years).

### 📊 Findings
| | Result | Evidence |
| :--- | :--- | :--- |
| Best model | **v14** (asinh): Gate A 14/18; vs v10 +9.4% volume, +2.8 min wet spells (10/10 years), +3.2 min storms (9/10); heavy-state rain −34% → −8% | paired years, conditioning table |
| Seasons | no-bridge calendar: monthly r **0.62 → 0.97** | §3.6 |
| vs classical baselines | diffusion wins storms (57 vs 20 min; real 62), extremes (12/15 vs 0/15 IDF cells), window coverage (0.86 vs 0.60), realism (classifier AUC 0.78 vs 0.94); copulas win hourly persistence (ACF RMSE 0.024–0.029 vs 0.051) and hourly forecasting utility | §5.4 |
| Noise floor | v14 inside real 2-year spread on intensity, dry fraction, volume; 2–3× outside on daily totals; far outside on long storms (0.13 vs 0.58–1.53) | §4.1 |
| Realism | coverage metrics saturate (v14 = real); a classifier still detects 5-h windows (AUC 0.78 vs 0.48 real): rain in 42% of windows vs 33% | §4.2 |
| Memorisation | none (nearest-training-window distance ratio 0.94–1.06; 0% too close) | §4.3 |
| Utility (TSTR) | v14 trains a forecaster to 97% (1 h) / 95% (6 h) of real data | §4.4 |
| Calibration | **two real held-out years pass only 10/18 Gate A checks**; the diurnal check fails for real rain | §2 |

**Verdict:** reportable improvements exist (transform, seasons, diffusion vs baselines,
methodology). The dominant failure is still the block-boundary cliff, which only E2 has moved.

### ✅ Decision
**One good idea (notebook §7): E2 without RePaint resampling**:
`RESAMPLE=1 sbatch scripts/submit_generation_v14_ctx.sh markov`, scored with
`scripts/score_context_generation.py --versions v14_ctx_u1 v14_ctx_u1_m1 v14_ctx_u1_m2 v14_ctx_u1_m3`
against the rules written in advance in `code_plan/NEXT_STEPS_AFTER_v14.md` §3. If it drifts or
runs dry: E3 (train the masked path).

---

## October 1, 2026 — E2 Context Chaining: Partial Results and Why It Ran Dry

### 🔍 Overview
E2 (`regime_training/generate_context.py`, K = 2 known columns, 10 RePaint passes, history noise
0.05) ran too slowly to finish (6–8 s/block on GPU, 28–40 h left) and was stopped. The progress
files hold 4 chains × 2.05 years (Markov) and 4 × 2.74 years (exact calendar). A CPU pilot (v14
checkpoint, same 11 days) then separated context from resampling
(`results/e2_pilot/e2_mechanism_pilot_11days.npz`).

### 📊 Findings
| | Wet steps | P(rain continues across a join) | Survival ratio at 5h20 | Volume ratio |
| :--- | :---: | :---: | :---: | :---: |
| v14 (independent blocks) | 9.0% | 0.11 (no-context pilot) | 0.13 | 1.01 |
| E2 full runs (K=2, U=10) | 4.8% | 0.63 | **0.73 / 0.79** | **0.49** |
| Pilot: context, no resampling (K=2, U=1), 8 chains | **10.1% ± 1.3** | **0.62** | — | — |
| Pilot: no context (K=0), 8 chains | 9.6% ± 1.1 | 0.11 | — | — |

* **Context removes the cliff**: storms continue across joins (0.62–0.63 vs 0.11; inside a block
  0.83–0.85), long storms 1.9 → 5.3 per year, 2.2% → 10.9% of rain, survival at 5h20 inside the
  real range (0.58–1.53). The pre-registered success targets (≥ 10/yr, ≥ 15%) are not met, and
  the must-not-break checks fail on volume and zero fraction.
* **The dry bias comes from resampling, not context.** Every chain-year of both runs is at
  0.40–0.66 of real volume; without resampling the pilot is as wet as v14. On 91%-zero data, each
  RePaint pass nudges the in-filled part towards the dry mode: resampling is mode-seeking.

**Verdict:** **mechanism confirmed, configuration wrong.** Next: the same run with `RESAMPLE=1`.

---

## October 1, 2026 — E1 Attention at 8×8: Result (negative)

### 🔍 Overview
v15 = v14 fine-tuned 200 more epochs with `attn_min_heads: 1`; v15_ctrl = the same fine-tune
without it. Same seed-42 plan, so the 10 generated years are paired.

### 📊 Findings
| v15 − v15_ctrl (10 paired years) | Mean [95% CI] | Years in that direction |
| :--- | :---: | :---: |
| Storm duration (min) | **−3.48 [−4.30, −2.67]** | shorter in 10/10 |
| Wet spell (min) | −1.70 [−2.60, −0.80] | shorter in 9/10 |
| Volume ratio | −0.035 [−0.057, −0.013] | lower in 8/10 |
| Storms > 5h20 per year | −0.7 [−1.6, +0.2] | n.s. |
| Best training loss | 0.2100 vs 0.2099 | identical |

Gate A: v15 11/18, v15_ctrl 14/18 (= v14). Heavy-state rain moves (state 2 −15%, state 3 +5%).
The control itself barely differs from v14 (volume −2%, hourly ACF RMSE +0.003).

**Verdict:** **negative.** The registered prediction ("better sub-block structure, cliff
unchanged") was wrong in its first half: full-resolution attention shortened storms in every
year without improving the fit. Do not carry attn_min_heads forward.

---

## October 1, 2026 — No-Bridge Calendar Run: Seasonal Cycle Restored

### 🔍 Overview
`MODES=calendar BRIDGE_BLOCKS=0 sbatch scripts/submit_generation_v14.sh` → v14_nobridge_cal,
testing the prediction written before the run: monthly correlation rises once the 39-day drift
from bridge blocks is gone.

### 📊 Findings
| | Monthly cycle r | Gate A | Storms > 5h20 /yr |
| :--- | :---: | :---: | :---: |
| v14_cal (bridges) | 0.62 | 13/18 | 1.7 |
| **v14_nobridge_cal** | **0.97** | **14/18** | 1.4 |
| seasonal copula AR(64), calendar | 0.94 | 7/18 | 0 |

**Verdict:** **prediction confirmed.** First diffusion version to pass the monthly-cycle check
(≥ 0.90). Use `--bridge_blocks 0` for all calendar runs from now on.

---

## September 30, 2026 — Next-Step Jobs Built: E1 (v15 + control), E2 (context chaining), Loose Ends (results pending)

### 🔍 Overview
Code and SLURM jobs for the next steps in `code_plan/NEXT_STEPS_AFTER_v14.md` (§5 steps 0
and 2), all built on the new v14 baseline and checked locally before submitting. The exact
commands and the predictions written before any results are at the top of that plan
("What to run now").

### 🛠️ What Changed
| File | Change |
| :--- | :--- |
| `models/ImagenFew/networks.py`, `models/ImagenFew/ImagenFew.py` | New opt-in `attn_min_heads` (default off: every existing config builds the same network). With 1, a requested attention layer gets at least 1 head instead of being silently dropped. |
| `regime_training/config_v15.yaml`, `config_v15_ctrl.yaml` | **E1** and its control: v14 fine-tuned for 200 more epochs from the v14 checkpoint, with / without `attn_min_heads: 1`. |
| `scripts/submit_job_v15.sh`, `submit_job_v15_ctrl.sh` | Fine-tune, then generate both assemblies with the production flags. |
| `regime_training/generate_context.py` | **E2**: each block generated with the previous block's last 16 steps as known context (RePaint with re-noised known pixels, 10 passes for σ ≥ 0.1, history noise 0.05). Labels by time on the production timeline; 4 chains per job; saves progress and resumes. |
| `scripts/submit_generation_v14_ctx.sh` | E2 job (`markov` or `calendar`); `K`, `RESAMPLE`, `MEMBERS`, `YEARS` settable. |
| `scripts/score_context_generation.py` | Scores E2 against the pre-registered §3 rules, including the year-by-year drift test against v14 (same plan seed). |
| `scripts/submit_generation_v14.sh` | Optional `MODES` and `BRIDGE_BLOCKS` (defaults unchanged); `BRIDGE_BLOCKS=0` writes `..._v14_nobridge*.csv`. |
| `scripts/submit_denoise_diagnostic.sh`, `scripts/submit_assembly_ablation.sh` | GPU versions of the two checks that were too slow for the laptop. |

### 📊 Findings (local checks)
**E1 starts exactly at v14.** Built through the training loader on the real v14 checkpoint,
the attention-enabled network gives **bit-identical** outputs to v14's generation weights at
σ = 0.01, 0.3, 2 and 80. Only the seven 8×8 blocks change (0 → 1 head, 30,016 new parameters);
the existing 4×4, 2×2 and bottleneck attention layers are untouched. The new output projection
receives gradients on the first training step.

**E2 does what it is for, with one risk.** CPU pilot, v14 checkpoint, 4 chains × 11 days,
same days and labels for both settings:

| | wet steps | P(wet \| wet before), inside a block | P(wet \| wet before), **across a join** |
| :--- | :---: | :---: | :---: |
| K = 0 (no context, same code) | 10.5% | 0.85 | **0.11** (19 cases) |
| K = 2 (E2 as registered) | 6.7% | 0.85 | **0.86** (14 cases) |
| v14, same 11 days | 10.7% | — | — |

Without context a join breaks the storm almost every time; with context rain continues across
the join as often as inside a block. **Risk:** the context chains were drier, and far more
variable from chain to chain (111–976 mm/yr). That is about 2 standard errors on this little
data, so it is flagged, not concluded. The §3 checks on the full run decide it.

**Also found (not changed): dropout.** `ImagenFew.py` never passes `dropout` to the network,
so every version so far trained with the network's default **0.10**, not the configs' `0.0`.
Left as is so E1 differs from v14 only in attention.

**Plumbing checked:** every job script passes a syntax check and a dry run with a stand-in
`python` (right configs, checkpoints, flags and output names). Resume was tested by
interrupting a context run (it restarted at block 4 and finished).

### ✅ Next
Run the seven jobs in "What to run now" (plan, top), sync, then score with
`gate_a_scorecard.py` and `score_context_generation.py`. Add one results entry per experiment.

---

## September 30, 2026 — Block-Assembly Ablation: Bridging × Crossfade (v14, paired)

### 🔍 Overview
Do the two hand-made joining rules matter? **Bridging** inserts one block with a 50/50
blended label (e.g. `[0, .5, 0, .5]`) at every state change; the model never saw blended
labels in training. **Crossfade** overlaps neighbouring blocks by 4 steps (8 at state
changes) and averages them. Because every block is generated independently, both rules can be
ablated **without retraining and without regenerating**: generate one set of blocks, then
join the *same* blocks four ways. Variants differ only in the joining, so small differences
are measurable.

New script: `scripts/ablate_block_assembly.py` (generation stage needs torch; the joining and
scoring stage is numpy, reusing the Gate A definitions). Run on CPU:
`python scripts/ablate_block_assembly.py --run v14 --years 2 --device cpu` → 3,566 blocks
(52 bridges), Markov plan seed 42 as in production. Outputs in `results/assembly_ablation/`.

### 📊 Findings
**1. Where the joins sit in the real 10-year outputs.** Rebuilding the exact block layout of
the v10/v12/v13/v14 CSVs from the plan code (same seed) shows a clear signature, which also
confirms the layout: **seams are 7% of steps**, and there the rain is **~37% more often wet
but ~30% weaker** than in block interiors (v14: wet 0.121 vs 0.088, mean wet 0.053 vs
0.078 mm). The crossfade turns "storm next to dry" into drizzle. **Storm ends fall inside
seams 2.5× more often than chance** (18% of ends in 7% of steps). v12, with no labels and no
bridges, shows the same signature.

**2. Paired ablation** (same blocks, 2 years):

| Metric | real | **A** bridge + xfade (production) | **B** no bridge + xfade | **C** bridge + hard join | **D** no bridge + hard join |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Zero fraction % | 90.80 | 91.37 | 91.41 | 91.60 | 91.62 |
| Wet spell (min) | 32.0 | 28.5 | 28.5 | 27.8 | 27.8 |
| Storms / yr | 697 | 731 | 730 | 716 | 712 |
| Storm duration (min) | 61.8 | 54.6 | 54.5 | 54.0 | 54.2 |
| Storms > 320 min / yr | 14.25 | 1.0 | 1.0 | 1.0 | 1.0 |
| Rain in storms > 320 min | 21.4% | 2.4% | 2.4% | 1.7% | 1.7% |
| P99 wet (mm) | 0.588 | 0.629 | 0.627 | 0.646 | 0.640 |
| P99.9 wet (mm) | 1.598 | 1.546 | 1.543 | 1.583 | 1.559 |
| Lag-1 ACF | 0.852 | 0.847 | 0.848 | 0.844 | 0.844 |
| Hourly ACF RMSE 1–24 h | — | 0.059 | 0.059 | 0.058 | 0.061 |

* **Bridging: no measurable effect** (A vs B, C vs D). The blended label, although never seen
  in training, does no visible harm at 1.5% of blocks.
* **Crossfade: small, two-sided effects** (A vs C). It adds a little drizzle (+0.2 pp wet),
  lengthens spells (+3%), lets a few storms survive the seam (rain in long storms 1.7% →
  2.4%), and softens peaks (P99 −3%). Every effect is 1–3%, far smaller than the gaps to the
  real record.
* **Neither touches the cliff:** 1.0 storm per year longer than 320 min under every variant
  (real 14.25).

**3. Bridges are also what makes the calendar drift.** Calendar-mode seasons fall
**39 days behind by year 10 with `--bridge_blocks 1`**, and **3 days ahead with
`--bridge_blocks 0`** (the flag already exists; no code change).

### ✅ Decision
1. **No cluster ablation of bridging or crossfade is needed.** Both are second-order
   heuristics at the seams; the cliff comes from the blocks being independent, which only a
   change to generation (E2) can address. This paired result is the thesis's robustness check
   that the conclusions do not depend on the joining rules. For tighter numbers, the same
   script runs 10 years on a GPU in minutes (`--years 10`).
2. **Drop bridges from new calendar runs** (`--bridge_blocks 0`): no measurable cost, and it
   removes the calendar drift. Cheap test with a prediction written in advance: regenerate
   v14_cal with `--bridge_blocks 0`; if drift was hurting seasonality, monthly Pearson r
   (0.597 now) should rise.
3. **Use variants A and D as E2's baselines.** E2 replaces the crossfade with real context,
   so it should beat both the production join (A) and the naive join (D) on the cliff.

---

## September 30, 2026 — v13 / v14 Transform Ablation: Results

### 🔍 Overview
Results for the Sept 27 setup. Three models, identical except for the elementwise map from
rainfall into model space: **v10** StandardScaler, **v13** $\log(1+x)$, **v14**
$\operatorname{asinh}(x/0.035)$, each standardised afterwards. Scored with
`python scripts/gate_a_scorecard.py --reference train --versions v8 v8_cal v10 v10_cal v11 v12 v13 v13_cal v14 v14_cal --json results/reference/gate_a_train_v8_v14.json`.
v13/v14 were generated with blocks converted to mm before stitching (`136f723`); for v10 that
change is a no-op (verified on the checkpoint, Sept 27).

### 📊 Findings
**Gate A tally:** v10 **11/18** → v13 **14/18** → v14 **14/18** (calendar assembly: 8 → 12 → 13).

| Metric (band) | v10 | v13 | v14 |
| :--- | :---: | :---: | :---: |
| Volume ratio (0.95–1.05) | 0.915 ✗ | 0.952 | **1.009** |
| Zero-fraction gap, pp (≤ 1.0) | 0.96 | 0.50 | **0.21** |
| Wet-spell ratio (0.90–1.10) | 0.866 ✗ | 0.925 | **0.953** |
| Storm duration ratio (0.90–1.10) | 0.875 ✗ | 0.919 | **0.927** |
| Storm volume ratio (0.90–1.10) | 0.908 | 0.931 | **0.959** |
| P99.9 wet ratio (0.85–1.15) | 0.900 | 0.958 | 0.956 |
| Max 5-min ratio (0.80–1.25) | 0.830 | 1.122 | 1.144 |
| Daily max ratio (0.85–1.15) | 0.720 ✗ | 0.862 | **0.920** |
| IDF cells in band (15/15) | 7 | 9 | **12** |
| Hourly ACF RMSE 1–24 h (≤ 0.05) | **0.049** | 0.054 ✗ | 0.050 ✗ (at the edge) |
| Storm count ratio (0.90–1.10) | 1.004 | 1.021 | 1.052 |
| Annual volume, z vs real years | −3.3 | −1.8 | **+0.4** |

The calendar versions move the same way (a second, independently sampled realisation of each
model), so the direction is not a single-draw accident.

**Mechanism: the transform brings back drizzle.** The intensity mix *within* wet steps barely
changes (share of wet steps below 0.1 mm: real 79.7%, v10 79.3%, v13 79.5%, v14 78.8%). What
changes is how many very light steps exist at all. Steps of 0.005–0.02 mm are **3.14%** of real
steps, **2.76%** in v10, 3.00% in v13 and **3.07%** in v14. These are the first and last steps
of storms. Under StandardScaler light rain gets only 1.7% of the model's value range, so the
model cannot place storm edges above the 0.005 mm threshold, and storms come out short and
dry. That one effect accounts for the wet fraction, wet-spell length, storm duration and
volume improvements together.

**Pre-registered expectation (Sept 28, item 5): half right.**

| Survival ratio P(dur > k), generated / real | 180 min | 240 min | **320 min (block)** | storms > 320 min /yr | rain in them |
| :--- | :---: | :---: | :---: | :---: | :---: |
| real | — | — | — | 14.25 | 21.4% |
| v10 | 0.74 | 0.63 | **0.08** | 1.10 | 1.6% |
| v13 | 0.84 | 0.73 | **0.12** | 1.70 | 2.2% |
| v14 | 0.89 | 0.77 | **0.13** | 1.90 | 2.2% |

The cliff at the block boundary did **not** move, as predicted. But the prediction placed the
movement in the intensity marginal; the largest movement is actually in **timing below the
wall** (the drizzle edges above).

**Denoising diagnostic** (`results/transform_diagnostics/denoise_test/`, held-out years, CPU):
light rain improves (bias −0.0042 → −0.0015 mm, volume error −13% → **−5%**). Heavy rain looks
much worse for v14 (RMSE 0.059 → 0.211 mm, volume error −1% → −11%). **This is the price of the
transform, not a worse model.** mm of rain per 0.01 model units:

| Rain rate | 0.01 mm | 0.2 mm | 1 mm | 3 mm |
| :--- | :---: | :---: | :---: | :---: |
| v10 standard | 0.00046 | 0.00046 | 0.00046 | 0.00046 |
| v13 log1p | 0.00035 | 0.00042 | 0.00069 | 0.00138 |
| v14 asinh | **0.00015** | 0.00081 | **0.00401** | **0.01204** |

At the lowest noise level the heavy-band error ratio v14/v10 is 7.1×, close to the 8.7× slope
ratio at 1 mm, so in its own space v14 is about as accurate as v10: it has moved its
precision from heavy rain to light rain. The generated tails still pass (P99.9, max, 12/15
IDF cells), but each heavy value is **~9× coarser at 1 mm and ~26× at 3 mm**. The σ-averaged
summary also overstates v14's heavy-rain bias: at high noise the denoiser's mean is taken in
compressed space and reads low once expanded, which does not happen to final samples.
The held-out years contain only 7 extreme steps. The training-years check (`--split train`,
40 extreme steps) was stopped by a local time limit before finishing and **still needs running
on the cluster**.

**Still failing for v14:** hourly ACF RMSE (0.050, at the edge), 24-h IDF (0.62–0.78; this is
the cliff), monthly seasonality (assembly), diurnal cycle (no version passes).

**Caveats:** one training run per transform (no training-seed spread). Every run's best epoch
is 460–500 of 500, so none has converged. One generation seed per assembly mode.

### ✅ Decision
1. **v14 is the new baseline.** It ties v13 on the tally but is better on volume, spells,
   daily max and IDF (12 vs 9 cells). E1/E2 build on v14.
2. **Keep v13 as the tail-precision reference.** If the downstream flood test turns out to
   need precise peak intensities, the next transform step is an asinh scale sweep between the
   two (s ≈ 0.1–0.3), not a new transform family.
3. **Stop transform work here.** The remaining failures are dominated by the block-boundary
   cliff, which no transform can fix.

---

## September 30, 2026 — v11 / v12 Conditioning Ablation: Results

### 🔍 Overview
v11 is v8 without the 4-state seasonal-phase label (seq_len 24); v12 is v10 without it
(seq_len 64). Both were generated with `--assembly_mode unconditional`: no labels, so no bridge
blocks, 4-step crossfade everywhere. Question: what does the label actually buy?

### 📊 Findings
| Metric | v8 (label, L=24) | v11 (no label) | v10 (label, L=64) | v12 (no label) |
| :--- | :---: | :---: | :---: | :---: |
| Gate A tally | 5/18 | 4/18 | **11/18** | 8/18 |
| Survival ratio at k = L | 0.12 | 0.12 | 0.08 | 0.08 |
| Storms > 320 min /yr (real 14.25) | 0.10 | 0.00 | 1.10 | 1.10 |
| IDF cells in band | 6 | **0** | 7 | **12** |
| Max 5-min ratio | 0.79 | **0.37** | 0.83 | 0.98 |
| Storm count ratio | 1.21 | 1.26 | 1.00 | 1.00 |
| Monthly Pearson r (Markov / uncond.) | 0.44 | −0.23 | −0.11 | 0.24 |

1. **Pre-registered prediction (Sept 28, item 4): confirmed.** Removing the label leaves the
   cliff in the same place and at the same depth (0.12 at L = 24, 0.08 at L = 64; v12's
   long-storm numbers are identical to v10's). The label carries no "storm in progress"
   information.
2. **At L = 64 the label buys nothing measurable.** v12's three extra failures are all at a
   band edge (zero-fraction gap 1.01 vs ≤ 1.0, hourly JSD 0.051 vs ≤ 0.05, ACF RMSE 0.052 vs
   ≤ 0.05), and v12 passes *more* IDF cells (12 vs 7) with a heavier tail. The IDF gain has
   no identified mechanism (the Markov plan's state mix matches training) and is one
   realisation.
3. **At L = 24 dropping the label hurts badly.** v11 caps out at 1.86 ± 0.15 mm per 5 min in
   every year (real 3.24 ± 1.12) and fails every IDF cell. With only 2 hours of context the
   block cannot tell a heavy regime from a light one; the label was supplying that.
4. **The label does not produce seasonality under Markov assembly** (v10 r = −0.11). Only
   calendar assembly does (v10_cal 0.65).

**Verdict:** the seasonal-phase label is **not a lever for long-range structure**, and at the
current block length it does little else.

### ✅ Decision
Keep conditioning in the v14 line: calendar seasonality needs it, and E2 will need an
external signal that is re-imposed every block to resist drift. Do not claim the v12 IDF
advantage without a second seed.

---

## September 28, 2026 — Architecture Audit: Does the UNet Attend? And Why Long Storms Vanish

### 🔍 Overview
Two questions, one of which turned out to answer the other. **(1) Does this fork's UNet
actually use self-attention, and over what?** Nobody had checked. **(2) Why does the model
reproduce short-range rain structure so well and long-range structure so badly?**

Full write-up with every derivation: [`docs/ATTENTION_AND_RECEPTIVE_FIELD_AUDIT.md`](../../docs/ATTENTION_AND_RECEPTIVE_FIELD_AUDIT.md).
Figure: `results/transform_diagnostics/storm_duration_cliff.png`
(regenerate with `python scripts/diagnose_storm_duration_cliff.py`).

### 📊 Findings — Part 1: the attention is mostly not there

The backbone is **`DhariwalUNet`**, not `SongUNet` — `EDMPrecond` defaults to it and
`ImagenFew.py` never overrides `model_type`. Verified against the v10 checkpoint
(`aebe363f`) by listing which blocks actually own `qkv`/`proj` tensors:

| Level | Resolution | Channels | In `attn_resolution: [8,4,2]`? | Heads | Attention runs? |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | 8×8 (64 tokens) | 32 | **yes** | `32 // 64 =` **0** | **No — silently dropped** |
| 1 | 4×4 (16 tokens) | 64 | yes | 1 | Yes |
| 2 | 2×2 (4 tokens) | 64 | yes | 1 | Yes |
| 3 | 1×1 (1 token) | 128 | no (bottleneck hardcodes it) | 2 | Instantiated, **no-op** |

Two defects, both invisible without looking in the checkpoint:

* **The res-8 request is silently dropped.** `num_heads = out_channels // channels_per_head`.
  `channels_per_head = 64` is inherited from Dhariwal & Nichol's 192-channel ImageNet config;
  with `unet_channels: 32` it truncates to **0**, which is falsy, so the attention modules are
  never constructed. **`attn_resolution: [8,4,2]` and `[4,2]` produce a bit-identical model**,
  and have done since v8.
* **The 1×1 bottleneck attention is a no-op.** Softmax over a single key gives weights of
  exactly `1.0`; confirmed numerically that `max|a − v| = 0.0`. It is a channel mixer, not
  attention.

**Verdict:** the model **never attends across a block at full resolution**. Its only real
attention is over 16 and 4 pooled tokens, and those tokens are not even contiguous in time.

Related representation finding: because `delay == embedding == 8`, the delay embedding is a
pure reshape, and one 3×3 conv at full resolution reaches time lags **{0, ±1, ±7, ±8, ±9}** —
it **cannot reach lags 2–6 (10–30 min)**, and three stacked blocks still leave holes at lags
4, 12, 20. The failing Gate A metrics (mean wet spell 30.4 min, storm duration 61.0 min) sit
in that band. Suggestive, not yet proven causal.

### 📊 Findings — Part 2: the storm-duration cliff (this is the big one)

Measured on the released CSVs with the **canonical** storm definition (contiguous wet runs
≥ 3 steps, as in `gate_a_scorecard.py`). Sanity check passed: this reproduces the storm
figures already in `docs/FINETUNING_ANALYSIS.md` §1 exactly (v10 storm-count ratio 1.004,
duration ratio 0.875).

Survival ratio $P(\text{dur} > k)_\text{gen} / P(\text{dur} > k)_\text{real}$:

| $k$ (min) | 15 | 30 | 60 | 90 | 120 | 150 | 180 | 240 | 320 | 400 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| real, absolute | 81.1% | 50.8% | 25.8% | 15.7% | 11.3% | 8.5% | 6.2% | 3.8% | 2.0% | 1.4% |
| **v8** (L=24) | 1.01 | 0.97 | 0.97 | 0.92 | **0.12** | 0.07 | 0.05 | 0.01 | 0.01 | 0.01 |
| **v9** (L=36) | 0.92 | 0.81 | 0.82 | 0.81 | 0.75 | 0.63 | **0.09** | 0.05 | 0.01 | 0.00 |
| **v10** (L=64) | 1.00 | 0.96 | 0.96 | 0.98 | 0.92 | 0.83 | 0.74 | 0.63 | **0.08** | 0.03 |

**Every version tracks the real curve to within a few percent up to ~¾ of its own block
length, then falls off a cliff by ~10× at exactly $k = L$.** Three different block lengths,
three cliffs, each in its own place. v7 (L=24) behaves like v8, as expected.

This rules out the alternatives by construction: capacity, loss, transform, conditioning and
checkpoint are all constant across pairs that straddle a cliff, and none of them predicts a
discontinuity that *moves with `seq_len`*.

**Cause:** block independence. `generate_hmm_v1.py` passes an **all-zero known-pixel mask**,
so every block is drawn with no knowledge of its neighbours. The only inter-block channel is
the 4-level seasonal-phase label, which changes on a multi-day scale and says nothing about
whether a storm is in progress. A storm outlasting one block needs two consecutive blocks to
be heavy *by coincidence*.

I tested and **rejected** the obvious rival explanation: seam truncation by the crossfade.
Forcing a hard dry step every 320 steps on the *real* record changes storm count by 0.0% and
mean duration by −0.4%. The 2026-09-27 stitching fix was correct and is not implicated.

**Cost of this:** storms > 320 min run **14.25/yr real vs 1.10/yr in v10** (ratio 0.077), and
they carry **21.4% of real rainfall volume vs 1.6% generated**. A fifth of the water is
missing from the events that matter most for flooding — while storm *count* (1.004) and
*mean* duration (0.875) both look healthy. The failure hides entirely in the tail.

**The fix is already built and switched off.** `ImagenFew.forward(x, mask, …)` implements
masked denoising and `DiffusionProcess.impute()` already re-imposes known pixels every solver
step — the RePaint rule, correctly written. But `train_regime.py` sets
`x_ts_mask = torch.zeros_like(x_ts)` on every batch and at 8×8 there is no padding, so across
v1–v14 **the model has never once seen a known pixel.**

### 📚 Literature (survey in the audit doc §4, non-weather included)
* **Diffusion Forcing** (NeurIPS 2024) — per-token noise levels; roll out past training length
  by conditioning on *slightly noisy* history. The key reference for this failure mode.
* **RePaint** (CVPR 2022) — condition an *unconditionally trained* model on known pixels with
  no retraining, via ~10 forward/backward resamplings. Our `impute()` is RePaint minus the
  resampling loop.
* **Lazy Diffusion** (2512.09572) — the counterweight: autoregressive diffusion rollout
  suffers **spectral collapse**. A 10-year rollout is ~16,400 blocks. Must be measured.
* **MIDiff** (Allerton 2026, mobile app usage) — closest non-weather analogue: sparse,
  imbalanced traces → images → UNet with **Triple Attention factorised along the imaging
  transform's own axes**. Best external argument that our attention placement is wrong.
  Its baseline **ZITS** splits zero-inflated generation into Bernoulli occurrence + magnitude.
* **t-EDM** (ICLR 2025) — Student-t prior for heavy tails, one scalar, validated on weather.
  Orthogonal to v13/v14: transform changes the *data map*, t-EDM changes the *noise prior*.
* **simple diffusion** (ICML 2023) — attention belongs at low resolution, but their stage has
  **256 tokens**; ours has 16. "Attention at low resolution" vs "attention at no resolution".

### ✅ Decision

> **Action plan:** [`code_plan/NEXT_STEPS_AFTER_v14.md`](../../code_plan/NEXT_STEPS_AFTER_v14.md)
> — written in plain terms, to be picked up **after v13/v14 finish**. It carries the gate,
> the exact file/line changes, and the pass/fail bands below.

1. **Reinterpret the context-window result.** v8 → v9 → v10 did **not** teach the model
   longer-range structure; it **moved the wall**. Everything below the cliff was already
   near-perfect at L = 24. Noted in `docs/FINETUNING_ANALYSIS.md` §3.
2. **Next experiments, in order** (audit §5): **E1** turn on res-8 attention — free, and
   `proj` is zero-initialised so the fine-tune starts bit-identical to v10; **E2** condition
   each block on the previous block's columns via the existing `impute()` path — inference
   only, no retraining, plus RePaint resampling and noisy history; **E3** train the masked
   path (one-line change to `x_ts_mask`); **E4** occurrence/intensity two-field split;
   **E5** t-EDM.
3. **Pre-registered targets for E2:** storms > 320 min from **1.10 → 14.25/yr**, volume share
   **1.6% → 21.4%**, with no loss in the wet-run histogram or zero fraction. Failure signal:
   annual volume or hourly ACF drifting monotonically across the 10 generated years.
4. **v11/v12 generation is now a decisive test, not just an ablation.** Prediction: the
   seasonal label never carried "storm in progress" information, so removing it should leave
   the cliff *in the same place and at similar depth* — v12 at k=64 ≈ 0.08, v11 at k=24 ≈ 0.12.
   Markedly deeper cliffs would falsify the stated mechanism.
5. **Expectation registered for v13/v14 before results arrive:** a transform reallocates value
   range across wet intensities and **cannot move a wall at the block boundary**. Expect
   movement in the intensity marginal and extreme quantiles, and the storm-duration cliff to
   stay exactly where it is. The ablation should not be credited or blamed for that.

---

## September 27, 2026 — Transform Ablation Re-based on v10: v13 (log1p) vs v14 (asinh) (results pending)

> **Results (2026-09-30):** see the Sept 30 entry "v13 / v14 Transform Ablation: Results".
> v13 and v14 both reach 14/18 (v10: 11/18); v14 becomes the new baseline.

### 🔍 Overview
The transform ablation now starts from **v10**, the best model so far (Gate A 11/18), instead
of the unconditional v12. Two runs, each **identical to v10** (seq_len 64, 4 HMM-state classes,
`ImagenFew_64.ckpt`, 500 epochs, same loss, same generation settings). The only change is the
elementwise map from rainfall to model space, applied before the delay-embedding image is
formed:

| Run | Mapping | Largest value on 2000–2007 |
| :--- | :--- | :---: |
| **v10** (baseline) | $z = (x - \mu)/\sigma$ (StandardScaler) | $z \approx 120$ |
| **v13** | $z = (\log(1 + x) - \mu)/\sigma$ | $z \approx 54$ |
| **v14** | $z = (\operatorname{asinh}(x/s) - \mu)/\sigma$, $s = 0.035$ mm/5min | $z \approx 14$ |

All three are standardised to mean 0 / std 1, so the EDM settings ($\sigma_\text{data} = 0.5$)
see the same overall scale.

**Question:** does squeezing the storm peaks alone help (v13), or do the gains also need
light rain to get more of the model's value range (v14)? With $x$ in mm/5min,
$\log(1+x)$ is nearly linear below ~1 mm, so v13 compresses the tail but leaves light rain
almost as cramped as v10. asinh with $s = 0.035$ does both. The pair therefore separates
the two effects.

### 🛠️ What Changed
| File | Change |
| :--- | :--- |
| `regime_training/transforms.py` | New `Log1pScaler`, selected with `data_transform: log1p`. |
| `regime_training/config_v13.yaml` | **Rewritten.** Now `config_v10.yaml` + `data_transform: log1p` (was v12 + asinh, see the Sept 24 note). |
| `regime_training/config_v14.yaml` | **New.** `config_v10.yaml` + `data_transform: asinh`, `asinh_scale: 0.035`. |
| `scripts/submit_v13.sh`, `scripts/submit_v14.sh` | One SLURM job each: train as v10 → generate 10 y with both assemblies (`_v13.csv` + `_v13_cal.csv`, same for v14). The diagnostic is no longer in the job; it runs once after both finish. |
| `scripts/diagnose_denoising_error.py` | **Bug fix** for conditional models (below). |
| `scripts/gate_a_scorecard.py` | Default versions now include `v13_cal`, `v14`, `v14_cal`. |
| `regime_training/generate_hmm_v1.py` | **Stitching fix:** each block is converted to mm *before* the crossfade (was: crossfade in model space, then convert). Same output for v10, v11, v12 (below). |
| `notebooks/bridging_and_stitching_tutorial.ipynb` | **New.** Walkthrough of bridging and stitching on the v10 checkpoint, with the checks below. |

Checked locally: a key-by-key comparison shows the v13/v14 configs differ from v10 only in
the transform keys and `run_dir`; both scalers invert exactly and survive pickling
(`scaler.pkl`).

### 📊 Findings So Far (data only, no training yet)
Share of the model's value range each rain band gets on 2000–2007 (same method as
`transform_range_shares.csv`):

| Band (mm/5min) | v10 standard | v13 log1p | v14 asinh |
| :--- | :---: | :---: | :---: |
| light (0.005–0.1) | **1.7%** | 4.8% | 28.3% |
| moderate (0.1–0.5) | 7.2% | 16.5% | 27.5% |
| heavy (0.5–2) | 27.0% | 36.9% | 24.0% |
| extreme (> 2) | **64.0%** | 41.6% | 17.7% |

**Diagnostic bug (found while re-basing on v10).** `diagnose_denoising_error.py` called the
network with no class label. For a conditional model this does not raise an error:
`networks.py` quietly substitutes an **all-zero label**, which the model never saw in
training. The fix gives each window its majority HMM state, the same rule `RegimeDataset`
uses. Effect on v10 (held-out years, training-weighted RMSE in mm/5min, CPU run):

| Band | blank label (old) | true label (fixed) | old / fixed |
| :--- | :---: | :---: | :---: |
| dry | 0.0048 | 0.0023 | **2.1×** |
| light | 0.0166 | 0.0163 | 1.0× |
| moderate | 0.0358 | 0.0339 | 1.1× |
| heavy | 0.0717 | 0.0588 | 1.2× |
| extreme | 0.1587 | 0.0830 | **1.9×** |

**Verdict:** the old script would have **overstated v10's error about 2× in the dry and
extreme bands**. Unconditional runs (v11, v12) were never affected. The Sept 24 caveat still
applies: the held-out years have only 7 extreme steps, so also check `--split train`.

**How blocks are joined (`notebooks/bridging_and_stitching_tutorial.ipynb`).** Every 64-step
block is generated independently (the sampler is given no known values from neighbours).
**Bridging happens before generation**: one extra block with a blended label (e.g. half state 1,
half state 3) is added at each state change. **Stitching happens after generation**: neighbours
are crossfaded over 4 steps, or 8 at state changes. About **7%** of all steps are such blends.

**Stitching bug for non-linear transforms (fixed before v13/v14 generate).** Stitching used to
average blocks in model space. That is harmless for StandardScaler, but for log1p/asinh an
average in the squashed space is less rain once converted back (a 50/50 blend of 0 and 1 mm gives
0.50 mm standard, 0.41 mm log1p, **0.13 mm asinh**). On shuffled real 64-step blocks this cost
**−0.19% (log1p) and −1.37% (asinh) of total rain**, a bias against v13/v14 that has nothing to
do with the models. Each block is now converted to mm before stitching. Old vs new code on the
v10 checkpoint (0.1 y, calendar and Markov, seed 42): same wet steps, same volume, largest
difference 1.4e-7 mm. So **v10 (and v11/v12) do not need regenerating**.

**Open issue, not fixed: calendar drift.** In calendar mode each bridge block adds time the
calendar index does not count, so the seasons fall **4.0 days behind per year, 39 days by year
10**. This may weaken the Tier 5 seasonality score of every `_cal` version. It is the same for
v10/v13/v14, so the ablation stays fair.

### ✅ Next
1. `sbatch scripts/submit_v13.sh` and `sbatch scripts/submit_v14.sh`
2. Once both finish (add `--device cpu` to run locally, ~2 min per model):
   `python scripts/diagnose_denoising_error.py --run v10=logs/ImagenFew/Rainfall_Regime/aebe363f --run v13=logs/ImagenFew/Rainfall_Regime/v13 --run v14=logs/ImagenFew/Rainfall_Regime/v14 --config regime_training/config_v14.yaml`, then again with `--split train`
3. `python scripts/gate_a_scorecard.py --reference train --versions v10 v10_cal v13 v13_cal v14 v14_cal`
4. `python scripts/visualize_transform.py --generated v10=results/generated_data/rainfall_synthetic_10y_v10.csv v13=results/generated_data/rainfall_synthetic_10y_v13.csv v14=results/generated_data/rainfall_synthetic_10y_v14.csv`
5. Add a results entry with the verdict.

---

## September 24, 2026 — asinh Data Transform: Setup for the v12 vs v13 Ablation (results pending)

> **Superseded (2026-09-27):** the v12-based v13 run planned below was replaced before any
> results were recorded. The ablation was re-based on v10 (the best model): **v13 is now
> log1p and asinh moved to v14**, both on v10's config. See the Sept 27 entry. The
> transform code added here (`transforms.py`, dataset and training changes) is unchanged
> and still used.

### 🔍 Overview
First step of the transform work: an **opt-in asinh transform** for mapping rainfall into model
space, set up as a single-variable ablation of v12. The only difference between the runs is
how rainfall becomes pixels:

| Run | Mapping | Largest value on 2000–2007 |
| :--- | :--- | :---: |
| **v12** (before) | $z = (x - \mu)/\sigma$ (StandardScaler) | $z \approx 120$ |
| **v13** (after) | $z = (\operatorname{asinh}(x/s) - \mu)/\sigma$, $s = 0.035$ mm/5min | $z \approx 14$ |

Both are standardised to mean 0 / std 1, so the EDM settings see the same overall scale.
$s = 0.035$ is the median wet step (≥ 0.005 mm) of the 2000–2007 training split.

**Hypothesis:** asinh gives more realistic heavy rain (wet P99 / P99.9, storm peaks)
without hurting dry/wet structure. **Expected:** partial success. The tail and light rain
improve, but the dry/wet problem is unchanged (91% of steps are still one value under both
transforms).

### 🛠️ What Changed
| File | Change |
| :--- | :--- |
| `regime_training/transforms.py` | **New.** `AsinhScaler` (sklearn-style fit / transform / inverse_transform) and `make_scaler()`. No torch. |
| `regime_training/regime_dataset.py` | New args `transform` (default `"standard"`) and `asinh_scale`. Default behaviour identical to before. |
| `regime_training/train_regime.py` | Reads `data_transform` / `asinh_scale` from the config (default `standard`), logs them. |
| `regime_training/config_v13.yaml` | **New.** Exact copy of `config_v12.yaml` + `data_transform: asinh`, `asinh_scale: 0.035`. |
| `scripts/submit_v13.sh` | **New.** One SLURM job: train → generate 10 y → denoising diagnostic. Own run_id, so v12 is not overwritten. |
| `scripts/visualize_transform.py` | **New.** Local. What the model sees under each transform + generated-vs-real tail plots. |
| `scripts/diagnose_denoising_error.py` | **New.** Cluster. Denoising error by true rain band, measured in mm, before vs after. |

Generation needs **no code change**: the pickled `AsinhScaler` does the inverse transform.

### 📊 Findings So Far (data only, no training yet)
Share of the model's value range each rain band gets (`results/transform_diagnostics/transform_range_shares.csv`):

| Band (mm/5min) | standard | asinh |
| :--- | :---: | :---: |
| light (0.005–0.1) | **2%** | 28% |
| moderate (0.1–0.5) | 7% | 27% |
| heavy (0.5–2) | 27% | 24% |
| extreme (> 2) | **64%** | 18% |

Under StandardScaler, light rain, which is most of the wet steps, gets only 2% of the range.
See `results/transform_diagnostics/transform_pixels.png`.

**Caveat for the diagnostic:** the held-out years contain only **7 extreme** and 157 heavy
steps, too few to judge the extreme band. The job also runs the diagnostic on 2000–2007
(40 extreme steps, in-sample) as a secondary check.

### ✅ Next
1. `sbatch scripts/submit_v13.sh`
2. `python scripts/gate_a_scorecard.py --reference train --versions v12 v13`
3. `python scripts/visualize_transform.py --generated v12=results/generated_data/rainfall_synthetic_10y_v12.csv v13=results/generated_data/rainfall_synthetic_10y_v13.csv`
4. Add a results entry with the verdict.

---

## September 22, 2026 — Comprehensive Evaluation Suite, Scorecard, and Fine-Tuning Analysis (v8–v10)

> **Correction (2026-09-22, later same day):** An independent audit
> ([`code_plan/AUDIT_2026-09-22.md`](../../code_plan/AUDIT_2026-09-22.md)) found that this
> entry's scorecard and verdict were computed against the **full 2000–2009 record**
> (`scripts/run_evaluation.py`'s default), not the 2000–2007 training partition
> `my notes/memo/day_1.md` §3.1 freezes as canonical — 80% of the full record is these
> models' own training data. Rescored against the canonical partition with the added
> Tier 4 (IDF) and Tier 5 (seasonality) tiers that this entry's tables do not cover:
> - **No version passes Gate A.** Best is v10 at 11/18 banded metrics
>   (`results/reference/gate_a_train.json`), not the near-passing picture this entry's
>   verdict implies.
> - **"v10 (Markov)" is corrected to "v10 (transition-sampled assembly)"** — the CLI flag
>   name `--assembly_mode markov` does not describe a physical Markov weather process (see
>   the Sept 4 entry below and `my notes/memo/day_1.md` §4).
> - The **Recommended Downstream RL Configuration** line below is withdrawn. `my notes/memo/day_1.md`
>   §7 bars claiming synthetic-data validity for RL control before Gate B (SWMM routing)
>   and Gate C (TSTR), neither of which has run.
> - **v10 under transition-sampled assembly has no seasonal cycle** (monthly Pearson r =
>   −0.11 against observed, computed 2026-09-22); only v10 under calendar-ordered assembly
>   (r = 0.65) and v8_cal (r = 0.91) carry one. "Seasonality Match: Flat" for v10 in the
>   table below was correctly logged at the time but undersells how large this gap is.
> - **v9/v9_cal rejection reasoning is corrected:** the quoted $2.83\text{ mm}$ max is
>   within the observed record's own per-year range ($1.92$–$5.55\text{ mm}$) and is not on
>   its own evidence of tail truncation; v9's decisive defect is its volume deficit
>   (ratio $0.831$, $z=-6.5$ against the interannual spread of the observed record), not
>   the single-draw maximum.
> - **"The EDM Generative Paradox"** (last bullet of the Loss Function section below) is
>   withdrawn: the loss values being compared are `.mean()`-reduced over a fixed 64-cell
>   grid regardless of how many cells are active at each `seq_len`, so raw scalars across
>   different `seq_len` are not on the same scale. Rescaled by $64/seq\_len$, v10 has the
>   **lowest** per-active-cell loss ($0.1556$) of the six versions compared — there is no
>   paradox, loss and sample quality agree once normalised correctly.
>
> The scorecard table, "Most Significant Change" section and loss bullets below are kept
> as originally written for the record of what was concluded at the time; read them with
> the five corrections above. Full derivation of every correction:
> [`code_plan/AUDIT_2026-09-22.md`](../../code_plan/AUDIT_2026-09-22.md).
> [`docs/RESEARCH_CONTRIBUTIONS.md`](../../docs/RESEARCH_CONTRIBUTIONS.md) and
> [`docs/FINETUNING_ANALYSIS.md`](../../docs/FINETUNING_ANALYSIS.md) have been revised in
> place to reflect them (they are living technical documents, not dated diary entries).

### 🔍 Overview
Completed the evaluation and synthesis across all six clean synthetic holdout datasets (**v8**, **v8_cal**, **v9**, **v9_cal**, **v10**, **v10_cal**) alongside historical benchmarks (v1–v7). Built the unified automated evaluation script [`scripts/run_evaluation.py`](../../scripts/run_evaluation.py) and interactive sanity notebook [`notebooks/synthetic_rainfall_evaluation.ipynb`](../../notebooks/synthetic_rainfall_evaluation.ipynb) (commit `e34a342`). Codified fine-tuning methodology, training loss dynamics, and physical storm metrics in [`docs/FINETUNING_ANALYSIS.md`](../../docs/FINETUNING_ANALYSIS.md), and authored the comparative research innovation and limitation report in [`docs/RESEARCH_CONTRIBUTIONS.md`](../../docs/RESEARCH_CONTRIBUTIONS.md) evaluating our contributions against the base paper (*Gonen et al., NeurIPS 2025*).


---

### 📊 Summary Scorecard & Version Rankings

| Evaluation Axis | Real Ground Truth | v8 (L=24) | v8_cal (L=24) | v9 (L=36) | v9_cal (L=36) | v10 (L=64) | v10_cal (L=64) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Annual Volume** | $709.3\text{ mm}$ | $688.0\text{ mm}$ ($97\%$) | $692.4\text{ mm}$ ($98\%$) | $589.5\text{ mm}$ ($83\%$) | $588.3\text{ mm}$ ($83\%$) | $649.2\text{ mm}$ ($92\%$) | $630.0\text{ mm}$ ($89\%$) |
| **Zero Fraction** | $90.96\%$ | $91.58\%$ | $91.56\%$ | $91.30\%$ | $91.29\%$ | $91.75\%$ | $91.84\%$ |
| **Mean Wet Intensity**| $0.0746\text{ mm}$ | $0.0778\text{ mm}$ | $0.0780\text{ mm}$ | $0.0645\text{ mm}$ | $0.0642\text{ mm}$ | $0.0749\text{ mm}$ | $0.0735\text{ mm}$ |
| **Storm Count ($N/\text{yr}$)**| **$679.7$** | $840.8$ ($+24\%$) | $848.0$ ($+25\%$) | $835.5$ ($+23\%$) | $835.4$ ($+23\%$) | **$699.8$** ($+2.9\%$) | **$693.3$** ($+2.0\%$) |
| **Mean Storm Duration**| **$62.3\text{ min}$** | $45.8\text{ min}$ ($-26\%$) | $45.7\text{ min}$ ($-27\%$) | $45.3\text{ min}$ ($-27\%$) | $45.2\text{ min}$ ($-27\%$) | **$54.1\text{ min}$** ($87\%$) | **$54.1\text{ min}$** ($87\%$) |
| **Hourly Lag-1 ACF** | **$0.4831$** | $0.3069$ | $0.2948$ | $0.3782$ | $0.3623$ | **$0.4475$** | **$0.4319$** |
| **Hourly ACF RMSE** | **$0.0000$** | $0.0926$ | $0.0936$ | $0.0784$ | $0.0797$ | **$0.0554$** | **$0.0564$** |
| **Hourly KS Distance** | $0.0000$ | $0.0312$ | $0.0320$ | $0.0415$ | $0.0421$ | **$0.0093$** | **$0.0114$** |
| **Wet-Only JSD** | $0.0000$ | $0.0624$ | $0.0634$ | $0.1007$ | $0.1001$ | **$0.0439$** | $0.0486$ |
| **Max 5-Min Burst** | $5.555\text{ mm}$ | $4.396\text{ mm}$ | $4.983\text{ mm}$ | $2.830\text{ mm}$ | $3.159\text{ mm}$ | $4.612\text{ mm}$ | **$5.638\text{ mm}$** |
| **Seasonality Match**| Ground Truth | Flat | **Strong** | Flat | Partial | Flat | Partial |

#### Verdict & Production Deployments:
- **Best Overall Generator:** **`v10` (Markov)** — Achieves the highest statistical fidelity, best hourly autocorrelation, lowest KS/JSD divergence, and restores storm geometry to within $+2.9\%$ of real storm frequency and $87\%$ of real storm duration.
- **Best for Seasonality:** **`v8_cal` (Calendar Assembly)** — Preserves the monthly annual precipitation cycle accurately across calendar months.
- **Recommended Downstream RL Configuration:** Primary training on **`v10`**, supplemented with **`v8_cal`** for seasonal cycle diversity.
- **Reject / Marginal:** **`v9` / `v9_cal`** rejected due to a $17\%$ volume deficit and truncated extreme tail ($2.83\text{ mm}$ max vs $5.555\text{ mm}$ real).

---

### 🏆 The Most Significant Change for the Research
The architectural expansion of the receptive context window from **$seq\_len = 24$ ($2.0\text{ hours}$)** to **$seq\_len = 64$ ($5.33\text{ hours}$)** in **v10** resolved the storm geometry failure that persisted through all prior iterations:
- **Annual Storm Count:** Normalized from $840.8\text{ storms/yr}$ (v8) down to **$699.8\text{ storms/yr}$** ($693.3/\text{yr}$ under calendar assembly), matching real ground truth ($679.7\text{ storms/yr}$) within **$+2.9\%$**.
- **Mean Storm Duration:** Rebounded from $45.8\text{ min}$ (v8) to **$54.1\text{ min}$** ($87\%$ of real $62.3\text{ min}$), reversing the $-26\%$ collapse.
- **Hourly Autocorrelation RMSE:** Slashed by **$40\%$** from $0.0926$ to **$0.0554$**, lifting hourly lag-1 ACF from $0.3069$ to **$0.4475$** (Real: $0.4831$).
- **Extreme Tail:** Peak burst maintained at **$5.638\text{ mm}/5\text{-min}$** (Real: $5.555\text{ mm}$).
- **Hourly KS Divergence:** Dropped from $0.0312$ to **$0.0093$** ($>3\times$ improvement).

---

### 📈 Loss Function Behavior Across Versions
- **Invariance across $L=24$ runs:** v5, v6, v7, and v8 all converged to a plateau between $0.0617$ and $0.0624$, proving that scalar loss is insensitive to conditioning or extreme tail health under EDM preconditioning.
- **Proportional loss scaling with context length:** Loss scaled directly with active delay-embedding tensor volume:
  - $L=24$ (v8): Initial $0.1276 \to$ Best **$0.0624$** (Ep 433)
  - $L=36$ (v9): Initial $0.1601 \to$ Best **$0.0909$** (Ep 453)
  - $L=64$ (v10): Initial $0.2545 \to$ Best **$0.1556$** (Ep 498)
- **Early optimization dynamics:** v10 experienced severe initial gradient shock (Epoch 1 mean grad norm $1.350$, $63.2\%$ clipped) before stabilizing by Epoch 25 ($10\%$ clipped) and settling under $0.45$.
- **The EDM Generative Paradox:** Higher scalar loss in v10 ($0.156$ vs $0.062$) corresponds to substantially superior generative hydrology.

---

## September 21, 2026 — Sequence Length Adaptation (v9 & v10) to Base Checkpoint Geometries

> **Correction (2026-09-22):** The claim below that base checkpoints are "constrained to
> these exact delay-embedding sequence lengths" and that targeting $seq\_len=144/288$
> "caused shape mismatches with base weights" is not what the training logs show. All four
> checkpoints (`ImagenFew_{12,24,36,64}.ckpt`) are byte-identical in size (54,577,102
> bytes), and every v8/v9/v10 run — including the failed 144/288 attempts — loads exactly
> 1006/1008 parameters, skipping only the two class-embedding heads, regardless of which
> checkpoint or target `seq_len` was used. The 144/288 runs instead failed with
> `IndexError: index 8 is out of bounds for dimension 3 with size 8` in
> `models/ImagenFew/img_transformations.py` — a limit set by the training config's
> `delay=embedding=8` grid size, not by the checkpoint. See
> [`code_plan/AUDIT_2026-09-22.md`](../../code_plan/AUDIT_2026-09-22.md) §3.1 and
> [`code_plan/PROVENANCE.md`](../../code_plan/PROVENANCE.md). A consequence: the v9/v10
> context-length comparison changed checkpoint initialisation and context length at the
> same time, which was not a necessary constraint, and the two have not yet been
> decoupled.

### 🔍 Overview
Following the decision to scale context length (Tasks T4.2 / Experiment 2), we audited the available pretrained base checkpoints in `models_ckpt/ImagenFew/` (`ImagenFew_12.ckpt`, `ImagenFew_24.ckpt`, `ImagenFew_36.ckpt`, `ImagenFew_64.ckpt`). Because base checkpoints are constrained to these exact delay-embedding sequence lengths, targeting $seq\_len = 144$ or $288$ caused shape mismatches with base weights.

---

### 🛠️ Modifications (Commits `aeec47e`, `6172f29`, `8b0db23`)
1. **Config Realignment:**
   - Realized v9 at **$seq\_len = 36$** ($3.0\text{ hours}$) fine-tuned from `ImagenFew_36.ckpt` ([`regime_training/config_v9.yaml`](../../regime_training/config_v9.yaml)).
   - Realized v10 at **$seq\_len = 64$** ($5.33\text{ hours}$) fine-tuned from `ImagenFew_64.ckpt` ([`regime_training/config_v10.yaml`](../../regime_training/config_v10.yaml)).
2. **Transition Matrix Derivation (`scripts/fit_seasonal_labels.py`):**
   - Added candidate block sizes $36$ and $64$ into the multi-scale block transition calculation:
     - `seasonal_transition_matrix_train_len36.pkl`
     - `seasonal_transition_matrix_train_len64.pkl`
   - Updated `data/rainfall/splits/MANIFEST.json`.
3. **Training & Inference Scripts:**
   - Updated training wrappers: [`scripts/run_v9_training.sh`](../../scripts/run_v9_training.sh) and [`scripts/run_v10_training.sh`](../../scripts/run_v10_training.sh).
   - Created generation scripts supporting dual Markov and Calendar assembly: [`scripts/run_v9_generation.sh`](../../scripts/run_v9_generation.sh) and [`scripts/run_v10_generation.sh`](../../scripts/run_v10_generation.sh).

---

## September 19–20, 2026 — Clean Chronological Holdout Split, De-Notebooked Pipeline & v8 Baseline

### 🔍 Overview
Executed Phase 0 and Phase 1 of the remediation plan ([`code_plan/REMEDIATION_PLAN.md`](../../code_plan/REMEDIATION_PLAN.md)) to eliminate cross-year data leakage and establish a defensible, reproducible experimental foundation (commits `fcd8e75`, `0933259`, `1a84713`, `47b4ffe`, `b846b3b`, `5044617`).

---

### 🛠️ Key Deliverables
1. **Canonical Partitioning (`data/rainfall/splits/`):**
   - **Training Set (2000–2007):** $840,960$ intervals ($80.00\%$), mean annual depth $709.6\text{ mm}$, zero fraction $91.03\%$.
   - **Validation Set (2008):** $105,120$ intervals ($10.00\%$, convective heavy-rain year), annual depth $755.7\text{ mm}$, wet mean $0.0923\text{ mm}$.
   - **Test Set (2009):** $105,120$ intervals ($10.00\%$, mild year), annual depth $661.0\text{ mm}$, maximum burst $2.005\text{ mm}$.
2. **De-Notebooked Seasonal-Phase Fitting (`scripts/fit_seasonal_labels.py`):**
   - Replaced interactive notebooks with a deterministic CLI script.
   - Fitted a 4-state `GaussianHMM` strictly on 2000–2007 training years' 14-day rolling mean climatology with 4-day minimum duration spell filtering (`smooth_hmm_states_min_duration`), stabilizing annual transitions to $21\text{ transitions/year}$.
   - Projected labels onto holdout years (2008–2009) out-of-sample via calendar slot index ($0$ to $105,119$).
3. **Frozen Governance Memo ([`my notes/memo/day_1.md`](../memo/day_1.md)):**
   - Established the One-Page Experiment Control Sheet for v1–v7.
   - Codified canonical ground-truth baseline targets and mathematical metric definitions.
   - Established the Claims Governance Table barring unsafe claims (retiring "HMM weather states" and "DDPM" terminology).
4. **Clean Baseline Pipeline (v8):**
   - Implemented [`regime_training/config_v8.yaml`](../../regime_training/config_v8.yaml) and [`scripts/run_v8_training.sh`](../../scripts/run_v8_training.sh).
   - Fine-tuned clean baseline (Run ID: `ed17d299`) for 500 epochs, confirming v7 parity on holdout splits without data contamination.

---

## September 7, 2026 — Empirical Baseline Audit & Ground-Truth Evaluation Targets

### 🔍 Overview
Conducted an exhaustive statistical and hydrological analysis of the 10-year Astlingen rainfall record partitioned into chronological holdouts:
- **Training Set (2000–2007):** 840,960 intervals (8 years, 80.05% of dataset), mean annual depth $709.6\text{ mm}$, $91.03\%$ zero intervals.
- **Validation Set (2008):** 105,120 intervals (1 year, 10.03%), annual depth $755.7\text{ mm}$ ($1.07\times$ train mean), $92.22\%$ zero intervals.
- **Held-Out Test Set (2009):** 105,120 intervals (1 year, 9.92%), annual depth $661.0\text{ mm}$ ($0.93\times$ train mean), $91.38\%$ zero intervals.

The findings are codified in `data_analysis/Train_Val_Test_Split_Analysis.ipynb` and establish our physical evaluation targets for synthetic rainfall generation.

---

### 📊 Key Findings & Ground-Truth Targets

1. **Intensity Marginals & Heavy Tails:**
   - Evaluated strictly on wet intervals ($> 0.005\text{ mm}$):
     - Train Mean: $0.0751\text{ mm}$ vs. Median: $0.0350\text{ mm}$ (proves severe right-skewness; drizzle dominates frequency while rare downpours drive volume).
     - Extremes: Train $P_{90} = 0.169\text{ mm}$, $P_{95} = 0.263\text{ mm}$, $P_{99} = 0.595\text{ mm}$, $P_{99.9} = 1.613\text{ mm}$, Max = $5.555\text{ mm}/5\text{min}$.
   - **Validation (2008) is a Heavy Convective Year:** Mean wet intensity reached $0.0923\text{ mm}$ ($+23\%$), $P_{99} = 0.719\text{ mm}$, $P_{99.9} = 1.734\text{ mm}$. Its survival curve sits consistently above training, making it an honest challenge for generative flood risk.
   - **Test (2009) is a Mild Year:** Max burst reached only $2.005\text{ mm}/5\text{min}$, testing whether generators over-predict extremes during calm years.
   - **Two-Sample KS Test:** Val vs. Train stat = $0.0758$ ($p = 2.56 \times 10^{-37}$); Test vs. Train stat = $0.0243$ ($p = 1.38 \times 10^{-4}$).

2. **Storm Dynamics & Spell Durations:**
   - **Mean Storm Duration ($\ge 15$ min):** $61.0\text{ minutes}$ (Train), $60.5\text{ minutes}$ (Val), $67.5\text{ minutes}$ (Test). Storm duration is an invariant physical characteristic of the Astlingen climate (${\approx} 1\text{ hour}$).
   - **Mean Wet Spell Length:** $30.4\text{ minutes}$ (${\approx} 6$ consecutive 5-min intervals).
   - **Storm Frequency:** Train averaged $682.6\text{ storms/yr}$ vs. Val $597.0\text{ storms/yr}$. 2008 concentrated greater annual rain volume into fewer, more intense downpours.
   - **Maximum Dry Spell:** Train captures a $33.40\text{ day}$ drought (Summer 2003 European heatwave), while single-year holdouts maxed out at $11.8$–$13.6$ days.

3. **Temporal Autocorrelation & Memory Decay:**
   - **Lag-1 5-Min ACF:** Train = $0.8518$, Val = $0.8665$, Test = $0.8535$. Rainfall exhibits massive short-term physical inertia, proving it cannot be modeled as memoryless noise.
   - **Convective Decay (0 to 2 Hours):** Rapid decay from $0.85 \to 0.20$ within 2 hours, matching the 60-minute storm cell lifecycle.
   - **Decorrelation Horizon (12 to 18 Hours):** Autocorrelation drops below $0.03$ by 12–18 hours, defining the maximum memory horizon of the physical process.

---

### 🎯 Synthesis: Evaluation Targets for Synthetic Series
These metrics give you concrete ground-truth targets to evaluate your generated synthetic rainfall:
- **Zero fraction:** Must match $91.0 \pm 1.0\%$.
- **Mean storm duration:** Must target $61.0 \pm 6.0\text{ min}$ (cannot collapse to 10 min or smear to 4 hours).
- **Mean wet spell:** Must target $30.4 \pm 3.0\text{ min}$.
- **Lag-1 ACF:** Must target $0.85 \pm 0.02$.
- **Extreme tail ($P_{99}$):** Must target $0.60 \pm 0.06\text{ mm}/5\text{min}$.

---

## September 4, 2026 — Renaming the Conditioning Mechanism

### 🔍 Overview
Following the panel audit (§0, §2), we audited the mathematical validity of framing our conditioning pipeline as a Hidden Markov Model (HMM) and 1st-order Markov chain. We formally retired this terminology across all documentation, guides, and plans in favor of descriptive definitions that accurately describe the statistical mechanism.

---

### 📌 Findings
The mechanism previously described as an HMM with Markovian block assembly is not a weather-state process for three concrete reasons:
1. **Periodic calendar function:** The 4-state labels assigned to timestamps are a periodic function of day-of-year, repeating identically across all 10 years rather than identifying stochastically evolving weather states.
2. **Autocorrelation & emission independence:** The HMM was fitted on a 14-day moving-average smoothed series. Successive 5-minute samples have ~100% autocorrelation, completely voiding the core HMM conditional emission-independence assumption ($P(X_t \mid S_t, X_{<t}) = P(X_t \mid S_t)$).
3. **Degenerate transition matrix:** The 4×4 block transition matrix is estimated from a few hundred near-deterministic seasonal boundary crossings with `smooth=1e-5` Laplace fill. Its off-diagonal entries merely capture "which calendar season follows which" rather than a stochastic 1st-order Markov chain.

---

### ⚖️ Decision
We have executed a global terminology shift across all documentation, guides, and plans:
- "HMM state" / "HMM state conditioning" $\rightarrow$ **"seasonal-phase index"** / **"seasonal-phase conditioning"**.
- "transition-aware Markov assembly" $\rightarrow$ **"calendar-ordered block assembly"**.
- "1st-order Markov chain" / "stationary distribution" $\rightarrow$ deleted or retained strictly with explicit caveats.
- Conditioning vector $c$ in the denoiser objective is formally designated as a **one-hot seasonal-phase index (4 levels, ascending mean intensity)**.

> **Verdict: The conditioning mechanism is seasonal-phase conditioning on smoothed climatology with calendar-ordered block assembly, not an HMM or Markovian weather state process.**

---

## September 1, 2026 — Loss-Function Audit & Formal Acceptance Criteria

> [!NOTE]
> **Correction (2026-09-04):** This entry referred to conditioning vector $c$ as an "HMM state" and recommended re-fitting the HMM on training years. The conditioning variable is actually a discrete seasonal-phase index (4 levels of smoothed climatological mean intensity), not a latent weather state. What was termed an HMM transition matrix is a calendar-ordered transition table between seasonal blocks.

### 🔍 Overview
With v7 in hand, we stopped generating and audited two things we had been carrying on faith:
what the training objective actually optimises, and what would count as "done". Findings are
written up in [LOSS_FUNCTION.md](../../code_plan/LOSS_FUNCTION.md) and
[ACCEPTANCE_CRITERIA.md](../../code_plan/ACCEPTANCE_CRITERIA.md).

---

### 📐 What we are optimising

ImagenFew is an **EDM** model (Karras et al. 2022), not a DDPM. The objective is a weighted
denoising regression in **x-prediction**:

$$\mathcal{L} = \mathbb{E}_{x,\sigma,n}\left[\lambda(\sigma)\,\|D_\theta(x+n;\sigma,c)-x\|^2\right],\quad
\lambda(\sigma)=\frac{\sigma^2+\sigma_d^2}{(\sigma\sigma_d)^2},\quad \sigma\sim\mathrm{LogNormal}(-1.2,\,1.2)$$

with $c$ the one-hot HMM state and $\sigma_d = 0.5$. $\lambda$ is chosen so
$\lambda c_{out}^2 = 1$, i.e. the raw network sees a unit-variance target at every noise
level — which is why the loss curve is nearly flat and its absolute value tells us very little.

---

### 📌 Audit findings

1. **"L2 regresses to the mean" is not the right explanation.** At the optimum the minimiser
   is $E[x\mid y]$, the exact score — EDM with an exact denoiser reproduces the tails. Our
   flattened extremes come from a **budget** problem in three parts:
   - Extremes are ~$10^{-6}$ frequent and contribute ~**1%** of expected loss, so finite
     capacity under-resolves them first.
   - `clip_grad_norm_(..., 1.0)` **systematically attenuates exactly those events**: a batch
     containing a cloudburst has a much larger gradient norm and gets rescaled down, while
     bulk dry batches pass through unscaled.
   - EMA at decay 0.9999 has a ~10,000-step memory and averages rare-batch corrections away.

2. **The FFT term is inert as a spectral prior.** `fft2(..., norm='forward')` divides by
   $N=64$; by Parseval the term is **exactly 1/64 (1.56%)** of the time term — measured
   0.01560 against a predicted 0.015625. Worse, `fft_loss` is indexed by *frequency* but is
   multiplied by `(1 - x_img_mask)`, a *spatial* mask. What it actually does is hold the
   zero-padded columns near zero: a 5× error confined to padding leaves the time term
   unchanged but multiplies the FFT term **16×**.

3. **Two code paths, two different losses.** `handler.py:44` (→ **v3, v4**) has **no FFT
   term**; `train_regime.py` (→ **v5, v6, v7**) has it. `ImagenFew.loss_fn` — the one that
   looks canonical — is dead code, called by nothing. Given finding 2 this shifts the loss by
   ~1.5%, but the results table must now say which objective produced which checkpoint.

4. **`sigma_data` is mis-tuned by 2×.** ImagenFew hardcodes $\sigma_d = 0.5$; `RegimeDataset`
   uses `StandardScaler`, which yields std **1.0**.

5. **⚠️ There is no held-out real data.** `config_hmm.yaml` sets `train_csv == test_csv`, and
   `hmm_105120_train.csv` spans **2000–2009** — the entire record. `hmm_105120_test.csv` (2009)
   is a *subset of the training range*, not a holdout. **Every TSTR number computed today would
   be contaminated.** Fix before any downstream claim: split by year (train 2000–2007, hold out
   2008–2009) and **re-fit the HMM on training years only** — the state labels are fitted
   quantities and leak exactly as the diffusion weights do.

---

### 🛠️ Code changes
- `intensity_weights()` in `train_regime.py`: $w(x) = 1 + \alpha\,\mathrm{relu}(x)^\gamma$ on the
  time term. `relu` because z-scoring puts dry steps at a small *negative* value, so the 91% dry
  mass stays at weight 1.0. $\gamma=0.5$ is sub-linear on purpose — $z$ reaches ~119 at the 10-yr
  maximum. Measured profile at $\alpha=0.3$: dry 1.00×, P99 1.54×, P99.9 2.04×, max 4.27×.
  Weights renormalise to batch-mean 1.0 so sweeping $\alpha$ does not silently sweep the LR.
- `grad_clip`, `fft_weight` now configurable; `grad_norm` and `clip_frac` logged per epoch.
- Per-regime **p99.9** logged + new `tail_health.png`. The mean can look right while the tail is
  dead — exactly how v6 passed epoch checks yet capped at 1.79 mm.
- **All defaults reproduce v5–v7 exactly** ($\alpha=0$, clip 1.0, fft_weight 1.0).

---

### 📊 Acceptance criteria (replaces the "RL Fitness Score")

The old equal-weighted score was gameable — it ranked v3 (0.733) above v4 (0.622) while v3
halved the flood peaks. Replaced by three staged gates:

- **Gate A — statistical screening.** Five tiers with hard pass bands: water balance &
  intermittency, intensity marginal, temporal structure, multi-scale extremes (IDF table over
  duration × return period — **not yet computed, biggest hole**), seasonality.
- **Gate B — hydraulic response.** Route real vs synthetic through SWMM-Astlingen under a
  **fixed** controller and compare CSO counts, spill volumes, tank levels, flooding.
  This is the first metric that is **non-linear in the rainfall** — sewer response is
  threshold-driven, so two series with near-identical marginals can spill very differently.
  No RL needed; cheap enough for every version. **Highest-value thing we are not yet doing.**
- **Gate C — TSTR.** Four PPO arms (real-full / real-scarce / synthetic-only /
  scarce+synthetic) evaluated on held-out real rainfall. Headline claim: scarce+synthetic
  within 10% of real-full. Plus a **non-negotiable safety check** on the largest held-out
  storm — an agent trained on too-weak extremes has learned that storms are survivable, and
  that must never be averaged into a mean return.

---

### 📌 Next actions, ranked
1. **`seq_len` 24 → 288.** Not a loss problem (see Aug 26). Run before any further loss tuning.
2. Read `clip_frac` on one diagnostic run; if high, raise `grad_clip` before touching $\alpha$.
3. Re-split by year and re-fit the HMM on training years only.
4. Build Gate B in `flood-control` — cheapest large gain available.

---

## August 26, 2026 — Full-Version Benchmark: v7 Wins on Intensity, Loses on Storm Shape

> [!NOTE]
> **Correction (2026-09-04):** This entry described v5–v7 as "HMM variants" and attributed volume calibration to "HMM conditioning". In reality, the model was conditioned on a 4-level seasonal-phase index derived from smoothed calendar climatology. The underlying sequence assembly was calendar-ordered block assembly rather than genuine Markovian weather state generation.

### 🔍 Overview
Regenerated all three HMM variants from the retrained checkpoints (SLURM 760225/760226/760227)
and built `scripts/compare_all_versions.py` — a single benchmark of all seven synthetic
versions against the 10-yr Astlingen gauge average across 8 evaluation axes. Outputs in
[results/comparison_all_versions/](../../results/comparison_all_versions/).

---

### 📊 Benchmark (5-min versions + real; v3/v4 excluded here — see resolution note)

| Version | Annual Vol (mm) | Vol Ratio | Zero % | Mean Wet | P99 | P99.9 | Max | Wet Spell (min) | N Storms | Storm Dur (min) | Lag-1 ACF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Real** | 709.4 | 1.000 | 90.95 | 0.0746 | 0.160 | 0.565 | 5.555 | 32.1 | 6,800 | 62.3 | 0.854 |
| v1 | 1365.9 | 1.926 | 90.03 | 0.1304 | 0.318 | 1.198 | 5.426 | 19.7 | 11,070 | 38.1 | 0.684 |
| v2 | 1363.3 | 1.922 | 90.10 | 0.1310 | 0.318 | 1.208 | 5.731 | 19.7 | 11,075 | 37.9 | 0.679 |
| v5 | 664.6 | 0.937 | 91.11 | 0.0711 | 0.151 | 0.542 | 4.134 | 25.5 | 8,715 | 46.6 | 0.855 |
| v6 | 644.7 | 0.909 | 91.11 | 0.0690 | 0.160 | 0.483 | **1.787** | 23.9 | 8,722 | 45.5 | 0.860 |
| **v7** | **709.5** | **1.000** | **90.97** | **0.0747** | **0.161** | **0.568** | **5.644** | 27.4 | 8,824 | 47.6 | **0.848** |

---

### 💡 Findings

1. **v7 is the best generator produced so far, by a clear margin.** Volume ratio 1.000, zero
   fraction within 0.02 pp, and — the first time in the project — the **entire intensity
   marginal** matches including the tail: P99 1.006×, P99.9 1.006×, maximum 1.016×. The
   May–June problem of halved cloudbursts is solved.
2. **HMM conditioning fixed the 2× volume bias.** v1/v2 generated 1.93× the real annual
   volume; all three HMM versions land in 0.91–1.00.
3. **v6 has a dead tail.** Maximum 1.787 mm against a real 5.555 mm, while its *mean* wet
   intensity is fine (0.069 vs 0.0746). The 365-day DoY-broadcast labels are too coarse to
   carry storm intensity. This is the textbook case for the p99.9 monitoring added Sep 1.
4. **Storm geometry still fails, in all seven versions.** v7: mean storm duration **47.6 min
   vs 62.3** (−24%), storm count **8,824 vs 6,800** (+30%), wet spell **27.4 vs 32.1** (−15%),
   hourly ACF RMSE **0.0906** — *worse* than v3's 0.0688. The rain is broken into too many,
   too-short pieces.

> **One-line verdict: v7 delivers the right amount of water, in the right sized drops,
> arriving in the wrong shaped storms.**

---

### 📌 Root cause: the context window shrank when we moved to 5-min

`seq_len` is **still 24** — unchanged since May. v3/v4 saw 24 × 10 min = **4 hours**; v5–v7 see
24 × 5 min = **2 hours**. Migrating to 5-min resolution (Aug 24) **halved the physical context
window** without anyone changing `seq_len`. Real mean storm duration is 62 minutes, so the model
is learning storm persistence through a window barely twice the length of a storm.

"Experiment 2: expand the context horizon" has been on the plan since **July 27** and has never
been run. It is now the top-priority experiment — and note this is *not* a loss-function issue,
so it should be run before any further loss tuning.

---

### ⚠️ Reporting trap identified
The benchmark table mixes 5-min and 10-min versions, and **per-step metrics are not
resolution-invariant**. v3's `Mean_Wet` of 0.130 mm/10-min is roughly *equal* to real, not
double it. Reading down that column across resolutions is wrong. All cross-version claims must
be made on hourly or daily aggregates, or restricted to one resolution.

---

## August 25, 2026 — HMM v2 Labels (Mean-Smoothed) & Training Instrumentation

> [!NOTE]
> **Correction (2026-09-04):** This entry discussed "HMM variant v2" and "hmm_105120_v2_transition_matrix.pkl". Fitting an HMM on a 14-day smoothed 1-D mean produces a periodic seasonal-phase index with near-100% sample autocorrelation, violating emission independence. The 4×4 transition matrix simply encodes deterministic calendar progression between seasonal phases with Laplace smoothing, not a physical Markovian weather process.

### 🔍 Overview
Two threads: a third HMM labelling variant, and proper monitoring so we stop flying blind
during 500-epoch fine-tunes.

---

### 🧪 HMM variant v2 (→ synthetic v7)
Notebooks: `Rainfall_10Yr_105120_Interval_HMM_v2_Mean.ipynb` and
`..._v2_Mean_smoothed.ipynb`.

- Refit the 105,120-interval HMM on a **single smoothed 1-D mean feature** rather than the
  3 daily features used in v1, then applied **minimum-duration post-processing** to remove
  physically implausible one-step state flips.
- Motivation: the v1 3-feature labels produced state sequences that switched far too fast for
  the block-level Markov assembly to be meaningful.
- Produced `dataset_with_105120_v2_fit_hmm.csv` and `hmm_105120_v2_transition_matrix.pkl`.

This turned out to be the change that mattered — v7 is the only version to match the intensity
tail (see Aug 26).

---

### 🛠️ Training instrumentation (`train_regime.py`)
Previously the fine-tunes logged a scalar loss and nothing else. Added:
- Per-epoch `history.csv` with loss and best-loss tracking.
- `loss_curve.png` with the best-epoch marked.
- `regime_convergence.png` — per-state generated mean intensity over epochs, so state
  separation collapse is visible during training rather than after a 10-year generation run.
- Persisted `scaler.pkl` and `regime_proportions.pkl` per run, removing the hardcoded regime
  frequencies the v4 generator relied on.

Retrained the 365 and 105120-v2 variants with monitoring (SLURM 756911/756912) and ran the
first generation pass (755849/755850).

---

## August 19–24, 2026 — Migration to 5-min Resolution & HMM State Conditioning

> [!NOTE]
> **Correction (2026-09-04):** This entry introduced "HMM state conditioning" and "transition-aware Markov assembly". In fact, the conditioning signal is a 4-level seasonal-phase index tracking day-of-year mean intensity. The block assembly was calendar-ordered assembly from an empirical transition matrix whose off-diagonals reflect seasonal boundaries, rather than a 1st-order Markov chain of meteorological weather states.

### 🔍 Overview
Executed **Experiment 1** and **Experiment 3** from the July 27 plan: replaced the GMM
14-day block conditioning with per-timestep HMM state conditioning, and replaced random block
shuffling with transition-aware Markov assembly.

---

### 🛠️ Implementation

1. **5-minute native resolution.** Rebuilt the datasets at the gauge's native 5-min interval
   (105,120 intervals/year) instead of the 10-min downsample used for v3/v4.
   *(Consequence not noticed at the time: this halved the effective context window — see Aug 26.)*

2. **HMM state conditioning replaces GMM regimes.** `RegimeDataset` generalised to accept
   `regime_col = hmm_state`, with per-window labels assigned by **majority state within the
   window** rather than by whole-block summary statistics. This closes the information-leakage
   loophole recorded on July 27, where 14-day window statistics leaked into conditioning.

3. **Two labelling variants trained** (a third followed on Aug 25):
   - **v5** — 105,120-interval HMM fit on 3 daily features.
   - **v6** — 365-day climatological HMM, day-of-year broadcast.

4. **Transition-aware generation** (`generate_hmm_v1.py`), directly addressing the "Random
   Block Shuffling" loophole:
   - Block-level 1st-order Markov sampling of the state sequence from the estimated
     $P(R_{t+1}\mid R_t)$, replacing `rng.permutation`.
   - **Overlap-add** stitching, with a **wider overlap at state-transition boundaries**
     (`--overlap 4`, `--transition_overlap 8`).
   - **Soft-label bridge blocks** inserted at transitions to blend between states.

5. **Retrained 500 epochs** from the base ImagenFew checkpoint with shape-safe loading
   (`load_checkpoint_safe`) to handle the 43 → 4 class change in `map_label`.

---

### 📌 Outcome
Both variants completed and generated 10-year series. Volume calibration improved dramatically
over v1/v2 (1.93× → ~0.91×) and lag-1 autocorrelation was restored to ~0.855 against a real
0.854. Storm duration improved but did not close (46.6 min vs 62.3). Full assessment on Aug 26.

---

## August 5, 2026 — Clustering Validation & Model Selection ($K=4$ HMM vs. GMM)

### 🔍 Overview
We conducted a comparative experiment testing a 4-state Hidden Markov Model (HMM) against a 4-component Gaussian Mixture Model (GMM) to validate our weather regime stratification approach.

---

### 🧪 Validation & Findings
- **GMM as Validation Baseline:** The 4-component GMM was used primarily as a baseline to validate and compare static state clustering against the temporal transition modeling of the HMM.
- **Model Selection ($K=4$):** Evaluated clustering behavior and Evidence Lower Bound (ELBO) metrics across the models.
- **Visual Plots & Analysis:** Refer to the plots and figures in [Astlingen_Rainfall_365Day_Climatological_Clustering.ipynb](file:///Users/tanyawarrier/Desktop/projects/ImagenFew/data_analysis/Astlingen_Rainfall_365Day_Climatological_Clustering.ipynb):
  - **Section 2.1:** *Side-by-Side Comparison: HMM Seasonal Bands vs Direct GMM Physical Bands (365 DOY)* (image comparing 4 HMM seasonal states against 4 GMM physical regimes over 365 DOY).
  - **Section 3:** *Extended Model Selection ($K \in [1, 15]$) Across All 365 DOY Feature Representations* (ELBO curves and model selection plots).

---

### 📌 Decision
- Confirmed **$K=4$** as the optimal number of regimes based on the 4 HMM vs. 4 GMM comparative analysis and ELBO metrics.

---

## July 27, 2026 — Seasonality Audit & Root Cause Analysis of Generation Loopholes

### 🔍 Overview
We audited how seasonal patterns and weather transitions behave in our synthetic datasets (`v3` unconditional and `v4` conditional) compared to real 10-minute historical data (`data/rainfall/train/rainfall_10min_labeled.csv`). We used an unsupervised 4-state Hidden Markov Model (HMM) in `data_analysis/HMM_Season_Analysis.ipynb` (mirrored in `/flood-control/src/rainfall_datagen/notebooks/HMM_Season_Analysis.ipynb`).

---

### 📌 Core Model Loopholes Discovered

1. **Random Block Shuffling Causes the "Daily Max Paradox"**
   - *Problem:* In `regime_training/generate_regime.py`, 14-day generated weather blocks were randomly shuffled (`generated[rng.permutation(...)]`) and glued together.
   - *Impact:* Real storms persist continuously over hours. Randomly shuffling blocks method probably generates the peak burst, but the block ends and gets randomly glued to a dry block.

2. **Short Context Horizon ($seq\_len = 24$ = 4 Hours)**
   - *Problem:* Training configs (`configs/finetune/Rainfall.yaml` and `regime_training/config.yaml`) set the context window to `seq_len: 24` (4 hours at 10-minute resolution).
   - *Impact:* The U-Net self-attention mechanism only sees 4 hours at a time, preventing the model from learning multi-hour storm growth, decay, or recovery.

3. **Static Window Conditioning & "Single-Regime Lock"**
   - *Problem:* During generation, reverse diffusion uses one static class label for the entire window.
   - *Impact:* Real weather continuously transitions between states. Static labels prevent the model from learning transitions ($R_t \rightarrow R_{t+1}$). Generated multi-year series stay permanently stuck in one state:
     - `v3` (Unconditional) stays locked in **State 2 (Dry/Winter)** for 365 of 365 days.
     - `v4` (Conditional) stays locked in **State 3 (Storm/Summer)** for 363 of 365 days.

4. **Biased Evaluation Scorecard**
   - *Problem:* The equal-weighted scoring formula penalizes `v4` heavily for storm duration mismatches (score: 0.622) while giving `v3` a higher score (0.733), even though `v3` caps extreme flood peaks by 50% (5.60 mm vs 10.68 mm real).
   - *Impact:* Relying purely on the `v3` score is dangerous; an RL agent trained on `v3` will fail during severe 1-in-10-year floods.

5. **Information Leakage in Data Labeling**
   - *Problem:* `scripts/label_10min_data.py` assigned labels using total 14-day window statistics, leaking summary metrics into window conditioning while ignoring transition probabilities $P(R_{t+1} \mid R_t)$.

---

### 🧪 Ground-Truth Seasonality vs. Synthetic Behavior

When fitting a 4-state `GaussianHMM` on daily aggregated real data, the model naturally identified 4 real-world seasons:
- **State 2 (Winter / Dry Baseline, Months 1–4 & 12):** Low volume, low intensity. Matches GMM Regimes 2 & 3.
- **State 0 (Monsoon Transition, Months 5–7):** Rising rain frequency and early storms. Matches GMM Regimes 1 & 3.
- **State 3 (Late Summer Peak, Months 8–10):** Extreme cloudbursts and maximum volumes. Matches GMM Regime 0.
- **State 1 (Autumn Decay, Month 11):** Post-storm decay back into winter.

**Synthetic Audit Result:**
- **Real Data:** Balanced distribution across States 0, 1, 2, and 3 across the year.
- **`v3` (Unconditional):** Generates State 2 (dry winter) 365/365 days because 89.5% of real time-steps are zero rain. The model defaults to calm weather to minimize loss.
- **`v4` (Conditional):** Generates State 3 (heavy summer storm) 363/365 days because GMM regime conditioning forces heavy rain without allowing transitions back to calm weather.

---

### 🚀 Planned Next Experiments

1. **Experiment 1: Transition-Aware Markovian Block Assembly**
   - Calculate the 1st-order transition matrix $P(R_{t+1} \mid R_t)$ from `rainfall_10min_labeled.csv`.
   - Update `regime_training/generate_regime.py` to sample sequential regimes via transition probabilities $P_{ij}$ instead of random shuffling, adding overlap-add smoothing at boundaries.
   - *Goal:* Increase average storm duration to ~100 mins and daily maximum rainfall to 50+ mm.

2. **Experiment 2: Expand Context Horizon ($seq\_len \rightarrow 144 / 288$)**
   - Increase `seq_len` in `configs/finetune/Rainfall.yaml` from 24 to 144 (24 hours) or 288 (48 hours).
   - Fine-tune ImagenFew with wider context windows.
   - *Goal:* Capture 24-hour storm dynamics and multi-day autocorrelation decay.

3. **Experiment 3: Continuous Time & Dynamic HMM Conditioning**
   - Feed continuous time features ($\sin/\cos$ of `day_of_year`) and continuous HMM state probabilities $P(S_t)$ into the U-Net conditioning layers.
   - *Goal:* Enable smooth 365-day weather transitions without artificial block cuts.

4. **Experiment 4: Classifier-Free Guidance (CFG) & Heavy-Tail Loss Weighting**
   - Tune CFG scale ($\omega \in [1.2, 2.5]$) and apply an intensity-weighted loss function $w(x) = 1 + \alpha |x|^\gamma$.
   - *Goal:* Allow unconditional models to naturally generate extreme peaks (>10 mm) without artificial regime forced labels.

---

## June 15, 2026 — Regime-Conditioned Model (v4) & The "Daily Max Paradox"

### 🔍 Overview
To fix the missing extreme peaks in unconditional models, we conditioned ImagenFew on 4 discrete weather regimes using a two-stage pipeline (HMM seasons + GMM storm clustering).
- **Code Locations:** `/flood-control/src/rainfall_datagen/notebooks/4_dataset_stratification.ipynb` and `regime_training/`.
- **Dataset Evaluated:** `results/generated_data/rainfall_synthetic_10y_v4.csv` (10-minute resolution, 10-year generation).

---

### 🛠️ Methodology

1. **Number of "K" decision with HMM:** Mapped daily rainfall profiles to $K=4$ seasonal states using a `GaussianHMM` (2 = Winter/Calm, 0 = Monsoon Transition, 3 = Late Summer Storm, 1 = Autumn Decay).
2. **Regime Clustering (GMM):** Segmented time series into 14-day blocks. Clustered blocks into 4 GMM regimes using intensity, volume, dry fraction, and HMM seasonal labels:
   - **Regime 0 (Extreme Cloudburst):** Max peak ~5.55 mm/10-min in data (generated up to 10.77 mm).
   - **Regime 1 (Secondary Heavy):** Max peak ~3.64 mm/10-min.
   - **Regime 3 (Moderate):** Max peak ~1.55 mm/10-min.
   - **Regime 2 (Dry Baseline):** Max peak ~0.70 mm/10-min.
3. **Training & Inference:** Fine-tuned diffusion model using one-hot encoded GMM regime vectors.
4. **Dataset Assembly:** Sampled regime blocks proportional to real-world frequencies (~25.5% R0, ~37.0% R1, ~27.9% R2, ~9.6% R3), then randomly shuffled and concatenated them into a 10-year dataset.

---

### 📊 Results & Key Trade-Offs

- **Overall RL Fitness Score:** **0.622 / 1.000** (Verdict: ⚠️ **MODERATELY FIT**)

#### Tri-Dataset Comparison Table (`v3` vs `v4` vs Real Data)

| Category | Metric | Real Data | Synthetic `v3` (Unconditional) | Synthetic `v4` (Regime-Conditioned) | Findings & Analysis |
|---|---|:---:|:---:|:---:|---|
| **Sparsity & Counts** | **Zero Fraction** | 89.5% | 89.9% | 90.7% | Excellent dry spell match across all models. |
| | **Storm Count** | 4,852 | 5,613 | 5,557 | Both models slightly overestimate storm count by ~15%. |
| **Storm Geometry** | **Mean Storm Duration** | **100 mins** | **77 mins** | **66 mins** | `v4` breaks duration because random block concatenation cuts storms short. |
| | **Mean Storm Volume** | 1.43 mm | 1.18 mm | 1.13 mm | Shortened storm durations reduce total volume per storm. |
| | **Peak Timing** | 36.8% | ~36–39% | 37.8% | Both models accurately front-load storm peaks (peaking near 1/3 duration). |
| **Extreme Peaks** | **P99 Intensity** | 0.315 mm | 0.314 mm | 0.329 mm | Both models accurately capture the 99th percentile. |
| | **Max Instantaneous Peak** | **10.68 mm** | **5.60 mm** | **10.77 mm** | **Major Win for `v4`:** GMM conditioning successfully forces realistic peak cloudbursts. |
| | **Daily Max Rainfall** | **52.1 mm** | 26.5 mm | **27.6 mm** | **The Daily Max Paradox:** `v4` matches peak 10-min bursts but fails 24-hr totals due to short storm durations. |

---

### 💡 Key Takeaways & RL Impact

1. **The "Daily Max" Paradox:** Matching peak instantaneous rain (10.77 mm) is not enough to generate 50mm+ daily floods. Real flood events require heavy rain sustained over 6–12 continuous hours. Shuffling independent 14-day blocks breaks this temporal continuity.
2. **RL Training Suitability:**
   - **`v3` (Unconditional):** Great for standard day-to-day policy training, but dangerous for flood defense because it underestimates flood volumes by 50%.
   - **`v4` (Regime-Conditioned):** Great for testing immediate flash-flood responses, but teaches the agent that severe storms always end quickly (66 mins vs 100 mins).

---

## June 12, 2026 — Unconditional Model Validation & The 0.005 mm Sparsity Breakthrough

### 🔍 Overview
We evaluated the 10-year unconditional synthetic dataset `results/generated_data/rainfall_synthetic_10y_v3.csv` at 10-minute resolution, using a post-processing rule that sets values $< 0.005\text{ mm}$ to `0.0`.
- *Earlier Iterations:* `v1` (5-min resolution with raw Gaussian noise) and `v2` (5-min resolution with negative values removed).

---

### 📈 Breakthrough Results (Score: 0.733 / 1.000)

Applying the $0.005\text{ mm}$ threshold eliminated low-level background noise, dramatically improving structural realism.

| Metric | Real Ground Truth | Synthetic (Raw Output) | Synthetic (With 0.005mm Threshold) | Key Improvement |
|---|:---:|:---:|:---:|---|
| **Zero Fraction** | 89.5% | 45.0% | **89.9%** | Removed background noise haze. Max drought expanded from 4 hrs to 93 hrs (3.8 days). |
| **Storm Count** | 4,852 | 35,248 | **5,613** | Eliminated tens of thousands of false noise interruptions. |
| **Mean Duration** | 100 mins | 55 mins | **77 mins** | 40% improvement in storm continuity over raw output. |
| **Mean Peak Intensity**| 0.302 mm | 0.054 mm | **0.323 mm** | Fixed intensity dilution caused by sub-millimeter noise. |
| **Mean Storm Volume** | 1.43 mm | 0.20 mm | **1.18 mm** | Reached realistic total rainfall volume per storm. |

---

### ⚠️ Main Flaw: Underestimated Extreme Cloudbursts

While `v3` captured average storms well, it severely underestimated 1-in-10-year extreme events:
- **Max Instantaneous Peak:** 5.60 mm/10-min (Synthetic) vs. **10.68 mm** (Real) — *Halved*
- **Daily Max Rainfall:** 26.5 mm (Synthetic) vs. **52.1 mm** (Real) — *Halved*

**Why This Happens:** Unconditional diffusion models suffer from "regression to the mean." Because cloudbursts (>10 mm) account for less than 0.1% of all data points, standard loss functions treat extreme spikes as noise variance and smooth them out to ~5.6 mm.

---

## May 28, 2026 — Designing the Two-Tier Evaluation Pipeline

### 🔍 Overview
Standard generative metrics (like global MSE) fail for rainfall because a model can achieve low error simply by outputting zero rain 90% of the time. To ensure synthetic data is safe for downstream RL training, we built a domain-specific evaluation framework in `evaluate_conditional.py`, `metrics/conditional_metrics.py`, `visualize.py`, and `utils/utils_vis.py` (documented in `guides/EVALUATION_GUIDE.md`).

---

### 📐 Two-Tier Evaluation Structure

1. **Tier 1: Global Sequence Metrics**
   - **RNN Discriminative Score** (`test/disc_mean`): Evaluates how easily a classifier distinguishes synthetic from real sequences.
   - **Predictive MAE** (`test/pred_mean`): Measures short-term step-by-step predictability.
   - **Context FID** (`test/context_fid`): Measures global feature distribution alignment using TS2Vec embeddings.

2. **Tier 2: Storm-Specific (Conditional) Metrics**
   - **Wet-Window Discriminative Score:** Evaluates sequence quality strictly during active rain windows.
   - **Intensity JSD (< 0.05):** Jensen-Shannon Divergence of non-zero rainfall values.
   - **P99 Extreme Ratio:** Compares 99th percentile rainfall intensities.
   - **Contiguous Event Duration & Volume Statistics:** Tracks continuous storm duration, peak timing, and total storm accumulation.

---

## May 22, 2026 — Initial Data Ingestion & Baseline Model Adaptation

### 🔍 Overview
Adapted the unconditional ImagenFew diffusion framework to ingest 10-minute empirical precipitation data (`rainfall_10min.csv`).

### ⚙️ Implementation Details
- **Data Characteristics:** High-resolution rainfall differs from standard continuous time-series benchmarks (e.g., ETT, ECG, Weather) because it has extreme zero-value sparsity (>89% zeros), heavy-tailed intensity spikes, and sudden intermittency.
- **Code Setup:** Implemented custom dataset handler (`data_provider/datasets/custom.py`) and config (`configs/finetune/Rainfall.yaml`) supporting normalization, configurable sliding windows (configured default `seq_len = 24` / 4 hours, with data loader supporting up to 144), and multi-year autoregressive sampling.
- **Key Insight:** Standard metrics reward models for producing constant light drizzle. Future iterations must evaluate dry spells and storm events separately.
