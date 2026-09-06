# Research Progress Diary & Changelog: Synthetic Rainfall Generation via Time-Series Diffusion

**Project:** Adapting ImagenFew and Time-Series Diffusion Models for Sparse Rainfall Data  
**Goal:** Create realistic synthetic precipitation datasets to train Reinforcement Learning (RL) agents for stormwater management, reservoir control, and flood regulation.

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
