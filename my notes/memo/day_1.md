# Experiment Control Sheet & Research Governance Memo

**Date:** September 19, 2026  
**Project:** Synthetic Precipitation Generation via Time-Series Diffusion (ImagenFew)  
**Document Status:** FROZEN — Canonical Baseline for Thesis & Evaluation  
**Directive:** **CUT IMMEDIATELY — DO NOT RUN ANOTHER MODEL TODAY.** The primary risk to this project is not lack of models; it is inconsistent definitions, contaminated holdouts, and ungrounded claims.

---

## 1. One-Page Experiment Control Sheet (v1 – v7)

Every historical model iteration is cataloged below with its temporal resolution, architectural context window, conditioning mechanism, generation protocol, and current lifecycle status.

| Version | Temporal Resolution | Architecture / Context Window | Conditioning Mechanism | Assembly / Generation Protocol | Key Outcome & Failure Mode | Lifecycle Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1** | **5-minute** | ImagenFew EDM ($seq\_len = 24$, 2 h) | None (Unconditional) | Unconditional sliding window | $1.93\times$ annual volume bias ($1365.9\text{ mm}$); pervasive Gaussian noise floor. | **ABANDONED** (Baseline only) |
| **v2** | **5-minute** | ImagenFew EDM ($seq\_len = 24$, 2 h) | None (Unconditional) | Post-hoc negative value thresholding | Volume bias persists ($1363.3\text{ mm}$); background noise haze remains. | **ABANDONED** (Baseline only) |
| **v3** | **10-minute** | ImagenFew EDM ($seq\_len = 24$, 4 h) | Class 0 conditioned (mislabelled unconditional) | Sliding window + $0.005\text{ mm}$ threshold | Volume calibrated ($692.3\text{ mm}$); flood peaks halved ($5.60\text{ mm}$ vs $10.68\text{ mm}$ real). Stuck in calm regime. | **ABANDONED** (Resolution deprecated) |
| **v4** | **10-minute** | ImagenFew EDM ($seq\_len = 24$, 4 h) | 4-class GMM regimes on 14-day window stats | Random block permutation (`rng.permutation`) | Captured peaks ($10.77\text{ mm}$); broke temporal continuity (Daily Max Paradox; storm duration collapsed to $66\text{ min}$). | **ABANDONED** (Random shuffling flaw) |
| **v5** | **5-minute** | ImagenFew EDM ($seq\_len = 24$, 2 h) | 4-level seasonal phase (3 daily features) | Calendar-ordered block assembly (overlap-add) | Volume controlled ($664.6\text{ mm}$); rapid label switching; context window halved from 4 h to 2 h. | **ABANDONED** (Superseded by v7) |
| **v6** | **5-minute** | ImagenFew EDM ($seq\_len = 24$, 2 h) | 4-level seasonal phase (365 DOY broadcast) | Calendar-ordered block assembly (overlap-add) | Volume controlled ($644.7\text{ mm}$); dead extreme tail (max capped at $1.787\text{ mm}$ vs $5.555\text{ mm}$ real). | **ABANDONED** (Tail collapse failure) |
| **v7** | **5-minute** | ImagenFew EDM ($seq\_len = 24$, 2 h) | 4-level seasonal phase (1-D smoothed mean intensity + min-duration) | Calendar-ordered block assembly (overlap-add) | **Best generator:** volume ratio $1.000$, zero % $90.97\%$, full marginal tail matched ($P_{99}, P_{99.9}, \text{max}$). **Fails storm geometry** (duration $-24\%$, count $+30\%$). | **ACTIVE BENCHMARK** (Single-draw production run; holdout retrain pending) |

---

## 2. Frozen Canonical Reference Dataset & Split

The canonical observed precipitation dataset is permanently frozen. No script may re-compute or redefine observed baseline statistics from unverified files.

### 2.1 Gauge Record & Intervals
- **Source Record:** 10-year Astlingen gauge record (`data/rainfall/real_rainfall_data.csv`), 4-gauge spatial arithmetic mean.
- **Native Temporal Resolution:** 5 minutes ($\Delta t = 5\text{ min}$).
- **Intervals Per Standard Year:** Exactly **105,120 intervals/year** ($365\text{ days} \times 24\text{ hours/day} \times 12\text{ intervals/hour}$).
- **Total Record Length:** 10 years = 3,650 days = **1,051,200 intervals** (leap days standardly excluded upstream to maintain uniform 365-day climatological calendar alignment).
- **Wet Threshold:** Permanently fixed at **$0.005\text{ mm}$**. Any interval with rainfall depth $< 0.005\text{ mm}$ is strictly defined as dry ($0.0\text{ mm}$). This threshold reflects the physical gauge measurement floor and eliminates generative background noise.

### 2.2 Frozen Chronological Data Split
Random shuffling or row-based 80/10/10 splitting across time is strictly prohibited to prevent multi-day temporal and seasonal leakage. The record is partitioned into discrete chronological calendar years:

- **Training Set (2000–2007):**
  - **Span:** 8 full calendar years (2000-01-01 00:00 to 2007-12-31 23:55).
  - **Sample Count:** 840,960 intervals ($80.00\%$ of dataset).
  - **Climatology:** Mean annual depth $709.6\text{ mm}$, zero fraction $91.03\%$, mean wet intensity $0.0751\text{ mm}/5\text{-min}$.
  - **Usage:** Model parameter optimization and training of seasonal-phase cluster definitions.
- **Validation Set (2008):**
  - **Span:** 1 full calendar year (2008-01-01 00:00 to 2008-12-31 23:55).
  - **Sample Count:** 105,120 intervals ($10.00\%$ of dataset).
  - **Climatology:** Convective / heavy-storm anomaly year; annual depth $755.7\text{ mm}$ ($1.07\times$ train mean), zero fraction $92.22\%$, mean wet intensity $0.0923\text{ mm}/5\text{-min}$ ($+23\%$), $P_{99.9} = 1.734\text{ mm}$.
  - **Usage:** Hyperparameter selection, checkpoint early stopping, and intermediate tuning.
- **Held-Out Test Set (2009):**
  - **Span:** 1 full calendar year (2009-01-01 00:00 to 2009-12-31 23:55).
  - **Sample Count:** 105,120 intervals ($10.00\%$ of dataset).
  - **Climatology:** Mild precipitation year; annual depth $661.0\text{ mm}$ ($0.93\times$ train mean), zero fraction $91.38\%$, maximum burst $2.005\text{ mm}/5\text{-min}$.
  - **Usage:** Locked. Evaluated strictly once for final thesis figures and Gate C (TSTR).

---

## 3. Frozen Metric Definitions & Resolution of Discrepancies

### 3.1 Resolution of Value Differences
An audit of historical reports identified two sets of apparent ground-truth targets:
1. Mean wet-spell duration of **$30.4\text{ minutes}$** vs **$32.1\text{ minutes}$**.
2. Mean storm duration of **$61.0\text{ minutes}$** vs **$62.3\text{ minutes}$** (and storm count of $682.6/\text{yr}$ vs $680.0/\text{yr}$).

#### Root Cause Analysis:
- **Target A ($30.4\text{ min}$ wet-spell / $61.0\text{ min}$ storm duration / $682.6\text{ storms/yr}$):**
  Computed strictly on the **8-Year Training Partition (2000–2007)** in `data_analysis/Train_Val_Test_Split_Analysis.ipynb`.
- **Target B ($32.1\text{ min}$ wet-spell / $62.3\text{ min}$ storm duration / $680.0\text{ storms/yr}$):**
  Computed across the **Entire 10-Year Historical Record (2000–2009)** in `scripts/compare_all_versions.py`. The inclusion of the convective holdout year 2008 and mild year 2009 (which averaged $67.5\text{ min}$ storm duration) slightly lifts the 10-year pooled averages.

#### Definitional Decision for the Thesis:
- **Canonical Thesis Ground-Truth Target:** **The Training Partition (2000–2007)** is frozen as the canonical ground-truth baseline for model development, statistical screening (Gate A), and holdout comparison:
  - **Mean Wet-Spell Duration:** **$30.4\text{ minutes}$** (Pass band: $27.4$–$33.4\text{ min}$, ratio $0.90$–$1.10$).
  - **Mean Storm Duration ($\ge 15\text{ min}$):** **$61.0\text{ minutes}$** (Pass band: $54.9$–$67.1\text{ min}$, ratio $0.90$–$1.10$).
  - **Mean Storm Frequency:** **$682.6\text{ storms/year}$**.
- **Documented Sensitivity Result:** The pooled 10-year whole-record values (**$32.1\text{ minutes}$** wet spell, **$62.3\text{ minutes}$** storm duration, **$6800\text{ total storms}$**) shall be documented strictly as a multi-year climate sensitivity analysis demonstrating interannual variation across wet convective and dry regimes.

---

### 3.2 Formal Mathematical Metric Definitions

1. **Wet Threshold ($I_{\text{wet}}$):**
   $$x_t \ge \theta,\quad \theta = 0.005\text{ mm}$$
   Intervals with $x_t < 0.005\text{ mm}$ are assigned $0.0\text{ mm}$.
2. **Annual Volume ($V_{\text{ann}}$):**
   $$V_{\text{ann}} = \frac{1}{N_{\text{yr}}} \sum_{t=1}^{N} x_t \quad (\text{mm/year})$$
3. **Zero Fraction ($f_0$):**
   $$f_0 = \frac{1}{N} \sum_{t=1}^N \mathbb{I}(x_t = 0.0) \times 100\%$$
4. **Wet Intensity ($\mu_{\text{wet}}$):**
   $$\mu_{\text{wet}} = \frac{1}{N_{\text{wet}}} \sum_{t: x_t > 0} x_t \quad (\text{mm/interval})$$
5. **Wet Spell:**
   A maximal contiguous sequence of time steps $\{x_s, x_{s+1}, \dots, x_e\}$ such that $x_t > 0.005\text{ mm}$ for all $t \in [s, e]$. Duration $= (e - s + 1) \times 5\text{ minutes}$.
6. **Storm Event:**
   A hydrologically significant contiguous wet spell with duration $\ge 15\text{ physical minutes}$ ($\ge 3$ consecutive 5-min intervals, or $L \ge 3$). Isolated 5-minute or 10-minute bursts are classified as transient drizzle spells, not storm events. Inter-storm separation requires an inter-event dry time (MIT) $\ge 15\text{ minutes}$.
7. **Tail Extremes ($P_{99}, P_{99.9}, \text{Max}$):**
   Empirical quantile values evaluated over the full series distribution (standard benchmark: $P_{99} = 0.160\text{ mm}$, $P_{99.9} = 0.565\text{ mm}$, $\text{Max} = 5.555\text{ mm}/5\text{-min}$) and over wet-only intervals ($P_{99,\text{wet}} = 0.595\text{ mm}$, $P_{99.9,\text{wet}} = 1.613\text{ mm}$ on train).
8. **Autocorrelation Function (ACF) & Hourly RMSE:**
   $$\rho_k = \frac{\sum_{t=1}^{N-k} (x_t - \mu)(x_{t+k} - \mu)}{\sum_{t=1}^N (x_t - \mu)^2}$$
   - **Lag-1 ACF (Native):** $\rho_1$ at $\Delta t = 5\text{ min}$.
   - **Hourly ACF RMSE:** Evaluated on hourly summed totals over lags $k = 1, \dots, 24\text{ hours}$:
     $$\text{RMSE}_{\text{ACF}} = \sqrt{\frac{1}{24} \sum_{k=1}^{24} \left(\rho_{k, \text{synthetic}} - \rho_{k, \text{observed}}\right)^2}$$
9. **Intensity-Duration-Frequency (IDF):**
   Rolling depth maxima over durations $D \in \{15\text{ min}, 1\text{ h}, 3\text{ h}, 6\text{ h}, 24\text{ h}\}$. Annual Maximum Series (AMS) fitted via Generalized Extreme Value (GEV) or Gumbel distribution to calculate return period depths for $T \in \{1, 2, 5, 10\}\text{ years}$.

---

## 4. Corrected Scientific Terminology

To maintain scientific integrity and resolve peer/panel audit critiques, the following terminology replacements are mandatory across all code, docstrings, papers, and thesis chapters:

| Deprecated / Incorrect Term | Mandatory Corrected Term | Physical & Mathematical Justification |
| :--- | :--- | :--- |
| **"HMM state" / "HMM conditioning"** | **"Seasonal-phase index" / "Seasonal-phase conditioning"** | The conditioning variable is a deterministic periodic function of day-of-year derived from 14-day smoothed climatological mean intensity, repeating identically each year. It does not represent a stochastic latent weather state. Successive 5-min samples have near 100% autocorrelation, completely violating HMM emission independence. |
| **"Markov assembly" / "Transition-aware Markov assembly"** | **"Calendar-ordered block assembly"** | Sequence assembly follows deterministic calendar progression across seasonal boundaries with Laplace-smoothed boundary transition tables, rather than a genuine first-order Markovian atmospheric process. |
| **"DDPM" (Denoising Diffusion Probabilistic Model)** | **"EDM diffusion model" (Elucidated Diffusion Model)** | The architecture is Karras et al. (2022) EDM. It uses continuous-time noise levels $\sigma$, an explicit preconditioned denoiser in $x$-prediction $D_\theta(x+n; \sigma, c)$, log-normal noise sampling $\ln(\sigma) \sim \mathcal{N}(-1.2, 1.2^2)$, and a 2nd-order deterministic Heun ODE sampler. It does not employ Ho et al. discrete-step $\epsilon$-prediction with linear $\beta$ schedules. |
| **"Unconditional v3"** | **"Class-0 conditioned v3"** | v3 retained active class-conditioning embeddings with $c=0$ hardcoded during reverse sampling. It was not structurally unconditional. |
| **"L2 regresses to the mean"** | **"Finite capacity tail under-resolution"** | Denoising score matching at optimum computes $\mathbb{E}[x \mid y]$ (the exact Bayesian posterior mean), which reproduces tails given sufficient capacity. Flattened extremes in early iterations stemmed from gradient clipping ($1.0$), 10,000-step EMA over-smoothing, and heavy sparsity ($10^{-6}$ frequency contributing $<1\%$ of loss). |

---

## 5. Performance Benchmark: v7 versus Real Ground Truth

Evaluation of the primary candidate generator (`v7`, 5-minute resolution, single 10-year realisation) against the 10-year Astlingen gauge reference:

| Evaluation Metric | Real Ground Truth (10-Yr Full) | Real Ground Truth (Train 2000–07) | Synthetic v7 Output | v7 / Real Ratio | Gate A Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Annual Volume ($V_{\text{ann}}$)** | $709.4\text{ mm}$ | $709.6\text{ mm}$ | **$709.5\text{ mm}$** | $1.000$ | **PASS** |
| **Zero Fraction ($f_0$)** | $90.95\%$ | $91.03\%$ | **$90.97\%$** | $1.000$ ($+0.02\text{ pp}$) | **PASS** |
| **Mean Wet Intensity** | $0.0746\text{ mm}$ | $0.0751\text{ mm}$ | **$0.0747\text{ mm}$** | $1.001$ | **PASS** |
| **Extreme $P_{99}$ Intensity** | $0.160\text{ mm}$ | $0.160\text{ mm}$ | **$0.161\text{ mm}$** | $1.006$ | **PASS** |
| **Extreme $P_{99.9}$ Intensity** | $0.565\text{ mm}$ | $0.565\text{ mm}$ | **$0.568\text{ mm}$** | $1.006$ | **PASS** |
| **Maximum 5-Min Burst** | $5.555\text{ mm}$ | $5.555\text{ mm}$ | **$5.644\text{ mm}$** | $1.016$ | **PASS** |
| **Storm Event Count** | $6,800\text{ storms}$ | $682.6/\text{yr}$ | **$8,824\text{ storms}$** | **$1.298$** ($+29.8\%$) | **FAIL** |
| **Mean Storm Duration ($\ge 15$m)** | $62.3\text{ min}$ | $61.0\text{ min}$ | **$47.6\text{ min}$** | **$0.764$** ($-23.6\%$) | **FAIL** |
| **Mean Wet-Spell Duration** | $32.1\text{ min}$ | $30.4\text{ min}$ | **$27.4\text{ min}$** | **$0.854$** ($-14.6\%$) | **FAIL** |
| **Lag-1 ACF (Native 5-Min)** | $0.8537$ | $0.8518$ | **$0.8478$** | $0.993$ (Diff: $-0.0059$) | **PASS** |
| **Hourly ACF RMSE (1–24 h)** | $0.0000$ | $0.0000$ | **$0.0906$** | $\text{RMSE} > 0.05$ | **FAIL** |

> **Diagnostic Summary:** v7 delivers the exact annual volume and completely resolves the tail collapse problem that plagued v1–v6. However, it **fails storm geometry**: rainfall is fragmented into too many short bursts ($+30\%$ storm count, $-24\%$ storm duration). The root cause is the short context window ($seq\_len = 24 = 2\text{ hours}$ at 5-minute sampling), which barely spans twice the physical length of a storm cell.

---

## 6. Claims Governance Table

To prevent overclaiming, every scientific assertion is audited against existing empirical evidence:

| Research Claim | Evidence Available | Evidence Still Missing | Current Status |
| :--- | :--- | :--- | :---: |
| **"Diffusion models can accurately reproduce rainfall annual volume and dry-fraction intermittency."** | v7 matches annual volume to $1.000\times$ ($709.5$ vs $709.4\text{ mm}$) and zero fraction within $0.02\text{ pp}$. | $\ge 30$-member ensemble variance under holdout split. | **ALLOWED (Conditioned on single-draw caveat)** |
| **"v7 resolves the cloudburst tail collapse and matches extreme precipitation up to 10-year maxima."** | Single-draw v7 reproduces $P_{99}$ ($1.006\times$), $P_{99.9}$ ($1.006\times$), and maximum burst ($1.016\times$). | Extreme value verification across multi-seed holdout ensembles and multi-scale IDF curves. | **ALLOWED (Conditioned on single-draw caveat)** |
| **"The conditioning pipeline is a Hidden Markov Model capturing stochastic weather states."** | HMM clustering was executed on historical data. | None — mathematically disproven. State sequence is a deterministic function of calendar DOY on 14-day smoothed data. | **DISALLOWED (UNSAFE CLAIM)** |
| **"Block generation utilizes a 1st-order Markov chain of meteorological transitions."** | Transition matrices were computed with Laplace smoothing. | None — disproven. Off-diagonal transitions reflect seasonal calendar progression. | **DISALLOWED (UNSAFE CLAIM)** |
| **"v7 faithfully reproduces storm geometry and temporal persistence."** | v7 native lag-1 ACF matches ($0.848$ vs $0.854$). | Contradicted by evidence: storm duration is $-24\%$ short, storm count is $+30\%$ excessive, and hourly ACF RMSE ($0.0906$) fails Gate A. | **DISALLOWED (UNSAFE CLAIM)** |
| **"The generative model is a DDPM."** | Denoising diffusion loss curves. | Contradicted by architecture: objective, preconditioning, and sampling are strictly EDM (Karras et al. 2022). | **DISALLOWED (UNSAFE CLAIM)** |
| **"The synthetic rainfall generator has been validated on an unpolluted held-out test split."** | Holdout split analysis notebook exists. | v5–v7 checkpoints were trained with `train_csv == test_csv` across 2000–2009. Retraining on 2000–2007 is pending. | **DISALLOWED (UNSAFE CLAIM)** |
| **"Synthetic rainfall improves downstream RL agent policy performance in SWMM-Astlingen."** | Theoretical premise and project proposal. | Gate B hydraulic routing has not been run; Gate C (TSTR PPO training) has not been run. | **DISALLOWED (UNSAFE CLAIM)** |
| **"Loss function reweighting is required to solve extreme tail flattening."** | Reweighted loss implementation in `train_regime.py`. | v7 solved the tail without loss reweighting ($\alpha = 0$). The open failure is temporal persistence, not loss weighting. | **DISALLOWED (UNSAFE CLAIM)** |
| **"The generator reproduces interannual climate variability (drought years and heavy convective years)."** | Multi-year concatenated output. | Model has no interannual conditioning; year-to-year variation is limited to block sampling noise. | **DISALLOWED (UNSAFE CLAIM)** |

---

## 7. Consolidated List of Unsafe Claims

The following claims are **strictly barred** from inclusion in the thesis or publications until formal empirical remediation is completed:
1. Claiming the conditioning mechanism is an **HMM** or that block assembly is a **Markovian weather-state transition process**.
2. Claiming that **storm dynamics or storm durations are solved** in v7.
3. Describing the model architecture as a **DDPM** instead of an **EDM**.
4. Quoting quantitative performance metrics as **unpolluted generalization results** before holdout retraining on 2000–2007.
5. Claiming synthetic data is **validated for RL stormwater control** prior to Gate B (SWMM routing) and Gate C (TSTR).
6. Asserting that **heavy-tail loss reweighting** is necessary to resolve extreme cloudbursts in the production model.
7. Claiming the model generates **realistic interannual climate variability** (such as the 2003 European drought).

---

## 8. Exact List of Abandoned Experiments & Methodologies

The following paths, code configurations, and metrics are permanently retired:
1. **Unconditional 10-Minute Generation (`v3` setup):** Abandoned due to severe extreme suppression (peaks halved to $5.60\text{ mm}$) and static regime locking in dry weather.
2. **GMM 14-Day Block Clustering with Random Permutation (`v4` setup):** Abandoned due to the "Daily Max Paradox" and destruction of multi-day autocorrelation caused by `rng.permutation`.
3. **Arbitrary Equal-Weighted "RL Fitness Score":** Abandoned because it rewarded physically dangerous models (ranking v3 at $0.733$ above v4 at $0.622$ despite v3 halving flood peaks). Superseded by staged physical gates (Gates A, B, C).
4. **Heavy-Tail Loss Reweighted Fine-Tuning as a Production Fix:** Deprecated from active production plans. v7 matched the intensity marginal without loss intervention; effort is redirected toward expanding temporal context length ($seq\_len \ge 144$).
5. **Spatial Masking of 2D Frequency Spectra (FFT Loss Bug):** The implementation multiplying `(1 - x_img_mask)` onto frequency columns is abandoned as mathematically invalid.
6. **Row-Based 80/10/10 Dataset Partitioning:** Leaked cross-year statistics. Permanently replaced by chronological calendar-year splits.
7. **Ad-Hoc Radar Scorecard Multipliers:** Abandoned unmotivated composite distance metrics ($1 - 20\times\text{Wasserstein}$) in favor of individual metric bootstrap bands.

---

**Sign-off:**  
This control sheet represents the canonical state of project governance. All subsequent modeling, evaluation, and documentation must adhere strictly to these frozen definitions.
