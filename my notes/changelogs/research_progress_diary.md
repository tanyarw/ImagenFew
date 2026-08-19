# Research Progress Diary & Changelog: Synthetic Rainfall Generation via Time-Series Diffusion

**Project:** Adapting ImagenFew and Time-Series Diffusion Models for Sparse Rainfall Data  
**Goal:** Create realistic synthetic precipitation datasets to train Reinforcement Learning (RL) agents for stormwater management, reservoir control, and flood regulation.

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
