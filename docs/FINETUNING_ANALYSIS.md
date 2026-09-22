# Fine-Tuning Methodology, Loss Dynamics, and Context-Length Comparison in ImagenFew Precipitation Diffusion

**Author:** Tanya Warrier
**Date:** September 22, 2026 (corrected 2026-09-22 — see banner below)
**Project:** Adapting ImagenFew Time-Series Diffusion for Sparse Precipitation Generation (Astlingen Record)
**Document Status:** Empirical study and methodological reference; several conclusions revised after independent verification.

> **Correction banner (2026-09-22).** This document originally described v10 as having
> "solved storm geometry," attributed a training-loss scaling pattern to model capacity
> rather than to a normalization artifact, described the base checkpoints as imposing a
> hard architectural ceiling, and recommended v10 for RL deployment. All four claims did
> not survive independent recomputation and have been corrected below; each correction is
> flagged inline. Full derivation: [`code_plan/AUDIT_2026-09-22.md`](../code_plan/AUDIT_2026-09-22.md).
> This document's companion, [`docs/RESEARCH_CONTRIBUTIONS.md`](RESEARCH_CONTRIBUTIONS.md),
> received the same corrections. `my notes/memo/day_1.md` (frozen 2026-09-19) governs
> where the two disagree.

---

## Executive Summary

This document analyses the fine-tuning methodology, training loss dynamics, and physical
performance of **ImagenFew** model iterations v1 through v10 for high-resolution synthetic
rainfall generation, all metrics recomputed against the canonical 2000–2007 training
partition (`my notes/memo/day_1.md` §3.1).

### Core findings, as they now stand:

1. **The context window comparison is the study's most promising lead, but is currently
   confounded and cannot yet be reported as causal.** v10 (context window expanded from
   `seq_len=24`, 2.0 h, to `seq_len=64`, 5.33 h) has the best storm-geometry numbers of any
   version tested: storm-count ratio 1.004× real (697.1 → 699.7/yr), storm duration 0.875×
   real, hourly ACF RMSE 0.049 (vs. 0.086 at L=24). But each context length was fine-tuned
   from a *different* pretrained checkpoint, and every checkpoint loads the identical
   1006/1008 parameters regardless of target `seq_len` — nothing in the fine-tuning
   procedure required this, and it means the L=24→64 comparison also changed the
   initialisation. See §3 for the evidence and the two-run experiment that would settle it.
2. **Methodological and governance corrections.** Retiring the invalid "stochastic Hidden
   Markov Model (HMM)" framing in favor of calendar-ordered / transition-sampled
   seasonal-phase conditioning, and establishing a strict chronological holdout split
   (2000–2007 Train, 2008 Validation, 2009 Test), are both real and completed. The
   validation split is not yet used for model selection — see §4.4.
3. **Loss behaviour under EDM preconditioning tracks context length once correctly
   normalised — there is no paradox.** The raw scalar loss increases with `seq_len` (0.062
   at L=24, 0.091 at L=36, 0.156 at L=64) because `train_regime.py`'s loss is averaged over
   the full 64-cell delay-embedding grid regardless of how many cells are active. Once
   rescaled to loss-per-active-cell, v10 has the **lowest** loss of the six versions
   compared (0.1556, vs. 0.1616–0.1664 for L=24/36). The earlier framing of this as "loss
   value ≠ sample quality, an EDM generative paradox" is withdrawn: correctly normalised,
   loss and sample quality point the same direction.

---

## 1. Fine-Tuning Methodology: How It Was Done

### 1.1 Underlying Architecture: Elucidated Diffusion Models (EDM)
ImagenFew is an **Elucidated Diffusion Model (EDM)** following Karras et al. (2022), distinct from classic discrete-step DDPM ($\epsilon$-prediction) models. It operates with continuous noise levels $\sigma$ and an explicit preconditioned denoiser in $x$-prediction:

$$\mathcal{L}(\theta) = \mathbb{E}_{x, \sigma, n} \left[ \lambda(\sigma) \| D_\theta(x + n; \sigma, c) - x \|^2 \right]$$

where:
- $x \in \mathbb{R}^{B \times 1 \times 8 \times 8}$ is the 1D rainfall sequence delay-embedded into an $8 \times 8$ grid and normalized via `StandardScaler` ($\mu \approx 0, \sigma \approx 1.0$).
- $c$ is a one-hot seasonal-phase conditioning vector ($4$ classes).
- $n \sim \mathcal{N}(0, \sigma^2 I)$ is additive Gaussian noise.
- $\sigma \sim \text{LogNormal}(P_{\text{mean}} = -1.2, P_{\text{std}} = 1.2)$ is sampled continuously per instance.
- $\lambda(\sigma) = \frac{\sigma^2 + \sigma_d^2}{(\sigma \cdot \sigma_d)^2}$ is the EDM loss weighting with $\sigma_d = 0.5$ — hardcoded in `models/ImagenFew/networks.py:712` and not yet updated to match the unit-variance `StandardScaler` output (`code_plan/LOSS_FUNCTION.md` T6.2, still open).

The network output is preconditioned such that:
$$D_\theta(y; \sigma, c) = c_{\text{skip}}(\sigma) y + c_{\text{out}}(\sigma) F_\theta(c_{\text{in}}(\sigma) y, c_{\text{noise}}(\sigma), c)$$

with:
$$c_{\text{skip}} = \frac{\sigma_d^2}{\sigma^2 + \sigma_d^2}, \quad c_{\text{out}} = \frac{\sigma \sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}}, \quad c_{\text{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_d^2}}$$

Because $\lambda(\sigma) c_{\text{out}}(\sigma)^2 = 1$, the denoiser $F_\theta$ sees a unit-variance regression target across all noise scales $\sigma \in [0.03, 3.0]$.

### 1.2 Delay-Coordinate Embedding (1D to 2D Image)
To leverage 2D convolutional and attention U-Net architectures, the 1D precipitation series is mapped to a 2D tensor via a delay-coordinate embedder (`models/ImagenFew/img_transformations.py`):
- Grid size: $8 \times 8$ (`img_resolution = 8`), set in each version's training config — **not** a property of the pretrained checkpoint (see §3).
- Stride and embedding parameters: `delay = 8`, `embedding = 8`.
- For $seq\_len = 24$, exactly $3$ of the $8$ columns are populated ($24$ steps); the remaining $5$ columns ($62.5\%$) are structural zero-padding enforced by a spatial mask $(1 - x_{\text{img\_mask}})$.
- For $seq\_len = 36$ (v9), $4.5$ columns are active.
- For $seq\_len = 64$ (v10), all $8$ columns are fully populated ($8 \times 8 = 64$ steps), eliminating structural padding.

### 1.3 Pretrained Base Checkpoints & Transfer Strategy

Fine-tuning initialised from pretrained checkpoints in `models_ckpt/ImagenFew/`:
`ImagenFew_12.ckpt`, `ImagenFew_24.ckpt`, `ImagenFew_36.ckpt`, `ImagenFew_64.ckpt`, each
54,577,102 bytes. Their exact pretraining corpus and hyperparameters are **not yet
confirmed** (`code_plan/PROVENANCE.md`, T0.3 open) — `configs/pretrain/pretrain.yaml`
suggests a single `stock` dataset, but this has not been cross-checked against the shipped
checkpoints' own training logs or metadata, which requires cluster access this repository's
local environment does not have (no torch installed).

**What is confirmed** is the shape-safe loading behaviour
(`load_checkpoint_safe` in `regime_training/train_regime.py`), which skips any tensor whose
shape does not match the target model and reinitialises it from a normal distribution.
Across all v8, v9 and v10 runs — targeting `seq_len` 24, 36 and 64 respectively, from three
differently-named checkpoint files — the load outcome is identical:

```
Skipping net.model.map_label.weight        (ckpt=[128, 32] vs model=[128, 4])
Skipping model_ema.modelmap_labelweight    (ckpt=[128, 32] vs model=[128, 4])
Loaded 1006 parameters, skipped 2
Loaded EMA weights (498 params)
```

Only the two class-embedding heads (43/32-class pretrained → 4-class target) are ever
skipped, in every run, at every `seq_len`. This means **no parameter in the U-Net backbone
depends on `seq_len` or on which of the four checkpoint files was loaded** — see §3 for why
this matters for the context-length comparison.

### 1.4 Optimizer, Hyperparameters, and Schedule
All fine-tuning runs across v5–v10 adhered to a unified optimization recipe:
- **Optimizer:** AdamW.
- **Learning Rate:** $\eta = 1 \times 10^{-4}$.
- **Weight Decay:** $1 \times 10^{-5}$.
- **Batch Size:** $2{,}048$ sequences per step.
- **Total Epochs:** $500$ epochs (${\approx}411$ iterations/epoch on the holdout training split).
- **EMA:** Decay $0.9999$, `ema_warmup` of $100$ epochs.
- **Gradient Clipping:** Maximum gradient norm $1.0$.
- **Diffusion Sampler:** Deterministic 2nd-order Heun ODE, $36$ steps, `S_churn = 0`.

### 1.5 Total Objective: Time-Domain + Spectral Term — Documented Defect, Analytically Inert at L=64

The total training loss minimized in `train_regime.py` is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{time}} + \beta_{\text{fft}} \mathcal{L}_{\text{fft}}$$

where:
$$\mathcal{L}_{\text{fft}} = \mathbb{E}_{x, \sigma, n} \left[ \lambda(\sigma) \| \text{FFT}_2(D_\theta(x+n)) - \text{FFT}_2(x) \|^2 \right]$$

**This term is not being presented as a contribution of this fork.**
`code_plan/LOSS_FUNCTION.md` §3.1 documented, before v8–v10 were trained, that with
`norm='forward'` the term is by Parseval's theorem exactly $1/64 = 1.5625\%$ of the time
term, and that the spatial mask applied to it (`(1 - x_img_mask)`) is in the wrong domain —
a spatial mask cannot meaningfully constrain a term already indexed by frequency.
`my notes/memo/day_1.md` §8 item 5 lists this as **permanently retired as mathematically
invalid**. v8, v9 and v10 were nevertheless all trained with `fft_weight: 1.0` and the bug
unchanged.

At $seq\_len = 64$ specifically, the masking bug is moot: the grid has no structural
padding left, so the mask is all-ones and the term reduces exactly to the Parseval limit.
The v10 training log confirms this at the best epoch (498):
$\mathcal{L}_{\text{time}} = 0.1532$, $\mathcal{L}_{\text{fft}} = 0.00239$, ratio $1.56\%$
— matching the theoretical $1.5625\%$ to three significant figures. This closes the
question of the FFT term's contribution to v10 analytically, without requiring a
`beta_fft=0` retrain: at L=64 the term is provably a constant rescale of the time loss and
could not have driven any of v10's storm-geometry improvement.

---

## 2. Chronological Evolution of Model Versions (v1 to v10)

| Version | Resolution | Context ($seq\_len$) | Conditioning | Assembly | Findings | Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **v1** | 5-min | 24 (2.0 h) | None | Sliding window | +93% volume bias (1365.9 mm); noise floor. | Retired |
| **v2** | 5-min | 24 (2.0 h) | None | Post-hoc negative clamp | Volume bias unchanged (1363.3 mm). | Retired |
| **v3** | 10-min | 24 (4.0 h) | Class 0 hardcoded | Sliding window + 0.005 mm floor | Volume calibrated (692.3 mm); peaks halved. | Retired |
| **v4** | 10-min | 24 (4.0 h) | 4 GMM regimes | Random permutation | Peaks captured; temporal continuity broken. | Retired |
| **v5** | 5-min | 24 (2.0 h) | 4 seasonal phases (3 daily features) | Calendar-ordered | Volume controlled; rapid label switching. | Retired |
| **v6** | 5-min | 24 (2.0 h) | 4 seasonal phases (365 DOY broadcast) | Calendar-ordered | Volume controlled; dead extreme tail (max 1.79 mm vs 5.56 mm real). | Retired |
| **v7** | 5-min | 24 (2.0 h) | 4 seasonal phases (1D mean + 4-day filter) | Calendar-ordered | Full intensity tail matched. Trained on all 10 years — **not a holdout result**. | Superseded (leakage) |
| **v8** | 5-min | 24 (2.0 h) | 4 seasonal phases (train-only fit) | Transition-sampled / Calendar | Clean holdout baseline. Storm count 1.206×, duration 0.742×, ACF RMSE 0.086 (train ref). | Clean baseline |
| **v9** | 5-min | 36 (3.0 h) | 4 seasonal phases (train-only fit) | Transition-sampled / Calendar | Volume dropped to 0.831×; duration remained short (0.732×). | Exploratory |
| **v10** | 5-min | 64 (5.33 h) | 4 seasonal phases (train-only fit) | Transition-sampled / Calendar | Best storm geometry: count 1.004×, duration 0.875×, ACF RMSE 0.049. Gate A 11/18. Confounded with checkpoint choice (§3). Fails Tier 4 (7/15 cells) and Tier 5 seasonality under transition-sampled assembly (monthly r = −0.11). | Best tested, not yet validated as causal |

All ratios above are against the 2000–2007 training reference, per `my notes/memo/day_1.md`
§3.1. Full Gate A results including Tiers 4–5 for every version:
`code_plan/AUDIT_2026-09-22.md` §4.

---

## 3. The Confound: Context Length Is Not Yet Separated From Checkpoint Choice

> **This section replaces an earlier claim that context-window expansion was "the single
> most impactful breakthrough" of the project.** That conclusion is premature. What follows
> is the evidence for and against it.

### 3.1 The evidence in favour

Hydrological audit of the Astlingen record establishes mean storm duration of 61.8 minutes
(training partition) with convective memory extending several hours further. At L=24
(2.0 h), the model observes barely twice a mean storm's length per training window; at
L=64 (5.33 h), several storm lifecycles fit inside one window. Consistent with this:

- **Storm count** fell from 1.206–1.266× real (v7, v8 at L=24) to **1.004×** (v10, L=64).
- **Mean storm duration** rose from 0.732–0.770× real to **0.875×**.
- **Hourly ACF RMSE** fell from 0.084–0.086 to **0.049** — the only version to pass this
  Gate A band.
- IDF analysis shows v10 is closest to observed at durations inside or near its receptive
  field (1 h ratio 0.94) and furthest at durations well beyond it (24 h ratio 0.60) — the
  pattern context-window expansion would predict.

### 3.2 The confound

Each context length was fine-tuned from a *different* base checkpoint
(`ImagenFew_24.ckpt`, `ImagenFew_36.ckpt`, `ImagenFew_64.ckpt`). All three files are
54,577,102 bytes; every training run's checkpoint-load log — independent of target
`seq_len` — reports the same outcome: **1006 parameters loaded, 2 skipped (the class-embedding
heads only)**. This means:

1. No weight in the U-Net backbone is actually different in shape or size across the three
   checkpoints in a way relevant to `seq_len`.
2. The v8→v9→v10 comparison changes both context length *and* initialisation
   simultaneously, in violation of `code_plan/REMEDIATION_PLAN.md`'s rule 5 ("one variable
   per experiment").
3. Two attempted extensions to L=144 and L=288 both failed — not with a checkpoint shape
   mismatch, but with `IndexError: index 8 is out of bounds for dimension 3 with size 8` in
   `img_transformations.py`, a data-transform limit set by the training config's
   `delay`/`embedding` values (fixed at 8 in every v8/v9/v10 config), independent of which
   checkpoint was loaded.

**Consequently:** the context-window result cannot presently be distinguished from a
checkpoint-initialisation effect, and the L=64 ceiling described in an earlier draft as
architectural is a config choice, not a property of the pretrained weights.

### 3.3 The two-run experiment this implies

- L=64 fine-tuned from `ImagenFew_24.ckpt` (same config as v8, only `seq_len`/grid changed).
- L=24 fine-tuned from `ImagenFew_64.ckpt` (same config as v10, only `seq_len`/grid changed).

If the storm-geometry gain tracks `seq_len` regardless of which checkpoint supplied the
initial weights, Change 2 in `docs/RESEARCH_CONTRIBUTIONS.md` is confirmed. If it tracks the
checkpoint instead, the context-window claim must be withdrawn and re-attributed. Neither
run has been made yet.

---

## 4. How Loss Values Behaved Across Versions

### 4.1 Training Trajectory Across Versions

| Version | Run ID | Base Checkpoint | $seq\_len$ | Ep 1 Loss | Best Loss (Epoch) | Final Loss (Ep 500) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| v5 | `1bff544a` | `ImagenFew_24` | 24 | 0.1227 | 0.0618 (Ep 495) | 0.0633 |
| v6 | `47d88206` | `ImagenFew_24` | 24 | 0.1212 | 0.0617 (Ep 397) | 0.0625 |
| v7 | `c3ef645f` | `ImagenFew_24` | 24 | 0.1207 | 0.0620 (Ep 499) | 0.0628 |
| v8 | `ed17d299` | `ImagenFew_24` | 24 | 0.1276 | 0.0624 (Ep 433) | 0.0631 |
| v9 | `160156d4` | `ImagenFew_36` | 36 | 0.1601 | 0.0909 (Ep 453) | 0.0921 |
| v10 | `aebe363f` | `ImagenFew_64` | 64 | 0.2545 | 0.1556 (Ep 498) | 0.1580 |

### 4.2 The Loss Scaling Is a Normalization Artifact, Not a Capacity Effect

**This subsection replaces an earlier "EDM Generative Paradox" framing.** `train_regime.py`
line 343 computes:

```python
loss = (weight * (w_int * time_loss + fft_weight * fft_loss) * signal).mean()
```

`.mean()` averages over **all 64 cells** of the delay-embedding grid, while `signal =
1 - x_img_mask` is nonzero only on the `seq_len` active cells. The reported scalar is
therefore proportional to `seq_len / 64`, not solely to how well the denoiser fits the
active data. Rescaling each version's best loss by `64/seq_len` to compare loss-per-active-cell:

| Version | $seq\_len$ | Best loss (raw) | × 64/$seq\_len$ (per active cell) |
|---|---|---|---|
| v5 | 24 | 0.0618 | 0.1647 |
| v6 | 24 | 0.0617 | 0.1645 |
| v7 | 24 | 0.0620 | 0.1654 |
| v8 | 24 | 0.0624 | 0.1664 |
| v9 | 36 | 0.0909 | 0.1616 |
| v10 | 64 | 0.1556 | **0.1556** |

Once normalised this way, **v10 has the lowest per-active-cell loss of the six versions
compared**, consistent with — not contradicting — its better storm-geometry outcome. The
earlier claim that "v10's higher scalar loss does not mean it is an inferior model" is
correct in spirit but attributed the effect to the wrong cause (denoising difficulty scaling
with dimensionality, as an intrinsic property of harder reconstruction) rather than to the
`.mean()` reduction's denominator. There is no paradox to explain once the normalization is
corrected.

### 4.3 Loss Is Insensitive to Conditioning Quality at Fixed Length

For all $seq\_len = 24$ runs (v5–v8), best converged loss lands within
$[0.0617, 0.0624]$ regardless of whether the intensity tail collapsed (v6) or storms
fragmented (v7, v8). Under EDM preconditioning with $\lambda(\sigma) c_{\text{out}}^2 = 1$,
the loss is dominated by regression over the ~91% zero-rain mass, making the scalar
insensitive to conditioning nuance. This finding is unchanged from earlier drafts.

### 4.4 Gradient Clipping and the Untested Validation Loop

v10 experienced substantial early optimisation pressure: mean gradient norm 1.350 with
63.2% of batches clipped at epoch 1, falling to 10.2% by epoch 25 and 1.5% by epoch 100.
v8 saw comparatively little clipping (4.1% at epoch 1). This is consistent with a wider
context window producing larger gradient spikes on heavy-rain batches early in training,
successfully stabilised by the optimiser.

Separately: `train_regime.py` builds a validation dataset from `val_years_labelled.csv`
(2008) and logs its size, but never computes a validation loss or uses one for checkpoint
selection — `is_best = avg_loss < best_loss` is evaluated on the training loss throughout.
Describing 2008 as used for "checkpoint early stopping" (as an earlier draft of this
document did) is not yet accurate for any v8–v10 run. `code_plan/REMEDIATION_PLAN.md` T6.5
is still open.

---

## 5. Quantitative Evaluation

Evaluated against the **2000–2007 training partition** (`my notes/memo/day_1.md` §3.1),
using `scripts/gate_a_scorecard.py`. This differs from evaluating against the full
2000–2009 record, which 80% of which is training data for v8–v10 — see
`code_plan/AUDIT_2026-09-22.md` §3.2 for the full three-reference comparison, which changes
several verdicts (e.g. v10's ACF RMSE is a Gate A PASS against the training partition and a
FAIL against the full record).

| Metric | Real (train) | v7 | v8 | v8_cal | v9 | v10 | v10_cal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Annual Volume ratio | 1.000 | 1.000 | 0.970 | 0.976 | 0.831 | 0.915 | 0.888 |
| Wet-spell ratio | 1.000 | 0.856 | 0.797 | 0.798 | 0.699 | 0.866 | 0.872 |
| Max ratio (single draw) | 1.000 | 1.016 | 0.791 | 0.897 | 0.509 | 0.830 | 1.015 |
| Hourly JSD | 0.000 | 0.055 | 0.062 | 0.063 | 0.100 | **0.044** | 0.049 |
| Hourly KS | 0.000 | 0.036 | 0.029 | 0.030 | 0.039 | **0.012** | 0.014 |
| Storm duration ratio | 1.000 | 0.770 | 0.742 | 0.739 | 0.732 | **0.875** | 0.875 |
| Storm count ratio | 1.000 | 1.266 | 1.206 | 1.216 | 1.198 | **1.004** | 0.995 |
| Hourly ACF RMSE | 0.000 | 0.084 | 0.086 | 0.087 | 0.072 | **0.049** | 0.050 |
| Tier 4 IDF cells passing (/15) | 15 | 6 | 6 | 6 | 0 | 7 | 7 |
| Monthly Pearson r | 1.000 | −0.36 | 0.44 | **0.91** | 0.11 | −0.11 | 0.65 |
| **Gate A bands passed (/18)** | 18 | 7 | 5 | 7 | 4 | **11** | 8 |

Single-draw values (annual max) are reported with the caveat that splitting each
realisation into its ten constituent years changes the ranking: v8's annual-max burst
(3.23 ± 0.55 mm) is closer to the observed interannual mean and spread (3.24 ± 1.12 mm)
than v10's (2.89 ± 0.95 mm). See `code_plan/AUDIT_2026-09-22.md` §3.3.

---

## 6. Status and Recommendations

1. **No version is currently recommended for RL training.** `code_plan/ACCEPTANCE_CRITERIA.md`
   Gate B (route rainfall through SWMM under a fixed controller) and Gate C (train-synthetic
   -test-real PPO comparison) have not been run, and `my notes/memo/day_1.md` §7 bars
   claiming synthetic-data validity for RL control before both. v10 is the best-scoring
   candidate on Gate A statistics alone (11/18 bands, best storm geometry of any version
   tested) and is the natural first candidate for Gate B once its confound with checkpoint
   choice (§3) is resolved — that is a recommendation about what to test next, not a
   deployment recommendation.
2. **Do not average v10 (transition-sampled) with v8_cal or v10_cal as a substitute for
   fixing either individually.** v10's seasonality is essentially absent (monthly r =
   −0.11); v8_cal's Gate A tally is weaker (7/18). Mixing the two has not been tested for
   producing a coherent combined distribution.
3. **Priority experiment: decouple context length from checkpoint choice** (§3.3). Until
   this runs, Change 2 of `docs/RESEARCH_CONTRIBUTIONS.md` is a strong correlation, not a
   demonstrated causal effect, and the thesis's central architectural claim rests on it.
4. **Fix the validation loop before further training**, so that future checkpoint selection
   is not conflated with training-loss overfitting (§4.4).
