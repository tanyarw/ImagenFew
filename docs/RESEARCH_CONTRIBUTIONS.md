# Research Contributions: Domain-Specific Precipitation Generation with ImagenFew

**Project:** ImagenFew × Astlingen Precipitation
**Date:** September 22, 2026 (corrected 2026-09-22 — see banner below)
**Author:** Tanya Warrier
**Base Paper:** *"Time Series Generation Under Data Scarcity: A Unified Generative Modeling Approach"* — Gonen, Pemper, Naiman, Berman, Azencot. NeurIPS 2025.
**Codebase:** Fork of [github.com/azencot-group/ImagenFew](https://github.com/azencot-group/ImagenFew)

> **Correction banner (2026-09-22).** The version of this document dated the same day
> overstated several results and misdescribed two engineering constraints. Every number
> below has been recomputed with `scripts/gate_a_scorecard.py` against the canonical
> 2000–2007 training reference (`my notes/memo/day_1.md` §3.1), not the full 2000–2009
> record the first draft used. Full derivation and the checks behind every correction:
> [`code_plan/AUDIT_2026-09-22.md`](../code_plan/AUDIT_2026-09-22.md). Where this document
> and `my notes/memo/day_1.md` (frozen 2026-09-19) disagree, `day_1.md` governs; its
> Claims Governance Table (§6) and Consolidated List of Unsafe Claims (§7) are binding on
> everything below.

---

## 1. What the Base Paper (ImagenFew) Does

ImagenFew (NeurIPS 2025) proposes a **unified diffusion-based generative framework** for time series under data scarcity, making three primary claims:

| Contribution | Details |
|---|---|
| **Benchmark** | First large-scale study of 4 generative models (TimeGAN, KoVAE, DiffusionTS, ImagenTime) on 12 datasets under 5%–15% data splits |
| **Architecture** | Dynamic Convolution (DyConv) layer for varying channel dims; dataset token conditioning for domain-aware generation |
| **Pre-Training** | Single model pre-trained on heterogeneous multi-domain corpus; fine-tuned per dataset with minimal examples |

The conditioning mechanism is a **dataset token** — a categorical label identifying which of the 12+ datasets the sample belongs to (e.g., "MuJoCo", "ETT"). This carries no domain-specific physical meaning.

Evaluation uses: Discriminative Score, Predictive Score, contextFID.
**None of these metrics address physical plausibility, storm geometry, or long-range autocorrelation structure.**

This fork has not run any of the base paper's four models or three metrics on Astlingen
rainfall. `models/KoVAE`, `models/ImagenTime` and `models/interpretable_diffusion`
(DiffusionTS) are vendored in this repository and `metrics/discriminative_torch.py`,
`metrics/predictive_metrics.py` and `metrics/context_fid.py` are implemented, with
`TS2Vec` encoders already trained for the Rainfall dataset
(`logs/TS2VEC/Rainfall_*.ckpt`) — but none has been invoked on this data. Everything below
compares ImagenFew configurations against each other, not against an external baseline.
That comparison is Open Research Direction 7 below and is a gap in the thesis, not a
claim already supported.

Our fork adapts ImagenFew to **high-resolution sparse precipitation generation** for urban
drainage simulation (Astlingen catchment, 4-gauge arithmetic mean, 5-minute resolution,
10-year record). The following describes what changed relative to the base paper, and —
separately, in §4 — what evidence currently supports each change.

---

## 2. What Was Changed Relative to the Base Paper

### Change 1 — Seasonal-Phase Conditioning (vs. Dataset Token Conditioning)

**Base paper:** Conditions on a categorical dataset identity token with no semantic content.

**This fork:** Replaces the generic dataset token with **4-level seasonal-phase
conditioning** derived from the physical climatology of the target gauge network:

1. Compute the 14-day rolling mean of daily precipitation totals over the 2000–2007
   training partition.
2. Fit a 4-state `GaussianHMM` on the smoothed, standardized signal, ordered monotonically
   by ascending mean intensity (Phase 0: Dry Baseline → Phase 3: Peak Cloudburst), with a
   4-day minimum-duration spell filter.
3. Broadcast the 365 resulting daily labels to 5-minute resolution and project them
   out-of-sample onto 2008/2009 by calendar interval slot.
4. At generation time, either **replay** these labels in true calendar order
   (`--assembly_mode calendar`) or **resample** the next block's phase stochastically from
   the empirical seasonal-phase transition matrix fit on the training years
   (`--assembly_mode markov`). Despite the flag name, the latter is not a model of a
   physical Markovian weather process — the transition matrix's off-diagonals encode which
   calendar season follows which, not stochastic meteorological transitions. See
   `my notes/memo/day_1.md` §4 for the full argument and the retired terminology this
   corrects.

**What the evidence currently shows:** this conditioning delivers the annual cycle only
under calendar-ordered assembly. Scored against the training reference, monthly-volume
correlation with the observed cycle is **0.91 (v8_cal)** but **−0.11 (v10, transition-sampled)**
and **0.65 (v10_cal)** — see §4 Gap 6. The mechanism carries a real physical signal, but as
currently deployed with transition-sampled assembly on the recommended model (v10), that
signal is not reaching the output. No version reproduces a diurnal cycle (r ranges
−0.50 to 0.23 across all seven versions against the ≥0.80 band); nothing in the
conditioning vector currently carries time-of-day information, so this is expected, not a
defect specific to any one version.

### Change 2 — Context-Window Expansion (Confounded with Checkpoint Choice)

**Base paper:** Treats seq_len as a fixed architectural parameter; does not investigate its
effect on physical temporal structure.

**This fork:** Compared three context windows, L ∈ {24, 36, 64} (2.0 h, 3.0 h, 5.33 h at
5-minute resolution), each fine-tuned from a differently-sized pretrained checkpoint
(`ImagenFew_24.ckpt`, `ImagenFew_36.ckpt`, `ImagenFew_64.ckpt`).

> At 5-minute resolution, L=24 spans **2.0 hours** — barely twice the mean storm duration of
> **61.8 minutes** (training partition), with convective memory extending several hours
> further. Models trained at L=24 fragmented storms: the storm-count ratio is 1.21–1.27×
> real and mean duration 0.74–0.77× real, regardless of conditioning quality.

| Metric (train reference) | v8 (L=24) | v9 (L=36) | v10 (L=64) | Real |
|---|---|---|---|---|
| Annual Storm Count (N/yr) | 840.8 (1.206×) | 835.5 (1.198×) | **699.7 (1.004×)** | 697.1 |
| Mean Storm Duration | 45.8 min (0.742×) | 45.3 min (0.732×) | **54.1 min (0.875×)** | 61.8 min |
| Hourly ACF RMSE | 0.0864 | 0.0716 | **0.0486** | 0.0000 |

**What the evidence currently shows, and what it does not:** L=64 is unambiguously the best
context length tested. What it does **not** yet show is that context length caused this,
because **the checkpoint choice is confounded with it**. All four base checkpoints
(`ImagenFew_{12,24,36,64}.ckpt`) are byte-identical in file size (54,577,102 bytes), and
every v8/v9/v10 training run — regardless of target `seq_len` — reports the identical
outcome:

```
Skipping net.model.map_label.weight        (ckpt=[128, 32] vs model=[128, 4])
Skipping model_ema.modelmap_labelweight    (ckpt=[128, 32] vs model=[128, 4])
Loaded 1006 parameters, skipped 2
```

No parameter shape in `models/ImagenFew/networks.py` depends on `seq_len`; the only tensors
that ever differ are the two class-embedding heads (43/32-class → 4-class), skipped and
reinitialised identically in all three runs. The v8→v10 comparison therefore changes two
variables at once — context length and initialisation — and cannot presently attribute the
gain to context length alone. Full detail and the two-run experiment that would decouple
them: `code_plan/AUDIT_2026-09-22.md` §3.1 and §6 (E1).

A second consequence: L=64 is **not** an architectural ceiling requiring an expensive new
foundation checkpoint (contrast §4 Gap 5). The delay-embedding grid
(`img_resolution=delay=embedding=8`) is set in the training config, not baked into any
checkpoint weight. Attempts to target L=144/288 with the current 8×8 grid failed with
`IndexError: index 8 is out of bounds for dimension 3 with size 8` — a config-level limit
(`delay × embedding = 64`), not a checkpoint limit. A 16×16 grid config should, in
principle, admit L=256 while still loading the same 1006/1008 parameters; this is an
unverified but cheap (one-epoch smoke test) prediction, not yet run.

**Where L=64 has and has not resolved storm geometry.** Intensity–duration–frequency (IDF)
analysis — computable from the committed data and never previously run for this project —
localises the remaining failure precisely. Fitting a Gumbel distribution to the 10-year
annual-maximum series at durations D ∈ {15 min, 1 h, 3 h, 6 h, 24 h} and comparing the T=10
year return-period depth to the training reference:

| D | v8 | v9 | v10 |
|---|---|---|---|
| 15 min | 0.82 | 0.48 | 0.81 |
| 1 h | 0.83 | 0.56 | **0.94** |
| 3 h | 0.70 | 0.54 | 0.77 |
| 6 h | 0.69 | 0.58 | 0.79 |
| 24 h | 0.57 | 0.47 | **0.60** |

v10 is close to observed at durations inside or near its 5.33 h receptive field (1 h) and
falls further short as the duration grows past it (24 h). That is the context-window
hypothesis in a stronger and more falsifiable form than the storm-count/duration numbers
alone: it predicts that extending the window further should improve longer durations
selectively, which E1/E2 in `code_plan/AUDIT_2026-09-22.md` §6 would test directly.
**All seven versions fail the Gate A Tier 4 IDF criterion** (every cell of a 5×3 table must
lie in 0.80–1.20; best case is v10/v10_cal at 7/15 cells).

### Change 3 — Physical Validity Evaluation Suite

**Base paper:** Discriminative Score / Predictive Score / contextFID — domain-agnostic;
does not test physical plausibility directly.

**This fork:** an evaluation harness following the tiered structure specified in
`code_plan/ACCEPTANCE_CRITERIA.md`:

| Tier | Metrics |
|---|---|
| **Tier 1 — Water Balance** | Annual volume, zero fraction, wet/dry spell duration |
| **Tier 2 — Intensity Marginal** | Wet-only P99/P99.9, max, hourly JSD, hourly KS |
| **Tier 3 — Temporal Structure** | Lag-1 ACF, hourly ACF RMSE, storm count/duration/volume |
| **Tier 4 — Multi-Scale Extremes** | Annual-maximum-series / IDF table across 5 durations × 3 return periods |
| **Tier 5 — Seasonality** | Monthly and diurnal Pearson correlation |

`scripts/run_evaluation.py` computes Tiers 1–3 descriptively (no pass/fail bands).
`scripts/gate_a_scorecard.py` (added 2026-09-22) computes all five tiers with the exact
bands from `ACCEPTANCE_CRITERIA.md` and a machine-readable verdict per metric — this is
what produced every ratio quoted in this document. Full Gate A tally against the training
reference, all 18 banded scalar metrics:

| | v7 | v8 | v8_cal | v9 | v9_cal | v10 | v10_cal |
|---|---|---|---|---|---|---|---|
| Bands passed | 7/18 | 5/18 | 7/18 | 4/18 | 3/18 | **11/18** | 8/18 |

v10 is the best-scoring version to date. **No version passes Gate A.**

Critical demonstration retained from earlier work: v6 scored well on Tier 2/3 distributional
metrics yet **failed catastrophically on maximum intensity** (1.79 mm vs 5.56 mm real),
which is exactly the kind of failure Tiers 1–3 alone can miss and Tier 4 (added this
revision) is designed to catch structurally rather than by inspection.

### Change 4 — Chronological Hold-Out Protocol

**Base paper:** Random subset sampling (5%, 10%, 15%) — not designed for temporally
correlated physical records.

**This fork:** Strict **chronological partitioning**, fit and enforced starting with v8:

- **Train:** 2000–2007 (8 years, 840,960 intervals, 80%)
- **Validation:** 2008 (1 convective-heavy year, 10%)
- **Test:** 2009 (1 mild year, 10%, locked)

All seasonal-phase labels and transition matrices are fit **exclusively on the training
partition** (`scripts/fit_seasonal_labels.py`). v7 and earlier trained with
`train_csv == test_csv` across all ten years; v8, v9 and v10 do not, and this is a real and
completed piece of remediation.

**What remains open:** the validation partition is loaded (`RegimeDataset` on
`val_years_labelled.csv`) and its size logged, but `regime_training/train_regime.py` never
computes a validation loss or uses it for checkpoint selection — `is_best = avg_loss <
best_loss` is evaluated on the **training** loss only. Describing 2008 as "used for
checkpoint early stopping and model selection" (as an earlier draft of this document and
`my notes/memo/day_1.md` §2.2 both do) is not yet true of any v8–v10 run. This is
`code_plan/REMEDIATION_PLAN.md` task T6.5, still open.

Separately, every reported ratio in this document is computed against the **2000–2007
training partition**, per `my notes/memo/day_1.md` §3.1. Evaluating instead against the
full 2000–2009 record (which `scripts/run_evaluation.py` does by default) changes several
verdicts materially — e.g. v10's hourly ACF RMSE is 0.049 (PASS) against the training
partition and 0.055 (FAIL) against the full record — because 80% of the full record is the
models' own training data. See `code_plan/AUDIT_2026-09-22.md` §3.2 for the complete
comparison across all three candidate references (train / full / locked test year).

### Change 5 — Calendar and Transition-Sampled Block Assembly

**Base paper:** Generates sequences of length T independently; concatenation strategy not
discussed.

**This fork:** Two assembly strategies for long-horizon synthesis from L-length blocks,
both using overlap-add (OLA) boundary smoothing (default 4-step = 20-minute overlap):

1. **Calendar-ordered assembly:** Replays the true climatological seasonal-phase label
   sequence in exact calendar order, year by year. Deterministic.
2. **Transition-sampled assembly** (CLI flag `--assembly_mode markov`, for historical
   reasons — see the terminology note in Change 1): the next block's phase is resampled
   stochastically from the empirical first-order transition frequency table estimated on
   the training years, with Laplace smoothing.

Calendar assembly reproduces the observed seasonal cycle (monthly r = 0.91 for v8_cal);
transition-sampled assembly does not (monthly r = −0.11 for v10). The two also have
slightly different storm-geometry outcomes (v10 storm count 1.004×, v10_cal 0.995× —
both within Gate A band), but the seasonality difference between them is the larger and
more decision-relevant finding of the two.

### Change 6 — Spectral (FFT) Loss Term — Documented Defect, Analytically Inert at L=64

**Base paper:** Standard EDM denoising objective only.

**This fork's training objective** (`regime_training/train_regime.py`) augments the loss
with a 2D FFT term:

  L_total = L_time + beta_fft * L_fft

**This is not being presented as a contribution.** `code_plan/LOSS_FUNCTION.md` §3.1
documented, before v8–v10 were trained, that with `norm='forward'` the term is by Parseval's
theorem exactly `1/64 = 1.5625%` of the time term, and that the spatial mask multiplied
onto it (`(1 - x_img_mask)`) is applied in the wrong domain — a spatial mask cannot
constrain a term already indexed by frequency. `my notes/memo/day_1.md` §8 item 5 lists this
as **permanently retired as mathematically invalid**. v8, v9 and v10 were nevertheless all
trained with `fft_weight: 1.0` and the bug unchanged.

At L=64 specifically, the spatial mask is all-ones (the 8×8 grid has no structural padding
left), so the masking bug has no effect and the term reduces exactly to the theoretical
1.5625% rescale — confirmed in the v10 training log at the best epoch (498):
`time_loss=0.1532, fft_loss=0.00239`, ratio 1.56%. This closes the ablation analytically for
v10 without needing a `beta_fft=0` retrain: the term could not have contributed to v10's
storm-geometry improvement, because at L=64 it is provably a constant rescale of the time
term.

### Change 7 — Downstream Application: Synthetic Weather for Stormwater RL Agents

**Base paper:** Evaluates against held-out real data; no downstream task.

**This fork's stated purpose:** generating multi-year synthetic precipitation records as
training environments for RL agents controlling the Astlingen SWMM urban drainage network.
`code_plan/ACCEPTANCE_CRITERIA.md` specifies this as Gate C, gated behind Gate B (routing
observed and synthetic rainfall through SWMM under a fixed baseline controller — no RL).
**Neither gate has been run.** No evidence currently exists that any version of this
generator improves, or even suffices for, RL training outcomes. Section 4's gaps and
§5 open directions below should not be read as implying otherwise.

---

## 3. Summary Matrix: This Fork vs. Base Paper

| Dimension | Base Paper | This Fork |
|---|---|---|
| **Conditioning** | Dataset identity token (semantic-free) | Seasonal-phase labels; delivers the annual cycle only under calendar-ordered assembly |
| **Context window** | Fixed; not ablated | Compared L∈{24,36,64}; confounded with checkpoint choice (unresolved) |
| **Evaluation** | Disc. / Predictive / contextFID | 18-metric, 5-tier physical harness with explicit pass/fail bands; no version passes |
| **Data split** | Random subset sampling | Strict chronological holdout (train/val/test); validation not yet used for model selection |
| **Long-horizon assembly** | Not addressed | Calendar-ordered (delivers seasonality) + transition-sampled (does not) |
| **Spectral loss** | Not applied | Present, documented as a retired defect, analytically inert at L=64 |
| **Downstream task** | None | Specified (Gate B/C); not run |
| **External baseline** | N/A (is the base paper) | None run; KoVAE/ImagenTime/DiffusionTS vendored but unused on this data |

---

## 4. Gaps and Limitations

### Gap 1 — Volume Deficit Is Real; the Stated Cause Is Not

v10's annual volume (649.2 mm/yr against the training reference of 709.6 mm/yr, ratio
0.915) fails the Gate A band (0.95–1.05). An earlier draft of this document attributed this
to the 0.005 mm wet/dry threshold suppressing drizzle accumulation. Measured directly: the
threshold removes **0.00 mm/yr** from every synthetic series in this study (the generator
does not emit sub-threshold intensities at all) and 0.02 mm/yr from the real record. The
stated mechanism does not operate, so the proposed fix (soften the threshold) would not
address the deficit. The deficit's actual source has not yet been identified.

A multiplicative volume correction (the other proposed fix, ×1/0.915) rescales every
accumulation duration equally, and Tier 4 shows the deficit is **not** uniform across
durations (0.81 at 15 min vs 0.60 at 24 h for v10) — such a correction would leave the 24 h
return-period depth failing Gate A even after "fixing" the annual total.

### Gap 2 — Storm Duration and Wet-Spell Deficits Persist at L=64

Mean storm duration (0.875×) and wet-spell duration (0.866×) both still fail their Gate A
bands (0.90–1.10) at L=64. The longest convective events remain under-represented. Tier 4
suggests this scales with the ratio of event duration to context window (§2, Change 2) —
untested beyond L=64.

### Gap 3 — Single Catchment, Single Assembly-Mode Recommendation Conflict

All experiments use the Astlingen 4-gauge network. There is no evidence seasonal-phase
conditioning or the context-window findings generalize to other catchments, climatic zones,
or resolutions. Separately, no single (version, assembly-mode) pair currently satisfies both
storm geometry and seasonality: v10 (transition-sampled) has the best Gate A tally (11/18)
but the worst seasonality (monthly r = −0.11); v8_cal has the best seasonality (monthly r =
0.91) but a worse Gate A tally (7/18). Recommending a "primary + supplementary" pairing (as
an earlier draft did) has not been validated as coherent training data — it has not been
tested whether mixing two differently-biased generators produces a well-calibrated combined
distribution, or just a wider one.

### Gap 4 — No Ensemble Uncertainty Quantification

All reported metrics are from single 10-year realizations. Splitting each realization into
its ten constituent years and comparing to the observed record's own interannual spread
(as a proxy, not a substitute, for the ≥30-realization ensemble `code_plan/REMEDIATION_PLAN.md`
T2.1 specifies) already changes some conclusions: on annual-maximum 5-minute burst, v8
(3.23 ± 0.55 mm, z = −0.0 vs. observed 3.24 ± 1.12 mm) is indistinguishable from real,
while v10 (2.89 ± 0.95 mm, z = −1.0) is slightly below it — the opposite of the ranking a
single-draw maximum comparison suggests. Credible hydrological claims require the full
ensemble with moving-block bootstrap confidence bands from the observed record.

### Gap 5 — Context-Window Extension Is a Config Change, Not a New Foundation Model

Corrected from an earlier draft, which described L=64 as a hard architectural ceiling
requiring a new pretrained checkpoint. As shown in §2 Change 2, the grid size is a training
config parameter (`img_resolution/delay/embedding`), and all four existing checkpoints load
1006/1008 parameters regardless of target `seq_len`. Extending to L=256 with a 16×16 grid
is an unverified but low-cost experiment (start with a one-epoch smoke test confirming the
load-parameter count), not a retrain-from-scratch undertaking.

### Gap 6 — Seasonal Label Quality and Assembly-Mode Interaction Are Not Ablated

The 14-day smoothing window and 4-class discretization are not ablated. More immediately:
the interaction between conditioning and assembly mode is now measured (§2 Change 1, Change
5) and shows the conditioning signal is largely carried by assembly order rather than by the
per-block class label alone. Whether the four class embeddings are meaningfully separated in
the model's own generation statistics (`history.csv` logs `regime_{0..3}_mean/max` per
epoch) has not been checked — `code_plan/AUDIT_2026-09-22.md` §6 (E6) proposes this test.

### Gap 7 — FFT Loss Contribution Is Resolved Analytically for L=64, Not for L<64

At L=64 the term is provably inert (§2 Change 6). At L=24/36, the documented masking defect
(`code_plan/LOSS_FUNCTION.md` §3.1) means the term's effect there has not been isolated and
its historical contribution to v5–v9 remains unclear. This does not need a further ablation
for v10 specifically, but the L<64 versions' training histories should not be read as
evidence the term was doing anything constructive.

### Gap 8 — Checkpoint-`seq_len` Coupling Is Not the Constraint It Was Described As

Corrected from an earlier draft, which stated that mismatched `(checkpoint, seq_len)` pairs
"produce silent shape mismatches" and that this was discovered during v9 development. The
v9_len144 and v10_len288 training logs show the actual failure mode is an `IndexError` in
the data transform at generation-grid boundaries, occurring identically regardless of which
checkpoint was loaded — checkpoints load without shape conflicts at every `seq_len` tested
(12, 24, 36, 64). See `code_plan/PROVENANCE.md` for the checkpoint hashes and load logs this
is based on.

### Gap 9 — Training Loss Cannot Guide Model Selection, But the Comparison Method Needs Correcting First

The scalar training loss does not track storm geometry. v6 (dead intensity tail) and v7
(storm fragmentation) converge to loss plateaus of ~0.062, indistinguishable from v8's.
Separately: comparing the *raw* logged loss across different `seq_len` values understates
how well higher-L models are actually fitting, because `train_regime.py`'s loss is
`.mean()`-reduced over the full 64-cell grid regardless of how many cells are active. Once
rescaled by `64/seq_len` to compare per-active-cell loss, v10's best training loss
(0.1556, native) becomes **0.1556 per active cell — the lowest of all six versions
compared** (v5–v8 at L=24: 0.1645–0.1664; v9 at L=36: 0.1616), reversing the "loss gets
worse with context length" reading of the raw numbers. There is no paradox between loss and
sample quality here once the normalization is corrected; both point the same direction.

### Gap 10 — No Version Passes Gate A

An earlier draft described v10 as "meeting storm count and nearly meeting ACF RMSE" against
a stated volume band of [0.90, 1.10] — a band `code_plan/ACCEPTANCE_CRITERIA.md` does not
specify (its Tier 1 volume band is 0.95–1.05). Scored against the bands actually specified,
across all 18 Gate A metrics, v10 passes 11/18 — the best of any version, and still not a
pass. Full scorecard: `code_plan/AUDIT_2026-09-22.md` §4, machine-readable output
`results/reference/gate_a_train.json`.

---

## 5. Open Research Directions

Ordered by expected evidence gained per unit of compute; full hypotheses and falsification
criteria for each are in `code_plan/AUDIT_2026-09-22.md` §6.

1. **Decouple context length from checkpoint initialisation (highest priority).** Two runs:
   L=64 fine-tuned from `ImagenFew_24.ckpt`, and L=24 fine-tuned from `ImagenFew_64.ckpt`.
   Until this runs, the central claim of Change 2 is not separable from a checkpoint effect.
2. **Extend past the storm-memory scale.** One smoke test (16×16 grid, one epoch, confirm
   parameter load count) then one full run at L=256, to test whether the Tier 4
   duration-dependent deficit closes selectively at longer durations.
3. **Ensemble evaluation.** ≥30 realizations per version with moving-block bootstrap
   confidence bands on the observed record.
4. **Gate B — route rainfall through SWMM-Astlingen** under a fixed baseline controller
   (no RL). The first non-linear-in-rainfall test, and the cheapest path to any evidence
   for the RL claim in Change 7.
5. **Fix and re-verify the validation loop**, then use it for model selection (closes Gap
   9's open item and `code_plan/REMEDIATION_PLAN.md` T6.5).
6. **Test whether conditioning delivers class separation** independent of assembly order
   (Gap 6).
7. **Run an external baseline** — KoVAE and/or DiffusionTS on the same chronological split,
   with the existing TS2Vec-based Discriminative/Predictive/contextFID metrics, alongside
   the physical tiers. Without this the thesis has no comparison class and never reports
   the base paper's own metrics on this data.
8. **Multi-catchment generalization**, once 1–7 are further along.
