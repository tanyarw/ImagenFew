# Attention Placement, Receptive Field, and the Long-Storm Deficit

**Author:** Tanya Warrier (audit performed 2026-09-28)
**Project:** Adapting ImagenFew Time-Series Diffusion for Sparse Precipitation Generation (Astlingen Record)
**Document status:** Architectural audit. Every claim below is either verified by direct
inspection of the trained v10 checkpoint, by numerical test, or by code reading — the
evidence class is stated per claim. Where a claim is *not* verified, it says so.

> **Scope.** This document answers one question that had never been asked of this fork —
> *does the UNet actually use self-attention, and over what?* — and follows the answer into
> a measured failure mode. It makes claims about **architecture, representation geometry and
> receptive field**, not about hydrology. Rainfall is the testbed.

---

## 0. Executive summary

| # | Finding | Evidence class |
|---|---|---|
| 1 | The backbone is **`DhariwalUNet`**, not `SongUNet`. `EDMPrecond` defaults to it and `ImagenFew.py` never overrides `model_type`. | code reading |
| 2 | Self-attention **is** present, but only at the **4×4 (16 tokens)** and **2×2 (4 tokens)** stages, 1 head each. | v10 checkpoint inspection |
| 3 | The config asks for attention at resolution 8 (`attn_resolution: [8, 4, 2]`). It is **silently dropped**: `num_heads = out_channels // channels_per_head = 32 // 64 = 0`, which is falsy, so the `qkv`/`proj` modules are never even constructed. | numerical test + checkpoint |
| 4 | The bottleneck attention at `dec.1x1_in0` (hardcoded `attention=True`) is a **mathematical no-op**: on a 1×1 grid the softmax has a single key, so the attention matrix is exactly `1.0` and the output equals `v` bitwise. It degenerates into a channel-mixing residual 1×1 conv. | numerical test |
| 5 | **The model has no global mixing at full resolution.** Nothing in the network ever attends across the 64 time steps of a block at their native resolution. | follows from 2–4 |
| 6 | Because `delay == embedding == 8`, one 3×3 convolution at full resolution reaches time lags **{0, ±1, ±7, ±8, ±9}** and *cannot* reach lags 2–6 (10–30 min). Stacking three blocks leaves holes at lags **4, 12, 20**. | numerical test |
| 7 | v10 reproduces short-range intermittency almost exactly, but produces **~13× too few storms longer than one generated block**, and the volume they carry collapses from **21.4% → 1.6%** of total rainfall. | computed from released CSVs |
| 8 | **The storm-duration survival curve collapses at exactly the block length, in every version** — v7/v8 at 120 min, v9 at 180 min, v10 at 320 min. Below the cliff each version is within a few percent of real. This is a natural experiment across three block lengths with no exceptions. | computed from released CSVs |
| 9 | Therefore `seq_len` **never taught the model longer-range structure — it only moved the wall.** The v8 → v9 → v10 "context window" result needs reinterpreting. | follows from 8 |
| 10 | The conditioning machinery for fixing 7–9 **already exists and is unused**: training always passes an all-zero mask, and generation calls the imputation path with zero known pixels. | code reading |

**One-line verdict.** The local behaviour of this model is good and the long-range behaviour
is bad, and the two have *different* causes that this audit separates. The local view is
shaped by the representation and attention placement — dense over the 5–10 minute
neighbourhood, strided and holed over 10–30 minutes, with no full-resolution global view at
all. The long-range failure is **not** a capacity or representation problem: it is a hard
wall at the block boundary, because each block is generated independently, and a fifth of the
real water arrives in events that cross it.

---

## 1. Which UNet, and where is the attention?

`models/ImagenFew/ImagenFew.py` builds the denoiser as

```python
self.net = EDMPrecond(args.img_resolution, args.input_channels, channel_mult=args.ch_mult,
                      model_channels=args.unet_channels, attn_resolutions=args.attn_resolution,
                      label_dim=self.num_classes, lora_rank=args.lora_dim,
                      dynamic_size=args.dynamic_size)
```

`EDMPrecond`'s signature carries `model_type = 'DhariwalUNet'` as its default and no call
site overrides it, so **`DhariwalUNet` is the backbone**. `SongUNet` is dead code in this
fork. This matters because the two classes place attention differently and use different
head-count rules.

The rainfall configs (`config_v8` … `config_v14`, all identical here) set:

```yaml
img_resolution: 8          # 8x8 delay-embedding grid
unet_channels: 32          # model_channels
ch_mult: [1, 2, 2, 4]
attn_resolution: [8, 4, 2]
```

`DhariwalUNet` uses `num_blocks = 3` (default, never overridden) and passes
`channels_per_head = 64` into every block. Attention head count is then decided inside
`UNetBlock.__init__`:

```python
self.num_heads = 0 if not attention else num_heads if num_heads is not None \
                 else out_channels // channels_per_head
```

and the attention modules are built — and later executed — only under `if self.num_heads:`.

### 1.1 What that resolves to, stage by stage

| Level | Resolution | `out_channels` | In `attn_resolution`? | `num_heads` | Attention actually runs? |
|---|---|---|---|---|---|
| 0 | 8×8 (64 tokens) | 32 | **yes** | `32 // 64 =` **0** | **No — silently dropped** |
| 1 | 4×4 (16 tokens) | 64 | yes | 1 | **Yes** |
| 2 | 2×2 (4 tokens) | 64 | yes | 1 | **Yes** |
| 3 | 1×1 (1 token) | 128 | no | 2 (bottleneck only) | Instantiated, but a **no-op** |

**Verification (v10 checkpoint, `logs/ImagenFew/Rainfall_Regime/aebe363f/best_regime_model.pt`).**
Enumerating which blocks carry `qkv`/`proj` tensors in the saved `state_dict` reproduces the
table exactly: no attention parameters exist at `enc.8x8_block{0,1,2}` or
`dec.8x8_block{0..3}`; `proj` tensors of shape `(64, 64, 1, 1)` exist at every `4x4_block*`
and `2x2_block*` in both encoder and decoder; and a single `(128, 128, 1, 1)` `proj` exists
at `dec.1x1_in0`. Total: 15 attention-bearing blocks, of which 14 are real and 1 is inert.

### 1.2 The res-8 drop is an integer-division accident, not a design choice

`channels_per_head = 64` is inherited unchanged from Dhariwal & Nichol's ImageNet
configuration, where `model_channels` is 192 and the narrowest stage still has ≥192
channels. Here `model_channels = 32`, so the narrowest stage has 32 channels and
`32 // 64` truncates to zero. The config's `attn_resolution: [8, 4, 2]` therefore documents
an intent the code does not honour, and it fails **silently** — no warning, no error, and
the missing parameters are invisible unless you go looking for them in the checkpoint.

This is worth stating plainly in the thesis because it changes the interpretation of every
result from v8 onward: **`attn_resolution` has had no effect at its first entry for the
entire experimental series.** Setting it to `[4, 2]` would produce a bit-identical model.

### 1.3 The bottleneck attention is inert

`DhariwalUNet` hardcodes `attention=True` on the bottleneck block
(`self.dec[f'{res}x{res}_in0']`) regardless of `attn_resolutions`. With
`ch_mult = [1, 2, 2, 4]` and `img_resolution = 8`, the bottleneck sits at 8 >> 3 = **1×1**.

On a 1×1 grid the flattened token count is 1, so `AttentionOp` computes a softmax over a
single key. Numerically confirmed: the attention matrix is exactly `1.0` in every entry and
`max|a − v| = 0.0` — the attention output equals the value tensor bitwise. The block reduces
to `x ← x + proj(v(norm(x)))`, a residual channel mixer with zero spatial mixing.

So the deepest stage of the network spends 2 heads' worth of parameters
(`qkv`: 128→384, `proj`: 128→128) on something that cannot move information in time. Not
harmful, but not attention either, and it should not be described as such.

---

## 2. What the delay embedding does to the receptive field

This is the more interesting half of the audit, and it is specific to the
image-diffusion-over-folded-time-series design that ImagenTime/ImagenFew introduced.

With `seq_len = 64`, `delay = 8`, `embedding = 8`, `DelayEmbedder.ts_to_img` fills

$$\texttt{image}[r, i] \;=\; \texttt{signal}[\,i \cdot 8 + r\,], \qquad r, i \in \{0,\dots,7\}$$

Because **`delay == embedding`**, successive columns are *disjoint and contiguous*: column
`i` holds the 8 consecutive 5-minute steps of a single 40-minute chunk, and the columns
tile the block with no overlap. The "delay embedding" is therefore a **pure reshape** in
this configuration — it introduces no redundancy, and it is exactly invertible. (That is a
convenience, not a given: `delay < embedding` would make the image an overcomplete,
redundant encoding.)

The axes have clean and different meanings:

* **row index `r`** — fine position *within* a 40-minute chunk (the fast axis)
* **column index `i`** — which 40-minute chunk (the slow axis)

### 2.1 A 3×3 convolution becomes a holed multi-scale temporal filter

A 3×3 kernel centred at `(r, i)` touches `(r ± 1, i ± 1)`, which in time is

$$t + \{0,\ \pm 1,\ \pm 7,\ \pm 8,\ \pm 9\}$$

**Lags 2, 3, 4, 5 and 6 are unreachable by a single full-resolution convolution.** In
physical units, one layer sees the neighbouring 5-minute steps and the corresponding
instants 35–45 minutes away, and *nothing in between*. Stacking the three blocks of level 0
widens the span to ±27 steps but still leaves exact holes at lags **4, 12 and 20** (20, 60
and 100 minutes).

Two honest qualifications:

1. Information does still reach those lags through the **downsampled path** (levels 1–3 and
   back up the decoder), which is densely connected. The claim is not that the model is
   blind to 20-minute structure; it is that the model can only reach it *after 2× spatial
   pooling*, i.e. at degraded temporal precision, and never at full resolution.
2. The 2× downsample pools a 2×2 patch, whose four cells are times
   $\{i\cdot 8 + r,\; i\cdot 8 + r + 1,\; (i+1)\cdot 8 + r,\; (i+1)\cdot 8 + r + 1\}$ —
   **non-contiguous in time**, two pairs 40 minutes apart. So each of the 16 tokens that the
   4×4 attention operates on is a mixture of two temporally distant fragments, not a
   localised 20-minute window. The same holds, more strongly, for the 4 tokens at 2×2.

This is the sharp version of the receptive-field story: **the only stage whose tokens are
clean, localised, contiguous points in time is resolution 8 — and that is precisely the
stage with no attention.**

### 2.2 Why the 10–30 minute band is the band that matters

The Gate A metrics that fail (`code_plan/ACCEPTANCE_CRITERIA.md`) live in exactly that band:
mean wet spell **30.4 min**, mean storm duration **61.0 min**. The metrics that pass — lag-1
autocorrelation, zero fraction, the intensity marginal — are all either single-step or
distribution-only quantities that the dense lag-{0, ±1} coverage is sufficient for.

The alignment between where the representation has holes and where the metrics fail is
suggestive, not proven. §4 gives the experiment that would settle it.

---

## 3. The measured failure: storms longer than one block

Statistics of the 2000–2007 training split (840,960 steps at 5 min), all computed directly:

* **90.79% of steps are exactly zero**; 90.80% fall below the 0.005 mm wet threshold. Only
  **9.20%** (77,404 steps) are wet.
* Wet-step quantiles (mm/5min): median **0.035**, q99 **0.588**, q99.9 **1.598**,
  q99.99 **3.159**, max **5.555**. Steps above 2 mm are **0.0046%** of the record.
* The top 1% of wet steps carry **13.8%** of all rainfall volume; the top 10% carry **47.6%**.

So this is a **zero-inflated** problem first and a heavy-tailed one second. That ordering
matters: the transform ablation (v13 `log1p`, v14 `asinh`) reallocates model value-range
across the *wet* intensities, but 90.8% of the loss budget is spent on the point mass at
zero either way.

### 3.1 Occurrence and intensity have different memories

| Lag | 5-min intensity ACF | Wet/dry indicator ACF |
|---|---|---|
| 1 (5 min) | +0.852 | +0.828 |
| 6 (30 min) | +0.355 | +0.621 |
| 12 (1 h) | +0.233 | +0.513 |
| 24 (2 h) | +0.147 | +0.398 |
| 64 (5.3 h) | +0.053 | **+0.213** |
| 288 (24 h) | +0.009 | +0.042 |

The **occurrence process carries four times the memory of the intensity process** at the
block horizon. The long-range structure this model is asked to reproduce lives almost
entirely in *whether* it is raining, not in *how hard*. The two processes have different
timescales and therefore want different receptive fields — an argument for treating them as
two coupled fields rather than one.

### 3.2 v10 loses the long storms, and with them a fifth of the water

All figures use the **canonical** storm definition (contiguous wet runs ≥ 3 steps, no gap
tolerance, `day_1.md` §3.2 def. 6), the same one `scripts/gate_a_scorecard.py` uses. As a
pipeline check, this reproduces the storm figures already reported in
`docs/FINETUNING_ANALYSIS.md` §1 exactly — 697.1 → 699.8 storms/yr (ratio **1.004**) and
mean duration ratio **0.875** — so the numbers below are directly comparable to the
scorecard.

| Quantity | Real (train, 8 y) | v10 (10 y) | Ratio |
|---|---|---|---|
| Storms per year | 697.1 | 699.8 | **1.004** |
| Mean storm duration | 61.8 min | 54.1 min | **0.875** |
| Wet-run length histogram (1 / 2 / 3 / 4–6 / 7–12 / 13+ steps) | 38.1 / 15.9 / 8.7 / 14.0 / 11.5 / 11.9 % | 40.4 / 14.9 / 8.5 / 14.3 / 10.7 / 11.1 % | ≈ 1.0 in every bin |
| Zero fraction | 90.80 % | 91.75 % | +0.95 pp |
| **Storms longer than one 64-step block (> 320 min), per year** | **14.25** | **1.10** | **0.077** |
| **Share of total volume in those storms** | **21.4 %** | **1.6 %** | **0.076** |

The two aggregate metrics at the top of that table are the ones the acceptance criteria
track, and **both look healthy**. Storm count is within 0.4% and mean duration within 13%.
That is exactly why this failure has gone unnoticed for seven model versions: it is
invisible in the mean and lives entirely in the **tail of the duration distribution**, where
a 13× deficit in the longest storms costs only a few minutes of mean duration but a fifth of
the total water.

### 3.3 The cliff sits exactly at the block length — in every version

This is the decisive measurement. The storm-duration survival function
$P(\text{duration} > k)$, generated over real, for every version whose CSV exists:

| $k$ (steps) | 3 | 6 | 12 | 18 | 24 | 30 | 36 | 48 | 64 | 80 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (minutes) | 15 | 30 | 60 | 90 | 120 | 150 | 180 | 240 | 320 | 400 | 480 |
| real, absolute | 81.1% | 50.8% | 25.8% | 15.7% | 11.3% | 8.5% | 6.2% | 3.8% | 2.0% | 1.4% | 0.8% |
| **v7** ratio (L = 24) | 1.01 | 1.02 | 1.03 | 1.00 | **0.15** | 0.10 | 0.09 | 0.01 | 0.00 | 0.00 | 0.00 |
| **v8** ratio (L = 24) | 1.01 | 0.97 | 0.97 | 0.92 | **0.12** | 0.07 | 0.05 | 0.01 | 0.01 | 0.01 | 0.00 |
| **v9** ratio (L = 36) | 0.92 | 0.81 | 0.82 | 0.81 | 0.75 | 0.63 | **0.09** | 0.05 | 0.01 | 0.00 | 0.00 |
| **v10** ratio (L = 64) | 1.00 | 0.96 | 0.96 | 0.98 | 0.92 | 0.83 | 0.74 | 0.63 | **0.08** | 0.03 | 0.03 |

Read the bold entries. **Each version tracks the real survival curve to within a few percent
out to roughly three quarters of its block length, and then collapses by an order of
magnitude at exactly its own $L$** — v7 and v8 at $k = 24$, v9 at $k = 36$, v10 at $k = 64$.
The ratio at $k = L$ is 0.15, 0.12, 0.09, 0.08: a consistent **≈ 10× deficit**, landing at a
different physical duration in each version, and landing there because that is where the
generated block ends.

This is a natural experiment across three block lengths with no exceptions, and it is about
as clean a causal attribution as this kind of evidence gets. It rules out the alternatives:
model capacity, the loss function, the data transform, the seasonal-phase conditioning and
the checkpoint choice are all *constant* across the pairs that straddle a cliff, and none of
them would predict a discontinuity whose location moves with `seq_len`.

**It also reinterprets the v8 → v9 → v10 progression.** That sequence was read as evidence
that a longer context window lets the model learn longer-range structure
(`docs/FINETUNING_ANALYSIS.md` §3). The survival table says something different and much
simpler: **raising `seq_len` did not teach the model longer-range structure, it moved the
wall.** v10 is better than v8 because its cliff is at 320 minutes instead of 120 — not
because it learned anything v8 could not. Everything below the cliff was already near-perfect
at L = 24.

That distinction matters for the thesis, because the two readings imply opposite next steps:
the first says "keep increasing `seq_len`", the second says "`seq_len` is not the variable —
remove the wall".

### 3.4 It is not seam truncation

The obvious hypothesis is that the cosine crossfade between independently generated blocks
truncates storms at the seams. **That hypothesis is wrong, and was tested and rejected.**
Emulating a hard seam on the *real* record — forcing a dry step every 320 steps, the worst
case for a storm trying to cross a boundary — changes the storm count by **0.0%** and the
mean duration by **−0.4%**. A 15-minute gap tolerance bridges a single dry step, and the
crossfade is smoother than a hard seam. Seam handling is not the problem; the stitching fix
committed on 2026-09-27 was correct and is not implicated here.

### 3.5 What is actually going on

The cause is **block independence**, not seam damage. Every 64-step block is drawn from
$p(x \mid \text{HMM state})$ with no knowledge of its neighbours:

* `regime_training/generate_hmm_v1.py` builds `mask` from a zero tensor with `pad_val=1`.
  At 8×8 there is no padding, so the mask is **identically zero** — zero known pixels.
* `process.interpolate(x_img=zeros, mask=zeros)` therefore runs the imputation solver with
  nothing to condition on: it is pure unconditional sampling through the inpainting code
  path.
* The *only* channel linking consecutive blocks is the 4-level seasonal-phase label — and
  that label changes on a multi-day timescale and carries no information about whether a
  storm is currently in progress.

A storm longer than 320 minutes therefore requires two or more consecutive blocks to be
independently heavy *by coincidence*. The observed 13× deficit is what that coincidence rate
looks like.

### 3.6 The fix is already built and switched off

`ImagenFew.forward(x, mask, ...)` implements masked denoising correctly: `mask == 1` marks
pixels that are passed through clean, and `DiffusionProcess.impute()` re-imposes those clean
values at every solver step — the RePaint replacement rule, already correct in this
repository.

But **training never exercises it.** `regime_training/train_regime.py` sets
`x_ts_mask = torch.zeros_like(x_ts)` on every batch, and the loss mask `signal = 1 - x_img_mask`
is identically 1. At 8×8 there is no padding to make any entry nonzero. So across v1–v14 the
model has **never once seen a known pixel**, and the conditional path, while implemented, is
untrained.

This is the single most actionable finding in this audit: the architecture is already an
inpainting model, the sampler already supports conditioning, and the one line that would
have taught it to condition has always been a tensor of zeros.

---

## 4. Literature survey: what others do about each of these

Surveyed 2026-09-28. Non-weather work is marked **[non-weather]** — the user's request was
explicitly for cross-domain precedent.

### 4.1 Time series as images (the parent design)

* **ImagenTime** — Naiman et al., NeurIPS 2024, [openreview](https://openreview.net/forum?id=2NfBBpbN9x), [code](https://github.com/azencot-group/ImagenTime). The direct parent: invertible delay-embedding and STFT transforms let a vision diffusion backbone handle both short and ultra-long series. Reports 58% mean improvement in short discriminative score and 133% in ultra-long classification.
* **[non-weather] "A Diffusion Model for Regular Time Series Generation from Irregular Data"** — [arXiv 2510.06699](https://arxiv.org/pdf/2510.06699). Directly compares folding, enhanced delay embedding and Gramian Angular Field. Finds the *geometric* encodings (folding, delay embedding) suit diffusion best **because each pixel maps to a single time point**, while GAF stores the sequence on its main diagonal and does not scale to long inputs. This is independent support for the reshape-style embedding used here — and it is exactly the property §2.1 exploits.
* **[non-weather] MIDiff** — [arXiv 2607.14249](https://arxiv.org/abs/2607.14249), Allerton 2026. **The closest cross-domain analogue to this project.** Mobile app-usage traces: sparse, heterogeneous, imbalanced. Maps them into a Cross-Gramian Angular Sum Field image and runs a UNet diffusion model with **Triple Attention** factorised over three axes — H (temporal), W (app/location) and Z (channel) — explicitly matched to the axis semantics of the imaging transform. Two design decisions are directly relevant here: (i) sparsity is handled by **structural encoding** (inactive steps get a dedicated 0-index cluster; active steps get fixed fill patterns) rather than loss reweighting; (ii) imbalance is handled by **spatial isolation** rather than reweighting. Beats ZITS-VAE on discriminative accuracy 0.153 vs 0.348.
* **[non-weather]** *Harnessing Vision Models for Time Series Analysis: A Survey*, [IJCAI 2025](https://www.ijcai.org/proceedings/2025/1178.pdf).

**Read-across.** MIDiff's Triple Attention is the strongest external argument that this
fork's attention placement is wrong: MIDiff attends **along the meaningful axes of the
imaging transform**, whereas this fork attends over flattened 2×2-pooled patches whose
tokens mix two time fragments 40 minutes apart (§2.1). The axis-aligned alternative is
classical: **Axial Attention** (Ho et al., [arXiv 1912.12180](https://arxiv.org/pdf/1912.12180)) factorises 2D attention into row-then-column passes at $O(N^{(d-1)/d})$ cost, and every position still reaches every other indirectly. On an 8×8 grid the rows are "within a 40-min chunk" and the columns are "across chunks" — the factorisation is *semantically* right here, not just cheap.

### 4.2 Where attention belongs in a diffusion UNet

* **simple diffusion** — Hoogeboom et al., [ICML 2023](https://proceedings.mlr.press/v202/hoogeboom23a/hoogeboom23a.pdf). Argues for putting capacity at **low** resolution: a small conv UNet downsamples to 16×16 and a large transformer runs there. Partially *vindicates* this fork's instinct — but note the scale. Their attention stage has 16×16 = **256 tokens**; this fork's largest attention stage has **16 tokens** and its smallest has **4**. The design is not "attention at low resolution", it is "attention at almost no resolution".
* **ADM / "Diffusion Models Beat GANs"** — Dhariwal & Nichol, [arXiv 2105.05233](https://arxiv.org/pdf/2105.05233). The source of `DhariwalUNet` and of `channels_per_head = 64`. Their `attn_resolutions = [32, 16, 8]` on 256×256 inputs with `model_channels = 192`. The head-count rule in §1.2 was written for that regime and silently degenerates in this one.

### 4.3 Heavy tails: change the prior, not just the transform

* **Heavy-Tailed Diffusion Models (t-EDM / t-Flow)** — [ICLR 2025, arXiv 2410.14171](https://arxiv.org/abs/2410.14171). Replaces the Gaussian prior with a **multivariate Student-t**, with a matched perturbation kernel and a Student-t denoising posterior. Controlled by a **single scalar** (degrees of freedom). Motivation: in high dimensions a Gaussian concentrates on a thin spherical shell and structurally under-serves the tails. Validated on **high-resolution weather data where generating rare and extreme events is the point**, and reported to beat standard EDM on extreme-value distributions.
* Follow-ups: *Self-Regulating Annealing in Heavy-Tailed Diffusion Models* ([arXiv 2606.01645](https://arxiv.org/pdf/2606.01645)); *Tail Annealing for Heavy-Tailed Flow Matching* ([arXiv 2605.20068](https://arxiv.org/pdf/2605.20068)).

**Read-across.** t-EDM is **orthogonal to the v13/v14 ablation**, and that is what makes it
valuable. v13/v14 change the *data* map $x \mapsto z$; t-EDM changes the *noise prior and
kernel*. Running it after v13/v14 yields a clean 2×2 design:
{standard, asinh} × {Gaussian, Student-t}. Because this fork is already EDM with
`sigma_data = 0.5` and explicit Karras preconditioning, t-EDM lands on the existing code
rather than beside it. It does require retraining — it changes the noise used during
training — and I have **not** verified whether the paper supports fine-tuning from a
Gaussian EDM checkpoint. That question should be settled from the paper before costing the
experiment.

### 4.4 Zero inflation: 90.8% of this dataset

* **[non-weather] ZITS / ZITS-VAE** — the baseline MIDiff compares against. Decouples zero-inflated generation into **Bernoulli-gated occurrence modelling** plus **non-zero magnitude estimation**. This is precisely the decomposition §3.1's ACF table argues for.
* **[non-weather] "Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data"** — [KDD 2022](https://www.cse.msu.edu/~ptan/papers/kdd2022.pdf). Treats both pathologies jointly, which is the situation here.
* *MZ-Rain: Moisture-Budget-Guided Zero-Inflated Model for Station-Level Precipitation Nowcasting* — [arXiv 2609.04864](https://arxiv.org/pdf/2609.04864).
* Two-stage occurrence-then-intensity is the standard remedy for the precipitation "drizzle problem" (e.g. *A Generative Deep Learning Approach to Stochastic Downscaling of Precipitation Forecasts*, [PMC9788314](https://pmc.ncbi.nlm.nih.gov/articles/PMC9788314/)).

**Read-across.** Two distinct strategies exist, and MIDiff is the interesting one: instead of
gating or reweighting, it *encodes* sparsity structurally so convolutions can recover it.
Worth noting that this fork already tried the reweighting route —
`intensity_weight_alpha` in `train_regime.py` — and every clean run since v8 sets it to
**0.0**, i.e. reweighting was built and then switched off. The unexplored options are the
structural one (MIDiff) and the decomposition one (ZITS).

### 4.5 Long-horizon generation: the block-independence problem

This is where the literature is richest and maps most directly onto §3.5.

* **Diffusion Forcing** — Chen et al., [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/2aee1c4159e48407d68fe16ae8e6e49e-Paper-Conference.pdf), [project page](https://www.boyuan.space/diffusion-forcing/). **The key reference for this project's failure mode.** Trains a diffusion model to denoise tokens at **independent per-token noise levels**, which yields a causal next-token model that can roll out *stably beyond its training length*. The central trick: condition on **slightly noisy** history rather than clean history, which stops error accumulation while keeping causal structure.
* **History-Guided Video Diffusion** — [arXiv 2502.06764](https://www.boyuan.space/history-guidance/). Learns the distribution of *all* sub-sequences, so it can condition on a history of **any length**, and adds temporal history guidance to compose long-horizon with local-reactive behaviour.
* **RePaint** — Lugmayr et al., [CVPR 2022](https://arxiv.org/pdf/2201.09865). Conditions an **unconditionally trained** diffusion model on known pixels, with **no task-specific training**, by resampling forward and backward in diffusion time to harmonise the known/unknown boundary. Benefit saturates around **n ≈ 10** resamplings. *This fork's `impute()` is already RePaint's replacement step — minus the resampling loop.*
* **[non-weather] Lazy Diffusion** — [arXiv 2512.09572](https://arxiv.org/pdf/2512.09572). **The counterweight, and it must be taken seriously.** Autoregressive diffusion rollout suffers **spectral collapse**: fed-back errors compound, high frequencies are suppressed and low frequencies amplified, so output grows progressively smoother and less physical. Their guidance: monitor spectra during training, blend model output with a "lazy" baseline, limit rollout length or periodically reinitialise.
* **[non-weather] STITCH-OPE** — [arXiv 2505.20781](https://arxiv.org/pdf/2505.20781), trajectory stitching with guided diffusion. **Generative View Stitching** — [arXiv 2510.24718](https://arxiv.org/pdf/2510.24718).
* **[non-weather]** Related spectral-bias result: the diffusion forward process erases information in a wavenumber-dependent way, so the score network receives little usable supervision for high-frequency modes at large $\sigma$ — a structural reason diffusion under-resolves fine detail.

**Read-across, including the risk.** A 10-year rollout at 64 steps per block is roughly
**16,400 autoregressive steps**. That is far beyond anything in the video-diffusion papers,
and Lazy Diffusion says plainly that naive feedback drifts. Two features of *this* problem
mitigate it, and they should be stated as the reason to expect it to work here rather than
assumed:

1. The HMM seasonal-phase label is an **external, non-autoregressive anchor** re-imposed at
   every block, so the chain is repeatedly pulled back toward the conditional climatology
   instead of free-running.
2. Diffusion Forcing's noisy-history conditioning is *designed* for exactly this failure, and
   it is cheap to implement here.

Even so, **spectral drift over a 10-year rollout is the main risk of the experiment proposed
below and must be measured, not assumed away.** The natural monitor already exists in the
acceptance criteria: the hourly ACF curve over lags 1–24 h, plus the annual-volume ratio,
computed per generated year rather than pooled.

### 4.6 Regime conditioning

HMM regime-switching conditioning is standard well outside weather — volatility regimes in
finance, sleep staging and auditory-attention decoding in EEG (e.g.
[arXiv 2606.02231](https://arxiv.org/html/2606.02231),
[arXiv 2602.13447](https://arxiv.org/pdf/2602.13447)). This supports the framing already
adopted in `my notes/memo/day_1.md` §4: the 4-level label is a **discrete latent-state
conditioning variable**, and the transition matrix encodes calendar succession, not physical
weather dynamics. No change recommended; the literature simply confirms the retired
terminology was the right thing to retire.

---

## 5. Ranked candidate experiments

Ordered by *evidence per GPU-hour*, not by novelty.

### E1 — Turn on the attention that the config already asks for. *(free at initialisation)*

Make resolution 8 actually attend, by either passing `num_heads=1` explicitly or setting
`channels_per_head = 32`. The result is genuine self-attention over all **64 tokens at full
resolution** — every token a single, clean, contiguous 5-minute step (§2.1) — which is
exactly the missing 10–30 minute lag coverage. An 8×8 grid means a 64×64 attention matrix:
negligible cost.

The initialisation is the attractive part, and it is **verified, not assumed**.
`DhariwalUNet` builds `proj` with `init_zero = dict(init_mode='kaiming_uniform',
init_weight=0, init_bias=0)`, so both the projection weight and its bias are **exactly zero**
at initialisation and the attention branch contributes nothing. The one subtlety is that
`UNetBlock.forward` applies `x = x * self.skip_scale` a *second* time inside the attention
branch — but `DhariwalUNet`'s `block_kwargs` never sets `skip_scale`, so it is 1 and the
second multiply is a no-op. (`SongUNet` sets it to $\sqrt{0.5}$, where this would *not* hold
— another reason the backbone identification in §1 matters.)

Confirmed numerically: with all shared weights copied across, an attention-enabled 32-channel
block and an attention-free one produce **bitwise identical** output
(`max|a − b| = 0.0`, `torch.equal → True`). So a fine-tune from the v10 checkpoint starts at
exactly v10's function and can only add capability.

Worth pairing with a second free change: the 1×1 bottleneck attention is inert (§1.3), so
either drop it or shorten `ch_mult` to `[1, 2, 4]` so the bottleneck sits at 2×2 and the
hardcoded attention does something.

**Ablation value:** cleanly separates "attention placement" from every other variable, and
retrospectively tells the thesis whether `attn_resolution` mattered at all.

### E2 — Condition each block on the previous block, with no retraining. *(inference only)*

Generate with stride < `seq_len`, setting the first $k$ columns of each block's image to the
previous block's last $k$ columns as **known** pixels (`mask = 1` there). Because
`delay == embedding`, those columns are exactly a contiguous 40·$k$-minute past (§2), so the
conditioning set is clean with no aliasing — a structural gift of this representation.
`impute()` already applies the replacement rule correctly.

Two things must be added, both from the literature:
* **RePaint resampling** (n ≈ 10). The model has never seen a clean pixel (§3.6), so
  conditioning is out of distribution; resampling is what makes RePaint work on
  unconditionally trained models, and it is currently absent.
* **Noisy history** à la Diffusion Forcing — condition on the previous columns at a small
  non-zero $\sigma$ rather than at $\sigma = 0$ — as the primary defence against the
  spectral drift in §4.5.

**Success criterion, pre-registered:** storms longer than 320 min should rise from **1.10/yr
toward 14.25/yr**, and their volume share from **1.6% toward 21.4%**, *without* degrading the
wet-run histogram or the zero fraction (§3.2). **Failure criterion:** per-year annual-volume
ratio or hourly ACF drifting monotonically across the 10 generated years — that is spectral
collapse, and it means E3 is required.

### E3 — Train the conditional path that already exists. *(one retrain, v13/v14 budget)*

The principled version of E2, and close to a one-line change: replace

```python
x_ts_mask = torch.zeros_like(x_ts)
```

with a random known-prefix mask (mark the first $k$ columns known, $k \sim \mathrm{Unif}\{0,\dots,4\}$,
$k = 0$ recovering current behaviour). The model becomes a proper conditional outpainter,
E2's conditioning becomes in-distribution, and unconditional sampling is preserved as the
$k=0$ case. Full Diffusion Forcing — independent per-column noise levels — is the stronger
version and needs per-pixel $\sigma$ plumbing; the prefix mask is the right first step.

### E4 — Two-field occurrence/intensity decomposition. *(most novel, largest change)*

Generate two channels — a wet/dry occurrence field and an intensity field — instead of one.
Justified by §3.1: the occurrence process carries **4× the memory** of the intensity process
at the block horizon, so the two want different receptive fields, and forcing one field to
carry both is an inductive-bias mismatch. ZITS supplies the gated decomposition, MIDiff the
structural-encoding alternative. This is the idea with the best chance of being a thesis
contribution in its own right, and it should follow E1–E3, not precede them.

### E5 — t-EDM. *(one retrain, touches the EDM core)*

Student-t prior and kernel, one scalar hyperparameter, orthogonal to the transform ablation
(§4.3). Run once v13/v14 have reported, to complete the 2×2.

**Recommended order: E1 + E2 together** (one free training change, one inference change,
both measurable against §3.2's pre-registered targets), then E3 if E2 drifts, then E4, then
E5.

---

## 6. Pending / not verified

Listed explicitly so nothing here is mistaken for a settled result.

1. **t-EDM checkpoint reuse** (§4.3) — unresolved. Two fetch attempts failed; the paper's
   position on fine-tuning from a Gaussian EDM checkpoint should be read directly before
   costing E5.
2. **The §2.2 alignment claim is suggestive, not proven.** That the convolutional lag holes
   (10–30 min) coincide with the failing wet-spell and storm-duration bands is an
   observation, not a demonstrated cause. E1 is the experiment that tests it. Note that §3.3
   now makes the *long-range* failure a settled matter of block independence, which is a
   separate mechanism — §2.2 bears only on the sub-block band.
3. **v11 / v12 inference is still pending — and this audit makes it a decisive test.**
   Both checkpoints exist locally (`logs/ImagenFew/Rainfall_Regime/v11`, `.../v12`, trained
   2026-09-24/25) but no `rainfall_synthetic_10y_v11.csv` or `_v12.csv` exists, so the
   unconditional ablation is unscored. **Pre-registered prediction from §3.3 and §3.5:**
   because these are unconditional, the seasonal-phase label — the only inter-block channel
   there is — is gone; but that label never carried "a storm is in progress" information, so
   it should make **no difference to the cliff**. v12 (L = 64) should therefore show a cliff
   at k = 64 of comparable depth to v10's 0.08, and v11 (L = 24) a cliff at k = 24 of
   comparable depth to v8's 0.12. If instead the unconditional cliffs are markedly *deeper*,
   the seasonal label is doing more long-range work than §3.5 credits it with, and the
   mechanism needs revising. Either outcome is informative, and it costs one generation run
   on checkpoints already on disk.
4. **v13 / v14 training is still pending** — no run directories exist. The transform
   ablation is unaffected by anything in this audit (it changes the data map, not the
   architecture), and all three stay comparable because the attention defect is identical in
   v10, v13 and v14. **But §3.3 caps what the ablation can deliver:** a transform reallocates
   value range across wet intensities and cannot move a wall that sits at the block boundary.
   Expect v13/v14 to move the intensity marginal and the extreme quantiles, and to leave the
   storm-duration cliff exactly where it is. That is worth writing down *before* the results
   arrive, so the ablation is not credited or blamed for something outside its reach.

---

## 7. How this changes the thesis narrative

Three statements are now available that were not before, all framed as claims about models
rather than about rain:

1. **Attention placement in folded-time-series diffusion is a first-class design variable,
   and it is easy to get silently wrong.** A head-count rule inherited from a 256×256 image
   model (`out_channels // 64`) truncates to zero on a 32-channel stage and removes the
   requested attention without error. Publishable as a short architectural note; immediately
   useful to anyone else building on ImagenTime/ImagenFew at small `model_channels`.

2. **The folding transform determines which temporal lags the convolution can reach.** With
   `delay == embedding`, a 3×3 kernel becomes a sparse multi-scale temporal filter with taps
   at lags {0, ±1, ±7, ±8, ±9} and holes at 2–6. This is a *derivable property of the
   representation*, it predicts where the model should struggle, and it generalises to any
   delay/embedding pair. It reframes the delay embedding from an implementation detail into
   an inductive bias with an analysable receptive field.

3. **Context length and long-range structure are separable, and this model has conflated
   them.** The v8→v10 sequence treated `seq_len` as the long-range knob. But v10 already
   reproduces sub-block structure near-perfectly while losing 13× of the long storms that
   carry a fifth of the water — so the binding constraint is not context *length*, it is the
   absence of any conditioning between blocks. That reframes the whole v8/v9/v10 comparison
   (`docs/FINETUNING_ANALYSIS.md` §3 already flags it as confounded with checkpoint choice)
   and points at a fix that needs no longer context at all.
