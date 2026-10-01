# Next Steps — after v10–v14 are finished

**Written:** 2026-09-28 · **Do not start until:** v13 and v14 have trained and been scored.
**Updated 2026-09-30:** the gate (§1) is done; results in the gate status below and in the
research diary (Sept 30 entries). **v14 is now the baseline: wherever this plan says v10, use
v14** (`regime_training/config_v14.yaml`, checkpoint `logs/ImagenFew/Rainfall_Regime/v14`).
**Background (read if this is cold):** [`docs/ATTENTION_AND_RECEPTIVE_FIELD_AUDIT.md`](../docs/ATTENTION_AND_RECEPTIVE_FIELD_AUDIT.md)
**Figure:** `results/transform_diagnostics/storm_duration_cliff.png`
(regenerate: `python scripts/diagnose_storm_duration_cliff.py`)

---

## ▶ What to run now (written 2026-09-30, before any of these results)

All code for §5 steps 0 and 2 is written, checked locally, and committed. On the cluster,
after `git pull`, the jobs are independent and can all be submitted at once:

| Job | Command | What it answers | Time |
| :--- | :--- | :--- | :--- |
| Calendar drift fix | `MODES=calendar BRIDGE_BLOCKS=0 sbatch scripts/submit_generation_v14.sh` | Does removing bridges (and so the 39-day drift) improve seasonality? → `..._v14_nobridge_cal.csv` | < 1 h |
| Denoising, training years | `sbatch scripts/submit_denoise_diagnostic.sh` | The tail check that did not finish on CPU → `results/transform_diagnostics/denoise_{train,test}/` | < 1 h |
| Assembly ablation, 10 y | `sbatch scripts/submit_assembly_ablation.sh` | Thesis-grade version of the 2-year paired test | < 1 h |
| **E1** attention | `sbatch scripts/submit_job_v15.sh` | 200-epoch fine-tune of v14 with 8×8 attention → `v15`, `v15_cal` | ~4–5 h |
| **E1 control** | `sbatch scripts/submit_job_v15_ctrl.sh` | Same fine-tune, no change → `v15_ctrl`, `v15_ctrl_cal` | ~4–5 h |
| **E2** context, Markov | `sbatch scripts/submit_generation_v14_ctx.sh markov` | Chained generation from v14 → `v14_ctx` (+ `_m1`–`_m3`) | ~4–8 h, ETA in log |
| **E2** context, calendar | `sbatch scripts/submit_generation_v14_ctx.sh calendar` | Same with the drift-free calendar → `v14_ctx_cal` (+ members) | ~4–8 h |

The E2 log prints the running wet % and mm/yr every 100 blocks (real: 9.2% wet, 710 mm/yr;
v14: 9.0%, 716). If a job hits its 16 h limit, submitting the same command resumes it.

**Scoring after syncing the CSVs:**
- `python scripts/gate_a_scorecard.py --reference train --versions v14 v14_cal v14_nobridge_cal v15 v15_cal v15_ctrl v15_ctrl_cal v14_ctx v14_ctx_cal`
- `python scripts/score_context_generation.py --versions v14_ctx v14_ctx_m1 v14_ctx_m2 v14_ctx_m3` (E2 against §3)
- `python scripts/score_context_generation.py --versions v14_ctx_cal v14_ctx_cal_m1 v14_ctx_cal_m2 v14_ctx_cal_m3 --baseline v14_nobridge_cal`
  (baseline without bridges: its calendar is drift-free like E2's, so the years line up)

**Predictions, written before the results:**
1. **No bridges:** v14_nobridge_cal's monthly Pearson r rises above v14_cal's 0.597; other
   metrics stay within noise (bridges had no measurable effect in the paired test).
2. **E1:** v15 beats v15_ctrl in the sub-block band (wet spells, storm duration, hourly ACF),
   and the survival ratio at 320 min stays about the same (~0.13) for both. If E1 moves the
   cliff, the diagnosis in §0 is wrong.
3. **E2:** joins stop breaking storms. In a CPU pilot (4 chains × 11 days), the chance that
   rain continues across a join rose from **0.11 (no context) to 0.86 (K = 2)**, the same as
   inside a block (0.85). **Risk seen in the same pilot:** context chains were drier (6.7% wet
   vs 10.5% without context and 10.7% for v14 on the same days, about 2 standard errors). The
   §3 checks (zero fraction within 1 pp, volume 0.95–1.05) decide it. A dry bias with no drift
   points to the RePaint settings (`RESAMPLE`, `K`) or to E3; drift points to E3.

**Results (2026-10-01; `notebooks/research_evaluation.ipynb`, diary Oct 1 entries):**
1. **No bridges: confirmed.** Monthly r 0.62 → **0.97** (Gate A 13 → 14/18). Use
   `--bridge_blocks 0` for every calendar run from now on.
2. **E1: refuted in its first half.** Attention made storms *shorter* (−3.5 min, 10/10 paired
   years) with identical training loss; the cliff did not move. Dropped.
3. **E2: cliff removed, rain halved.** Survival at 5h20 0.13 → 0.73–0.79 (inside the real range),
   but volume ratio 0.49. A CPU pilot showed the dry bias comes from RePaint resampling
   (no resampling: 10.1% wet vs 4.8%; storms still continue across joins, 0.62 vs 0.11).

**▶ Next (the one idea from the evaluation):** E2 without resampling.
`RESAMPLE=1 sbatch scripts/submit_generation_v14_ctx.sh markov` (and `calendar`), ~10–12 h
each, then
`python scripts/score_context_generation.py --versions v14_ctx_u1 v14_ctx_u1_m1 v14_ctx_u1_m2 v14_ctx_u1_m3`
against §3. If it drifts or still runs dry → E3 (§4).

**Done 2026-10-01 (from the literature review, `docs/LITERATURE_RAINFALL_GENERATORS.md`):**
* **Stronger classical baseline**: storm-and-cell (randomised Bartlett-Lewis) model,
  `scripts/baselines/run_bartlett_lewis.py`. It fails where diffusion succeeds and vice versa:
  realistic long storms (13.8/yr), daily totals, persistence and seasons, but wrong 5-minute
  texture and weak short bursts; Gate A 2/18.
* **Official heavy-rain table** (KOSTRA-DWD-2020, Erft cells), `scripts/kostra_compare.py`: the
  real gauges match it within ~5% from 15 minutes up; the 4-gauge average used for training has
  5-minute peaks ~40% below a single gauge.

**New items these raised:**
* **Four gauges for the sewer test.** SWMM-Astlingen reads four separate gauges; the generator
  makes one average series. Before the agent-in-the-loop test, generate the four gauges (jointly,
  e.g. as 4 channels, or by spreading the average back out), or 5-minute peaks will be ~40% weak.
* **Two-level generator (later).** A storm-scale model decides when and for how long it rains; the
  diffusion model fills in the 5-minute detail. Motivated by the complementary failures above.

**Also found while building E1 (not changed):** `ImagenFew.py` never passes `dropout` to the
network, so every version so far trained with the network's default **dropout = 0.10**, not the
configs' `dropout: 0.0`. Left as is so E1 differs from v14 only in attention.

---

## 0. The one-paragraph reminder

The model does not generate 10 years of rain. It generates **one 5h20m chunk at a time**
and glues ~16,400 of them end to end. Each chunk is made **from scratch, knowing nothing
about the chunk before it**. So storms shorter than one chunk come out excellent, and storms
longer than one chunk barely exist — a long storm would need two chunks in a row to be stormy
by coincidence.

Measured: the storm-duration curve matches reality closely right up to the chunk length, then
falls off a cliff by about 10× **at exactly the chunk length, in every version** (v7/v8 at
120 min, v9 at 180 min, v10 at 320 min). So raising `seq_len` never taught the model
long-range structure — **it just moved the wall.**

The cost: **v10 makes 1.10 storms/year longer than 320 min; reality has 14.25.** Those storms
carry **21.4% of real rainfall but only 1.6% of generated rainfall.** A fifth of the water is
missing from exactly the events that matter for flooding.

The good news: the fix is mostly already written and switched off.

---

## 1. Gate — finish these first

Do not start §2 until all three are done. They are cheap and they change what §2 should look
like.

| # | Task | Why it has to come first |
| :-- | :--- | :--- |
| 1.1 | **Generate v11 and v12.** Checkpoints are already on disk (`logs/ImagenFew/Rainfall_Regime/v11`, `.../v12`). | This is now a **test of the diagnosis**, not just an ablation. See the prediction in §1.1 below. |
| 1.2 | **Finish v13 / v14 training + scoring.** | The transform ablation is independent of everything here and should be closed out cleanly first. |
| 1.3 | **Run the cliff diagnostic on all of them:** `python scripts/diagnose_storm_duration_cliff.py` (add v11–v14 to `DEFAULT_RUNS`). | Confirms the cliff is where it should be in six more versions before we act on it. |

### 1.1 Prediction to check, written down in advance

**v11/v12 are unconditional** — no seasonal-phase label at all. The label was the only thing
connecting one chunk to the next, but it changes on a multi-day timescale and never said
"a storm is happening right now."

**So removing it should make no difference to the cliff.** Expect:

- v12 (chunk = 64): cliff at k = 64, depth around 0.08 — same as v10
- v11 (chunk = 24): cliff at k = 24, depth around 0.12 — same as v8

**If the unconditional cliffs are clearly deeper than that**, the seasonal label is doing more
long-range work than the diagnosis credits, and §2 needs rethinking before spending anything
on it.

### 1.2 Prediction for v13 / v14, also written down in advance

A transform changes how rainfall values are spread across the model's number range. **It
cannot move a wall that sits at a chunk boundary.**

Expect v13/v14 to move the **intensity** numbers (P99, P99.9, max, the wet-intensity
histogram) and to leave the **storm-duration cliff exactly where it is**. Do not credit or
blame the transform for the cliff either way.

### Gate status (2026-09-30): done

| Version | Gate A | Survival ratio at k = L | Storms > 320 min /yr | Rain in them |
| :--- | :---: | :---: | :---: | :---: |
| real (2000–2007) | — | — | 14.25 | 21.4% |
| v8 (L = 24) / v11 (no label) | 5 / 4 | 0.12 / 0.12 | 0.10 / 0.00 | 0.1% / 0.0% |
| v10 / v12 (no label) | 11 / 8 | 0.08 / 0.08 | 1.10 / 1.10 | 1.6% / 1.6% |
| v13 (log1p) | 14 | 0.12 | 1.70 | 2.2% |
| **v14 (asinh)** | **14** | **0.13** | **1.90** | **2.2%** |

- **1.1 confirmed.** The unconditional cliffs are identical to the conditional ones. The label
  does no long-range work, so §2 stands as written.
- **1.2 half right.** The cliff did not move. But the main gain was not in the intensity
  histogram: the transforms brought back very light rain at storm edges (0.005–0.02 mm steps:
  real 3.14% of steps, v10 2.76%, v14 3.07%), which lengthened spells and storms *below* the
  wall (survival ratio at 240 min 0.63 → 0.77) and fixed the volume deficit.
- **1.3 done** for v8 and v10–v14 (numbers above; computed with the functions in
  `scripts/diagnose_storm_duration_cliff.py`, whose figure still plots only v8–v10).
- **New baseline for E2 (§3): v14 makes 1.90 storms/yr longer than 320 min, carrying 2.2% of
  rain.**

### Bridging and crossfade: answered, no cluster ablation needed (2026-09-30)

Question: should we ablate how blocks are joined (the blended `[0, .5, 0, .5]` bridge label;
the overlapping crossfade)? Answered with a **paired** test that needs no retraining and no
cluster: generate one set of v14 blocks and join the *same* blocks four ways
(`scripts/ablate_block_assembly.py`, 2 years on CPU; diary entry "Block-Assembly Ablation").

| | bridge + xfade (now) | no bridge + xfade | bridge + hard join | no bridge + hard join |
| :--- | :---: | :---: | :---: | :---: |
| Storms > 320 min / yr (real 14.25) | 1.0 | 1.0 | 1.0 | 1.0 |
| Rain in those storms (real 21.4%) | 2.4% | 2.4% | 1.7% | 1.7% |
| Wet spell, min (real 32.0) | 28.5 | 28.5 | 27.8 | 27.8 |
| P99 wet, mm (real 0.588) | 0.629 | 0.627 | 0.646 | 0.640 |

- **Bridges do nothing measurable.** **Crossfade** shifts a few metrics by 1–3% in both
  directions. **Neither moves the cliff**, which is set by the blocks being independent.
- **Bridges cause the calendar drift** (39 days late by year 10; 3 days early without them).
  Use `--bridge_blocks 0` for new calendar runs.
- **E2 baselines:** compare E2 against both "bridge + xfade" (production) and "no bridge + hard
  join" (naive), with the same plan seed.

---

## 2. The two main experiments

### E1 — Turn the attention back on *(free; one config knob + one small code change)*

**The problem.** The configs ask for attention at the finest level (`attn_resolution: [8,4,2]`)
and the code silently ignores it. Head count is computed as
`out_channels // channels_per_head` = `32 // 64` = **0**, which is falsy, so the attention
layers are never even built. `attn_resolution: [8,4,2]` and `[4,2]` currently produce a
**bit-identical model**.

**Why it matters here.** The 8×8 stage is the only one whose 64 tokens are clean, contiguous,
single points in time. Every coarser stage pools 2×2 patches whose cells are two time
fragments 40 minutes apart. So the model has never had a proper global view of a chunk at
full resolution.

**The change.** `models/ImagenFew/networks.py:461` hardcodes `channels_per_head=64` inside
`DhariwalUNet`'s `block_kwargs`. Make it a constructor argument (default 64, so nothing else
in the repo changes) and pass it down from the config as e.g. `channels_per_head: 32`.
Equivalent alternative: pass `num_heads=1` explicitly.

**Why this is free — already verified.** The attention branch ends in a `proj` layer whose
weight *and* bias are initialised to exactly zero, and `skip_scale` is 1 for `DhariwalUNet`.
I confirmed numerically that an attention-enabled block and an attention-free block produce
**bitwise identical** output at initialisation (`torch.equal → True`). So a fine-tune from the
v10 checkpoint **starts at exactly v10's behaviour** and can only add capability. There is no
risk of starting off worse.

**Also worth doing while in there:** the bottleneck attention at `dec.1x1_in0` runs on a 1×1
grid, where softmax over a single key is exactly 1.0 — it does nothing spatially. Either drop
it, or shorten `ch_mult` to `[1, 2, 4]` so the bottleneck lands at 2×2 and the attention there
has something to attend to.

**As built (2026-09-30).** Not `channels_per_head: 32`: that would also give the *existing*
4×4 and 2×2 attention layers (64 channels) 2 heads instead of 1 and change what their trained
weights compute. Instead `attn_min_heads: 1` (config key → `DhariwalUNet` → `UNetBlock.min_heads`)
only turns 0 heads into 1, so only the 8×8 blocks change (7 blocks, 30,016 new parameters).
Checked locally on the v14 checkpoint through the training loader: outputs are bit-identical to
v14's generation weights at σ = 0.01–80, and the new `proj` gets gradients on the first step.
Configs `config_v15.yaml` / `config_v15_ctrl.yaml`; jobs `submit_job_v15.sh` / `submit_job_v15_ctrl.sh`.

**Control needed (added 2026-09-30).** v10, v12, v13 and v14 all reached their best loss at
epoch 460–500 of 500, so none had converged. A fine-tune from v14 with attention on will
also get *more training*. Run the same fine-tune **without** the attention change (same
epochs, same settings) and compare E1 against that, not against v14 itself.

**How to judge it.** Run the normal Gate A scorecard plus the cliff diagnostic. E1 targets the
**sub-chunk** band (wet spells ~30 min, storm duration ~60 min), **not** the cliff. If E1 alone
moves the cliff, the diagnosis in §0 is wrong.

---

### E2 — Let each chunk see the one before it *(no retraining at all)*

**The idea.** Instead of generating chunk N from nothing, fill in the last few **columns** of
chunk N−1 first and let the model generate the rest around them.

**Why the columns line up perfectly.** Each 64-step chunk is folded into an 8×8 picture. Since
`delay` and `embedding` are both 8, **each column is exactly 8 consecutive 5-minute steps =
40 minutes**, and the columns sit in time order left to right. So "the last 2 columns" means
exactly "the last 80 minutes" — nothing overlapping, nothing scrambled. This is a lucky
property of the current folding; it would not hold if `delay ≠ embedding`.

**What already works.** `ImagenFew.forward(x, mask, …)` implements masked denoising, and
`DiffusionProcess.impute()` already re-imposes the known pixels at every solver step. That is
RePaint's rule, correctly written. The machinery is fine.

**What is switched off.** The `mask` is all zeros in **both** places:

| File | Line | What it says today |
| :--- | :--- | :--- |
| `regime_training/train_regime.py` | **348** | `x_ts_mask = torch.zeros_like(x_ts)` → nothing is ever known during training |
| `regime_training/generate_hmm_v1.py` | **277** (single block) and **315** (batched) | mask built from zeros → nothing is ever known during generation |

**The change.** Generate with a stride shorter than the chunk, and mark the first *k* columns
as known (`mask = 1` there), filled with the previous chunk's last *k* columns. Start with
k = 2 (80 minutes of history).

**Two things must be added — do not skip these.**

1. **RePaint resampling (n ≈ 10).** The model has never practised keeping known pixels
   (see the table above), so this is unfamiliar territory for it. RePaint's fix is to step
   back and forth through the denoising process about ten times so the known part and the
   invented part blend. Without it there is a visible mismatch at the join. The benefit
   saturates around n = 10.
2. **Noisy history, not clean history.** Hand the model the previous chunk's columns with a
   small amount of noise added rather than perfectly clean. This is Diffusion Forcing's trick.
   Reason: if a model eats its own output 16,400 times in a row, small errors compound and the
   whole series slowly drifts. Noisy history teaches it not to over-trust the past.

**As built (2026-09-30): `regime_training/generate_context.py`.** One correction to the
above: `impute()` keeps the known pixels *clean* while the rest is noisy, an input the network
never saw in training. The new generator follows RePaint instead: known pixels are re-noised to
the current σ at every step. Settings: K = 2 known columns (48 new steps per block), 10 RePaint
passes on every step with σ ≥ 0.1 (Euler on intermediate passes, Heun on the last), history
noise 0.05 (model units), labels looked up by time on the production plan's timeline (so E2 year
y pairs with v14 year y), 4 independent chains per job, ~323 network evaluations per block
(production: 71). `--context_cols 0` runs the same code with no context as a control.

**The risk, stated honestly.** A 10-year run is ~16,400 chained chunks — far longer than
anything in the video-diffusion papers this borrows from. The known failure mode is
*spectral collapse*: output gradually smooths out as errors accumulate. Your seasonal-phase
label helps, because it is re-imposed from outside at every chunk and keeps pulling the chain
back. **But this has to be measured, not assumed.**

---

## 3. Pass / fail, decided in advance

### E2 succeeds if

| Metric | Real | v10 | **v14 now** | Target |
| :--- | :---: | :---: | :---: | :---: |
| Storms > 320 min, per year | 14.25 | 1.10 | **1.90** | ≥ 10 |
| Share of total rain in those storms | 21.4 % | 1.6 % | **2.2 %** | ≥ 15 % |

### E2 must not break

The short-range behaviour is currently **excellent** and must stay that way. Check all of:

- wet-run length histogram (1 / 2 / 3 / 4–6 / 7–12 / 13+ steps) stays within ~2 pp per bin
- zero fraction stays within 1.0 pp of 90.80 %
- lag-1 autocorrelation stays within 0.02
- annual volume ratio stays inside 0.95–1.05

### E2 is drifting (spectral collapse) if

Score **each generated year separately** instead of pooling all ten:

- annual volume ratio moves steadily in one direction from year 1 to year 10, **or**
- hourly ACF over lags 1–24 h degrades steadily year on year

**Compare year by year against v14 with the same plan seed (added 2026-09-30).** The
wet and dry years come from the state plan, not the model: v10 and v14 share seed 42's plan,
and their annual volume ratios rise and fall together (both low in years 2 and 9). Without
chaining, v14's annual volume ratio already swings by about ±0.12 from year to year, with a
slope of only −0.002 per year (hourly ACF RMSE: +0.001 per year). So run E2 with `--seed 42`
and test the **per-year difference E2 − v14**: that cancels the plan's pattern, and drift shows
up as a trend in the difference well above those baseline slopes.

Either of those means the chain is drifting. If so, stop and go to E3 — the noisy-history
trick alone was not enough.

---

## 4. If E2 drifts — the follow-ups

**E3 — Train the masked path.** Nearly a one-line change at `train_regime.py:348`: instead of
an all-zero mask, mark the first *k* columns as known with *k* drawn at random from
{0, 1, 2, 3, 4}. `k = 0` reproduces current behaviour exactly, so unconditional generation
still works. The model then actually *learns* to continue from a known past, and E2 stops
being unfamiliar territory. Costs one retrain, same budget as v13/v14.

**E4 — Split occurrence from intensity.** Generate two channels — a wet/dry field and an
intensity field — instead of one. Justified by a measurement: **90.8 % of steps are exactly
zero**, and the wet/dry signal carries about **4× the memory** of the intensity signal at the
chunk horizon (ACF at lag 64: 0.213 vs 0.053). The two behave differently and probably want
different treatment. This is the biggest change here and the one most likely to stand on its
own as a contribution — but it should come *after* E1–E3, not before.

**E5 — t-EDM (heavy-tailed noise).** Swaps the Gaussian noise for a Student-t, controlled by
one number. It is **orthogonal to v13/v14**: the transform changes how the data is mapped,
t-EDM changes the noise itself. Running it afterwards gives a clean 2×2: {standard, asinh} ×
{Gaussian, Student-t}. Open question to settle from the paper first: whether an existing EDM
checkpoint can be fine-tuned or a full retrain is required.

---

## 5. Suggested order

0. **Cheap loose ends (GPU minutes, no training; scripts ready, see the top):**
   - regenerate v14_cal with `--bridge_blocks 0` and re-score. Prediction: monthly Pearson r
     (0.597) rises if the drift was hurting seasonality;
   - run the denoising diagnostic with `--split train` on the cluster (did not finish on CPU);
   - optional: `scripts/ablate_block_assembly.py --run v14 --years 10` for the thesis table.
1. ~~**Gate (§1)** — v11/v12 generation, v13/v14 scoring, cliff diagnostic on all~~ (done 2026-09-30)
2. **E1 + E2 together** — one free training change, one inference-only change
   (code ready: `attn_min_heads` in `models/ImagenFew/networks.py`, `config_v15*.yaml`,
   `regime_training/generate_context.py`; commands in "What to run now" at the top)
3. **E3** — only if E2 drifts
4. **E4**, then **E5**

E1 and E2 target different things (sub-chunk structure vs the cliff), so running them together
is fine — they will not confound each other, and the cliff diagnostic separates them cleanly.
