# Two-Resolution Generator: Coarse Storms, Fine Texture

**Written:** 2026-10-14, before any code or run. **Status:** proposed.
**Question:** can a generator that sees 32 hours at once at 30-minute resolution, and then fills in 5-minute detail
with a v14-like model, reproduce long storms and sewer loading while keeping v14's 5-minute calibration?
**Why this design:** the window-horizon pilot says the sewer needs about a one-day horizon (one-day real windows
recover ~94% of overflow; 5h20 windows ~60–70%). Every attempt to stitch 5h20 pieces up to that horizon failed,
because the continue-or-stop decision at each seam compounds (pasting too weak, guidance runs away, v17 loses
calibration). Here long storms form inside one coarse piece; there is no hourly chaining.
**Literature:** `docs/LITERATURE_TWO_LEVEL_GENERATION.md`. The structure is established (classical disaggregation in
hydrology; cascaded diffusion in ML); a learned generator of long continuous sub-hourly point records was not found.

---

## 1. The design

| Level | Model | Sees | Decides |
| :--- | :--- | :--- | :--- |
| Coarse | diffusion on 30-minute totals; v14's recipe and network; 64 steps = **32 hours** per piece | the seasonal phase | when it rains, for how long, how much, across days |
| Fine | v14-like diffusion on 5-minute rain, with the 30-minute totals as a second input channel | the coarse totals of its 5h20 window | the 5-minute pattern inside each half-hour |

**Generation:** the coarse model makes 10 years of 30-minute totals (independent 32-hour pieces on the calendar
timeline, joined end to end). The fine model then turns every 10 half-hours into 60 five-minute steps, given those
totals. Each half-hour's six values are scaled to its coarse total exactly. Fine windows are **independent given the
coarse series**: nothing is chained, so no seam decision can compound.

**What each level must get right:**

| Property | Level |
| :--- | :--- |
| Long storms, storm depth, rain in storms > 5h20, daily totals, sewer loading | coarse |
| Bursts, single-step showers, 5-minute peaks, wet/dry texture inside a storm | fine |

---

## 2. Settings fixed now (rules from `sitetransfer/rules.py`, applied to 2000–2007 at 30 min)

| Setting | Value | Rule or reason |
| :--- | :--- | :--- |
| Coarse resolution | 30 min | 64 steps = 32 h, past the ~1-day horizon; fits the pretrained 8×8 network (`ImagenFew_64.ckpt`) unchanged |
| Coarse wet threshold | **0.0075 mm per 30 min** | smallest value making up ≥ 1% of wet half-hours |
| Coarse asinh scale | **0.1025 mm** | median wet half-hour |
| Coarse training | v14 config otherwise (asinh, 4 seasonal phases, batch 2048, lr 1e-4, dropout 0.10, EMA), **500 epochs** | 140,097 windows give 68 steps per epoch; matching v14's 205,000 steps would need ~3,000 passes over the data and risk memorising it, so the number of passes is matched instead, and memorisation is checked |
| Seasonal phases | v14's calendar profile, taken at every 6th 5-minute slot | same seasons as v14 |
| Fine model | v14's network with a second input channel; starts from v14's weights, new channel's weights zero (so at the start it **is** v14) | keeps v14's calibration from the outset |
| Fine input channel | each 5-minute step carries its half-hour's total ÷ 6, in v14's model units | same units as the rain channel; never noised |
| Fine training | v14's recipe and loss, 5h20 windows from 2000–2007 at every offset, fine-tuned from v14 (epochs: section 5) | |
| Fine windows at generation | start every 60 steps (10 half-hours), 64-step window, last 4 steps discarded | windows aligned to half-hours, so every half-hour lies inside one window |
| Mass matching | fine values below the 5-minute wet threshold (0.005 mm) set to zero, then the six values of each half-hour scaled to the coarse total; a scaled value still below 0.005 mm is added to the half-hour's largest step; dry coarse half-hour → six zeros | rain conserved exactly; no sub-gauge drizzle (the cascade's known fault) |
| Edge case | coarse wet, fine all zero → the total goes to the one step where the fine model's raw value is largest; counted and reported | |
| Split | train 2000–2007, never touch 2008–2009 except as held-out reference | as v14 |

---

## 3. Stage 1: the coarse model alone (cluster)

**Build:** 30-minute split files; config `config_coarse30.yaml` (v14's with only the data, scale and epoch keys
changed); a small generator for 30-minute series (independent pieces, calendar timeline); evaluation at 30 minutes.

**How it is judged.** Both the generated and the real series are compared at 30-minute resolution. For the existing
metric code, each half-hour total is spread evenly over its six 5-minute steps (real and generated alike), so only
structure at 30 minutes and longer is compared. Bands are resampled from the real years spread the same way, as in
`sitetransfer`. Spreading lengthens storms, so the real targets are recomputed at 30 minutes (below).

| Comparator | Why |
| :--- | :--- |
| real 2000–2007 (reference) and 2008–2009 (held out) | the target and its noise |
| v14 aggregated to 30 minutes | what we have now |
| perfect 32-hour-window pilot at 30 minutes | the best any 32-hour piece generator can do |

**Stop rule.** Continue only if the coarse model:
1. passes family D (long-scale structure: deep storms, rain in storms > 5h20, top storms, long runs, variance
   across scales, daily totals) under resampled bands; **and**
2. has at least 60% of rain in storms longer than 5h20 (2-hour gap), measured at 30 minutes, where real has 71% and
   v14 44% (at 5 minutes: real 65%, v14 25%, v17 53%); 60% closes at least 60% of v14's gap; **and**
3. keeps volume within ±5% of real; **and**
4. shows no memorisation (no generated 32-hour window closer to a training window than the closest 5% of held-out
   real windows, the test of `eval_suite.py`).

Otherwise stop: a coarse diffusion model cannot learn day-scale storm structure from 8 years, and that is reported.

---

## 4. Stage 2: the fine model on real coarse input ("perfect coarse" test, cluster)

Trained in parallel with stage 1, since it learns from real data only.

**Build:** package `tworesolution/` with the two-channel wrapper (`EDMPrecond` with 2 inputs and 1 output; the
condition is joined after the network's input scaling, so it is never shrunk), its training loop (v14's loss, EMA,
optimiser), generation and mass matching. **Test first:** with the new channel's weights at zero, the wrapper's output
equals v14's exactly.

**The test:** disaggregate the **real** 2008–2009 half-hour totals with the fine model, and compare with the real
5-minute rain of those years. This isolates the fine model: any failure is the fine model's, not the coarse model's.
Same test with a **microcanonical cascade** (classical, fitted on 2000–2007 by counting split patterns; no training):
the standard tool for this step.

**Stop rule.** Continue only if the fine model passes families B (intensity and extremes) and C (short-scale
timing) against real 2008–2009 under resampled bands, **and** rain carries across the joins between fine windows
(every 5 h) as often as inside them: P(wet | wet) at joins within 0.05 of the inside value (real 0.84). Fine windows
do not see their neighbours, so a fine model that starts or ends windows dry would break long storms at every join. If the cascade passes and the fine model does not, the full
chain uses the cascade, and the paper reports that diffusion adds value at the coarse level only.

---

## 5. Stage 3: the full chain (cluster)

Coarse model (10 years) → fine model → 5-minute series `twores_v1`; also coarse model → cascade (`twores_cascade`).
Fine generation needs no chaining: about 17,500 independent windows, all batched.

**Judged by the rules already used for v17** (comparable across all attempts):

| Check | Target |
| :--- | :--- |
| Unbroken runs > 5h20 / yr | ≥ 10 (real 14.3; v14 1.4; v17 8.2) |
| Share of rain in them | ≥ 15% (real 21.4%) |
| Volume | 0.95–1.05 of real |
| Wet spell | 0.90–1.10 of real |
| Gate A (fixed bands) | ≥ 12 / 18 (v14 14; v17 5–9) |
| Sewer overflow, BC and EFD | 0.85–1.15 of real (v14 0.66; v17 0.82) |

Plus the standard evaluation (families A–E, resampled bands) and diffusion vs cascade on families B and C.

---

## 6. Order of work and time

| Step | Where | Time |
| :---: | :--- | :--- |
| 1 | laptop | plan and predictions in the diary |
| 2 | laptop (< 30 min of running) | 30-min data and splits; coarse config; coarse generator; `tworesolution/` package with the two-channel wrapper and its identity test; cascade; evaluation at 30 min; SLURM jobs; smoke tests |
| 3 | cluster | coarse training (~2–3 h, 500 epochs × 68 steps) and fine training (~8–11 h) **in parallel** |
| 4 | cluster | stage 1 evaluation (minutes); stage 2 perfect-coarse test (minutes) |
| 5 | cluster | stage 3 generation (~1 h) and evaluation; sewer check |

About 3–4 days of code, then ~1 day of cluster time. Each stop rule is applied before the next stage is sent.

---

## 7. Predictions (written before any run)

1. Stage 1 passes: the coarse model puts 60–71% of rain in storms longer than 5h20 (at 30 minutes), volume within 5%.
2. Stage 2: the fine model passes families B and C on real coarse input; the cascade fails C (autocorrelation too low,
   too many tiny intensities, as reported in the cascade literature).
3. Stage 3: ≥ 10 unbroken runs > 5h20 per year, Gate A ≥ 12, sewer overflow 0.9–1.1.

## 8. Not covered

Singapore (the recipe can be transferred later with `sitetransfer`), multiple gauges, time-of-day conditioning,
storms longer than 32 hours (4.6% of Astlingen's rain in 2000–2007).
