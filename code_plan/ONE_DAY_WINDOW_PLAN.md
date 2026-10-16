# v19: The v14 Recipe with a One-Day Window

**Written:** 2026-10-15, before any GPU run. **Status:** proposed; smoke job first.
**Question:** does a diffusion model that generates one whole day at a time at 5-minute resolution reproduce the
long storms and the sewer loading that every 5h20 model missed, while keeping v14's 5-minute calibration?

## 1. Why this run

The thesis's central claim is that the window length sets the horizon: per-window statistics look right, but a
generator of independent 5h20 pieces cannot make storms longer than its window, and the sewer needs about a day.
The window-horizon pilot measured the ceiling of each window length with real windows ("perfect generators"):

| Perfect generator | Unbroken runs > 5h20 / yr | Rain in them | Deep storms (≥ 10 mm) / yr | Sewer overflow (BC) |
| :--- | :---: | :---: | :---: | :---: |
| 5h20 windows | 1.9–2.8 | 1.8–3.6% | 8.3–9.6 | 0.57–0.71 |
| one-day windows | 11.3–13.8 | 16.1–20.5% | 13.0–15.2 | 0.78–1.05 (mean 0.94) |
| real 2000–2007 | 14.3 | 21.4% | 15.8 | 1 |

Every released ImagenFew checkpoint is an 8×8 network (64 steps), so a learned one-day model could not be built
until now. The user's own pretraining (2026-10-15 note below) gives a 24×24 checkpoint for 288-step series.
Learned models have tracked their window's perfect-generator line on long-scale statistics (v14 on the 5h20 line;
the two-resolution coarse model on the 32-hour line for storm duration and depth), so v19 tests the claim directly:
no stitching inside a day, no second model.

## 2. The one change

`regime_training/config_v19.yaml` (written by `python -m oneday config`) is `config_v14.yaml` with only the window
changed: `seq_len` 288, `delay` = `embedding` = 24 (an image of 24 rows × 12 columns, padded to 24 × 24; the
padding is marked known and never scored), `img_resolution` 24, `attn_resolution` [12, 6, 3]. These match the
checkpoint exactly; only its label layer (32 pretraining datasets) is replaced by v14's 4 seasonal phases.
Initial weights: `models_ckpt/ImagenFew/ImagenFew_288_807epochs.ckpt` (sha256 `3fd96271c481d301…`; not in git).
Everything else is v14's: asinh transform and scale 0.035 mm, seasonal phases, loss, AdamW, learning rate, EMA,
dropout, 2000–2007 training split, training seed 0, the production generator (calendar timeline, no bridge
blocks, 4-step overlap; the one-day calendar file `seasonal_transition_matrix_train_len288.pkl`).

**The checkpoint:** pretrained with `run.py` on the ImagenFew pretraining datasets at 288 steps, batch 1024,
stopped at epoch 807 of the 1,094 configured (user, 2026-10-15); `configs/pretrain/pretrain.yaml` unedited, with
the window settings given on the command line. Its quality is not separately measured: a caveat to report.

## 3. Epochs and batch size: fixed by a rule, from a GPU smoke job

A 24×24 image has 9× the pixels of v14's, so a training step is slower. The smoke job (`oneday/slurm/submit.sh
smoke`, about 10 minutes) times real training steps at batch 2048 and 1024, sampling, and reads the partition's
time limit (`results/oneday/smoke.json`). Then:

* **batch 2048** (v14's) if it runs with peak memory at most 90% of the GPU's; otherwise 1024;
* **500 epochs** (v14's number of passes: 205,000 steps at batch 2048) if training, the training checks and the
  generation of 4 members fit in 85% of the time limit;
* if fewer than 500 epochs fit, training is **not** silently shortened: we decide with the numbers in front of
  us (a shorter run reported as a deviation, or splitting training across jobs), and record the decision before
  training. The two-resolution coarse model showed that undertraining confounds the result.

## 4. Rules (as v17 and the two-resolution stage 3; means over 4 generation seeds, 42–45)

| Check | Target |
| :--- | :--- |
| Unbroken runs > 5h20 / yr | ≥ 10 (real 14.3, perfect one-day 11.3–13.8, v14 1.4) |
| Share of rain in them | ≥ 15% (real 21.4%) |
| Volume | 0.95–1.05 of real |
| Wet spell | 0.90–1.10 of real |
| Gate A (fixed bands) | ≥ 12 / 18 (v14 14) |
| Memorisation: rainy days (≥ 1 mm) closer to a training day than the closest 5% of 2008–2009 days | ≤ 10% (held-out real 5.1%, v14 5.6%, copies of real days 100%) |
| Sewer overflow, BC and EFD (laptop) | 0.85–1.15 of real (v14 0.66, v17 0.82) |

Reported, not required: Tier 6 (perfect one-day 3/3), P99.9, families A–E under the resampled bands, the
comparators above. Code: package `oneday/` (`python -m oneday evaluate` → `results/oneday/evaluation.json`).

## 5. Predictions (written before any run)

1. Unbroken runs > 5h20: 8–13 per year, mean at least 10.
2. Share of rain in them: 13–20%.
3. Volume 0.92–1.02. The least certain: the two-resolution coarse model made too few rainy periods (0.93).
4. Wet spell 0.90–1.05; Gate A at least 12 / 18; Tier 6 at least 2 / 3; too-close days at most 10%.
5. Sewer overflow (BC) 0.80–1.05 of real, near the perfect one-day line (0.78–1.05).
6. Overall: the statistical rule passes, at modest confidence (about 55%).

## 6. What each outcome means (fixed now)

* **Passes, sewer in range:** a learned one-day window closes the horizon gap; v19 replaces v14 as the deliverable,
  and the thesis's central claim is confirmed by a learned model, not only by the perfect-window pilot.
* **Long storms reach the one-day line but volume or calibration fails:** the horizon claim holds; calibration at
  the longer window is the open problem (as for the two-resolution coarse model). v14 stays the deliverable.
* **Long storms stay near v14's:** a long window alone does not make a model carry storms; reported as such.
* **Memorisation above 10%:** rejected whatever else it shows.

This is the last experiment before writing up. No v20.

## 7. Order and time

| Step | Where | Time |
| :--- | :--- | :--- |
| plan, code, tests, laptop smoke (one CPU batch; a few generated pieces; evaluation on v14 as stand-in) | laptop | done |
| GPU smoke job | cluster | ~10 min |
| epochs and batch size by section 3; config and diary addendum committed | laptop | minutes |
| training, 4 members, evaluation | cluster | measured by the smoke job |
| sewer, BC and EFD | laptop | ~1 h |

## 8. Settings fixed by the smoke job (2026-10-16)

Smoke job 951333: batch 2048 runs at 0.475 s per step with 26.4 GB peak (of 39.5 GB); 500 epochs = 27.0 h. The
`gpu-long` limit is 3 days. By section 3: **batch 2048, 500 epochs** (v14's recipe, no deviation); job time 36 h.

