# Missing Experiments for a Paper

**Written:** 2026-10-06 · **Status:** proposed, none started.
**Context:** the single-site generator work (v1–v16, the AR / copula / storm-and-cell baselines and
the sewer check) supports a paper framed as an evaluation-and-design study: *generators fail in
complementary ways, and only a two-level design reproduces sewer loading* (v16, BC overflow 1.00×
real). It does not yet support "diffusion beats classical generators". These experiments close
the gaps a reviewer would raise. Background: research diary, entries from 2026-10-01 to 2026-10-06.

**Current headline numbers** (one realisation each unless noted; `results/evaluation/eval_cache.pkl`,
`results/reference/gate_b_sewer.json`):

| | Gate A | Tier 6 | Storms > 5h20 / yr (real 14.3) | IDF in band | Sewer overflow, BC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| v14 (diffusion) | 14/18 | 0/3 | 1.9 | 12/15 | 0.62× |
| v16 (storm model + v14 texture), seed 0 | 16/18 | 3/3 | 6.6 | 5/15 | 1.00× (seeds 1–4: 0.86–1.08) |
| Copula AR(64), occurrence-matched | 12/18 | 1/3 | 15.8 | 3/15 | 1.47× |
| Storm-and-cell, relaxed α, wet-fraction objective (10 seeds) | 6.5/18 | 1/3 | 32 | 10/15 | not run |

Time estimates are rough.

---

## Priority 1: decide the paper's message

### 1. Fully classical two-level generator
* **Question:** does v16 work because of the two-level structure, or because of the diffusion
  texture?
* **How:** keep v16's storm renewal model (`regime_training/generate_two_level.py`) and replace
  the v14 texture inside each storm with a classical one: (a) the occurrence-matched copula,
  (b) real storm segments resampled from 2000–2007, (c) storm-and-cell. Score each with Gate A,
  Tier 6 and the sewer check (BC), next to v16.
* **Cost:** about a day of coding; minutes to run locally.
* **What it decides:** if a classical texture matches v16, the message is "structure matters,
  diffusion is optional". If it falls clearly short on extremes, texture or sewer loading, it is
  direct evidence that diffusion adds value. **The most likely examiner question.**

### 2. Multi-seed ensembles for the headline models
* **Question:** are the differences larger than seed-to-seed spread?
* **How:** 10 seeds each for the occurrence-matched copula (seconds each), v16 (5 exist; add 5)
  and v14 (about 10 generation runs on the cluster); storm-and-cell already has 10. Sewer check
  (BC) on every member, about 1 minute per seed locally (`scripts/gate_b_sewer.py`). Report means
  with ranges; paired tests where seeds align.
* **Cost:** under an hour locally, plus cluster time for v14.
* **What it decides:** which differences can be claimed.

### 3. One established baseline tool
* **Question:** are our classical baselines as strong as the field's standard tools?
* **How:** fit pyBL (randomised Bartlett–Lewis) on 2000–2007 and score it with
  `scripts/eval_suite.py`, or validate `scripts/baselines/run_bartlett_lewis.py` against pyBL on
  the same statistics. CoSMoS (R) is the copula-side equivalent.
* **Cost:** about a day, mostly installing and fitting.
* **What it decides:** whether "your baselines are weak" is a valid objection. The copula
  history (first fit 5/18, refit 12/18) shows why a reviewer would ask.

---

## Priority 2: strengthen the evidence

### 4. Two-process copula
* **Question:** can a classical single-level model fix the copula's extremes overshoot?
* **How:** one latent process for occurrence (lag-1 ρ 0.976, as in
  `run_copula_arima_occ.py`), a second, less persistent one for amounts (wet-to-wet latent
  correlation about 0.85), joined by the threshold.
* **Cost:** about a day; seconds to run.
* **What it decides:** if it matches v16 on the sewer check, a classical single-level model
  competes, and the claim that a two-level design is necessary has to be weakened.

### 5. Sewer check with a second controller
* **Question:** do the overflow results hold beyond the baseline controller?
* **How:** rerun the sewer check with `--controllers BC EFD` for every headline model and seed;
  report per-year spread.
* **Cost:** about 1 minute per series; under an hour in total.
* **What it decides:** whether "only v16 matches the sewer" is robust.

### 6. The four individual gauges
* **Question:** do the conclusions hold at gauge level, where 5-minute peaks are about 40% higher
  than in the 4-gauge average (diary, KOSTRA entry, 2026-10-01)?
* **How:** generate four gauge series, jointly or by spreading the average with flood-control's
  `src/rain/split.py`, and evaluate each against its real gauge and KOSTRA
  (`scripts/kostra_compare.py`).
* **Cost:** a few days, depending on the method.
* **What it decides:** whether the gauge-averaging limitation weakens the conclusions.

### 7. Held-out years as the reference
* **Question:** do the rankings hold when 2008–2009 is the reference instead of the training years?
* **How:** rerun the suite with 2008–2009 as the reference (noisier: 2 years).
* **Cost:** under an hour.
* **What it decides:** guards against "you scored on the data you trained on".

---

## Priority 3: purpose and generality

### 8. Agent-in-the-loop test
* **Question:** does training controllers on synthetic rain help?
* **How:** in flood-control, train PPO/DQN on real 2000–2007 rain versus v16 (and one baseline);
  test all on real 2008–2009; compare overflow.
* **Cost:** large: days of compute and setup.
* **What it decides:** turns "statistically right" into "useful", the project's actual goal.

### 9. A second site
* **Question:** is the pattern specific to Astlingen?
* **How:** repeat the core comparison on another long 5-minute record (e.g. the Bochum series used
  in the Bartlett–Lewis literature), if its licence allows.
* **Cost:** weeks, including retraining diffusion.
* **What it decides:** the generality of every claim.

### 10. Diffusion training-run variance
* **Question:** is v14 a typical training run or a lucky one?
* **How:** retrain v14 with 2–3 training seeds on the cluster; generate and score each.
* **Cost:** cluster days.
* **What it decides:** whether diffusion results reflect the method or one run.

---

## Suggested order

| Target | Experiments | Effort |
| :--- | :--- | :--- |
| Defence | 1 | about a day |
| Workshop paper | 1, 2, 5 | about a week, mostly local plus cluster time for v14 seeds |
| Journal paper (e.g. HESS, WRR, J. Hydrology, EMS) | add 3, 4, 7; ideally 6 or 8 | a few months |

**Before writing:** check thesis timing, authorship and the data terms for the Erftverband /
Astlingen record with the supervisor.
