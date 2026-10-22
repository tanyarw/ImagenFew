# The test year: v19 scratch (epoch 50) against real 2009

Written 2026-10-22, before the generator's epoch-50 series exist and before any 2009 number in this test is
computed. Code: `testyear/` (README there for the order of work).

## 1. The question

Is real 2009, a year the generator never trained on, a plausible year of the generator's rain at the scales the
sewer responds to, without the generator copying its training days?

One generator is tested: **v19 scratch at epoch 50** (`logs/ImagenFew/Rainfall_Regime/v19_scratch/
epoch_0050_regime_model.pt`), 4 members of 10 calendar years. Epoch 50 was chosen on 2026-10-22 (diary) because
held-out denoising loss is lowest there and rises at every later snapshot. 2009 is used once, for this generator
only. No RL agent is trained or scored here; flood-control's own test-year rule (its D13) is untouched.

## 2. What 2009 has already touched (disclosed, not hidden)

- No model was trained on 2009. Since v8 every generator trains on 2000-2007.
- 2009 has been **seen pooled with 2008**: the real-vs-real noise floor of `scripts/eval_suite.py`, the
  memorisation threshold and comparator row of `oneday/evaluate.py`, and the held-out loss check that chose
  epoch 50. The epoch-50 choice is the same on 2008 alone (EMA held-out loss on 2008: 0.098 at epoch 50, then
  0.104, 0.110, 0.115, 0.120, 0.124, 0.125), so it is recorded as a 2008 decision.
- 2009's overflow under BC (228,140 m3) is printed in flood-control's `docs/ENV_VERIFICATION.md`. No generator
  choice used it.
- From now until the test runs, no ImagenFew evaluation loads 2009: the epoch-50 job evaluates with
  `--heldout 2008`.

## 3. The test

The generator's 40 years (4 members x 10) form a distribution for each per-year quantity. For each quantity, the
real year's **percentile** among the 40 is `(#below + 0.5 #equal) / 40`. It is **inside** when it lies in
[0.05, 0.95]. A generator that produces years like the real ones puts a real year inside 90% of the time.

Per-year quantities (one value per calendar year, every year simulated from empty tanks):

| Kind | Quantity | Source of the definition |
| :--- | :--- | :--- |
| sewer | total overflow under BC (**primary**), under EFD; creek overflow under BC; overflow events >= 1000 m3 under BC | flood-control `bc_efd_check.one` (real), `scripts/gate_b_sewer.one` (generated), the same protocol |
| rain | annual depth; wet share of steps; mean wet spell; P99.9 of wet steps; largest hour; largest calendar day; storms >= 10 mm; storms longer than 5 h 20; share of rain in those; largest storm | `sitetransfer.metrics` (Gate A and Tier 6 definitions); storms are wet runs split by 2 h dry |
| copying | share of the generator's one-day windows nearer a 2000-2007 day than the nearest 5% of real-year days are | `tworesolution.evaluate.Memorisation` (one-day windows, 15-min totals, >= 1 mm), real year as reference |

## 4. The rule (fixed now; `testyear/config.py`)

**PASS** when all three hold:

1. **Sewer:** 2009's total overflow under BC is inside.
2. **Rain:** at most 2 of the 10 rain quantities are outside. If the generator is right and the quantities were
   independent, a real year fails this by chance about 7% of the time.
3. **Copying:** too-close share, mean over the 4 members, at most 10% (a real year gives 5% by construction).

Reported but not part of the verdict: the other three sewer quantities; the same percentiles for the eight
training years 2000-2007 (how often the training years themselves fall inside: in-sample calibration); 2008
(rehearsal).

## 5. Safety: how 2009 stays single-use

1. **Register** (`python -m testyear register`): writes `testyear/preregistration.json` with the series and
   their sha256, the generator's 40 years of every quantity (sewer values copied from
   `results/reference/gate_b_sewer.json`), the training-year calibration and the rule. Nothing about the
   generator is computed after this. Refuses to overwrite.
2. **Commit** the preregistration and `testyear/`. The test refuses to run while either has uncommitted changes,
   so neither the rule nor the code can change after the year is opened.
3. **Rehearse** (`python -m testyear rehearse`): the whole test on 2008, which must also reproduce
   flood-control's saved 2008 BC and EFD rows exactly. The test refuses to run without a rehearsal made from
   the same preregistration.
4. **Test** (`python -m testyear test --spend-test-year`): checks the flag, the commit state, the series
   hashes and the rehearsal; creates `results/testyear/test_2009.claim.json` exclusively (a second or concurrent
   run refuses; a crashed run leaves the claim for a person to inspect); only then loads 2009. The 2009 loader
   refuses without the key the guard hands out.
5. The result `results/testyear/test_2009.json` is final. There is no `--force`.

## 6. What each outcome means

- **PASS:** the generator's year-to-year range covers a year it never saw, at the sewer and in the storm
  quantities, without copying. That is the claim the thesis needs before synthetic rain trains agents.
- **Sewer outside, rain inside:** the rain quantities are not the ones the sewer responds to; the sewer rule
  is the one that counts.
- **Rain outside (3 or more):** the named quantities say where the generator's range is too narrow or shifted.
- **Copying above 10%:** the result is not evidence of generalisation, whatever the other two say.
- One year is one draw: a PASS says 2009 is plausible, not that every year would be. The training-year
  calibration shows how often the generator's range covers the years it trained on.
