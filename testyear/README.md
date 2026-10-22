# testyear: v19 scratch (epoch 50) against real 2009, once

Plan, rule and what each outcome means: `code_plan/TEST_YEAR_PLAN.md`.

| Module | What it does |
| :--- | :--- |
| `config.py` | the series, the years, the rule; standard library only |
| `guard.py` | keeps 2009 single-use: flag, committed preregistration and code, unchanged series, rehearsal, exclusive claim; hands out the only key the 2009 loaders accept |
| `score.py` | per-year quantities, percentiles, the verdict (pure functions) |
| `run.py` | register, rehearse (2008), test (2009): one function for both years |
| `sewer_real.py` | a real year under BC and EFD with flood-control's own `bc_efd_check.one`; started by `run.py` in flood-control's environment |

Tests: `python -m unittest discover -s testyear/tests -t .`

## Order of work

1. Cluster: generate the four members from epoch 50 under their own name, evaluated against 2008 only:
   ```bash
   RUN_ID=v19_scratch SERIES_ID=v19_scratch_ep50 CKPT_NAME=epoch_0050_regime_model.pt HELDOUT=2008 GENERATE_ONLY=1 \
       TRAIN_TIME=02:00:00 oneday/slurm/submit.sh train
   ```
2. Laptop: rsync `results/generated_data/rainfall_synthetic_10y_v19_scratch_ep50_*` and
   `results/oneday/evaluation_v19_scratch_ep50.json`; then the sewer on all members:
   ```bash
   cd ../flood-control && PYTHONPATH=. .venv/bin/python ../ImagenFew/scripts/gate_b_sewer.py \
       v19_scratch_ep50_nobridge_cal v19_scratch_ep50_nobridge_cal_m1 v19_scratch_ep50_nobridge_cal_m2 \
       v19_scratch_ep50_nobridge_cal_m3 --controllers BC EFD
   ```
3. `python -m testyear register`, then commit `testyear/preregistration.json` (with any rule change made before
   this point; none after).
4. `python -m testyear rehearse`: the whole test on 2008 (about a minute). It must print
   `ok check: sewer_reproduces_flood_control`. Its verdict does not matter: 2008 has the largest BC overflow
   of 2000-2008 (413,768 m3).
5. `python -m testyear test --spend-test-year`: once. Then the diary entry, from `results/testyear/test_2009.json`.

`python -m testyear status` shows which of these exist and whether everything is committed. Until step 5 is done,
nothing in this repository should load 2009: evaluate with `--heldout 2008`, and do not run `scripts/eval_suite.py`.
