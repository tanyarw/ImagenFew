# Common Commands

Updated 2026-09-22. The original single-line note below (kept for reference) predates the
chronological-holdout pipeline and points at the v1-era workflow. It still runs, but it
fine-tunes on a random subset split, not the frozen 2000–2007/2008/2009 split — do not use
it for anything that will be reported as a result.

**Original note:** To perform finetuning: `nohup python run.py --subset_p 0.1 --model_ckpt ./models_ckpt/ImagenFew/ImagenFew_12.ckpt --config configs/finetune/Rainfall.yaml > rainfall_finetune.log 2>&1 &`

---

## Current pipeline (v8 onward — chronological holdout split)

### Fit the seasonal-phase labels (train years only)

```bash
python scripts/fit_seasonal_labels.py --years 2000-2007 --grid 105120
```

### Fine-tune a version

```bash
python regime_training/train_regime.py \
    --model_ckpt models_ckpt/ImagenFew/ImagenFew_24.ckpt \
    --config regime_training/config_v8.yaml
```

Swap the checkpoint/config pair for v9 (`ImagenFew_36.ckpt` / `config_v9.yaml`) or v10
(`ImagenFew_64.ckpt` / `config_v10.yaml`). All three configs point at the same
`train_years_labelled.csv` / `val_years_labelled.csv` split; only `seq_len` and the base
checkpoint differ between them (see `code_plan/AUDIT_2026-09-22.md` §3.1 for why that pairing
is currently confounded).

Cluster wrappers: `scripts/run_v{8,9,10}_training.sh`, submitted via
`scripts/submit_job_v{8,9,10}.sh`.

### Generate a 10-year synthetic realisation

```bash
python regime_training/generate_hmm_v1.py \
    --model_ckpt logs/ImagenFew/Rainfall_v10/<run_id>/best_model.pt \
    --scaler_path logs/ImagenFew/Rainfall_v10/<run_id>/scaler.pkl \
    --years 10 --assembly_mode calendar \
    --transition_matrix_path data/rainfall/splits/seasonal_transition_matrix_train_len64.pkl
```

`--assembly_mode` is `markov` (stochastic, resampled from the empirical seasonal-phase
transition matrix — despite the flag name, this is not a physical Markov weather process;
see `my notes/memo/day_1.md` §4) or `calendar` (deterministic replay of the true seasonal
order). Cluster wrappers: `scripts/run_v{8,9,10}_generation.sh`.

### Evaluate against Gate A

```bash
python scripts/gate_a_scorecard.py --reference train --json results/reference/gate_a_train.json
```

`--reference` is `train` (2000–2007, canonical per `my notes/memo/day_1.md` §3.1), `test`
(locked 2009 year), or `full` (2000–2009, descriptive only — see
`scripts/run_evaluation.py`'s module docstring for why `full` should not be used for
reported results). `scripts/run_evaluation.py` remains available for the same descriptive
metrics without pass/fail bands or Tiers 4–5.

### Interactive sanity check

```bash
jupyter notebook notebooks/synthetic_rainfall_evaluation.ipynb
```
