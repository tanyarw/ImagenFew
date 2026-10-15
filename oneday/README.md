# oneday: v19, the v14 recipe with a one-day window

Plan, rules and predictions: `code_plan/ONE_DAY_WINDOW_PLAN.md`.

| Module | What it does |
| :--- | :--- |
| `config.py` | writes `regime_training/config_v19.yaml` (v14's, window keys changed) and checks it and the checkpoint |
| `smoke.py` | GPU: speed and memory at batch 2048 / 1024, sampling speed, partition time limit |
| `evaluate.py` | the verdict: v17's rule + Gate A + memorisation, member means, comparators |

Training and generation use the existing `regime_training/train_regime.py` and `generate_hmm_v1.py` unchanged.
Tests: `python -m unittest discover -s oneday/tests -t .`

## Order of work

1. Copy the checkpoint to the cluster (it is not in git):
   ```bash
   rsync -avz models_ckpt/ImagenFew/ImagenFew_288_807epochs.ckpt \
       tanyawar@xlogin1:/home/t/tanyawar/thesis/ImagenFew/models_ckpt/ImagenFew/
   ```
2. Cluster: `git pull && oneday/slurm/submit.sh smoke` (~10 min); paste the end of `logs/v19_smoke_<id>.out`.
3. Laptop: `python -m oneday config --epochs N --batch-size B --reason "..."` by the plan's rule (section 3);
   commit with a diary addendum.
4. Cluster: `git pull && TRAIN_TIME=<hh:mm:ss> oneday/slurm/submit.sh train`.
5. Laptop: rsync `results/oneday/`, `results/generated_data/rainfall_synthetic_10y_v19_*`,
   `logs/ImagenFew/Rainfall_Regime/v19`; sewer from flood-control:
   ```bash
   cd ../flood-control && PYTHONPATH=. .venv/bin/python ../ImagenFew/scripts/gate_b_sewer.py \
       v19_nobridge_cal v19_nobridge_cal_m1 v19_nobridge_cal_m2 v19_nobridge_cal_m3 --controllers BC EFD
   ```
