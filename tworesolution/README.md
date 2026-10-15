# tworesolution: coarse storms, fine texture

Plan, rules and predictions: `code_plan/TWO_RESOLUTION_PLAN.md` (section 10 lists what was settled while coding).

A **coarse** diffusion model makes 30-minute totals 32 hours at a time (v14's recipe and network, trained by
`regime_training/train_regime.py` on 30-minute data). A **fine** model, v14 with the half-hour totals as an extra
input, fills in the 5-minute detail one window per 5 hours. Each half-hour's six values are then scaled to its
total. Fine windows are independent given the coarse series, so nothing is chained.

## Modules

| Module | What it does | Torch |
| :--- | :--- | :---: |
| `grid.py` | resolutions, aggregation and spreading, window layouts, seasonal phase of a window | no |
| `data.py` | 30-minute split files and `derived.json` from the 5-minute splits; both training configs | no |
| `mass.py` | mass matching: each half-hour's six values sum to its total, nothing below 0.005 mm | no |
| `cascade.py` | the classical comparator: microcanonical cascade 30 -> 15 -> 5 min, splits drawn from 2000-2007 | no |
| `evaluate.py` | the three stage verdicts and their rules (`RULES`), memorisation and window-join checks | no |
| `invariants.py` | checks on data and configs; with `--torch`, the fine model starts as v14 exactly | optional |
| `network.py` | the fine model's network: v14's U-Net plus a zero-initialised input path for the totals | yes |
| `sampling.py` | the production sampler (identical to `DiffusionProcess.impute`), coarse and fine generation | yes |
| `train_fine.py` | fine-model training: v14's loss and optimiser, from v14's weights | yes |

Tests: `python -m unittest discover -s tworesolution/tests -t .` (22 tests; the torch ones are skipped where
torch cannot load).

## Order of work

On the laptop (seconds), once, then commit:

```bash
python -m tworesolution build-data      # data/rainfall/splits_30min/*.csv + results/tworesolution/derived.json
python -m tworesolution configs         # tworesolution/configs/{coarse30,fine}.yaml (v14's, listed keys changed)
python -m tworesolution check --torch --device cpu
python -m unittest discover -s tworesolution/tests -t .
```

On the cluster (repository root, after `git pull`):

```bash
tworesolution/slurm/submit.sh train --dry-run     # look first
tworesolution/slurm/submit.sh train               # coarse and fine train in parallel (~2-3 h each);
                                                  # stage 1 runs after the coarse model, stage 2 after the fine one
# read results/tworesolution/stage1.json and stage2.json (their verdicts are printed at the end of the logs)
tworesolution/slurm/submit.sh stage3              # refuses unless stages 1 and 2 allow it
# follow-up (plan section 11): the coarse model for v14's 205,000 steps, then its stage 1 (outputs tagged _long)
tworesolution/slurm/submit.sh retrain
COARSE_RUN=twores_coarse30_long tworesolution/slurm/submit.sh stage3     # only if stage1_long.json passes
```

The cluster needs, outside git: `data/rainfall/splits/{train,val,test}_years_labelled.csv` (already there for
v14), `models_ckpt/ImagenFew/ImagenFew_64.ckpt` and `logs/ImagenFew/Rainfall_Regime/v14/`. Every job first
rebuilds the 30-minute files; the split files must hash as recorded in the committed `derived.json`.

Back on the laptop:

```bash
rsync -avz tanyawar@xlogin1:/home/t/tanyawar/thesis/ImagenFew/results/tworesolution/ results/tworesolution/
rsync -avz 'tanyawar@xlogin1:/home/t/tanyawar/thesis/ImagenFew/results/generated_data/rainfall_synthetic_10y_twores_*' results/generated_data/
rsync -avz 'tanyawar@xlogin1:/home/t/tanyawar/thesis/ImagenFew/logs/ImagenFew/Rainfall_Regime/twores_*' logs/ImagenFew/Rainfall_Regime/
# the sewer check (stage 3), from flood-control, both controllers together:
cd ../flood-control && PYTHONPATH=. .venv/bin/python ../ImagenFew/scripts/gate_b_sewer.py \
    twores_v1 twores_v1_m1 twores_v1_m2 twores_v1_m3 --controllers BC EFD
```

## Outputs

| Path | Contents |
| :--- | :--- |
| `results/tworesolution/derived.json` | thresholds, asinh scale, window counts; hashes of the files they came from |
| `results/tworesolution/stage{1,2,3}.json` | every series' metrics, checks and families; the rules, tests and verdict |
| `results/tworesolution/cascade_fit.json` | the cascade's classes: sizes, medians, share of all-in-one-child splits |
| `results/tworesolution/records/` | one run record per command: commit, inputs and outputs with hashes, results |
| `results/tworesolution/{coarse,disaggregated}/*.npz` | generated arrays (not in git; regenerable) |
| `results/generated_data/rainfall_synthetic_10y_twores_{v1,cascade}[_mK].csv` | the 10-year 5-minute series |

## Smoke test (laptop, about 20 minutes in all)

```bash
python regime_training/train_regime.py --model_ckpt models_ckpt/ImagenFew/ImagenFew_64.ckpt \
    --config tworesolution/configs/coarse30.yaml --epochs 1 --limit_batches 2 --device cpu --run_id smoke_twores_coarse30
python -m tworesolution train-fine --device cpu --epochs 1 --limit-batches 2 --batch-size 256 --run-id smoke_twores_fine
export TWORES_OUT=<scratch folder>          # keeps smoke output out of results/
python -m tworesolution generate-coarse --coarse-run smoke_twores_coarse30 --years 2 --members 2 --device cpu
python -m tworesolution disaggregate --source heldout --method fine --fine-run smoke_twores_fine --members 2 --steps 4 --device cpu
python -m tworesolution disaggregate --source heldout --method cascade --members 2
python -m tworesolution evaluate --stage 2 --draws 4 && python -m tworesolution evaluate --stage 1 --draws 4
```
