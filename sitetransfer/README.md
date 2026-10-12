# sitetransfer

Applies the v14 rainfall-generator recipe to a second site and judges whether it transfers.
Plan and decision rules: [`code_plan/SINGAPORE_TRANSFER_PLAN.md`](../code_plan/SINGAPORE_TRANSFER_PLAN.md).

## Principles

* **Segregated.** Everything new lives in this package. Existing code is imported, never edited, with one
  exception: an opt-in `mask_col` in `regime_training/regime_dataset.py` (skip windows that touch a data
  outage; off by default, tested to change nothing when off).
* **Same procedure, re-measured inputs, no tuning.** Hand-written choices live in `sites/<site>.yaml`.
  Data-derived values come from pure rule functions (`rules.py`) and are written to
  `results/transfer/<site>/derived.json`, never typed.
* **Reproduce before extending.** On Astlingen, every port reproduces the existing pipeline exactly:
  the rules give the frozen values (0.005 mm, 0.035 mm, the 10 mm storm classification), the seasonal
  phases and labels are identical, all 45 evaluation metrics and the Gate A / Tier 6 tallies match
  `scripts/eval_suite.py`, and the latent Gaussian AR series is identical value for value.
* **Traceable.** Every command writes a run record (`results/transfer/<site>/records/`): command,
  parameters, git commit and dirty state, library versions, host and SLURM job, SHA-256 of inputs and
  outputs, status. Build outputs carry an audit file and a manifest with hashes.

## Layout

| File | Role |
| :--- | :--- |
| `sites/*.yaml` | site descriptions (home: frozen values; transfer: ingestion choices, years, rules settings) |
| `site.py` | loads a site; the single source of data-derived parameters (`Site.params`) |
| `timegrid.py` | the 365-day, 5-minute calendar (29 February dropped; labels = interval start) |
| `rules.py` | wet threshold, asinh scale, storm-depth threshold, reference years, matched epochs |
| `ingest_singapore.py` | raw files -> four-gauge series with a fill flag per step |
| `labels.py` | seasonal phases (port of `scripts/fit_seasonal_labels.py`) and the transition bundle |
| `metrics.py` | hydrological metrics with site parameters (port of `eval_suite.py` / `gate_a_scorecard.py`) |
| `checks.py` | Gate A + Tier 6 checks, metric families, fixed and resampled bands, the recipe verdict |
| `pilot.py` | perfect-window generators (5h20 with/without time of day, one day) |
| `baselines.py` | occurrence-matched latent Gaussian AR for any site (calls the existing script's functions) |
| `training.py` | writes `regime_training/config_<run>.yaml` from v14's config (whitelisted keys only) |
| `pipeline.py` | the commands; `__main__.py` the command line; `invariants.py` checks on built outputs |
| `provenance.py` | run records and hashing |
| `tests/` | unit tests and Astlingen regression tests (`unittest`, no extra dependencies) |
| `slurm/` | cluster jobs: `train_generate.sbatch` (GPU), `evaluate.sbatch` (CPU), `submit.sh` |

## Running

From the repository root, with the project venv:

```bash
python -m unittest discover -s sitetransfer/tests -t .        # 17 tests, ~6 s

# Singapore (raw files in data/singapore/raw/, ~2 min to build)
python -m sitetransfer build-data --site singapore
python -m sitetransfer prepare --site singapore --workers 6    # derive, labels, train-config, baseline,
                                                               # pilot, bands, evaluate, check (~6 min)
python -m sitetransfer prepare --site astlingen --workers 6    # derive (invariant 1), pilot, bands, evaluate
python -m sitetransfer check --site singapore --dataset        # 14 invariants, including the training windows

# after the trained model's series exists
python -m sitetransfer evaluate --site singapore
python -m sitetransfer verdict                                  # results/transfer/verdict.md
```

## Cluster

Only training and generation need a GPU. The data under `data/` is not in git, so sync it once:

```bash
# on the laptop: commit, push, then
rsync -av data/singapore/processed data/singapore/splits <user>@<cluster>:/home/t/tanyawar/thesis/ImagenFew/data/singapore/
# on the cluster, from the repository root
git pull && sitetransfer/slurm/submit.sh --dry-run && sitetransfer/slurm/submit.sh
```

`submit.sh` refuses uncommitted code, runs the tests and invariants, then sends the GPU job (train ~8.5 h
plus ~1 min generation) and a dependent CPU job (pilot, evaluate both sites, verdict).

## Outputs

| Path | Content | In git |
| :--- | :--- | :---: |
| `data/singapore/processed/` | `cluster4.csv` (series + flags), `cluster_gauges.npz`, `audit.json` | no |
| `data/singapore/splits/` | split CSVs in the training format, transition bundle, `MANIFEST.json` | no |
| `results/transfer/<site>/derived.json` | rule-derived parameters (home: invariant 1 result) | yes |
| `results/transfer/<site>/bands.json`, `pilot.json`, `evaluation.json`, `lgar_model_card.json` | results | yes |
| `results/transfer/<site>/pilot/*.npy` | pilot series (regenerable) | no |
| `results/transfer/<site>/records/` | run records | yes |
| `results/transfer/verdict.md`, `verdict.json` | the verdict | yes |
| `results/generated_data/singapore/` | generated series (baseline, trained model) | no |
| `regime_training/config_sg_v14.yaml` | training config, written by `train-config` | yes |

## Not covered

The storm-and-cell baseline is not adapted (its 0.005 mm cut-off and parameter search box are inside
`scripts/baselines/run_bartlett_lewis.py`). Classifier, coverage and forecasting metrics (family F of
the plan) are not computed. Singapore has no sewer model, so no sewer or agent test.
