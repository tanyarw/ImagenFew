# Base Checkpoint Provenance

Status: **T0.3 open, partially answered locally on 2026-09-22.** This file records only
what is verifiable from files present in this repository. It does not answer the question
T0.3 actually asks (what corpus and hyperparameters produced `ImagenFew_{12,24,36,64}.ckpt`)
— that requires the original pretraining run's logs or the checkpoint's own metadata,
inspected on the cluster where torch is available (this repo's `.venv` has no torch).

## What is verifiable locally

- **`configs/pretrain/pretrain.yaml`** sets `train_on_datasets: [stock]` (line 92), with a
  single dataset entry `name: stock, data: stock` (lines 29–30). If this is the config that
  actually produced the shipped checkpoints, they are **not** a multi-domain foundation
  model — they are transferred from a single stock-price time series. This has not been
  cross-checked against the checkpoint's own metadata or training logs, so treat it as a
  lead, not a confirmed fact. Do not describe the checkpoints as "pretrained on a
  heterogeneous multi-domain corpus" (as `docs/RESEARCH_CONTRIBUTIONS.md` §1 describes the
  *base paper's* pretraining) until this is confirmed for *this fork's* checkpoints
  specifically — the base paper's own pretraining and this repo's shipped `.ckpt` files are
  not necessarily the same run.

- **The four checkpoints are identical in every way that matters to fine-tuning at
  different `seq_len`.** All four are byte-identical in file size
  (54,577,102 bytes) and have distinct MD5 hashes (so they are not literally the same
  file, but nothing shape-dependent differs between them):

  | File | MD5 |
  |---|---|
  | `ImagenFew_12.ckpt` | `1f372984b5f33f591d045987ef42645` |
  | `ImagenFew_24.ckpt` | `19bfb8f42ad7c576cb4387a33d4dd3a1` |
  | `ImagenFew_36.ckpt` | `2a5db25bbc5d5b4f473dde96d4d0f197` |
  | `ImagenFew_64.ckpt` | `77560dc20aeef2e7370798195ac46457` |

  Every v8/v9/v10 training log reports the identical load outcome regardless of which
  checkpoint or target `seq_len` was used:
  ```
  Skipping net.model.map_label.weight        (ckpt=[128, 32] vs model=[128, 4])
  Skipping model_ema.modelmap_labelweight    (ckpt=[128, 32] vs model=[128, 4])
  Loaded 1006 parameters, skipped 2
  Loaded EMA weights (498 params)
  ```
  The only parameters that ever differ by target class count are the two class-embedding
  heads. No parameter shape in `models/ImagenFew/networks.py` depends on `seq_len` — the
  delay-embedding grid is fixed at `img_resolution=embedding=delay=8` in every v8/v9/v10
  config, and `seq_len` only changes how many of the grid's columns the data transform
  populates (`models/ImagenFew/img_transformations.py:73-100`). See
  `code_plan/AUDIT_2026-09-22.md` §3.1 for the consequence this has for the v8→v10
  ablation.

- **`dyConv_Basic_24.ckpt`**, referenced by `scripts/run_visualization.sh`,
  `scripts/run_finetune_count.sh` and `scripts/run_finetune_percentage.sh`, does not exist
  anywhere in this repository (`models_ckpt/ImagenFew/` or otherwise). These three scripts
  are **not runnable as written**. If a copy exists on cluster scratch or in an old SLURM
  job directory, retrieve and commit it (or its hash and origin, if it is too large to
  commit); otherwise point these scripts at `ImagenFew_24.ckpt` and note the substitution.

## Still open

- Exact pretraining corpus, hyperparameters, epoch count and EMA setting for the shipped
  `ImagenFew_{12,24,36,64}.ckpt` files — requires cluster access
  (`python -c "import torch; print(torch.load(p, map_location='cpu').keys())"` on each,
  and cross-referencing against `logs/` for the original pretrain job, if it still exists).
- `sha256` of each `.ckpt` for the permanent record (MD5 above is sufficient to prove the
  four files are distinct, but is not the project's standard hash — see
  `data/rainfall/splits/MANIFEST.json` for the sha256 convention used elsewhere).
- Whether `dyConv_Basic_24.ckpt` ever existed in this project's history, or was inherited
  from the upstream ImagenFew repository and never adapted here.
