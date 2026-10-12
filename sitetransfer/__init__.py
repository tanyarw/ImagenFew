"""sitetransfer: apply the ImagenFew rainfall-generator recipe to a second site and judge whether it transfers.

Everything the transfer test needs lives in this package: site descriptions, the rules that
re-measure data-derived parameters, data ingestion, seasonal phases, evaluation, the perfect-window
pilot, the classical baseline adapter, training-config generation, tests and SLURM jobs. Existing
project code is imported, never edited, except for one opt-in switch in
`regime_training/regime_dataset.py` (skip windows that touch a data outage).

Plan: code_plan/SINGAPORE_TRANSFER_PLAN.md. Usage: sitetransfer/README.md.
"""

__version__ = "0.1.0"
