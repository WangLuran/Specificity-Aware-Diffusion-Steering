# Negative guidance experiments

This repository contains the two experiment packages used for the reported
text-to-image and specificity-aware pMHC binder results.

- `t2i/` runs only **Ours** on the fixed five prompts with
  related and unrelated negatives (10 conditions). It does not include the
  historical hyperparameter search or T2I baseline launchers.
- `binder/` runs the matched 64-binder experiment for Proposition 2 and the
  three reported baselines: target-A-only BoltzGen, fixed CFG, and DNG.

The committed reference CSVs contain the measured per-condition and per-binder
results, not hard-coded table values. The summary programs recompute the paper
rows from those CSVs. Run the
lightweight reproducibility check with:

```bash
python -m pytest -q
```

## Installation

T2I used Python 3.10 and can use one environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Binder generation and Boltz-2 verification are kept as separate environments
because their upstream dependency stacks differ. See
[`binder/README.md`](binder/README.md) for setup and execution.

## Reproduce the tables

```bash
# Recompute the published T2I row from the committed raw condition results.
python t2i/summarize_results.py --condition-csv t2i/reference/condition_results.csv

# Generate the 10 T2I conditions (one condition per available GPU at a time).
bash t2i/run_10_pairs.sh

# Recompute the binder table from the committed raw per-binder results.
python binder/summarize_table.py --reference
```

The GPU experiments are deterministic at the fixed seeds in their launchers,
subject to the usual numerical differences across CUDA, driver, and GPU
versions. Exact software versions and every experimental constant are recorded
next to the corresponding launcher.
