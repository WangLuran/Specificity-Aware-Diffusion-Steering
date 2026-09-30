# Specificity-Aware Diffusion Steering via Variance-Reduced Sequential Monte Carlo

This repository reproduce the main results in the NeurlPS2026 Paper: Specificity-Aware Diffusion Steering via Variance-Reduced Sequential Monte Carlo

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
