# Specificity-aware pMHC binder design

This package reproduces the matched binder comparison. Every method uses the
same eight seed blocks (`20261230 + 1000 * shard`), with eight backbones per
shard, one ProteinMPNN inverse fold per backbone, and three Boltz-2 predictions
for each wanted and unwanted target. Thus each table row has `N=64`, while
`K=3` reduces verifier noise for each binder/target pair.

The reported statistic for each binder is

```text
mean wanted binder--peptide iPTM - mean unwanted binder--peptide iPTM.
```

## Environments

BoltzGen generation used Python 3.12 and upstream commit
`a3149cf18eeb58648d1abbb27539bd73f746cdda`. Create a generation environment,
then apply and test the committed negative-guidance overlay:

```bash
python3.12 -m venv .venv-boltzgen
source .venv-boltzgen/bin/activate
BOLTZGEN_PY=$PWD/.venv-boltzgen/bin/python bash binder/setup_boltzgen.sh
```

Boltz-2 verification used a separate Python 3.10 environment:

```bash
python3.10 -m venv .venv-boltz2
source .venv-boltz2/bin/activate
pip install -r binder/requirements-boltz2.txt
```

The launchers accept separate executables through `BOLTZGEN_PY`, `BOLTZ_PY`,
and `BOLTZ_EXECUTABLE`. Set them before running, for example:

```bash
export BOLTZGEN_PY=$PWD/.venv-boltzgen/bin/python
export BOLTZ_PY=$PWD/.venv-boltz2/bin/python
export BOLTZ_EXECUTABLE=$PWD/.venv-boltz2/bin/boltz
export CUDA_DEVICES=0,1,2,3,4,5,6,7
```

Model downloads are stored under `HF_HOME` (default
`~/.cache/huggingface`) and `BOLTZ_CACHE` (default `~/.cache/boltz`).

## Run

```bash
# Proposition-2 method only
bash binder/run_ours.sh

# All three matched baselines
bash binder/run_baselines.sh

# Both groups and the final table
bash binder/run_all.sh
```

Each launcher supports resumption. `run_ours.sh` also accepts
`PHASE=generation`, `prepare`, or `evaluate`; the baseline launcher accepts
`PHASE=generation` or `evaluate`. Run the phases in that order when submitting
them separately.

The exact Proposition-2 generation setting is `c=6`, `rho1=-1`, 25 full-ESS
`rho2` candidates on `[0,1.2]`, Proposition-2 JVP shrinkage `0.75`, source-wise
clips `(0.1, 0.25, 0.1)`, and an ancestor resampling cooldown of 10 steps. The
baselines are target-A-only BoltzGen, fixed CFG `0.25`, and DNG gain `18` with
prior `0.01`, temperature `0.2`, and maximum posterior `0.8`.

To recompute the paper table without GPUs:

```bash
python binder/summarize_table.py --reference
```

The committed overlay contains only the files modified relative to the pinned
BoltzGen commit. Its upstream MIT license is included as
`boltzgen_overlay/LICENSE.boltzgen`.

