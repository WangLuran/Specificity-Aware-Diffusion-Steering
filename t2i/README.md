# Text-to-image experiment

`run_10_pairs.sh` reproduces the single reported Proposition-2 setting on five
positive prompts, each paired with its related and unrelated negative. The
fixed setting is the selected `central_clip15` run:

- Stable Diffusion v1.4, 100 reverse steps, 8 particles, seed 70000;
- fixed `rho1 = -2` and 101 candidates for `rho2` on `[0, 1.2]`;
- Proposition-2 weights for both candidate selection and propagation;
- full cumulative ESS as the `rho2` objective;
- source-wise rank/value clipping of gate (`0.1`), JVP (`0.25`), and local
  kernel (`0.1`) terms, one extreme particle per source;
- full-ESS resampling trigger `0.30`, median-centered ancestor clip `1.5`,
  minimum resampling gap 10, and no terminal resampling exclusion.

The launcher never performs a hyperparameter scan. With `CUDA_DEVICES=0,1,...`
it schedules at most one condition on each listed GPU and processes additional
conditions in a second wave.

```bash
export CUDA_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=$HOME/.cache/huggingface
bash t2i/run_10_pairs.sh
```

Set `PROP2_LOCAL_FILES_ONLY=1` for a pre-populated offline Hugging Face cache.
Outputs default to `outputs/t2i_prop2_10_pairs`. To summarize an existing run:

```bash
python t2i/summarize_results.py --run-root outputs/t2i_prop2_10_pairs
```

Expected four-decimal row:

```text
Ours (Proposition 2)  0.2722  0.1413  0.1309  0.2679  0.0655  0.2024  0.1666
```

`src/` contains the research runner and its runtime modules. The extra modules
are imports needed by the Proposition-2 entry point; no T2I baseline launcher
or search script is included.

