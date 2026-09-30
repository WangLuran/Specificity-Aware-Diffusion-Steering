#!/usr/bin/env bash
set -euo pipefail

T2I_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${T2I_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/t2i_prop2_10_pairs}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-prop2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

IFS=',' read -r -a GPU_LIST <<< "${CUDA_DEVICES}"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "CUDA_DEVICES must contain at least one GPU index." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs"

common_args=(
  --mode fixed_rho1
  --model-id CompVis/stable-diffusion-v1-4
  --prompts-file "${T2I_DIR}/dng_prompts.json"
  --steps 100
  --n-particles 8
  --seed 70000
  --unet-chunk-size 8
  --attention-slicing auto
  --c0 5.944276e26
  --c1 5.944276e26
  --c-schedule constant
  --bs0 0.75
  --bs1 0.75
  --bs-schedule constant
  --gamma-c 1.0
  --gamma-bs 1.0
  --theta-mid-logit-shift -0.05
  --c-path-shape plateau
  --c-path-logit-shift -7.0
  --c-path-down-end-frac 0.25
  --c-path-return-start-frac 0.75
  --gate-power 26.0
  --gate-power-schedule linear
  --gate-power-start-frac 0.0
  --gate-power-end-frac 1.0
  --gate-power-schedule-gamma 1.0
  --lhat-temp 0.75
  --gate-logw-rank-clip-topk 1
  --gate-logw-clip 0.1
  --jvp-logw-rank-clip-topk 1
  --jvp-logw-clip 0.25
  --kernel-logw-rank-clip-topk 1
  --kernel-logw-clip 0.1
  --resample-ess 0.30
  --resample-logw-clip 1.5
  --resample-logw-clip-center median
  --resample-trigger-mode full
  --resample-min-gap 10
  --no-resample-last-steps 0
  --no-rho-last-steps 1
  --early-stop-roots 1
  --optimizer-field-mode frozen
  --optimizer-weight-mode prop2
  --rho2-grid-objective full_ess
  --prop-weight-mode prop2
  --jvp-mode forward-ad
  --jvp-eps 0.01
  --jvp-shrink-alpha 0.75
  --fixed-rho1 -2.0
  --fixed-rho1-schedule constant
  --rho2-min 0.0
  --rho2-max 1.2
  --rho2-points 101
  --log-every 10
  --save-images
  --clip-score
  --clip-model-id openai/clip-vit-large-patch14
  --clip-batch-size 16
  --clip-text-batch-size 16
)

if [[ "${PROP2_LOCAL_FILES_ONLY:-0}" != "1" ]]; then
  common_args+=(--no-clip-local-files-only)
fi

run_condition() {
  local prompt_index="$1" negative_kind="$2" gpu="$3" job_index="$4"
  local condition="p${prompt_index}_${negative_kind}"
  local out_dir="${OUT_ROOT}/${condition}"
  local summary="${out_dir}/summary.json"
  if [[ -s "${summary}" ]] && rg -q '"status": "complete"' "${summary}" \
      && rg -q '"clip_score_n_images"' "${summary}"; then
    echo "[skip] ${condition} is complete"
    return 0
  fi
  mkdir -p "${out_dir}"
  echo "[run] ${condition} on cuda:${gpu}"
  "${PYTHON_BIN}" "${T2I_DIR}/src/run_smc_two_rho_exact_prompt1.py" \
    "${common_args[@]}" \
    --devices "cuda:${gpu}" \
    --clip-score-device "cuda:${gpu}" \
    --dist-port "$((63600 + job_index))" \
    --prompt-index "${prompt_index}" \
    --negative-kind "${negative_kind}" \
    --out-dir "${out_dir}" \
    >"${OUT_ROOT}/logs/${condition}.log" 2>&1
}

prompt_indices=(1 1 2 2 3 3 4 4 5 5)
negative_kinds=(related unrelated related unrelated related unrelated related unrelated related unrelated)
pids=()
labels=()
failed=0

for job_index in "${!prompt_indices[@]}"; do
  gpu="${GPU_LIST[job_index % ${#GPU_LIST[@]}]}"
  run_condition "${prompt_indices[job_index]}" "${negative_kinds[job_index]}" \
    "${gpu}" "${job_index}" &
  pids+=("$!")
  labels+=("p${prompt_indices[job_index]}_${negative_kinds[job_index]}")
  if (( (job_index + 1) % ${#GPU_LIST[@]} == 0 || job_index == ${#prompt_indices[@]} - 1 )); then
    for index in "${!pids[@]}"; do
      if ! wait "${pids[index]}"; then
        echo "[failure] ${labels[index]} (see ${OUT_ROOT}/logs/${labels[index]}.log)" >&2
        failed=$((failed + 1))
      fi
    done
    pids=()
    labels=()
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "${failed} condition(s) failed." >&2
  exit 1
fi

"${PYTHON_BIN}" "${T2I_DIR}/summarize_results.py" --run-root "${OUT_ROOT}"

