#!/usr/bin/env bash
set -euo pipefail

BINDER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BINDER_DIR}/.." && pwd)"
PIPELINE="${BINDER_DIR}/pipeline.sh"
RUN_BASE="${RUN_BASE:-${REPO_ROOT}/outputs/binder/baselines}"
SEED_BASE_START="${SEED_BASE_START:-20261230}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
PHASE="${PHASE:-all}"

IFS=',' read -r -a GPU_LIST <<< "${CUDA_DEVICES}"
if [[ "${#GPU_LIST[@]}" -ne 8 ]]; then
  echo "The matched experiment requires exactly eight CUDA device entries." >&2
  exit 2
fi
mkdir -p "${RUN_BASE}/launcher_logs"

run_method_phase() {
  local method="$1" phase="$2" gpu="$3" shard="$4" seed_base="$5"
  local mode scale target_a=0 run_summarize=1 run_generation=0 run_inverse_fold=0
  local run_crossfold=0 run_crossfold_execute=0 run_score=0 refresh=0
  case "${method}" in
    boltzgen_target_a_only) mode=target_a_only; scale=0; target_a=1; run_summarize=0 ;;
    fixed_cfg) mode=fixed; scale=0.25 ;;
    dng) mode=dng; scale=18 ;;
    *) echo "Unknown method: ${method}" >&2; return 2 ;;
  esac
  case "${phase}" in
    generation) run_generation=1; run_inverse_fold=1 ;;
    evaluate) run_crossfold=1; run_crossfold_execute=1; run_score=1; refresh=1 ;;
    *) echo "Unknown phase: ${phase}" >&2; return 2 ;;
  esac
  RUN_ROOT="${RUN_BASE}/${method}/${shard}" \
  GEN_GPUS="${gpu}" CROSSFOLD_GPUS="${gpu}" SCALES="${scale}" \
  GUIDANCE_MODE="${mode}" TARGET_A_ONLY="${target_a}" \
  TRIALS=1 TRIAL_START=0 NUM_DESIGNS=8 DIFFUSION_BATCH_SIZE=1 \
  IF_SEQUENCES=1 GEN_STEPS=200 CROSSFOLD_SAMPLES=3 CROSSFOLD_STEPS=200 \
  CROSSFOLD_MAX_JOBS_PER_GPU=2 MATCHED_SEED_BASE="${seed_base}" \
  DESIGN_STEP_SCALE=1.0 DESIGN_NOISE_SCALE=1.0 \
  GUIDANCE_SCHEDULE=cosine GUIDANCE_RAMP_START=0.20 GUIDANCE_RAMP_END=0.65 \
  GUIDANCE_MAX_DELTA_RATIO=1.0 DNG_PRIOR=0.01 DNG_TEMPERATURE=0.2 \
  DNG_OFFSET=0.0 DNG_P_MIN=0.000001 DNG_P_MAX=0.8 \
  DNG_POSTERIOR_EPS=0.000001 DNG_VARIANCE_FLOOR=0.0001 DNG_VARIANCE_SCALE=1.0 \
  RUN_GENERATION="${run_generation}" RUN_SUMMARIZE="${run_summarize}" \
  RUN_INVERSE_FOLD="${run_inverse_fold}" RUN_CROSSFOLD="${run_crossfold}" \
  RUN_CROSSFOLD_EXECUTE="${run_crossfold_execute}" RUN_SCORE="${run_score}" \
  REFRESH_CROSSFOLD_MANIFEST="${refresh}" RESUME=1 \
    "${PIPELINE}"
}

case "${PHASE}" in
  all) phases=(generation evaluate) ;;
  generation|evaluate) phases=("${PHASE}") ;;
  *) echo "PHASE must be all, generation, or evaluate" >&2; exit 2 ;;
esac

pids=() labels=()
for index in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[index]}"
  shard="shard$(printf '%02d' "${index}")"
  seed_base=$((SEED_BASE_START + 1000 * index))
  labels+=("${shard}")
  (
    for phase in "${phases[@]}"; do
      for method in boltzgen_target_a_only fixed_cfg dng; do
        echo "[baseline] ${method} ${phase} ${shard}"
        run_method_phase "${method}" "${phase}" "${gpu}" "${shard}" "${seed_base}"
      done
    done
  ) >"${RUN_BASE}/launcher_logs/${shard}_${PHASE}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    echo "[failure] ${labels[index]}" >&2
    failed=$((failed + 1))
  fi
done
[[ "${failed}" -eq 0 ]]
