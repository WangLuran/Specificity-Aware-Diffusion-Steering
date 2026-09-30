#!/usr/bin/env bash
set -euo pipefail

BINDER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BINDER_DIR}/.." && pwd)"
PIPELINE="${BINDER_DIR}/pipeline.sh"
RUN_BASE="${RUN_BASE:-${REPO_ROOT}/outputs/binder/ours_prop2}"
SEED_BASE_START="${SEED_BASE_START:-20261230}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
PHASE="${PHASE:-all}"

IFS=',' read -r -a GPU_LIST <<< "${CUDA_DEVICES}"
if [[ "${#GPU_LIST[@]}" -ne 8 ]]; then
  echo "The matched experiment requires exactly eight CUDA device entries." >&2
  exit 2
fi
mkdir -p "${RUN_BASE}/launcher_logs"

run_phase() {
  local phase="$1" run_generation=0 run_inverse_fold=0 run_crossfold=0
  local run_crossfold_execute=0 run_score=0 refresh_crossfold=0
  case "${phase}" in
    generation) run_generation=1; run_inverse_fold=1 ;;
    prepare) run_crossfold=1; refresh_crossfold=1 ;;
    evaluate) run_crossfold=1; run_crossfold_execute=1; run_score=1 ;;
    *) echo "Unknown phase: ${phase}" >&2; return 2 ;;
  esac

  local pids=() labels=() failed=0 index gpu shard seed_base
  for index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[index]}"
    shard="shard$(printf '%02d' "${index}")"
    seed_base=$((SEED_BASE_START + 1000 * index))
    labels+=("${shard}")
    (
      RUN_ROOT="${RUN_BASE}/${shard}" \
      GEN_GPUS="${gpu}" CROSSFOLD_GPUS="${gpu}" \
      SCALES=6.5 GUIDANCE_MODE=pa_gate_smc \
      TRIALS=1 NUM_DESIGNS=8 DIFFUSION_BATCH_SIZE=8 IF_SEQUENCES=1 \
      GEN_STEPS=200 CROSSFOLD_SAMPLES=3 CROSSFOLD_STEPS=200 \
      CROSSFOLD_MAX_JOBS_PER_GPU=2 MATCHED_SEED_BASE="${seed_base}" \
      DESIGN_STEP_SCALE=1.0 DESIGN_NOISE_SCALE=1.0 \
      PA_GATE_C=6 PA_GATE_ETA=0.005 PA_GATE_POWER=2022.222222222222 \
      PA_LHAT_TEMPERATURE=0.75 PA_LHAT_CLIP=100 PA_RHO1=-1 \
      PA_RHO2_MIN=0 PA_RHO2_MAX=1.2 PA_RHO2_CANDIDATES=25 \
      PA_RHO_OBJECTIVE=full_ess PA_RHO_ESS_MIN_GAIN=0 \
      PA_NO_RHO_LAST_STEPS=2 PA_JVP_EPS=0.01 PA_JVP_SHRINK_ALPHA=0.75 \
      DNG_VARIANCE_SCALE=0.10 \
      PA_GATE_LOGW_RANK_CLIP_TOPK=1 PA_GATE_LOGW_CLIP=0.1 \
      PA_JVP_LOGW_RANK_CLIP_TOPK=1 PA_JVP_LOGW_CLIP=0.25 \
      PA_KERNEL_LOGW_RANK_CLIP_TOPK=1 PA_KERNEL_LOGW_CLIP=0.1 \
      PA_LOGW_RANK_CLIP_TOPK=0 PA_LOGW_CLIP=0 \
      PA_RESAMPLE_ESS=0.30 PA_RESAMPLE_LOGW_CLIP=1.5 \
      PA_RESAMPLE_CARRY_CORRECTION=1 PA_RESAMPLE_TEMPER_ESS=0 \
      PA_RESAMPLE_COOLDOWN_STEPS=10 PA_MAX_RESAMPLES=10 \
      PA_NO_RESAMPLE_LAST_STEPS=10 PA_MIN_UNIQUE_ROOTS=0 \
      PA_NETWORK_CHUNK_PARTICLES=4 \
      RUN_GENERATION="${run_generation}" RUN_INVERSE_FOLD="${run_inverse_fold}" \
      RUN_CROSSFOLD="${run_crossfold}" RUN_CROSSFOLD_EXECUTE="${run_crossfold_execute}" \
      RUN_SCORE="${run_score}" RUN_SUMMARIZE=1 RESUME=1 \
      REFRESH_CROSSFOLD_MANIFEST="${refresh_crossfold}" \
        "${PIPELINE}"
    ) >"${RUN_BASE}/launcher_logs/${shard}_${phase}.log" 2>&1 &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
      echo "[failure] ${labels[index]} ${phase}" >&2
      failed=$((failed + 1))
    fi
  done
  [[ "${failed}" -eq 0 ]]
}

case "${PHASE}" in
  all) phases=(generation prepare evaluate) ;;
  generation|prepare|evaluate) phases=("${PHASE}") ;;
  *) echo "PHASE must be all, generation, prepare, or evaluate" >&2; exit 2 ;;
esac
for phase in "${phases[@]}"; do
  echo "[ours] ${phase}"
  run_phase "${phase}"
done

