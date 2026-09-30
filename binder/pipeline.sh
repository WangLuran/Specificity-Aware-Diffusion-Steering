#!/usr/bin/env bash
set -euo pipefail

BINDER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BINDER_DIR}/.." && pwd)"
ROOT="${BINDER_DIR}"
EXP_DIR="${BINDER_DIR}"
BOLTZGEN_REPO="${BOLTZGEN_REPO:-${REPO_ROOT}/external/boltzgen}"
BOLTZGEN_PY="${BOLTZGEN_PY:-python}"
BOLTZ_PY="${BOLTZ_PY:-python}"
BOLTZ_EXECUTABLE="${BOLTZ_EXECUTABLE:-}"
BOLTZ_CACHE="${BOLTZ_CACHE:-${HOME}/.cache/boltz}"

export PYTHONPATH="${BOLTZGEN_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-prop2-binder}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/binder/default}"
GEN_GPUS_CSV="${GEN_GPUS:-0,1,2,3,4,5,6,7}"
CROSSFOLD_GPUS_CSV="${CROSSFOLD_GPUS:-0,1,2,3,4,5,6,7}"
SCALES_CSV="${SCALES:-0,0.25,0.5,1.0}"
TRIALS="${TRIALS:-5}"
TRIAL_START="${TRIAL_START:-0}"
NUM_DESIGNS="${NUM_DESIGNS:-1}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-1}"
GEN_STEPS="${GEN_STEPS:-200}"
IF_SEQUENCES="${IF_SEQUENCES:-3}"
DESIGN_STEP_SCALE="${DESIGN_STEP_SCALE:-}"
DESIGN_NOISE_SCALE="${DESIGN_NOISE_SCALE:-}"
CROSSFOLD_SAMPLES="${CROSSFOLD_SAMPLES:-5}"
CROSSFOLD_STEPS="${CROSSFOLD_STEPS:-200}"
CROSSFOLD_MAX_JOBS_PER_GPU="${CROSSFOLD_MAX_JOBS_PER_GPU:-1}"
MUTATION_POSITION="${MUTATION_POSITION:-4}"
GUIDANCE_SCHEDULE="${GUIDANCE_SCHEDULE:-cosine}"
GUIDANCE_RAMP_START="${GUIDANCE_RAMP_START:-0.20}"
GUIDANCE_RAMP_END="${GUIDANCE_RAMP_END:-0.65}"
GUIDANCE_MAX_DELTA_RATIO="${GUIDANCE_MAX_DELTA_RATIO:-1.0}"
GUIDANCE_MODE="${GUIDANCE_MODE:-fixed}"
TARGET_A_ONLY="${TARGET_A_ONLY:-0}"
DNG_PRIOR="${DNG_PRIOR:-0.01}"
DNG_TEMPERATURE="${DNG_TEMPERATURE:-0.2}"
DNG_OFFSET="${DNG_OFFSET:-0.0}"
DNG_P_MIN="${DNG_P_MIN:-0.000001}"
DNG_P_MAX="${DNG_P_MAX:-0.8}"
DNG_POSTERIOR_EPS="${DNG_POSTERIOR_EPS:-0.000001}"
DNG_VARIANCE_FLOOR="${DNG_VARIANCE_FLOOR:-0.0001}"
DNG_VARIANCE_SCALE="${DNG_VARIANCE_SCALE:-1.0}"
PA_GATE_C="${PA_GATE_C:-9.0}"
PA_GATE_ETA="${PA_GATE_ETA:-0.005}"
PA_GATE_POWER="${PA_GATE_POWER:-1925.92592593}"
PA_LHAT_TEMPERATURE="${PA_LHAT_TEMPERATURE:-0.75}"
PA_LHAT_CLIP="${PA_LHAT_CLIP:-0.0}"
PA_RHO1="${PA_RHO1:--2.0}"
PA_RHO2_MIN="${PA_RHO2_MIN:-0.0}"
PA_RHO2_MAX="${PA_RHO2_MAX:-1.2}"
PA_RHO2_CANDIDATES="${PA_RHO2_CANDIDATES:-25}"
PA_RHO_ESS_MIN_GAIN="${PA_RHO_ESS_MIN_GAIN:-0.0001}"
PA_RHO_OBJECTIVE="${PA_RHO_OBJECTIVE:-full_ess}"
PA_RHO_VARIANCE_MIN_GAIN="${PA_RHO_VARIANCE_MIN_GAIN:-0.0}"
PA_NO_RHO_LAST_STEPS="${PA_NO_RHO_LAST_STEPS:-2}"
PA_JVP_EPS="${PA_JVP_EPS:-0.01}"
PA_JVP_SHRINK_ALPHA="${PA_JVP_SHRINK_ALPHA:-0.75}"
PA_GATE_LOGW_RANK_CLIP_TOPK="${PA_GATE_LOGW_RANK_CLIP_TOPK:-0}"
PA_GATE_LOGW_CLIP="${PA_GATE_LOGW_CLIP:-0.0}"
PA_JVP_LOGW_RANK_CLIP_TOPK="${PA_JVP_LOGW_RANK_CLIP_TOPK:-0}"
PA_JVP_LOGW_CLIP="${PA_JVP_LOGW_CLIP:-0.0}"
PA_KERNEL_LOGW_RANK_CLIP_TOPK="${PA_KERNEL_LOGW_RANK_CLIP_TOPK:-0}"
PA_KERNEL_LOGW_CLIP="${PA_KERNEL_LOGW_CLIP:-0.0}"
PA_LOGW_RANK_CLIP_TOPK="${PA_LOGW_RANK_CLIP_TOPK:-0}"
PA_LOGW_CLIP="${PA_LOGW_CLIP:-0.0}"
PA_RESAMPLE_CARRY_CORRECTION="${PA_RESAMPLE_CARRY_CORRECTION:-1}"
PA_RESAMPLE_LOGW_RANK_CLIP_TOPK="${PA_RESAMPLE_LOGW_RANK_CLIP_TOPK:-0}"
PA_RESAMPLE_LOGW_CLIP="${PA_RESAMPLE_LOGW_CLIP:-1.5}"
PA_MAX_RESAMPLES="${PA_MAX_RESAMPLES:-10}"
PA_RESAMPLE_ESS="${PA_RESAMPLE_ESS:-0.85}"
PA_RESAMPLE_TEMPER_ESS="${PA_RESAMPLE_TEMPER_ESS:-0.0}"
PA_RESAMPLE_COOLDOWN_STEPS="${PA_RESAMPLE_COOLDOWN_STEPS:-0}"
PA_NO_RESAMPLE_LAST_STEPS="${PA_NO_RESAMPLE_LAST_STEPS:-10}"
PA_MIN_UNIQUE_ROOTS="${PA_MIN_UNIQUE_ROOTS:-11}"
PA_NETWORK_CHUNK_PARTICLES="${PA_NETWORK_CHUNK_PARTICLES:-1}"
MATCHED_SEED_BASE="${MATCHED_SEED_BASE:-20260727}"
USE_KERNELS="${USE_KERNELS:-false}"
RESUME="${RESUME:-1}"
REFRESH_CROSSFOLD_MANIFEST="${REFRESH_CROSSFOLD_MANIFEST:-0}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_SUMMARIZE="${RUN_SUMMARIZE:-1}"
RUN_INVERSE_FOLD="${RUN_INVERSE_FOLD:-1}"
RUN_CROSSFOLD="${RUN_CROSSFOLD:-1}"
RUN_CROSSFOLD_EXECUTE="${RUN_CROSSFOLD_EXECUTE:-1}"
RUN_SCORE="${RUN_SCORE:-1}"

WANTED_SPEC="${WANTED_SPEC:-${EXP_DIR}/specs/01_wanted_nlvpmvatv.yaml}"
UNWANTED_SPEC="${UNWANTED_SPEC:-${EXP_DIR}/specs/02_unwanted_nlvfmvatv_exact_shared_context.yaml}"
WANTED_TARGET="${WANTED_TARGET:-${EXP_DIR}/targets/wanted_nlvpmvatv.pdb}"
UNWANTED_TARGET="${UNWANTED_TARGET:-${EXP_DIR}/targets/unwanted_nlvfmvatv_exact_shared_context.pdb}"
WANTED_STEM="${WANTED_STEM:-$(basename "${WANTED_SPEC}" .yaml)}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${MPLCONFIGDIR}"

IFS=',' read -r -a GEN_GPU_LIST <<< "${GEN_GPUS_CSV}"
IFS=',' read -r -a SCALE_LIST <<< "${SCALES_CSV}"
if [[ "${#GEN_GPU_LIST[@]}" -lt 1 || "${#SCALE_LIST[@]}" -lt 1 ]]; then
  echo "At least one generation GPU and scale are required." >&2
  exit 2
fi

boltzgen() {
  "${BOLTZGEN_PY}" -m boltzgen.cli.boltzgen "$@"
}

run_generation_trial() {
  local gpu="$1"
  local scale="$2"
  local trial="$3"
  local scale_tag="${scale/./p}"
  local trial_tag
  trial_tag="$(printf '%03d' "${trial}")"
  local seed="$((MATCHED_SEED_BASE + trial))"
  local trial_dir="${RUN_ROOT}/generation/lambda_${scale_tag}/trial_${trial_tag}"
  local prefix="${LOG_DIR}/lambda_${scale_tag}_trial_${trial_tag}"
  local diagnostics_dir="${trial_dir}/negative_guidance_diagnostics"
  mkdir -p "${trial_dir}" "${diagnostics_dir}"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export TRITON_CACHE_DIR="${RUN_ROOT}/gpu_cache/triton_gpu_${gpu}"
  export TORCHINDUCTOR_CACHE_DIR="${RUN_ROOT}/gpu_cache/torchinductor_gpu_${gpu}"
  mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

  local -a reuse_args=()
  if [[ "${RESUME}" == "1" ]]; then
    reuse_args=(--reuse)
  fi
  local -a sampler_args=()
  if [[ -n "${DESIGN_STEP_SCALE}" ]]; then
    sampler_args+=(--step_scale "${DESIGN_STEP_SCALE}")
  fi
  if [[ -n "${DESIGN_NOISE_SCALE}" ]]; then
    sampler_args+=(--noise_scale "${DESIGN_NOISE_SCALE}")
  fi
  local -a target_specs=("${WANTED_SPEC}")
  if [[ "${TARGET_A_ONLY}" != "1" ]]; then
    target_specs+=("${UNWANTED_SPEC}")
  fi
  boltzgen configure \
    "${target_specs[@]}" \
    --output "${trial_dir}" \
    --protocol protein-anything \
    --steps design inverse_folding \
    --num_designs "${NUM_DESIGNS}" \
    --diffusion_batch_size "${DIFFUSION_BATCH_SIZE}" \
    --inverse_fold_num_sequences "${IF_SEQUENCES}" \
    --devices 1 \
    --num_workers 1 \
    --use_kernels "${USE_KERNELS}" \
    --cache "${HF_HOME}" \
    "${sampler_args[@]}" \
    --config design \
      "data.batch_size=2" \
      "data.cfg.feature_seed=${seed}" \
      "sampling_steps=${GEN_STEPS}" \
    "${reuse_args[@]}" \
    >"${prefix}_configure.log" 2>&1

  local design_count
  design_count=0
  if [[ -d "${trial_dir}/intermediate_designs" ]]; then
    design_count="$(
      find "${trial_dir}/intermediate_designs" \
        -maxdepth 1 -type f -name "${WANTED_STEM}*.cif" \
        ! -name '*_native.cif' | wc -l
    )"
  fi
  if [[ "${RESUME}" == "1" && "${design_count}" -ge "${NUM_DESIGNS}" ]]; then
    echo "[skip] existing ${design_count} designs" >"${prefix}_design.log"
  else
    if [[ "${TARGET_A_ONLY}" == "1" ]]; then
      "${BOLTZGEN_PY}" -m boltzgen.cli.boltzgen execute "${trial_dir}" \
        --steps design \
        >"${prefix}_design.log" 2>&1
    else
      env \
      BOLTZGEN_BINDER_NEG_GUIDANCE=1 \
      BOLTZGEN_NEG_GUIDANCE_MODE="${GUIDANCE_MODE}" \
      BOLTZGEN_NEG_GUIDANCE_SCALE="${scale}" \
      BOLTZGEN_NEG_GUIDANCE_SCHEDULE="${GUIDANCE_SCHEDULE}" \
      BOLTZGEN_NEG_GUIDANCE_RAMP_START="${GUIDANCE_RAMP_START}" \
      BOLTZGEN_NEG_GUIDANCE_RAMP_END="${GUIDANCE_RAMP_END}" \
      BOLTZGEN_NEG_GUIDANCE_MAX_DELTA_RATIO="${GUIDANCE_MAX_DELTA_RATIO}" \
      BOLTZGEN_NEG_GUIDANCE_SEED="${seed}" \
      BOLTZGEN_NEG_GUIDANCE_DIAGNOSTICS_DIR="${diagnostics_dir}" \
      BOLTZGEN_DNG_PRIOR="${DNG_PRIOR}" \
      BOLTZGEN_DNG_TEMPERATURE="${DNG_TEMPERATURE}" \
      BOLTZGEN_DNG_OFFSET="${DNG_OFFSET}" \
      BOLTZGEN_DNG_P_MIN="${DNG_P_MIN}" \
      BOLTZGEN_DNG_P_MAX="${DNG_P_MAX}" \
      BOLTZGEN_DNG_POSTERIOR_EPS="${DNG_POSTERIOR_EPS}" \
      BOLTZGEN_DNG_VARIANCE_FLOOR="${DNG_VARIANCE_FLOOR}" \
      BOLTZGEN_DNG_VARIANCE_SCALE="${DNG_VARIANCE_SCALE}" \
      BOLTZGEN_PA_GATE_C="${PA_GATE_C}" \
      BOLTZGEN_PA_GATE_ETA="${PA_GATE_ETA}" \
      BOLTZGEN_PA_GATE_POWER="${PA_GATE_POWER}" \
      BOLTZGEN_PA_LHAT_TEMPERATURE="${PA_LHAT_TEMPERATURE}" \
      BOLTZGEN_PA_LHAT_CLIP="${PA_LHAT_CLIP}" \
      BOLTZGEN_PA_RHO1="${PA_RHO1}" \
      BOLTZGEN_PA_RHO2_MIN="${PA_RHO2_MIN}" \
      BOLTZGEN_PA_RHO2_MAX="${PA_RHO2_MAX}" \
      BOLTZGEN_PA_RHO2_CANDIDATES="${PA_RHO2_CANDIDATES}" \
      BOLTZGEN_PA_RHO_ESS_MIN_GAIN="${PA_RHO_ESS_MIN_GAIN}" \
      BOLTZGEN_PA_RHO_OBJECTIVE="${PA_RHO_OBJECTIVE}" \
      BOLTZGEN_PA_RHO_VARIANCE_MIN_GAIN="${PA_RHO_VARIANCE_MIN_GAIN}" \
      BOLTZGEN_PA_NO_RHO_LAST_STEPS="${PA_NO_RHO_LAST_STEPS}" \
      BOLTZGEN_PA_JVP_EPS="${PA_JVP_EPS}" \
      BOLTZGEN_PA_JVP_SHRINK_ALPHA="${PA_JVP_SHRINK_ALPHA}" \
      BOLTZGEN_PA_GATE_LOGW_RANK_CLIP_TOPK="${PA_GATE_LOGW_RANK_CLIP_TOPK}" \
      BOLTZGEN_PA_GATE_LOGW_CLIP="${PA_GATE_LOGW_CLIP}" \
      BOLTZGEN_PA_JVP_LOGW_RANK_CLIP_TOPK="${PA_JVP_LOGW_RANK_CLIP_TOPK}" \
      BOLTZGEN_PA_JVP_LOGW_CLIP="${PA_JVP_LOGW_CLIP}" \
      BOLTZGEN_PA_KERNEL_LOGW_RANK_CLIP_TOPK="${PA_KERNEL_LOGW_RANK_CLIP_TOPK}" \
      BOLTZGEN_PA_KERNEL_LOGW_CLIP="${PA_KERNEL_LOGW_CLIP}" \
      BOLTZGEN_PA_LOGW_RANK_CLIP_TOPK="${PA_LOGW_RANK_CLIP_TOPK}" \
      BOLTZGEN_PA_LOGW_CLIP="${PA_LOGW_CLIP}" \
      BOLTZGEN_PA_RESAMPLE_CARRY_CORRECTION="${PA_RESAMPLE_CARRY_CORRECTION}" \
      BOLTZGEN_PA_RESAMPLE_LOGW_RANK_CLIP_TOPK="${PA_RESAMPLE_LOGW_RANK_CLIP_TOPK}" \
      BOLTZGEN_PA_RESAMPLE_LOGW_CLIP="${PA_RESAMPLE_LOGW_CLIP}" \
      BOLTZGEN_PA_MAX_RESAMPLES="${PA_MAX_RESAMPLES}" \
      BOLTZGEN_PA_RESAMPLE_ESS="${PA_RESAMPLE_ESS}" \
      BOLTZGEN_PA_RESAMPLE_TEMPER_ESS="${PA_RESAMPLE_TEMPER_ESS}" \
      BOLTZGEN_PA_RESAMPLE_COOLDOWN_STEPS="${PA_RESAMPLE_COOLDOWN_STEPS}" \
      BOLTZGEN_PA_NO_RESAMPLE_LAST_STEPS="${PA_NO_RESAMPLE_LAST_STEPS}" \
      BOLTZGEN_PA_MIN_UNIQUE_ROOTS="${PA_MIN_UNIQUE_ROOTS}" \
      BOLTZGEN_PA_NETWORK_CHUNK_PARTICLES="${PA_NETWORK_CHUNK_PARTICLES}" \
        "${BOLTZGEN_PY}" -m boltzgen.cli.boltzgen execute "${trial_dir}" \
          --steps design \
          >"${prefix}_design.log" 2>&1
    fi
  fi

  if [[ "${RUN_INVERSE_FOLD}" == "1" ]]; then
    local if_count
    if_count=0
    if [[ -d "${trial_dir}/intermediate_designs_inverse_folded" ]]; then
      if_count="$(
      find "${trial_dir}/intermediate_designs_inverse_folded" \
          -maxdepth 1 -type f -name "${WANTED_STEM}_[0-9]*.cif" \
          | wc -l
      )"
    fi
    local expected_if_count="$((NUM_DESIGNS * IF_SEQUENCES))"
    if [[ "${RESUME}" == "1" && "${if_count}" -ge "${expected_if_count}" ]]; then
      echo "[skip] existing ${if_count} inverse folds" \
        >"${prefix}_inverse_fold.log"
    else
      boltzgen execute "${trial_dir}" --steps inverse_folding \
        >"${prefix}_inverse_fold.log" 2>&1
    fi
  fi
}

generation_worker() {
  local gpu="$1"
  local worker_index="$2"
  local worker_count="$3"
  local task_index=0
  local trial
  local scale
  local trial_end="$((TRIAL_START + TRIALS))"
  for ((trial = TRIAL_START; trial < trial_end; trial++)); do
    for scale in "${SCALE_LIST[@]}"; do
      if ((task_index % worker_count == worker_index)); then
        echo "[launch] gpu=${gpu} scale=${scale} trial=${trial}"
        run_generation_trial "${gpu}" "${scale}" "${trial}"
        echo "[done] gpu=${gpu} scale=${scale} trial=${trial}"
      fi
      task_index=$((task_index + 1))
    done
  done
}

if [[ "${RUN_GENERATION}" == "1" ]]; then
  worker_pids=()
  for index in "${!GEN_GPU_LIST[@]}"; do
    generation_worker \
      "${GEN_GPU_LIST[index]}" "${index}" "${#GEN_GPU_LIST[@]}" &
    worker_pids+=("$!")
  done
  failed=0
  for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "One or more generation workers failed; inspect ${LOG_DIR}." >&2
    exit 1
  fi
  if [[ "${RUN_SUMMARIZE}" == "1" ]]; then
    "${BOLTZ_PY}" "${ROOT}/scripts/32_summarize_alignment_rotations.py" \
      --run-root "${RUN_ROOT}"
    if [[ "${GUIDANCE_MODE}" == "dng" ]]; then
      "${BOLTZ_PY}" "${ROOT}/scripts/33_summarize_dng_diagnostics.py" \
        --run-root "${RUN_ROOT}"
    elif [[ "${GUIDANCE_MODE}" == "pa_gate" ]]; then
      "${BOLTZ_PY}" "${ROOT}/scripts/34_summarize_pa_gate_diagnostics.py" \
        --run-root "${RUN_ROOT}"
    elif [[ "${GUIDANCE_MODE}" == "pa_gate_smc" ]]; then
      "${BOLTZ_PY}" \
        "${ROOT}/scripts/35_summarize_pa_gate_smc_diagnostics.py" \
        --run-root "${RUN_ROOT}"
    fi
  fi
fi

if [[ "${RUN_CROSSFOLD}" == "1" ]]; then
  prepare_args=(
    "${ROOT}/scripts/25_prepare_pmhc_binder_crossfold.py"
    --run_root "${RUN_ROOT}"
    --wanted_target "${WANTED_TARGET}"
    --unwanted_target "${UNWANTED_TARGET}"
    --diffusion_samples "${CROSSFOLD_SAMPLES}"
    --sampling_steps "${CROSSFOLD_STEPS}"
  )
  if [[ "${RESUME}" != "1" || "${REFRESH_CROSSFOLD_MANIFEST}" == "1" ]]; then
    prepare_args+=(--force)
  fi
  if [[ ! -s "${RUN_ROOT}/binder_manifest.csv" || "${RESUME}" != "1" || "${REFRESH_CROSSFOLD_MANIFEST}" == "1" ]]; then
    "${BOLTZ_PY}" "${prepare_args[@]}"
  fi
  if [[ "${RUN_CROSSFOLD_EXECUTE}" == "1" ]]; then
    if [[ -z "${BOLTZ_EXECUTABLE}" ]]; then
      BOLTZ_EXECUTABLE="$(command -v boltz || true)"
    fi
    if [[ -z "${BOLTZ_EXECUTABLE}" || ! -x "${BOLTZ_EXECUTABLE}" ]]; then
      echo "Set BOLTZ_EXECUTABLE to the executable in the Boltz-2 environment." >&2
      exit 2
    fi
    crossfold_args=(
      "${ROOT}/scripts/04_run_boltz_jobs.py"
      --config "${RUN_ROOT}/crossfold_config.yaml"
      --gpus "${CROSSFOLD_GPUS_CSV}"
      --max_jobs_per_gpu "${CROSSFOLD_MAX_JOBS_PER_GPU}"
      --status_interval 30
      --boltz_executable "${BOLTZ_EXECUTABLE}"
      --boltz_cache "${BOLTZ_CACHE}"
    )
    if [[ "${RESUME}" == "1" ]]; then
      crossfold_args+=(--resume)
    fi
    "${BOLTZ_PY}" "${crossfold_args[@]}"
  fi
fi

if [[ "${RUN_SCORE}" == "1" ]]; then
  "${BOLTZ_PY}" "${ROOT}/scripts/26_score_pmhc_binder_specificity.py" \
    --run_root "${RUN_ROOT}" \
    --mutation_position "${MUTATION_POSITION}" \
    --expected_models "${CROSSFOLD_SAMPLES}"
  "${BOLTZ_PY}" "${ROOT}/scripts/31_summarize_negative_guidance_sweep.py" \
    --run-root "${RUN_ROOT}"
fi

echo "[done] ${RUN_ROOT}"
echo "[result] ${RUN_ROOT}/specificity_summary.csv"
