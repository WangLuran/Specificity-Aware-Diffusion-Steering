#!/usr/bin/env bash
set -euo pipefail

BINDER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BINDER_DIR}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/binder}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_BASE="${OUT_ROOT}/ours_prop2" PHASE=all "${BINDER_DIR}/run_ours.sh"
RUN_BASE="${OUT_ROOT}/baselines" PHASE=all "${BINDER_DIR}/run_baselines.sh"
"${PYTHON_BIN}" "${BINDER_DIR}/summarize_table.py" \
  --ours-root "${OUT_ROOT}/ours_prop2" \
  --target-a-root "${OUT_ROOT}/baselines/boltzgen_target_a_only" \
  --cfg-root "${OUT_ROOT}/baselines/fixed_cfg" \
  --dng-root "${OUT_ROOT}/baselines/dng" \
  --output-dir "${OUT_ROOT}"
