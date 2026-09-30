#!/usr/bin/env bash
set -euo pipefail

BINDER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BINDER_DIR}/.." && pwd)"
BOLTZGEN_REPO="${BOLTZGEN_REPO:-${REPO_ROOT}/external/boltzgen}"
BOLTZGEN_PY="${BOLTZGEN_PY:-python}"
UPSTREAM_URL="https://github.com/HannesStark/boltzgen.git"
UPSTREAM_COMMIT="a3149cf18eeb58648d1abbb27539bd73f746cdda"

if [[ ! -d "${BOLTZGEN_REPO}/.git" ]]; then
  mkdir -p "$(dirname "${BOLTZGEN_REPO}")"
  git clone "${UPSTREAM_URL}" "${BOLTZGEN_REPO}"
  git -C "${BOLTZGEN_REPO}" checkout "${UPSTREAM_COMMIT}"
elif [[ "$(git -C "${BOLTZGEN_REPO}" rev-parse HEAD)" != "${UPSTREAM_COMMIT}" ]]; then
  echo "Existing checkout is not at ${UPSTREAM_COMMIT}: ${BOLTZGEN_REPO}" >&2
  echo "Use a fresh BOLTZGEN_REPO or put that checkout at the pinned commit." >&2
  exit 2
fi

cp "${BINDER_DIR}/boltzgen_overlay/src/boltzgen/cli/boltzgen.py" \
  "${BOLTZGEN_REPO}/src/boltzgen/cli/boltzgen.py"
cp "${BINDER_DIR}/boltzgen_overlay/src/boltzgen/model/modules/diffusion.py" \
  "${BOLTZGEN_REPO}/src/boltzgen/model/modules/diffusion.py"
cp "${BINDER_DIR}/boltzgen_overlay/src/boltzgen/task/predict/data_from_yaml.py" \
  "${BOLTZGEN_REPO}/src/boltzgen/task/predict/data_from_yaml.py"
cp "${BINDER_DIR}/boltzgen_overlay/tests/test_binder_negative_guidance.py" \
  "${BOLTZGEN_REPO}/tests/test_binder_negative_guidance.py"

"${BOLTZGEN_PY}" -m pip install -e "${BOLTZGEN_REPO}"
"${BOLTZGEN_PY}" -m pip install "pytest>=8,<9"
"${BOLTZGEN_PY}" -m pytest -q \
  "${BOLTZGEN_REPO}/tests/test_binder_negative_guidance.py"

echo "Prepared patched BoltzGen at ${BOLTZGEN_REPO}"
