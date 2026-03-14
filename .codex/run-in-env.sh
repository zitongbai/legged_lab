#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/env.local.sh"
ENV_FILE="${CODEX_ENV_FILE:-${DEFAULT_ENV_FILE}}"

error() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 1
}

if [[ $# -eq 0 ]]; then
    error "No command provided. Usage: bash .codex/run-in-env.sh <command> [args...]"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    error "Missing ${ENV_FILE}. Create .codex/env.local.sh from .codex/env.local.sh.example and fill in the machine-local values."
fi

# shellcheck source=/dev/null
source "${ENV_FILE}"

if [[ -z "${CODEX_CONDA_ENV:-}" ]]; then
    error "CODEX_CONDA_ENV is not set in ${ENV_FILE}."
fi

if [[ -z "${ISAACLAB_PATH:-}" ]]; then
    error "ISAACLAB_PATH is not set in ${ENV_FILE}."
fi

if [[ ! -d "${ISAACLAB_PATH}" ]]; then
    error "ISAACLAB_PATH does not exist or is not a directory: ${ISAACLAB_PATH}"
fi

find_conda_sh() {
    if [[ -n "${CODEX_CONDA_SH:-}" ]]; then
        printf '%s\n' "${CODEX_CONDA_SH}"
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${conda_base}" ]]; then
            printf '%s\n' "${conda_base}/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    return 1
}

CONDA_SH="$(find_conda_sh || true)"
if [[ -z "${CONDA_SH}" || ! -f "${CONDA_SH}" ]]; then
    error "Unable to locate conda.sh. Set CODEX_CONDA_SH in ${ENV_FILE}."
fi

# shellcheck source=/dev/null
source "${CONDA_SH}"

if ! conda activate "${CODEX_CONDA_ENV}"; then
    error "Failed to activate conda environment: ${CODEX_CONDA_ENV}"
fi

export ISAACLAB_PATH
export PYTHONPATH="${REPO_ROOT}/source/legged_lab:${ISAACLAB_PATH}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
exec "$@"
