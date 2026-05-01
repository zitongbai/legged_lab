#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd "${SKILL_DIR}/../../.." && pwd -P)"
ENV_FILE="${REPO_ROOT}/.agents/.local.env"
EXAMPLE_ENV_FILE="${REPO_ROOT}/.agents/.local.env.example"

DOCKER_WORKDIR="/workspace/legged_lab"
DOCKER_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash .agents/skills/legged-lab-runner/scripts/run.sh python <args...>
  bash .agents/skills/legged-lab-runner/scripts/run.sh pytest <args...>
  bash .agents/skills/legged-lab-runner/scripts/run.sh pip <args...>
  bash .agents/skills/legged-lab-runner/scripts/run.sh pre-commit <args...>
EOF
}

error() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    error "Missing ${ENV_FILE}. Create it from ${EXAMPLE_ENV_FILE}, or run ${SCRIPT_DIR}/configure-env.sh."
fi

# shellcheck source=/dev/null
source "${ENV_FILE}"

COMMAND="$1"
shift

case "${COMMAND}" in
    python | pytest | pip | pre-commit)
        ;;
    *)
        usage
        error "Unsupported command '${COMMAND}'."
        ;;
esac

if [[ -z "${DEV_TYPE:-}" ]]; then
    error "DEV_TYPE is not set in ${ENV_FILE}. Expected 'conda' or 'docker'."
fi

find_conda_sh() {
    if [[ -n "${CONDA_SH:-}" ]]; then
        printf '%s\n' "${CONDA_SH}"
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

run_with_conda() {
    if [[ -z "${CONDA_ENV_NAME:-}" ]]; then
        error "CONDA_ENV_NAME is not set in ${ENV_FILE}."
    fi

    local conda_sh
    conda_sh="$(find_conda_sh || true)"
    if [[ -z "${conda_sh}" || ! -f "${conda_sh}" ]]; then
        error "Unable to locate conda.sh. Set CONDA_SH in ${ENV_FILE}."
    fi

    # shellcheck source=/dev/null
    source "${conda_sh}"

    if ! conda activate "${CONDA_ENV_NAME}"; then
        error "Failed to activate conda environment: ${CONDA_ENV_NAME}"
    fi

    cd "${REPO_ROOT}"
    case "${COMMAND}" in
        python)
            exec python "$@"
            ;;
        pytest)
            exec python -m pytest "$@"
            ;;
        pip)
            exec python -m pip "$@"
            ;;
        pre-commit)
            exec pre-commit "$@"
            ;;
    esac
}

run_with_docker() {
    if [[ -z "${DOCKER_CONTAINER_NAME:-}" ]]; then
        error "DOCKER_CONTAINER_NAME is not set in ${ENV_FILE}."
    fi

    if ! docker container inspect "${DOCKER_CONTAINER_NAME}" >/dev/null 2>&1; then
        error "Docker container '${DOCKER_CONTAINER_NAME}' does not exist. Start it with docker/run.sh or update DOCKER_CONTAINER_NAME."
    fi

    local container_state
    container_state="$(docker inspect -f '{{.State.Running}}' "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true)"
    if [[ "${container_state}" != "true" ]]; then
        error "Docker container '${DOCKER_CONTAINER_NAME}' is not running. Start it with docker/run.sh."
    fi

    case "${COMMAND}" in
        python)
            exec docker exec --workdir "${DOCKER_WORKDIR}" "${DOCKER_CONTAINER_NAME}" "${DOCKER_PYTHON}" "$@"
            ;;
        pytest)
            exec docker exec --workdir "${DOCKER_WORKDIR}" "${DOCKER_CONTAINER_NAME}" "${DOCKER_PYTHON}" -m pytest "$@"
            ;;
        pip)
            exec docker exec --workdir "${DOCKER_WORKDIR}" "${DOCKER_CONTAINER_NAME}" "${DOCKER_PYTHON}" -m pip "$@"
            ;;
        pre-commit)
            exec docker exec --workdir "${DOCKER_WORKDIR}" "${DOCKER_CONTAINER_NAME}" "${DOCKER_PYTHON}" -m pre_commit "$@"
            ;;
    esac
}

case "${DEV_TYPE}" in
    conda)
        run_with_conda "$@"
        ;;
    docker)
        run_with_docker "$@"
        ;;
    *)
        error "Invalid DEV_TYPE='${DEV_TYPE}' in ${ENV_FILE}. Expected 'conda' or 'docker'."
        ;;
esac
