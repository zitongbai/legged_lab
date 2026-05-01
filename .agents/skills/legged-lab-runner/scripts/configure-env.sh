#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd "${SKILL_DIR}/../../.." && pwd -P)"
ENV_FILE="${REPO_ROOT}/.agents/.local.env"

prompt() {
    local name="$1"
    local default_value="${2:-}"
    local value

    if [[ -n "${default_value}" ]]; then
        read -r -p "${name} [${default_value}]: " value
        printf '%s\n' "${value:-${default_value}}"
    else
        read -r -p "${name}: " value
        printf '%s\n' "${value}"
    fi
}

validate_name() {
    local label="$1"
    local value="$2"

    if [[ -z "${value}" ]]; then
        echo "ERROR: ${label} cannot be empty." >&2
        exit 1
    fi

    if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: ${label} may only contain letters, digits, '.', '_', and '-'." >&2
        exit 1
    fi
}

write_assignment() {
    local key="$1"
    local value="$2"

    printf '%s=%q\n' "${key}" "${value}"
}

if [[ -f "${ENV_FILE}" ]]; then
    read -r -p "${ENV_FILE} already exists. Overwrite? [y/N]: " overwrite
    case "${overwrite}" in
        y | Y | yes | YES)
            ;;
        *)
            echo "Canceled."
            exit 0
            ;;
    esac
fi

echo "Configure Legged Lab runner environment."
DEV_TYPE="$(prompt "DEV_TYPE (conda or docker)")"

case "${DEV_TYPE}" in
    conda)
        CONDA_ENV_NAME="$(prompt "CONDA_ENV_NAME")"
        validate_name "CONDA_ENV_NAME" "${CONDA_ENV_NAME}"
        CONDA_SH="$(prompt "CONDA_SH (optional)")"
        DOCKER_CONTAINER_NAME="legged-lab"
        ;;
    docker)
        CONDA_ENV_NAME=""
        CONDA_SH=""
        DOCKER_CONTAINER_NAME="$(prompt "DOCKER_CONTAINER_NAME" "legged-lab")"
        validate_name "DOCKER_CONTAINER_NAME" "${DOCKER_CONTAINER_NAME}"
        ;;
    *)
        echo "ERROR: DEV_TYPE must be 'conda' or 'docker'." >&2
        exit 1
        ;;
esac

{
    printf '# conda or docker\n'
    write_assignment "DEV_TYPE" "${DEV_TYPE}"
    printf '\n# conda only\n'
    write_assignment "CONDA_ENV_NAME" "${CONDA_ENV_NAME}"
    write_assignment "CONDA_SH" "${CONDA_SH}"
    printf '\n# docker only\n'
    write_assignment "DOCKER_CONTAINER_NAME" "${DOCKER_CONTAINER_NAME}"
} > "${ENV_FILE}"

echo "Wrote ${ENV_FILE}"
