#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

set -a
source "${SCRIPT_DIR}/.env.base"
set +a

CONTAINER_NAME="${CONTAINER_NAME:-legged-lab}"
LEGGED_LAB_IMAGE="${LEGGED_LAB_IMAGE:-legged-lab:latest}"
NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"

resolve_path() {
    local path="$1"
    local base="$2"

    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${base}/${path}"
    fi
}

RSL_RL_HOST_PATH="$(resolve_path "${RSL_RL_PATH:-../../rsl_rl}" "${SCRIPT_DIR}")"
CACHE_HOST_ROOT="${HOME}/docker/isaac-sim"
GIT_METADATA_VOLUME=()

if [[ -z "${LEGGED_LAB_DATA_DIR:-}" ]]; then
    echo "LEGGED_LAB_DATA_DIR is not set. Please set it to the host datasets directory before starting Docker." >&2
    exit 1
fi

if [[ ! -d "${LEGGED_LAB_DATA_DIR}" ]]; then
    echo "LEGGED_LAB_DATA_DIR does not point to an existing directory: ${LEGGED_LAB_DATA_DIR}" >&2
    exit 1
fi

DATASETS_HOST_PATH="$(cd "${LEGGED_LAB_DATA_DIR}" && pwd -P)"

if [[ ! -d "${RSL_RL_HOST_PATH}" ]]; then
    echo "RSL_RL_PATH does not point to an existing directory: ${RSL_RL_HOST_PATH}" >&2
    exit 1
fi

RSL_RL_HOST_PATH="$(cd "${RSL_RL_HOST_PATH}" && pwd -P)"

if [[ -f "${REPO_ROOT}/.git" ]]; then
    GITDIR_LINE="$(sed -n 's/^gitdir: //p' "${REPO_ROOT}/.git")"

    if [[ "${GITDIR_LINE}" == ../.git/modules/* ]]; then
        GIT_METADATA_HOST_PATH="$(cd "${REPO_ROOT}/.." && pwd -P)/.git"

        if [[ ! -d "${GIT_METADATA_HOST_PATH}" ]]; then
            echo "Git metadata directory does not exist: ${GIT_METADATA_HOST_PATH}" >&2
            exit 1
        fi

        GIT_METADATA_VOLUME=(--volume "${GIT_METADATA_HOST_PATH}:/workspace/.git:rw")
    elif [[ -n "${GITDIR_LINE}" ]]; then
        echo "Unsupported .git gitdir layout for Docker workflow: ${GITDIR_LINE}" >&2
        exit 1
    fi
fi

mkdir -p \
    "${CACHE_HOST_ROOT}/cache/kit" \
    "${CACHE_HOST_ROOT}/cache/ov" \
    "${CACHE_HOST_ROOT}/cache/pip" \
    "${CACHE_HOST_ROOT}/cache/glcache" \
    "${CACHE_HOST_ROOT}/cache/computecache" \
    "${CACHE_HOST_ROOT}/logs" \
    "${CACHE_HOST_ROOT}/data" \
    "${CACHE_HOST_ROOT}/documents"

CACHE_HOST_ROOT="$(cd "${CACHE_HOST_ROOT}" && pwd -P)"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_ARGS=(
    --detach
    --interactive
    --tty
    --name "${CONTAINER_NAME}"
    --entrypoint bash
    --network host
    --gpus all
    --workdir /workspace/legged_lab
    --volume "${REPO_ROOT}:/workspace/legged_lab"
    --volume "${RSL_RL_HOST_PATH}:/workspace/rsl_rl"
    --volume "${DATASETS_HOST_PATH}:/workspace/datasets:rw"
    "${GIT_METADATA_VOLUME[@]}"
    --volume "${CACHE_HOST_ROOT}/cache/kit:/isaac-sim/kit/cache:rw"
    --volume "${CACHE_HOST_ROOT}/cache/ov:/root/.cache/ov:rw"
    --volume "${CACHE_HOST_ROOT}/cache/pip:/root/.cache/pip:rw"
    --volume "${CACHE_HOST_ROOT}/cache/glcache:/root/.cache/nvidia/GLCache:rw"
    --volume "${CACHE_HOST_ROOT}/cache/computecache:/root/.nv/ComputeCache:rw"
    --volume "${CACHE_HOST_ROOT}/logs:/root/.nvidia-omniverse/logs:rw"
    --volume "${CACHE_HOST_ROOT}/data:/root/.local/share/ov/data:rw"
    --volume "${CACHE_HOST_ROOT}/documents:/root/Documents:rw"
    --env OMNI_KIT_ALLOW_ROOT=1
    --env ACCEPT_EULA=Y
    --env PRIVACY_CONSENT=Y
    --env LEGGED_LAB_DATA_DIR=/workspace/datasets
    --env "DISPLAY=${DISPLAY:-}"
    --env "HTTP_PROXY=${HTTP_PROXY:-}"
    --env "HTTPS_PROXY=${HTTPS_PROXY:-}"
    --env "NO_PROXY=${NO_PROXY}"
    --env "http_proxy=${HTTP_PROXY:-}"
    --env "https_proxy=${HTTPS_PROXY:-}"
    --env "no_proxy=${NO_PROXY}"
)

if [[ -f "${HOME}/.Xauthority" ]]; then
    DOCKER_ARGS+=(--volume "${HOME}/.Xauthority:/root/.Xauthority")
fi

docker run "${DOCKER_ARGS[@]}" "${LEGGED_LAB_IMAGE}" -lc '
    set -euo pipefail
    git config --global safe.directory /workspace/legged_lab || true
    mkdir -p /workspace/legged_lab/.vscode
    cp /opt/legged_lab/vscode/settings.json /workspace/legged_lab/.vscode/settings.json
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install -e /workspace/rsl_rl --no-deps
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install -e /workspace/legged_lab/source/legged_lab
    exec bash
'
