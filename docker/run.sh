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

if [[ ! -d "${RSL_RL_HOST_PATH}" ]]; then
    echo "RSL_RL_PATH does not point to an existing directory: ${RSL_RL_HOST_PATH}" >&2
    exit 1
fi

RSL_RL_HOST_PATH="$(cd "${RSL_RL_HOST_PATH}" && pwd -P)"

DATASETS_HOST_PATH="$(resolve_path "${DATASETS_PATH:-../../datasets}" "${SCRIPT_DIR}")"

if [[ ! -d "${DATASETS_HOST_PATH}" ]]; then
    echo "WARNING: DATASETS_PATH does not point to an existing directory: ${DATASETS_HOST_PATH} — skipping datasets mount." >&2
else
    DATASETS_HOST_PATH="$(cd "${DATASETS_HOST_PATH}" && pwd -P)"
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

if [[ -d "${DATASETS_HOST_PATH}" ]]; then
    DOCKER_ARGS+=(--volume "${DATASETS_HOST_PATH}:/workspace/datasets:rw")
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
