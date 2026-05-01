#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

set -a
source "${SCRIPT_DIR}/.env.base"
set +a

if [[ -f "${HOME}/.bashrc" ]]; then
    set +u
    source "${HOME}/.bashrc" >/dev/null 2>&1 || true
    set -u
fi

ISAACLAB_IMAGE="${ISAACLAB_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.1}"
LEGGED_LAB_IMAGE="${LEGGED_LAB_IMAGE:-legged-lab:latest}"

PROXY_BUILD_ARGS=()
for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
    proxy_value="${!proxy_var-}"
    if [[ -n "${proxy_value}" ]]; then
        PROXY_BUILD_ARGS+=(--build-arg "${proxy_var}=${proxy_value}")
    fi
done

if [[ ${#PROXY_BUILD_ARGS[@]} -gt 0 ]]; then
    for proxy_var in NO_PROXY no_proxy; do
        proxy_value="${!proxy_var-}"
        if [[ -n "${proxy_value}" ]]; then
            PROXY_BUILD_ARGS+=(--build-arg "${proxy_var}=${proxy_value}")
        fi
    done
    PROXY_BUILD_ARGS+=(--network host)
fi

docker build \
    "${PROXY_BUILD_ARGS[@]}" \
    --build-arg "ISAACLAB_IMAGE=${ISAACLAB_IMAGE}" \
    -t "${LEGGED_LAB_IMAGE}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${REPO_ROOT}"
