ARG ISAACLAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.1
FROM ${ISAACLAB_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy

SHELL ["/bin/bash", "-lc"]

RUN set -euo pipefail; \
    codex_installer=/tmp/codex-install.sh; \
    codex_install_url=https://github.com/openai/codex/releases/latest/download/install.sh; \
    if command -v curl >/dev/null 2>&1; then \
        curl -fsSL "${codex_install_url}" -o "${codex_installer}"; \
    elif command -v wget >/dev/null 2>&1; then \
        wget -qO "${codex_installer}" "${codex_install_url}"; \
    else \
        echo "Either curl or wget is required to install Codex." >&2; \
        exit 1; \
    fi; \
    bash "${codex_installer}"; \
    rm -f "${codex_installer}"

RUN mkdir -p /opt/legged_lab/vscode && \
    printf '%s\n' \
        '{' \
        '    "python-envs.defaultEnvManager": "ms-python.python:system",' \
        '    "python.defaultInterpreterPath": "/workspace/isaaclab/_isaac_sim/kit/python/bin/python3",' \
        '    "python.pythonPath": "/workspace/isaaclab/_isaac_sim/kit/python/bin/python3",' \
        '    "python.analysis.extraPaths": [' \
        '        "/workspace/legged_lab/source/legged_lab",' \
        '        "/workspace/isaaclab/source/isaaclab",' \
        '        "/workspace/isaaclab/source/isaaclab_assets",' \
        '        "/workspace/isaaclab/source/isaaclab_mimic",' \
        '        "/workspace/isaaclab/source/isaaclab_rl",' \
        '        "/workspace/isaaclab/source/isaaclab_tasks"' \
        '    ]' \
        '}' \
        > /opt/legged_lab/vscode/settings.json

WORKDIR /workspace/legged_lab
