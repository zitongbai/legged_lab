#!/usr/bin/env bash
set -euo pipefail

codex_installer=/tmp/codex-install.sh
codex_install_url=https://github.com/openai/codex/releases/latest/download/install.sh

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "${codex_install_url}" -o "${codex_installer}"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "${codex_installer}" "${codex_install_url}"
else
  echo "Either curl or wget is required to install Codex." >&2
  exit 1
fi

bash "${codex_installer}"
