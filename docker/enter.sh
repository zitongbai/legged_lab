#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

set -a
source "${SCRIPT_DIR}/.env.base"
set +a

CONTAINER_NAME="${CONTAINER_NAME:-legged-lab}"

docker exec -it "${CONTAINER_NAME}" bash
