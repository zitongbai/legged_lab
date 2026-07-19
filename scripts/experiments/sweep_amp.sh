#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sweep_amp.sh — Single-parameter AMP hyper-parameter sweep on one machine.
#
# Sweeps ONE AMP parameter over a list of values. Each value gets its own GPU,
# its own tmux session, and a distinct --run_name so TensorBoard curves are
# easy to compare. All runs share a fixed seed so the only thing that varies
# is the swept parameter.
#
# Machine-specific info (Python env, proxy) is NOT hard-coded here. It is read
# from a local, git-ignored config file so this script stays portable/open:
#   scripts/env.local.sh   (copy from env.local.sh.example)
# Every setting can also be overridden via an environment variable.
#
# Usage:
#   scripts/experiments/sweep_amp.sh <param> <val1> <val2> ... <valN>
#
# Examples:
#   scripts/experiments/sweep_amp.sh disc_update_interval 1 3 5 8
#   scripts/experiments/sweep_amp.sh disc_learning_rate   1e-4 5e-5 1e-5
#   scripts/experiments/sweep_amp.sh grad_penalty_scale   10 20 40
#   scripts/experiments/sweep_amp.sh style_reward_scale   5 3 2
#   scripts/experiments/sweep_amp.sh task_style_lerp      0.3 0.4 0.5
#
# Supported <param> keys (mapped to their Hydra override paths below):
#   disc_update_interval, disc_learning_rate, grad_penalty_scale,
#   style_reward_scale, task_style_lerp
#
# Config (env.local.sh or environment variable):
#   VENV_ACTIVATE   path to the Python env activate script (required)
#   PROXY_URL       proxy for remote assets; empty = no proxy (optional)
#
# Extra environment overrides (optional):
#   SEED=42                 fixed seed shared by all runs
#   MAX_ITER=20000          training iterations per run
#   TASK=LeggedLab-Isaac-AMP-Flat-G1-v0
#   DRY_RUN=1               print commands instead of launching tmux
# ---------------------------------------------------------------------------
set -euo pipefail

# --- resolve paths (script lives in <root>/scripts/experiments) ------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# --- load machine-local config (does not override already-set env vars) ----
LOCAL_ENV="${REPO_ROOT}/scripts/env.local.sh"
if [[ -f "${LOCAL_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV}"
fi

# --- tunables --------------------------------------------------------------
SEED="${SEED:-42}"
MAX_ITER="${MAX_ITER:-20000}"
TASK="${TASK:-LeggedLab-Isaac-AMP-Flat-G1-v0}"
DRY_RUN="${DRY_RUN:-0}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
PROXY_URL="${PROXY_URL:-}"

# --- validate machine config ----------------------------------------------
if [[ -z "${VENV_ACTIVATE}" ]]; then
  echo "ERROR: VENV_ACTIVATE is not set." >&2
  echo "       Copy scripts/env.local.sh.example -> scripts/env.local.sh and fill it in," >&2
  echo "       or pass it inline: VENV_ACTIVATE=/path/bin/activate $0 ..." >&2
  exit 1
fi
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "ERROR: VENV_ACTIVATE points to a missing file: ${VENV_ACTIVATE}" >&2
  exit 1
fi

# --- param name -> Hydra override path -------------------------------------
declare -A OVERRIDE_PATH=(
  [disc_update_interval]="agent.algorithm.amp_cfg.disc_update_interval"
  [disc_learning_rate]="agent.algorithm.amp_cfg.disc_learning_rate"
  [grad_penalty_scale]="agent.algorithm.amp_cfg.grad_penalty_scale"
  [style_reward_scale]="agent.algorithm.amp_cfg.amp_discriminator.style_reward_scale"
  [task_style_lerp]="agent.algorithm.amp_cfg.amp_discriminator.task_style_lerp"
)

# --- args ------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <param> <val1> <val2> ... <valN>" >&2
  echo "Supported params: ${!OVERRIDE_PATH[*]}" >&2
  exit 1
fi

PARAM="$1"; shift
VALUES=("$@")

HYDRA_PATH="${OVERRIDE_PATH[$PARAM]:-}"
if [[ -z "${HYDRA_PATH}" ]]; then
  echo "ERROR: unknown param '${PARAM}'." >&2
  echo "Supported params: ${!OVERRIDE_PATH[*]}" >&2
  exit 1
fi

# --- GPU count / capacity check --------------------------------------------
NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -eq 0 ]]; then
  echo "ERROR: no GPUs detected via nvidia-smi." >&2
  exit 1
fi
if [[ "${#VALUES[@]}" -gt "${NUM_GPUS}" ]]; then
  echo "ERROR: ${#VALUES[@]} values but only ${NUM_GPUS} GPUs." >&2
  echo "       This script runs one value per GPU (no queuing). Reduce values or free GPUs." >&2
  exit 1
fi

# --- build the per-run shell prologue (activate env + optional proxy) ------
# This runs inside each tmux pane before training. Equivalent to sourcing the
# uv env and `proxy_on`.
PROLOGUE="source '${VENV_ACTIVATE}'"
if [[ -n "${PROXY_URL}" ]]; then
  PROLOGUE+=" && export http_proxy='${PROXY_URL}' https_proxy='${PROXY_URL}'"
  PROLOGUE+=" HTTP_PROXY='${PROXY_URL}' HTTPS_PROXY='${PROXY_URL}'"
fi

# --- launch ----------------------------------------------------------------
echo "=================================================================="
echo " AMP sweep:  ${PARAM}  over  [${VALUES[*]}]"
echo " Hydra path: ${HYDRA_PATH}"
echo " task=${TASK}  seed=${SEED}  max_iterations=${MAX_ITER}"
echo " venv=${VENV_ACTIVATE}"
echo " proxy=${PROXY_URL:-<none>}"
echo " GPUs available: ${NUM_GPUS}  |  runs to launch: ${#VALUES[@]}"
echo "=================================================================="

for i in "${!VALUES[@]}"; do
  VAL="${VALUES[$i]}"
  GPU="${i}"
  # run_name: sanitize the value (dots/dashes) for a clean folder name
  SAFE_VAL="$(echo "${VAL}" | tr '.-' 'p_')"
  RUN_NAME="${PARAM}-${SAFE_VAL}-s${SEED}"
  SESSION="amp_${PARAM}_${SAFE_VAL}"

  TRAIN_CMD="python scripts/rsl_rl/train.py \
--task ${TASK} \
--headless \
--device cuda:${GPU} \
--seed ${SEED} \
--max_iterations ${MAX_ITER} \
--run_name ${RUN_NAME} \
--logger tensorboard \
agent.device=cuda:${GPU} \
${HYDRA_PATH}=${VAL}"

  echo
  echo "[GPU ${GPU}] session=${SESSION}  run_name=${RUN_NAME}"
  echo "   ${TRAIN_CMD}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  # full command run inside the tmux pane
  PANE_CMD="cd '${REPO_ROOT}' && ${PROLOGUE} && ${TRAIN_CMD} 2>&1 | tee 'logs/sweep_${SESSION}.log'; echo '=== run exited (code $?) — press enter to close ==='; read"

  # kill a stale session of the same name, then start fresh
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" "${PANE_CMD}"
done

echo
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — nothing launched. Re-run without DRY_RUN to start."
else
  echo "Launched ${#VALUES[@]} tmux session(s). Inspect with:"
  echo "   tmux ls"
  echo "   tmux attach -t amp_${PARAM}_<safeval>"
  echo "Per-run stdout also tee'd to logs/sweep_amp_${PARAM}_*.log"
  echo "TensorBoard:  tensorboard --logdir logs/rsl_rl/g1_amp"
fi
