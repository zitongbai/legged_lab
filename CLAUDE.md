# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`legged_lab` is an Isaac Lab v3.0.0 extension for legged locomotion and humanoid motion imitation (DeepMimic / AMP) on the Unitree G1. RL uses upstream `rsl-rl-lib`; the AMP algorithm lives in-project under `source/legged_lab/legged_lab/rsl_rl/amp/`.

## Environment

- **Python interpreter** — do NOT hardcode a path. Read `.vscode/settings.json` → `python.defaultInterpreterPath` and use that interpreter (the Isaac Lab env) for every `python` / `pip` command, not a bare `python`.
- If that key is missing or the file can't be read, **ask the user which environment to use**, then write it back into `.vscode/settings.json` as `python.defaultInterpreterPath` so it's configured for next time.
- **Official source code for reference** — Isaac Lab / upstream packages are not vendored here. When you need to read the official source (to check an API, base class, or upstream behavior), look up its location in `.vscode/settings.json` → `python.analysis.extraPaths` and read from there. If that key is absent or the path doesn't resolve, **ask the user where the source lives**.
- Requires an NVIDIA GPU + Isaac Sim runtime. Git LFS holds USD robots + motion data — run `git lfs install && git lfs pull` after clone.
- **Network proxy** — the proxy is only for pulling code/assets and must be off during training/sim. Don't assume specific proxy on/off commands. If you hit a network problem (e.g. a download fails, or a training/sim job misbehaves in a way that looks proxy-related), stop and ask the user for the proxy on/off commands to use.

## Install

Using the interpreter resolved above (call it `$PY`):

```bash
$PY -m pip install -e source/legged_lab          # editable install of the extension package
```

Package metadata is read from `source/legged_lab/config/extension.toml`. Do not `pip install` the repo root — the package is under `source/legged_lab`.

## Train / play

Entry points: `scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py`.

```bash
# Train (headless)
python scripts/rsl_rl/train.py --task <TASK> --headless --max_iterations 50000

# Non-default GPU: pass BOTH the launcher flag and the agent override
python scripts/rsl_rl/train.py --task <TASK> --headless --device cuda:1 agent.device=cuda:1

# Play — record video (kit renderer)
python scripts/rsl_rl/play.py --task <TASK> --num_envs 64 --video --viz kit \
    --checkpoint logs/rsl_rl/<exp>/<run>/model_xxx.pt

# Play — interactive Viser (no --video / --headless)
python scripts/rsl_rl/play.py --task <TASK> --viz viser --num_envs 16
```

- Task IDs are gym-registered under `tasks/locomotion/*/config/g1/__init__.py`, e.g. `LeggedLab-Isaac-AMP-Flat-G1-v0`, `LeggedLab-Isaac-AMP-Rough-G1-v0` (+ `-Play-v0` variants), `LeggedLab-Isaac-Deepmimic-G1-v0`. List them with `python scripts/list_envs.py`.
- Hydra overrides work (e.g. `agent.algorithm.amp_cfg.*`). Logs go to `logs/rsl_rl/<experiment>/<run>/`.
- Interactive launcher / sweeps: `python -m scripts.launch` (TUI), `scripts/experiments/sweep_amp.sh` (one tmux session per GPU).
- `train.py` prints a `DeprecationWarning` suggesting `./isaaclab.sh train ...`; the direct invocation above is what's used here.

## Lint / format

Line length is **120** everywhere. Config is split across `.pre-commit-config.yaml`, `.flake8`, and `pyproject.toml`. Run all checks with:

```bash
pre-commit run --all-files
```

Tools: black (`--preview`), flake8, isort (black profile, custom `ISAACLABPARTY` section), pyupgrade, codespell, pyright (`typeCheckingMode=off`). Note: `setup.py` declares `python_requires>=3.12` while isort/pyright configs target 3.11 — a known inconsistency, not a bug to "fix" casually.

## Tests

`pytest` (config in `pytest.ini`, marker `isaacsim_ci`). Tests live in `source/legged_lab/test/` and mostly require the Isaac Sim runtime, so they are not plain unit tests.

## Layout

- `source/legged_lab/legged_lab/` — the package: `tasks/locomotion/{amp,deepmimic,animation,velocity}/` (each with `config/g1`, `mdp/`), `rsl_rl/amp/` (in-project AMP: `ppo_amp:PPOAMP`), `envs/`, `managers/`, `terrains/`, `sensors/`, `assets/`, `data/{MotionData,Robots}` (LFS).
- `scripts/` — `rsl_rl/` (train/play), `launch/` (TUI), `experiments/` (sweeps), `tools/retarget/` (GMR retargeting), plus `env.local.sh` (git-ignored machine-local config shared by the launcher and sweeps; copy from `env.local.sh.example`).
- `docker/` scripts are **currently not usable** — they're the old pre-v3 setup and haven't been migrated yet.
