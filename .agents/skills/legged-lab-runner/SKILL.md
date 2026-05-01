---
name: legged-lab-runner
description: Use when Codex needs to run Python-related commands in this legged_lab repository, including python entrypoints, pytest, pip, pre-commit, training, playing, or repository scripts that require the configured conda or Docker development environment. Do not use for git or generic shell commands.
---

# Legged Lab Runner

Use this skill for Python-related command execution in this repository. Route commands through the bundled runner so the same request works with either the conda or Docker development setup.

## Required workflow

1. Before running a Python-related command, check that `.agents/.local.env` exists and is valid.
2. If it is missing or invalid, ask the user for the missing local values or run `.agents/skills/legged-lab-runner/scripts/configure-env.sh` to create it interactively.
3. Do not run `python`, `pytest`, `pip`, `pre-commit`, or repository Python entrypoints directly.
4. Run commands through:

```bash
bash .agents/skills/legged-lab-runner/scripts/run.sh <command> [args...]
```

## Command mapping

Use these forms:

```bash
bash .agents/skills/legged-lab-runner/scripts/run.sh python <args...>
bash .agents/skills/legged-lab-runner/scripts/run.sh pytest <args...>
bash .agents/skills/legged-lab-runner/scripts/run.sh pip <args...>
bash .agents/skills/legged-lab-runner/scripts/run.sh pre-commit <args...>
```

For repository entrypoints, use the `python` command:

```bash
bash .agents/skills/legged-lab-runner/scripts/run.sh python scripts/rsl_rl/train.py ...
bash .agents/skills/legged-lab-runner/scripts/run.sh python scripts/rsl_rl/play.py ...
```

## Environment contract

The local config file is `.agents/.local.env`. It is machine-local and must not be committed.

Supported values:

- `DEV_TYPE=conda`: use `CONDA_ENV_NAME` and optional `CONDA_SH`.
- `DEV_TYPE=docker`: use `DOCKER_CONTAINER_NAME`.

Do not add or require `ISAACLAB_PATH`, `PYTHONPATH`, `DOCKER_WORKDIR`, or `DOCKER_PYTHON` in `.agents/.local.env`. Docker internal paths are fixed by this repository's container contract.

### Docker permissions

For `DEV_TYPE=docker`, the runner needs permission to access the host Docker daemon, usually through `/var/run/docker.sock`. In sandboxed or restricted environments, Docker-related checks or commands may fail because of Docker access permissions, even when the container itself is correctly configured.
You may need to ask the user to grant Docker permissions.

## Boundaries

- Do not use this skill for `git` commands.
- Do not use this skill for generic shell commands that do not need the repository Python environment.
- Do not use the legacy `.codex/run-in-env.sh` bootstrap for new commands in this repository.
