# Codex Environment Bootstrap Design

**Date:** 2026-03-12

**Goal:** Ensure Codex always runs Python-related commands in this repository through a validated, machine-local bootstrap layer that activates the correct conda environment and exposes the local Isaac Lab installation.

## Problem

This repository depends on Isaac Lab installed in a user-managed conda environment, while the repository itself is installed with `pip install -e`. Today there is no repository-local mechanism that tells Codex how to enter the correct environment before running `python`, `pytest`, `pip`, `pre-commit`, or Isaac Lab scripts. That creates avoidable failure modes:

- Codex may run commands with the wrong Python interpreter.
- Isaac Lab imports may fail even when the user machine is correctly configured.
- Machine-specific paths cannot be safely hardcoded into the repository.
- The agent currently has no deterministic error path when local environment configuration is missing or invalid.

## Requirements

- All Python-related commands in this repository must go through one bootstrap entrypoint.
- Machine-specific settings must live outside tracked repository logic.
- Missing configuration must produce a clear stop condition, not implicit guesses.
- Invalid configuration must report exactly what the user needs to fix.
- The mechanism must be simple enough to enforce via `AGENTS.md`.

## Chosen Approach

Use a two-layer shell bootstrap:

1. A tracked template file at `.codex/env.local.sh.example`
2. An untracked machine-local file at `.codex/env.local.sh`
3. A tracked wrapper entrypoint at `.codex/run-in-env.sh`

Codex must run all Python-related commands through:

```bash
bash .codex/run-in-env.sh <command> [args...]
```

Examples:

```bash
bash .codex/run-in-env.sh python -m pip install -e source/legged_lab
bash .codex/run-in-env.sh pytest source/legged_lab/test
bash .codex/run-in-env.sh pre-commit run --all-files
bash .codex/run-in-env.sh python scripts/rsl_rl/play.py --task ...
```

## Configuration Contract

The local config file `.codex/env.local.sh` should define at least:

```bash
export CODEX_CONDA_ENV="isaaclab"
export ISAACLAB_PATH="/absolute/path/to/IsaacLab"
```

Optional variable if `conda` init location is not discoverable on the machine:

```bash
export CODEX_CONDA_SH="/absolute/path/to/conda.sh"
```

Repository policy:

- `.codex/env.local.sh` is user-specific and should stay untracked.
- `.codex/env.local.sh.example` is tracked and documents the required variables.

## Bootstrap Behavior

`run-in-env.sh` should perform the following steps:

1. Resolve repository root and `.codex` directory.
2. Check whether `.codex/env.local.sh` exists.
3. If missing:
   - Print a clear message pointing to `.codex/env.local.sh.example`
   - Exit non-zero
4. Source `.codex/env.local.sh`
5. Validate required variables:
   - `CODEX_CONDA_ENV` is non-empty
   - `ISAACLAB_PATH` is non-empty
6. Validate filesystem state:
   - `ISAACLAB_PATH` exists and is a directory
7. Initialize conda:
   - Prefer `CODEX_CONDA_SH` when provided
   - Otherwise fall back to `conda info --base` or common initialization logic
8. Activate `CODEX_CONDA_ENV`
9. Export any Isaac Lab related environment glue that this repository requires
10. `exec "$@"`

The script should fail fast with messages that name the broken variable or missing path.

## Why This Approach

### Option A: Only `source .codex/env.sh` in each command

Rejected because it is too easy for Codex to bypass accidentally, and validation logic becomes duplicated or inconsistent.

### Option B: Python launcher script

Rejected because the launcher itself depends on a usable Python interpreter before the correct environment is active.

### Option C: Shell wrapper plus local config

Chosen because it is explicit, robust, easy to document, and does not depend on pre-existing Python correctness.

## AGENTS.md Policy

`AGENTS.md` should state:

- Never run `python`, `pytest`, `pip`, `pre-commit`, or repository Python entrypoints directly.
- Always run them via `.codex/run-in-env.sh`.
- If `.codex/env.local.sh` is missing, stop and instruct the user to create it from `.codex/env.local.sh.example`.
- If conda activation fails or `ISAACLAB_PATH` is invalid, stop and report the exact variable or path that must be fixed.
- Do not guess machine-specific values.

## Verification Strategy

Verification should cover both shell bootstrap behavior and Python import behavior.

### Shell-level checks

- Missing `.codex/env.local.sh` returns a non-zero exit code and points to the example file.
- Empty `CODEX_CONDA_ENV` returns a non-zero exit code with a targeted message.
- Invalid `ISAACLAB_PATH` returns a non-zero exit code with a targeted message.

### Python-level checks

Add a small test module that confirms the bootstrapped environment can:

- import `isaaclab`
- import `legged_lab`
- observe required environment variables when applicable

This test should be run through the wrapper, not directly.

## Non-Goals

- Automatically inventing or repairing machine-local paths
- Replacing Isaac Lab's own installation instructions
- Supporting direct execution without the wrapper

## Open Implementation Detail

The only detail still to choose during implementation is how `run-in-env.sh` should discover `conda.sh` when `CODEX_CONDA_SH` is not set. The preferred implementation order is:

1. `CODEX_CONDA_SH`
2. `$(conda info --base)/etc/profile.d/conda.sh`
3. Clear error asking the user to set `CODEX_CONDA_SH`

That keeps the default ergonomic while preserving deterministic failure when automatic discovery is not possible.
