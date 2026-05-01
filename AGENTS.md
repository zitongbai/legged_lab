# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives in `source/legged_lab/legged_lab`. Use `tasks/` for environment and task definitions, `envs/`, `managers/`, and `sensors/` for runtime components, and `data/` plus `assets/` for motion files and robot resources. Test modules are under `source/legged_lab/test`. Developer entrypoints live in `scripts/rsl_rl` for training and playback, and `scripts/tools/retarget` for motion-data conversion. Container support is in `docker/`; generated run artifacts usually land in `logs/`, `outputs/`, and `temp/`.


## Coding Style & Naming Conventions
Follow Python 3.10+ style with 4-space indentation and a 120-character line limit. `black` is the formatter; `isort` uses the Black profile and treats `legged_lab` as first-party. Prefer `snake_case` for modules, functions, and variables, `PascalCase` for classes, and task/config names that match the existing Isaac Lab pattern such as `g1_amp_env_cfg.py`. Keep new scripts under `scripts/` task-focused and narrowly scoped.

## Commit & Pull Request Guidelines
- Run pre-commit before every commit.
- When the user asks for a commit, first analyze the pending changes and determine an appropriate commit scope. If the scope is not specified, prefer atomic commits that group only closely related changes.
- Before committing, show the proposed scope and commit message (in language that the user use) to the user and wait for confirmation.
- Use English for all commit messages when committing.

## Codex Environment Bootstrap
This repository depends on a user-local Isaac Lab installation inside a conda environment.
Never run Python-related commands directly in this repository.

- Always run `python`, `pytest`, `pip`, `pre-commit`, and repository Python entrypoints through `bash .codex/run-in-env.sh ...`
- Before executing commands, `.codex/run-in-env.sh` must load `.codex/env.local.sh`
- If `.codex/env.local.sh` is missing, stop and ask the user to create it from `.codex/env.local.sh.example`
- If `CODEX_CONDA_ENV` is missing, invalid, or cannot be activated, stop and report the exact issue
- If `ISAACLAB_PATH` is missing or invalid, stop and report the exact issue
- Never guess or auto-fill machine-specific values for the user
- Strictly forbid using `git worktree` for this repository
- Do not create, use, or suggest worktree-based workflows here
- Reason: this repository is developed via `pip install -e` inside a conda environment, so changing the checkout path breaks or conflicts with the editable install target

## Configuration & Assets
Pull large assets with `git lfs pull` after cloning. Avoid committing generated files from `logs/`, `outputs/`, or `temp/` unless the change explicitly updates tracked sample data or documentation.

## Language & Communication
- Always write code in English
- Talk to the user with the language they used to ask the question, if possible. If the user explicitly requests another language, switch to that language.
