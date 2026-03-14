# Codex Environment Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repository-local bootstrap layer so Codex always activates the correct conda environment and Isaac Lab path before running Python-related commands.

**Architecture:** Store machine-specific settings in `.codex/env.local.sh`, validate and activate them through `.codex/run-in-env.sh`, and enforce usage through `AGENTS.md`. Verify behavior with shell-level checks and a focused Python environment test.

**Tech Stack:** Bash, conda, pytest, repository documentation in `AGENTS.md`

---

### Task 1: Add the local environment template

**Files:**
- Create: `.codex/env.local.sh.example`

**Step 1: Write the template file**

Add commented variables for:

```bash
export CODEX_CONDA_ENV="isaaclab"
export ISAACLAB_PATH="/absolute/path/to/IsaacLab"
# export CODEX_CONDA_SH="/absolute/path/to/miniconda3/etc/profile.d/conda.sh"
```

Include short comments explaining that:

- users should copy it to `.codex/env.local.sh`
- `.codex/env.local.sh` is machine-local and should not be committed

**Step 2: Verify the file content**

Run: `sed -n '1,200p' .codex/env.local.sh.example`
Expected: The file documents required variables and local setup instructions.

**Step 3: Commit**

```bash
git add .codex/env.local.sh.example
git commit -m "docs: add codex environment template"
```

### Task 2: Implement the bootstrap wrapper

**Files:**
- Create: `.codex/run-in-env.sh`

**Step 1: Write the failing shell checks manually**

Define the target behaviors before implementation:

- missing `.codex/env.local.sh` exits non-zero
- empty `CODEX_CONDA_ENV` exits non-zero
- invalid `ISAACLAB_PATH` exits non-zero

**Step 2: Write minimal wrapper implementation**

Implement:

- repository root resolution
- local config loading
- variable validation
- `conda.sh` discovery
- `conda activate "$CODEX_CONDA_ENV"`
- `exec "$@"`

**Step 3: Make the script executable**

Run: `chmod +x .codex/run-in-env.sh`
Expected: The wrapper is executable.

**Step 4: Run manual shell verification**

Run:

```bash
bash .codex/run-in-env.sh python -V
```

Expected: Either the configured Python version prints, or the script stops with a targeted configuration error.

**Step 5: Commit**

```bash
git add .codex/run-in-env.sh
git commit -m "feat: add codex environment bootstrap wrapper"
```

### Task 3: Enforce the policy in AGENTS.md

**Files:**
- Modify: `AGENTS.md`

**Step 1: Add a Codex environment bootstrap section**

Document:

- all Python-related commands must go through `.codex/run-in-env.sh`
- do not run `python`, `pytest`, `pip`, `pre-commit`, or repository Python scripts directly
- missing or invalid `.codex/env.local.sh` is a stop condition
- do not guess machine-specific values

**Step 2: Verify the rendered text**

Run: `sed -n '1,260p' AGENTS.md`
Expected: The new section is clear, concise, and consistent with existing repository guidelines.

**Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: enforce codex environment bootstrap workflow"
```

### Task 4: Add Python-level environment verification

**Files:**
- Create: `source/legged_lab/test/test_codex_env.py`

**Step 1: Write the failing test**

```python
def test_codex_environment_can_import_required_packages():
    import importlib

    assert importlib.import_module("isaaclab") is not None
    assert importlib.import_module("legged_lab") is not None
```

**Step 2: Run the targeted test to verify failure mode**

Run:

```bash
bash .codex/run-in-env.sh pytest source/legged_lab/test/test_codex_env.py -v
```

Expected: The test fails until the environment bootstrap is correctly configured or the file is implemented.

**Step 3: Write the minimal test implementation**

Add any small assertions needed for:

- import success
- expected interpreter or environment markers if stable

Do not assert machine-specific absolute paths unless they are intentionally surfaced as environment variables.

**Step 4: Run the targeted test again**

Run:

```bash
bash .codex/run-in-env.sh pytest source/legged_lab/test/test_codex_env.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add source/legged_lab/test/test_codex_env.py
git commit -m "test: verify codex environment bootstrap"
```

### Task 5: Verify the end-to-end developer workflow

**Files:**
- Modify: `AGENTS.md` if wording needs correction after validation
- Modify: `.codex/run-in-env.sh` if validation gaps appear

**Step 1: Run repository environment commands through the wrapper**

Run:

```bash
bash .codex/run-in-env.sh python -m pip --version
bash .codex/run-in-env.sh pytest source/legged_lab/test/test_codex_env.py -v
bash .codex/run-in-env.sh pre-commit run --all-files
```

Expected: Commands run inside the configured environment, or fail with targeted actionable messages.

**Step 2: Tighten wording only if needed**

Adjust documentation or wrapper messages based on the actual failure modes observed.

**Step 3: Commit**

```bash
git add AGENTS.md .codex/run-in-env.sh source/legged_lab/test/test_codex_env.py
git commit -m "chore: finalize codex environment bootstrap workflow"
```
