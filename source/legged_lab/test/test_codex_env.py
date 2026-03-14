from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
CODEX_DIR = REPO_ROOT / ".codex"
RUN_IN_ENV = CODEX_DIR / "run-in-env.sh"
ENV_TEMPLATE = CODEX_DIR / "env.local.sh.example"


def run_wrapper(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", str(RUN_IN_ENV), *args],
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_env_template_exists() -> None:
    assert ENV_TEMPLATE.exists(), f"Missing environment template: {ENV_TEMPLATE}"


def test_wrapper_fails_when_local_env_file_is_missing() -> None:
    result = run_wrapper("python", "-V")

    assert result.returncode != 0
    assert ".codex/env.local.sh" in result.stderr
    assert ".codex/env.local.sh.example" in result.stderr


def test_wrapper_reports_missing_conda_env_variable(tmp_path: Path) -> None:
    env_file = tmp_path / "env.local.sh"
    env_file.write_text('export ISAACLAB_PATH="/tmp"\n', encoding="utf-8")

    result = run_wrapper("python", "-V", env={"CODEX_ENV_FILE": str(env_file)})

    assert result.returncode != 0
    assert "CODEX_CONDA_ENV" in result.stderr


def test_wrapper_reports_invalid_isaaclab_path(tmp_path: Path) -> None:
    env_file = tmp_path / "env.local.sh"
    env_file.write_text(
        'export CODEX_CONDA_ENV="isaaclab"\nexport ISAACLAB_PATH="/path/that/does/not/exist"\n',
        encoding="utf-8",
    )

    result = run_wrapper("python", "-V", env={"CODEX_ENV_FILE": str(env_file)})

    assert result.returncode != 0
    assert "ISAACLAB_PATH" in result.stderr


@pytest.mark.skipif(not (CODEX_DIR / "env.local.sh").exists(), reason="Local Codex environment is not configured")
def test_wrapper_can_import_isaaclab_and_legged_lab_when_local_env_exists() -> None:
    result = run_wrapper(
        "python",
        "-c",
        "import importlib; importlib.import_module('isaaclab'); importlib.import_module('legged_lab')",
    )

    assert result.returncode == 0, result.stderr
