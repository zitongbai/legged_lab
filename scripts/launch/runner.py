"""Execute an assembled command: print it, or launch it inside tmux.

The tmux path mirrors ``scripts/experiments/sweep_amp.sh``:

* machine-local config (venv activate path + optional proxy) is read from
  ``scripts/env.local.sh`` — never hard-coded here;
* each run gets its own tmux session named after the run;
* the pane sources the venv, exports the proxy, runs the command, tees output
  to a log, and holds open with ``read`` so failures stay visible.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .command import to_shell_string
from .discovery import REPO_ROOT

ENV_LOCAL = os.path.join(REPO_ROOT, "scripts", "env.local.sh")
ENV_LOCAL_EXAMPLE = os.path.join(REPO_ROOT, "scripts", "env.local.sh.example")


# --- command printing -------------------------------------------------------


def print_command(argv: list[str], console: Console | None = None, note: str | None = None) -> None:
    """Pretty-print the assembled command for the user to copy."""
    console = console or Console()
    shell = to_shell_string(argv)
    # Render as a multi-line, backslash-continued command for readability.
    pretty = shell.replace(" --", " \\\n    --").replace(" agent.", " \\\n    agent.")
    syntax = Syntax(pretty, "bash", theme="ansi_dark", word_wrap=True)
    console.print(Panel(syntax, title="Command", border_style="cyan"))
    if note:
        console.print(note)


def print_copyable(argv: list[str], console: Console | None = None) -> None:
    """Print the command as a bare single line with no box/markup/highlight.

    The boxed :func:`print_command` output is nice to read but its border
    characters get caught when you select the text. This prints the exact same
    command as one plain line so it can be copied cleanly.
    """
    console = console or Console()
    shell = to_shell_string(argv)
    # soft_wrap=True avoids rich inserting hard line breaks; markup/highlight
    # off so nothing but the command itself is emitted.
    console.print("\n[dim]# copy-paste:[/]")
    console.print(shell, markup=False, highlight=False, soft_wrap=True)


# --- env.local.sh parsing ---------------------------------------------------


def _parse_env_local(path: str = ENV_LOCAL) -> dict[str, str]:
    """Extract ``VENV_ACTIVATE`` / ``PROXY_URL`` from env.local.sh.

    Only simple ``export KEY="value"`` / ``KEY=value`` lines are parsed; this is
    intentionally shallow to avoid sourcing arbitrary shell.
    """
    values: dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    assign_re = re.compile(r'^\s*(?:export\s+)?(\w+)\s*=\s*(.*)\s*$')
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].rstrip()
                m = assign_re.match(line)
                if not m:
                    continue
                key, raw = m.group(1), m.group(2).strip()
                if raw and raw[0] in "\"'" and raw[-1:] == raw[0]:
                    raw = raw[1:-1]
                values[key] = raw
    except OSError:
        return values
    return values


def get_env_config(console: Console | None = None) -> dict[str, str] | None:
    """Return {'VENV_ACTIVATE':..., 'PROXY_URL':...} or ``None`` if unusable.

    Prints a helpful hint when ``env.local.sh`` is missing or the venv path is
    unset/invalid.
    """
    console = console or Console()
    if not os.path.isfile(ENV_LOCAL):
        console.print(
            Panel(
                "[yellow]scripts/env.local.sh not found.[/]\n"
                "The tmux launcher sources it to activate your Python env (and optional proxy).\n\n"
                "Create it from the template:\n"
                "    [cyan]cp scripts/env.local.sh.example scripts/env.local.sh[/]\n"
                "then set [cyan]VENV_ACTIVATE[/] (and optionally [cyan]PROXY_URL[/]).",
                title="Missing env.local.sh",
                border_style="yellow",
            )
        )
        return None

    cfg = _parse_env_local()
    venv = cfg.get("VENV_ACTIVATE", "").strip()
    if not venv:
        console.print("[yellow]VENV_ACTIVATE is empty in env.local.sh — please set it.[/]")
        return None
    if not os.path.isfile(venv):
        console.print(f"[yellow]VENV_ACTIVATE points to a missing file:[/] {venv}")
        return None
    return {"VENV_ACTIVATE": venv, "PROXY_URL": cfg.get("PROXY_URL", "").strip()}


# --- tmux session naming ----------------------------------------------------


def sanitize_session_name(name: str) -> str:
    """Make a string safe for a tmux session name (no '.' or ':')."""
    # tmux forbids '.' and ':' in names; mirror sweep_amp.sh's dot/dash mapping.
    name = name.replace(".", "p").replace("-", "_").replace(":", "_")
    name = re.sub(r"[^\w]", "_", name)
    return name.strip("_") or "run"


def tmux_available() -> bool:
    from shutil import which

    return which("tmux") is not None


def _existing_sessions() -> set[str]:
    try:
        out = subprocess.check_output(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def unique_session_name(base: str) -> str:
    """Return ``base`` or ``base_2``/``base_3``... if already taken."""
    base = sanitize_session_name(base)
    existing = _existing_sessions()
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def build_pane_command(argv: list[str], env_cfg: dict[str, str], session: str, repo_root: str = REPO_ROOT) -> str:
    """Build the shell string executed inside the tmux pane."""
    cmd = to_shell_string(argv)
    prologue = f"source {shlex.quote(env_cfg['VENV_ACTIVATE'])}"
    proxy = env_cfg.get("PROXY_URL", "")
    if proxy:
        pq = shlex.quote(proxy)
        prologue += (
            f" && export http_proxy={pq} https_proxy={pq}"
            f" HTTP_PROXY={pq} HTTPS_PROXY={pq}"
        )
    log_path = f"logs/launch_{session}.log"
    pane = (
        f"cd {shlex.quote(repo_root)} && {prologue} && {cmd} 2>&1 | tee {shlex.quote(log_path)}; "
        f"echo '=== run exited (code $?) — press enter to close ==='; read"
    )
    return pane


def run_in_tmux(
    argv: list[str],
    session_base: str,
    env_cfg: dict[str, str],
    repo_root: str = REPO_ROOT,
    console: Console | None = None,
    dry_run: bool = False,
) -> str | None:
    """Launch ``argv`` in a fresh detached tmux session. Returns session name."""
    console = console or Console()
    if not tmux_available():
        console.print("[red]tmux is not available on this machine.[/]")
        return None

    session = unique_session_name(session_base)
    pane = build_pane_command(argv, env_cfg, session, repo_root)

    if dry_run:
        console.print(f"[dim]DRY RUN — would create tmux session[/] [cyan]{session}[/]")
        console.print(Syntax(pane, "bash", theme="ansi_dark", word_wrap=True))
        return session

    os.makedirs(os.path.join(repo_root, "logs"), exist_ok=True)
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, pane], check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        console.print(f"[red]Failed to start tmux session:[/] {exc}")
        return None

    console.print(
        Panel(
            f"Launched tmux session [bold cyan]{session}[/].\n"
            f"Attach:  [cyan]tmux attach -t {session}[/]\n"
            f"Log:     [cyan]logs/launch_{session}.log[/]",
            title="Running",
            border_style="green",
        )
    )
    return session
