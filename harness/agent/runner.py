"""Wrap the Claude CLI as a subprocess: build argv, stream stdout to a transcript
file, capture stderr, enforce a wall-clock timeout (macOS has no GNU `timeout`), and
report how the process exited.

The CLI binary is resolved from HARNESS_CLAUDE_BIN, then PATH, then the known nvm path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.config import RunConfig

_NVM_FALLBACK = "/Users/allenjiang/.nvm/versions/node/v24.12.0/bin/claude"


def resolve_claude_bin() -> str:
    env = os.environ.get("HARNESS_CLAUDE_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("claude")
    if found:
        return found
    if Path(_NVM_FALLBACK).exists():
        return _NVM_FALLBACK
    raise FileNotFoundError(
        "claude CLI not found. Set HARNESS_CLAUDE_BIN to its absolute path."
    )


@dataclass
class RunOutcome:
    exit_code: int | None      # None => timed out (killed)
    timed_out: bool
    transcript_path: Path
    stderr_path: Path
    cli_version: str


def _cli_version(claude_bin: str) -> str:
    try:
        out = subprocess.run([claude_bin, "--version"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        return out.split()[0] if out else "unknown"
    except Exception:
        return "unknown"


def build_argv(claude_bin: str, prompt: str, workspace: Path, cfg: RunConfig) -> list[str]:
    return [
        claude_bin,
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--model", cfg.model,
        "--permission-mode", cfg.permission_mode,
        "--max-turns", str(cfg.max_turns),
        "--add-dir", str(workspace),
        "--allowedTools", ",".join(cfg.allowed_tools),
    ]


def run_agent(prompt: str, workspace: Path, transcript_path: Path,
              stderr_path: Path, cfg: RunConfig) -> RunOutcome:
    claude_bin = resolve_claude_bin()
    argv = build_argv(claude_bin, prompt, workspace, cfg)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    timed_out = False
    exit_code: int | None = None
    with open(transcript_path, "w") as out, open(stderr_path, "w") as err, \
            open(os.devnull) as devnull:
        proc = subprocess.Popen(
            argv, cwd=str(workspace), stdin=devnull, stdout=out, stderr=err,
        )
        try:
            exit_code = proc.wait(timeout=cfg.wall_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()

    return RunOutcome(
        exit_code=exit_code,
        timed_out=timed_out,
        transcript_path=transcript_path,
        stderr_path=stderr_path,
        cli_version=_cli_version(claude_bin),
    )
