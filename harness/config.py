"""Configuration dataclasses for the harness. Defaults are chosen for the Phase-0
Opus sweep; every field is overridable via the orchestrator CLI or env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = two levels up from this file (harness/config.py -> ambig-SWE/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PathsConfig:
    """Where things live on disk."""
    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = PROJECT_ROOT / "data" / "interactive-swe"
    dataset_split: str = "test"
    runs_dir: Path = PROJECT_ROOT / "runs"          # per-run result.json trees
    repos_dir: Path = PROJECT_ROOT / "repos"        # cached bare mirror clones
    worktrees_dir: Path = PROJECT_ROOT / ".worktrees"  # ephemeral per-run checkouts
    eval_log_dir: Path = PROJECT_ROOT / "eval_logs"
    manifest_dir: Path = PROJECT_ROOT / "manifests"  # frozen sample manifests

    def ensure(self) -> "PathsConfig":
        for p in (self.runs_dir, self.repos_dir, self.worktrees_dir,
                  self.eval_log_dir, self.manifest_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class RunConfig:
    """Parameters controlling a single Claude-CLI agent run."""
    model: str = os.environ.get("HARNESS_MODEL", "opus")
    permission_mode: str = "acceptEdits"   # edit files in the sandbox without prompts
    max_turns: int = 40
    wall_timeout_s: int = 1800             # kill a run after 30 min
    # Tools the agent is allowed to use. AskUserQuestion is intentionally NOT
    # pre-denied: we want it to be able to ask so we can detect the decision.
    allowed_tools: tuple[str, ...] = (
        "Read", "Grep", "Glob", "Edit", "Write", "MultiEdit", "Bash",
        "AskUserQuestion",
    )
    n_repeats_ambiguous: int = 3           # N for the repeated-runs variance study
    n_repeats_full: int = 1


@dataclass(frozen=True)
class EvalConfig:
    """Parameters for the swebench evaluation pass."""
    max_workers: int = 4
    cache_level: str = "env"       # keep env images, discard instance images
    force_rebuild: bool = False
    timeout_s: int = 1800
    # "swebench" => PULL prebuilt images from the published registry instead of building
    # locally. Verified on an 8 GB machine: local builds OOM at the conda step, but
    # pulling + running the prebuilt image fits (gold-patch check reports resolved=True).
    namespace: str | None = "swebench"
    run_id_prefix: str = "ambigswe"


@dataclass(frozen=True)
class ConcurrencyConfig:
    cli_workers: int = 4     # parallel Claude-CLI subprocesses (I/O bound)
    eval_workers: int = 4    # parallel docker eval containers (CPU/docker bound)


@dataclass(frozen=True)
class HarnessConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    global_cost_ceiling_usd: float | None = None  # abort sweep if exceeded; None=off
