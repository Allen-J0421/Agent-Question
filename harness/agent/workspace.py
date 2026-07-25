"""Per-run isolated agent workspace via git worktrees over a cached bare mirror.

One bare mirror clone per repo lives in repos/<owner__name>.git; each run adds a
detached worktree at base_commit. Worktrees share object storage (cheap) and give a
guaranteed-clean tree. Worktree add/remove on a shared repo is serialized under a
per-repo lock; the agent run itself is fully parallel.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from harness.config import PathsConfig

_REPO_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _repo_lock(repo: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if repo not in _REPO_LOCKS:
            _REPO_LOCKS[repo] = threading.Lock()
        return _REPO_LOCKS[repo]


def _mirror_name(repo: str) -> str:
    return repo.replace("/", "__") + ".git"


# Some machines set safe.bareRepository=explicit (a security default) which blocks
# operations on bare repos. We pass safe.bareRepository=all for our own cached mirrors.
_GIT_SAFE = ["-c", "safe.bareRepository=all"]


def _run(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    # inject the safe-bare override right after the `git` executable
    if args and args[0] == "git":
        args = [args[0], *_GIT_SAFE, *args[1:]]
    return subprocess.run(args, cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True, check=check)


class Workspace:
    def __init__(self, path: Path, base_commit: str, repo: str):
        self.path = path
        self.base_commit = base_commit
        self.repo = repo


def _ensure_mirror(repo: str, paths: PathsConfig) -> Path:
    """Ensure a bare mirror clone of the repo exists; fetch base_commit on demand."""
    mirror = paths.repos_dir / _mirror_name(repo)
    with _repo_lock(repo):
        if not mirror.exists():
            url = f"https://github.com/{repo}.git"
            _run(["git", "clone", "--bare", url, str(mirror)])
    return mirror


def create_workspace(instance_id: str, repo: str, base_commit: str,
                     paths: PathsConfig | None = None) -> Workspace:
    paths = (paths or PathsConfig()).ensure()
    mirror = _ensure_mirror(repo, paths)
    ws_path = paths.worktrees_dir / instance_id

    with _repo_lock(repo):
        # make sure the commit is present in the mirror (older commits may need a fetch)
        have = _run(["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
                    cwd=mirror, check=False)
        if have.returncode != 0:
            _run(["git", "fetch", "origin", base_commit], cwd=mirror, check=False)

        if ws_path.exists():
            _run(["git", "worktree", "remove", "--force", str(ws_path)],
                 cwd=mirror, check=False)
        _run(["git", "worktree", "add", "--detach", str(ws_path), base_commit],
             cwd=mirror)

    return Workspace(path=ws_path, base_commit=base_commit, repo=repo)


def teardown_workspace(ws: Workspace, paths: PathsConfig | None = None) -> None:
    paths = paths or PathsConfig()
    mirror = paths.repos_dir / _mirror_name(ws.repo)
    with _repo_lock(ws.repo):
        _run(["git", "worktree", "remove", "--force", str(ws.path)],
             cwd=mirror, check=False)
