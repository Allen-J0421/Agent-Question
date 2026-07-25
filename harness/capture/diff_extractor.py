"""Extract the agent's code change as a unified diff from its workspace, relative to
base_commit. Captures tracked edits AND new untracked files (via `git add -N`), then
parses the diff for touched files and LOC counts.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from harness.record.schema import PatchInfo

_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    ).stdout


def extract_diff(workspace: Path, base_commit: str) -> PatchInfo:
    """Diff the workspace tree against base_commit. Include untracked files by
    intent-to-add so newly created files appear in the diff."""
    ws = Path(workspace)
    # stage intent-to-add for untracked files so `git diff` shows them
    _git(["add", "-A", "-N"], ws)
    diff = _git(["diff", base_commit], ws)

    if not diff.strip():
        return PatchInfo(produced_patch=False)

    files = []
    for m in _DIFF_FILE_RE.finditer(diff):
        p = m.group(1).strip()
        if p and p != "/dev/null" and p not in files:
            files.append(p)

    loc_added = sum(
        1 for ln in diff.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    )
    loc_removed = sum(
        1 for ln in diff.splitlines()
        if ln.startswith("-") and not ln.startswith("---")
    )

    return PatchInfo(
        produced_patch=True,
        diff=diff,
        diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        files_touched=files,
        n_files_touched=len(files),
        loc_added=loc_added,
        loc_removed=loc_removed,
    )
