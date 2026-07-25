"""Load the interactive-swe dataset and parse its all-string fields into a typed
Instance model. Handles the documented data caveats:
  - FAIL_TO_PASS / PASS_TO_PASS are JSON strings -> list[str]
  - files is comma/newline-separated and None for 26/500 -> list[str], with a
    fallback that derives paths from the gold patch's `+++ b/<path>` headers
  - hints_text is empty (not None) for 165/500
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from datasets import load_from_disk

from harness.config import PathsConfig

# Matches the post-image path in a unified-diff file header: `+++ b/path/to/file.py`.
_DIFF_NEW_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    environment_setup_commit: str
    version: str
    difficulty: str
    problem_statement: str      # AMBIGUOUS prompt (agent-facing in the ambiguous condition)
    original_issue: str         # FULL prompt (agent-facing in the full condition)
    hints_text: str             # may be ""
    patch: str                  # gold fix — harness-only, never shown to the agent
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    gold_files: list[str]       # localization ground truth (patch-header fallback applied)
    created_at: str


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return list(json.loads(raw))


def _files_from_patch(patch: str | None) -> list[str]:
    """Fallback for the 26 instances whose `files` field is None: read the edited
    paths straight out of the gold patch's diff headers."""
    if not patch:
        return []
    seen: list[str] = []
    for m in _DIFF_NEW_FILE_RE.finditer(patch):
        path = m.group(1).strip()
        if path and path != "/dev/null" and path not in seen:
            seen.append(path)
    return seen


def _parse_files(raw: str | None, patch: str | None) -> list[str]:
    if raw and raw.strip():
        parts = re.split(r"[,\n]", raw)
        return [p.strip() for p in parts if p.strip()]
    return _files_from_patch(patch)


def _to_instance(row: dict) -> Instance:
    return Instance(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        environment_setup_commit=row["environment_setup_commit"],
        version=row["version"],
        difficulty=row["difficulty"],
        problem_statement=row["problem_statement"] or "",
        original_issue=row["original_issue"] or "",
        hints_text=row["hints_text"] or "",
        patch=row["patch"] or "",
        test_patch=row["test_patch"] or "",
        fail_to_pass=_parse_json_list(row["FAIL_TO_PASS"]),
        pass_to_pass=_parse_json_list(row["PASS_TO_PASS"]),
        gold_files=_parse_files(row.get("files"), row.get("patch")),
        created_at=row["created_at"],
    )


@lru_cache(maxsize=1)
def load_instances(dataset_dir: str | None = None, split: str | None = None) -> tuple[Instance, ...]:
    """Load all instances as an immutable tuple. Cached so repeated calls are free."""
    paths = PathsConfig()
    ddir = Path(dataset_dir) if dataset_dir else paths.dataset_dir
    sp = split or paths.dataset_split
    ds = load_from_disk(str(ddir))[sp]
    return tuple(_to_instance(ds[i]) for i in range(len(ds)))


def instances_by_id(dataset_dir: str | None = None, split: str | None = None) -> dict[str, Instance]:
    return {inst.instance_id: inst for inst in load_instances(dataset_dir, split)}
