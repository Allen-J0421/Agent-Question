"""Grade stored agent patches with the official SWE-bench evaluation harness.

This is the study's only grader. Runs *capture* the agent's diff
(:func:`capture_agent_patch`) and grading happens afterwards inside the
official harness (``swebench.harness.run_evaluation``): each instance is
graded in its own Docker image with the pinned interpreter, dependency set,
compiled extensions, and era-correct test runner. Every repository's runner
is supported (django's ``runtests.py``, sympy's ``bin/test``, pytest), so all
500 dataset instances grade.

An earlier in-process grader ran the official test ids under this
repository's own interpreter in a raw checkout; that test-bed produced
``env_unavailable`` for 8/10 runs of the 2026-07-30 batch (astropy's logger
refuses to initialize under a modern pytest that has already patched
``warnings.showwarning``) and could not address django/sympy at all. It was
removed in favor of this module.

Division of labor:

* Pure helpers (patch splitting, prediction building, duplicate-instance
  bucketing, report parsing) are offline-testable and hold all mapping logic.
* :func:`run_harness` is the one thin wrapper that shells out to the harness
  and requires a running Docker daemon.

Grading semantics:

* **Only the agent's source edits are graded** (:func:`source_only_patch`) —
  agent-authored tests are never an oracle and would collide with the gold
  ``test_patch`` the harness applies. The full patch stays on disk as
  evidence. Verified against the dataset: no gold *source* patch touches a
  path :func:`is_test_path` matches, so the split cannot discard a real fix.
* **An empty source patch is unresolved by definition** and never costs a
  container.
* **Localization** is computed here from the stored patch against the
  dataset's gold files, since the harness does not report it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# The official SWE-bench Verified dataset; instance_ids in data/interactive-swe
# are the standard princeton ids, so the harness resolves them directly.
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
MODEL_NAME = "ambig-swe"  # model_name_or_path; also the harness's log dir name
DEFAULT_NAMESPACE = "swebench"  # pull prebuilt per-instance images from Docker Hub
DEFAULT_MAX_WORKERS = 2
DEFAULT_TIMEOUT = 1800

STATUS_SCORED = "scored"
STATUS_ERROR = "error"
STATUS_NOT_EVALUATED = "not_evaluated"

# Directory names that mark a test tree across the dataset's repositories.
TEST_PATH_PARTS = ("tests", "testing")


# --------------------------------------------------------------------------
# Patch and oracle helpers (pure)
# --------------------------------------------------------------------------


def is_test_path(path: str) -> bool:
    """Return whether ``path`` belongs to a repository's test tree.

    Verified against the dataset: this matches every test file referenced by a
    ``test_patch`` or a ``FAIL_TO_PASS``/``PASS_TO_PASS`` node id, and no gold
    *source* patch anywhere in the 500 instances touches a path it matches. So
    stripping these files can only ever discard agent-authored tests.
    """
    parts = Path(path).parts
    name = Path(path).name
    return (
        any(part in TEST_PATH_PARTS for part in parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def parse_node_ids(raw: str) -> list[str]:
    """Parse a JSON-encoded ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` column."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [item for item in value if isinstance(item, str)]


def patch_files(patch: str) -> list[str]:
    """Return the post-image paths named by a unified diff's headers."""
    files: list[str] = []
    for line in (patch or "").splitlines():
        if line.startswith("diff --git ") and " b/" in line:
            path = line.split(" b/", 1)[1].strip()
            if path and path not in files:
                files.append(path)
    return files


def source_only_patch(patch: str) -> str:
    """Drop every file section of a diff that targets a test path.

    Agent-authored tests are never an oracle, and would also conflict with the
    gold ``test_patch`` the harness applies. Splitting the diff here keeps the
    stored patch complete (the agent's test edits remain inspectable as
    evidence) while grading only its source changes.
    """
    sections: list[list[str]] = []
    for line in (patch or "").splitlines(keepends=True):
        if line.startswith("diff --git "):
            sections.append([line])
        elif sections:
            sections[-1].append(line)
    kept = [
        section
        for section in sections
        if not (
            " b/" in section[0]
            and is_test_path(section[0].split(" b/", 1)[1].strip())
        )
    ]
    return "".join("".join(section) for section in kept)


def gold_files(row: dict[str, Any]) -> list[str]:
    """Return the gold-edited source paths for localization scoring.

    ``files`` is empty for 26 of the 500 instances, so the gold ``patch``
    diff headers are used as the documented fallback.
    """
    raw = row.get("files") or ""
    listed = [
        part.strip()
        for part in raw.replace(",", "\n").splitlines()
        if part.strip()
    ]
    if listed:
        return listed
    return patch_files(row.get("patch") or "")


# --------------------------------------------------------------------------
# Patch capture (the only step that needs the live workspace)
# --------------------------------------------------------------------------


def _changed_paths(run_git: Callable[..., Any], workspace: Path) -> list[str]:
    """Return every path the agent added, modified, or deleted."""
    status = run_git(["status", "--porcelain"], cwd=workspace, check=False).stdout
    paths: list[str] = []
    for line in status.splitlines():
        entry = line[3:].strip() if len(line) > 3 else ""
        # Renames are reported as "old -> new"; the new path is what exists.
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        if entry and entry not in paths:
            paths.append(entry)
    return paths


def capture_agent_patch(
    *,
    workspace: Path,
    run_git: Callable[..., Any],
    logs_root: Path,
    run_id: str,
) -> dict[str, Any]:
    """Persist the agent's diff, then reset the workspace.

    This is the only step that must happen while the run's workspace still
    holds the agent's edits. Everything the grader needs is captured here, so
    grading itself can run later -- and be re-run after the grading rules
    change -- without repeating the (expensive, non-deterministic) session.
    """
    changed = _changed_paths(run_git, workspace)
    # Plain `git diff` omits files the agent *created* (untracked paths) and
    # anything the agent happened to `git add` mid-session, and the clean-up
    # below would then destroy the only copy of that work. Intent-to-add
    # registers untracked files in the index without content, and diffing
    # against HEAD captures staged and unstaged edits alike.
    run_git(["add", "--intent-to-add", "."], cwd=workspace, check=False)
    diff = run_git(["diff", "HEAD"], cwd=workspace, check=False).stdout

    patch_path = None
    if diff:
        destination = Path(logs_root) / "patches" / f"{run_id}.patch"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(diff, encoding="utf-8")
        patch_path = str(destination)

    # Leave a clean checkout: prepare_workspace refuses a dirty tree. The
    # reset drops the intent-to-add entries first so checkout/clean see them
    # as ordinary untracked files again.
    run_git(["reset", "--quiet"], cwd=workspace, check=False)
    run_git(["checkout", "--", "."], cwd=workspace, check=False)
    run_git(["clean", "-fd"], cwd=workspace, check=False)

    return {
        "patch_path": patch_path,
        "patch": diff,
        "changed_paths": changed,
        "patch_bytes": len(diff.encode("utf-8")),
    }


# --------------------------------------------------------------------------
# Harness orchestration
# --------------------------------------------------------------------------


def harness_root(logs_root: Path) -> Path:
    """Directory the harness runs in, so its logs/ and reports stay in logs_root."""
    return Path(logs_root) / "swebench"


def localization_fields(
    row: dict[str, Any], patch: str, patch_path: str | None
) -> dict[str, Any]:
    """Compute the patch-derived fields the harness does not report."""
    changed = patch_files(patch)
    agent_test_files = [path for path in changed if is_test_path(path)]
    agent_source_files = [path for path in changed if not is_test_path(path)]
    gold = gold_files(row)
    return {
        "localization_hit": bool(set(agent_source_files) & set(gold)),
        "gold_files": gold,
        "agent_source_files": agent_source_files,
        "agent_test_files": agent_test_files,
        "patch_path": patch_path,
        "patch_bytes": len((patch or "").encode("utf-8")),
        "empty_patch": not agent_source_files,
    }


def bucket_by_instance(
    summaries: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split runs into groups whose instance ids are unique.

    The harness keys predictions by ``instance_id``, but this study runs the
    same instance under multiple conditions (and retries), so duplicates are
    spread across sequential harness invocations rather than silently dropped.
    """
    buckets: list[dict[str, dict[str, Any]]] = []
    for summary in summaries:
        instance_id = summary["task"]["instance_id"]
        for bucket in buckets:
            if instance_id not in bucket:
                bucket[instance_id] = summary
                break
        else:
            buckets.append({instance_id: summary})
    return [list(bucket.values()) for bucket in buckets]


def predictions_for(
    bucket: list[dict[str, Any]], patches: dict[str, str]
) -> list[dict[str, str]]:
    """Build the harness predictions rows for one bucket (source edits only)."""
    rows = []
    for summary in bucket:
        run_id = summary["run_id"]
        rows.append(
            {
                "instance_id": summary["task"]["instance_id"],
                "model_name_or_path": MODEL_NAME,
                "model_patch": source_only_patch(patches.get(run_id, "")),
            }
        )
    return rows


def empty_patch_evaluation(
    row: dict[str, Any], patch: str, patch_path: str | None
) -> dict[str, Any]:
    """Grade an empty source patch without spending a container.

    SWE-bench semantics: a prediction with no source change cannot flip any
    FAIL_TO_PASS test, so it is unresolved by definition. ``f2p/p2p_passed``
    are 0 because nothing ran; the ``error`` note keeps that distinguishable
    from a real all-fail test run.
    """
    f2p = parse_node_ids(row.get("FAIL_TO_PASS", ""))
    p2p = parse_node_ids(row.get("PASS_TO_PASS", ""))
    return {
        "harness": "swebench",
        "swebench_run_id": None,
        "status": STATUS_SCORED,
        "resolved": False,
        "f2p_total": len(f2p),
        "f2p_passed": 0,
        "p2p_total": len(p2p),
        "p2p_passed": 0,
        "missing_node_ids": [],
        "duration_seconds": None,
        "error": "empty source patch: unresolved by definition; tests not run",
        **localization_fields(row, patch, patch_path),
    }


def evaluation_from_report(
    row: dict[str, Any],
    report: dict[str, Any] | None,
    patch: str,
    patch_path: str | None,
    swebench_run_id: str,
) -> dict[str, Any]:
    """Map one harness per-instance ``report.json`` entry onto the study schema."""
    f2p = parse_node_ids(row.get("FAIL_TO_PASS", ""))
    p2p = parse_node_ids(row.get("PASS_TO_PASS", ""))
    base = {
        "harness": "swebench",
        "swebench_run_id": swebench_run_id,
        "f2p_total": len(f2p),
        "p2p_total": len(p2p),
        "missing_node_ids": [],
        "duration_seconds": None,
        **localization_fields(row, patch, patch_path),
    }

    if report is None:
        return {
            **base,
            "status": STATUS_ERROR,
            "resolved": None,
            "f2p_passed": 0,
            "p2p_passed": 0,
            "error": (
                "the harness produced no report.json for this instance; "
                f"see logs/run_evaluation/{swebench_run_id}/{MODEL_NAME}/ "
                "under the swebench harness directory"
            ),
        }

    tests = report.get("tests_status") or {}
    f2p_status = tests.get("FAIL_TO_PASS") or {}
    p2p_status = tests.get("PASS_TO_PASS") or {}
    seen: set[str] = set()
    for group in (f2p_status, p2p_status):
        seen.update(group.get("success") or [])
        seen.update(group.get("failure") or [])

    error = None
    if report.get("patch_successfully_applied") is False:
        error = "harness could not apply the model patch"

    return {
        **base,
        "status": STATUS_SCORED,
        "resolved": bool(report.get("resolved")),
        "f2p_passed": len(f2p_status.get("success") or []),
        "p2p_passed": len(p2p_status.get("success") or []),
        "missing_node_ids": [node for node in (*f2p, *p2p) if node not in seen],
        "error": error,
    }


def load_instance_report(
    root: Path, swebench_run_id: str, instance_id: str
) -> dict[str, Any] | None:
    """Read the harness's per-instance report, tolerant of a missing file."""
    path = (
        root
        / "logs"
        / "run_evaluation"
        / swebench_run_id
        / MODEL_NAME
        / instance_id
        / "report.json"
    )
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = data.get(instance_id)
    return entry if isinstance(entry, dict) else None


def docker_ready() -> tuple[bool, str]:
    """Return whether a Docker daemon is reachable, with a printable reason."""
    if shutil.which("docker") is None:
        return False, "the `docker` executable is not on PATH"
    probe = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=30
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        return False, detail[-1] if detail else "docker info failed"
    return True, ""


def run_harness(
    *,
    root: Path,
    predictions_path: Path,
    swebench_run_id: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: int = DEFAULT_TIMEOUT,
    namespace: str | None = DEFAULT_NAMESPACE,
) -> int:
    """Invoke the official harness; its output streams to the console.

    Runs with ``cwd=root`` so the harness's ``logs/`` tree and report land
    inside the experiment's log directory instead of the repository.
    """
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASET_NAME,
        "--split",
        "test",
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--run_id",
        swebench_run_id,
        "--report_dir",
        str(root),
    ]
    if namespace:
        command.extend(["--namespace", namespace])
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=str(root)).returncode


def evaluate_with_harness(
    *,
    logs_root: Path,
    rows: dict[str, dict[str, Any]],
    summaries: list[dict[str, Any]],
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: int = DEFAULT_TIMEOUT,
    namespace: str | None = DEFAULT_NAMESPACE,
    harness=run_harness,
) -> dict[str, dict[str, Any]]:
    """Grade the selected runs' stored patches; return run_id -> evaluation.

    ``harness`` is injectable so the orchestration (bucketing, empty-patch
    short-circuit, report mapping) is testable without Docker.
    """
    root = harness_root(logs_root)
    root.mkdir(parents=True, exist_ok=True)

    patches: dict[str, str] = {}
    for summary in summaries:
        run_id = summary["run_id"]
        patch_file = Path(logs_root) / "patches" / f"{run_id}.patch"
        patches[run_id] = (
            patch_file.read_text(encoding="utf-8") if patch_file.exists() else ""
        )

    evaluations: dict[str, dict[str, Any]] = {}

    def patch_path_for(run_id: str) -> str | None:
        path = Path(logs_root) / "patches" / f"{run_id}.patch"
        return str(path) if path.exists() else None

    # Empty source patches never need a container.
    runnable: list[dict[str, Any]] = []
    for summary in summaries:
        run_id = summary["run_id"]
        row = rows[summary["task"]["instance_id"]]
        if source_only_patch(patches[run_id]).strip():
            runnable.append(summary)
        else:
            evaluations[run_id] = empty_patch_evaluation(
                row, patches[run_id], patch_path_for(run_id)
            )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for index, bucket in enumerate(bucket_by_instance(runnable)):
        swebench_run_id = f"{stamp}-{index}"
        predictions_path = root / f"predictions-{swebench_run_id}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row_dict in predictions_for(bucket, patches):
                handle.write(json.dumps(row_dict) + "\n")

        print(
            f"SWE-bench harness invocation {index + 1}: "
            f"{len(bucket)} instance(s), run_id={swebench_run_id}",
            flush=True,
        )
        returncode = harness(
            root=root,
            predictions_path=predictions_path,
            swebench_run_id=swebench_run_id,
            max_workers=max_workers,
            timeout=timeout,
            namespace=namespace,
        )
        if returncode != 0:
            print(
                f"WARNING: harness exited {returncode}; per-instance reports "
                "that were produced are still collected.",
                flush=True,
            )

        for summary in bucket:
            run_id = summary["run_id"]
            instance_id = summary["task"]["instance_id"]
            report = load_instance_report(root, swebench_run_id, instance_id)
            evaluations[run_id] = evaluation_from_report(
                rows[instance_id],
                report,
                patches[run_id],
                patch_path_for(run_id),
                swebench_run_id,
            )

    return evaluations
