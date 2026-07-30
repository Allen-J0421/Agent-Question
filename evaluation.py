"""Grade an agent's patch against the dataset's own SWE-bench oracles.

The dataset ships every field this module needs: ``FAIL_TO_PASS`` and
``PASS_TO_PASS`` (the pass/fail oracle), ``test_patch`` (the gold tests), and
``files`` / ``patch`` (localization ground truth). Nothing here invents a test.

Three behaviors are load-bearing and were each established by direct
experiment rather than assumption:

* **Whole files are handed to pytest, never node ids.** Passing a node id that
  no longer exists makes pytest report ``no tests ran`` and silently discard
  every *other* id in the same invocation -- ``--continue-on-collection-errors``
  does not rescue it. Results are therefore collected by running the test files
  with ``-rA`` and looking each node id up in the output.

* **Agent-authored test edits are discarded before grading.** The agent may
  rewrite the very tests it is judged by, and such edits can also break
  ``git apply`` of the gold ``test_patch``. Every file matching
  :func:`is_test_path` is restored first. This can never throw away a real fix:
  no gold source patch in the dataset touches such a path.

* **Un-gradable runs are recorded, never scored.** django and sympy use their
  own test runners, and this harness provisions no per-instance environment, so
  a repo may simply not import here. Those outcomes get their own ``status`` and
  a ``resolved`` of ``None`` -- a missing dependency must never masquerade as a
  patch that failed.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


# Directory names that mark a test tree across the dataset's repositories.
TEST_PATH_PARTS = ("tests", "testing")

DEFAULT_TIMEOUT_SECONDS = 1800

# `-rA` short-summary lines, e.g. "PASSED path/to/test_x.py::test_y".
_OUTCOME_LINE = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL)\s+(\S+)")

STATUS_SCORED = "scored"
STATUS_UNSUPPORTED_RUNNER = "unsupported_runner"
STATUS_ENV_UNAVAILABLE = "env_unavailable"
STATUS_TEST_PATCH_FAILED = "test_patch_failed"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"
STATUS_NOT_EVALUATED = "not_evaluated"


def is_test_path(path: str) -> bool:
    """Return whether ``path`` belongs to a repository's test tree.

    Verified against the dataset: this matches every test file referenced by a
    ``test_patch`` or a ``FAIL_TO_PASS``/``PASS_TO_PASS`` node id, and no gold
    *source* patch anywhere in the 500 instances touches a path it matches. So
    restoring these files can only ever discard agent-authored tests.
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


def uses_pytest_ids(node_ids: list[str]) -> bool:
    """Return whether these ids are pytest node ids this module can grade.

    django and sympy record ids in their own runners' formats (e.g.
    ``test_x (module.Class)``), which pytest cannot address. 194 of the 500
    instances use pytest ids; the rest are reported as
    ``unsupported_runner`` rather than silently scored as failures.
    """
    return bool(node_ids) and any("::" in node_id for node_id in node_ids)


def test_files_for(node_ids: list[str]) -> list[str]:
    """Return the deduplicated test files owning these node ids, in order."""
    files: list[str] = []
    for node_id in node_ids:
        if "::" not in node_id:
            continue
        path = node_id.split("::", 1)[0]
        if path not in files:
            files.append(path)
    return files


def patch_files(patch: str) -> list[str]:
    """Return the post-image paths named by a unified diff's headers."""
    files: list[str] = []
    for line in (patch or "").splitlines():
        if line.startswith("diff --git ") and " b/" in line:
            path = line.split(" b/", 1)[1].strip()
            if path and path not in files:
                files.append(path)
    return files


def _source_only_patch(patch: str) -> str:
    """Drop every file section of a diff that targets a test path.

    Agent-authored tests are never an oracle, and re-applying them would also
    conflict with the gold ``test_patch``. Splitting the diff here keeps the
    stored patch complete (so the agent's test edits remain inspectable as
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


def probe_environment(
    workspace: Path, test_files: list[str], runner: list[str], timeout: int
) -> tuple[bool, str | None]:
    """Check that pytest can import and collect the graded test files.

    This harness pins no per-instance environment, so a repository may not be
    importable in the shared virtualenv. Probing first keeps an ``ImportError``
    from being recorded as an agent failure.
    """
    if not test_files:
        return False, "no test files to collect"
    try:
        completed = subprocess.run(
            [*runner, "-m", "pytest", "--collect-only", "-q", *test_files],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "collection timed out"
    except OSError as exc:
        return False, f"could not launch pytest: {exc}"
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr).strip().splitlines()
        return False, "\n".join(tail[-15:]) or "collection failed"
    return True, None


def parse_pytest_outcomes(output: str) -> dict[str, str]:
    """Map node id -> outcome from ``pytest -rA`` short-summary lines."""
    outcomes: dict[str, str] = {}
    for line in output.splitlines():
        match = _OUTCOME_LINE.match(line.strip())
        if match:
            outcome, node_id = match.group(1), match.group(2)
            outcomes[node_id] = outcome
    return outcomes


def run_pytest_files(
    workspace: Path, test_files: list[str], runner: list[str], timeout: int
) -> tuple[dict[str, str], str | None]:
    """Run whole test files and return ``(node id -> outcome, error)``.

    Whole files are used deliberately: a single stale node id on the command
    line makes pytest discard the results of every other id in the run.
    """
    try:
        completed = subprocess.run(
            [*runner, "-m", "pytest", "-rA", "--tb=no", "-q", *test_files],
            cwd=str(workspace),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {}, "timeout"
    except OSError as exc:
        return {}, f"could not launch pytest: {exc}"
    return parse_pytest_outcomes(completed.stdout + completed.stderr), None


def _tally(node_ids: list[str], outcomes: dict[str, str]) -> tuple[int, list[str]]:
    """Return how many ids passed, plus ids absent from the pytest output."""
    passed = 0
    missing: list[str] = []
    for node_id in node_ids:
        outcome = outcomes.get(node_id)
        if outcome is None:
            missing.append(node_id)
        elif outcome in ("PASSED", "XFAIL"):
            passed += 1
    return passed, missing


def _result(
    status: str,
    *,
    resolved: bool | None = None,
    error: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": status,
        "resolved": resolved,
        "f2p_total": 0,
        "f2p_passed": 0,
        "p2p_total": 0,
        "p2p_passed": 0,
        "missing_node_ids": [],
        "localization_hit": None,
        "gold_files": [],
        "agent_source_files": [],
        "agent_test_files": [],
        "patch_path": None,
        "patch_bytes": 0,
        "empty_patch": True,
        "duration_seconds": None,
        "error": error,
    }
    base.update(fields)
    return base


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
    diff = run_git(["diff"], cwd=workspace, check=False).stdout

    patch_path = None
    if diff:
        destination = Path(logs_root) / "patches" / f"{run_id}.patch"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(diff, encoding="utf-8")
        patch_path = str(destination)

    # Leave a clean checkout: prepare_workspace refuses a dirty tree.
    run_git(["checkout", "--", "."], cwd=workspace, check=False)
    run_git(["clean", "-fd"], cwd=workspace, check=False)

    return {
        "patch_path": patch_path,
        "patch": diff,
        "changed_paths": changed,
        "patch_bytes": len(diff.encode("utf-8")),
    }


def evaluate_patch(
    *,
    row: dict[str, Any],
    workspace: Path,
    run_git: Callable[..., Any],
    patch: str,
    changed_paths: list[str] | None = None,
    patch_path: str | None = None,
    runner: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Grade a captured patch in a checkout, then restore the checkout.

    Takes the patch as data rather than reading a working tree, so a stored
    patch can be re-graded at any time. ``workspace`` must be a clean checkout
    of the instance's ``base_commit``; the patch is applied here, graded, and
    reverted.
    """
    import sys

    started = time.monotonic()
    runner = runner or [sys.executable]

    f2p = parse_node_ids(row.get("FAIL_TO_PASS", ""))
    p2p = parse_node_ids(row.get("PASS_TO_PASS", ""))
    gold = gold_files(row)

    if changed_paths is None:
        changed_paths = patch_files(patch)
    agent_test_files = [path for path in changed_paths if is_test_path(path)]
    agent_source_files = [path for path in changed_paths if not is_test_path(path)]

    common = {
        "localization_hit": bool(set(agent_source_files) & set(gold)),
        "gold_files": gold,
        "agent_source_files": agent_source_files,
        "agent_test_files": agent_test_files,
        "patch_path": patch_path,
        "patch_bytes": len(patch.encode("utf-8")),
        "empty_patch": not agent_source_files,
        "f2p_total": len(f2p),
        "p2p_total": len(p2p),
    }

    try:
        # Reconstruct the agent's tree from the stored patch. Source edits only:
        # agent-authored tests are never an oracle, and would also collide with
        # the gold test_patch applied below.
        source_only = _source_only_patch(patch)
        if source_only:
            applied = run_git(
                ["apply", "-"], cwd=workspace, check=False, input=source_only
            )
            if applied.returncode != 0:
                return _result(
                    STATUS_ERROR,
                    error=(
                        "could not re-apply the stored agent patch: "
                        + (applied.stderr or "").strip()
                    )[:2000],
                    duration_seconds=time.monotonic() - started,
                    **common,
                )

        if not uses_pytest_ids(f2p + p2p):
            return _result(
                STATUS_UNSUPPORTED_RUNNER,
                error="test ids are not pytest node ids (django/sympy runner)",
                duration_seconds=time.monotonic() - started,
                **common,
            )

        test_patch = row.get("test_patch") or ""
        if test_patch:
            applied = run_git(
                ["apply", "-"], cwd=workspace, check=False, input=test_patch
            )
            if applied.returncode != 0:
                return _result(
                    STATUS_TEST_PATCH_FAILED,
                    error=(applied.stderr or "git apply failed").strip()[:2000],
                    duration_seconds=time.monotonic() - started,
                    **common,
                )

        graded_files = test_files_for(f2p + p2p)
        ok, probe_error = probe_environment(workspace, graded_files, runner, timeout)
        if not ok:
            return _result(
                STATUS_ENV_UNAVAILABLE,
                error=probe_error,
                duration_seconds=time.monotonic() - started,
                **common,
            )

        outcomes, run_error = run_pytest_files(workspace, graded_files, runner, timeout)
        if run_error == "timeout":
            return _result(
                STATUS_TIMEOUT,
                error=f"pytest exceeded {timeout}s",
                duration_seconds=time.monotonic() - started,
                **common,
            )
        if run_error:
            return _result(
                STATUS_ERROR,
                error=run_error,
                duration_seconds=time.monotonic() - started,
                **common,
            )

        f2p_passed, f2p_missing = _tally(f2p, outcomes)
        p2p_passed, p2p_missing = _tally(p2p, outcomes)
        return _result(
            STATUS_SCORED,
            resolved=(f2p_passed == len(f2p) and p2p_passed == len(p2p)),
            f2p_passed=f2p_passed,
            p2p_passed=p2p_passed,
            missing_node_ids=f2p_missing + p2p_missing,
            duration_seconds=time.monotonic() - started,
            **common,
        )
    except Exception as exc:  # noqa: BLE001 - a grading bug must not lose the run
        return _result(
            STATUS_ERROR,
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.monotonic() - started,
            **common,
        )
    finally:
        # Undo everything grading applied, so the checkout can be graded again.
        run_git(["checkout", "--", "."], cwd=workspace, check=False)
        run_git(["clean", "-fd"], cwd=workspace, check=False)


def evaluate_run(
    *,
    row: dict[str, Any],
    workspace: Path,
    run_git: Callable[..., Any],
    logs_root: Path | None = None,
    run_id: str | None = None,
    runner: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Capture the agent's patch from a live workspace, then grade it.

    Convenience wrapper over :func:`capture_agent_patch` and
    :func:`evaluate_patch` for the common case of grading immediately after a
    session. The two halves stay independently callable so grading can be
    redone later from the stored patch alone.
    """
    captured = capture_agent_patch(
        workspace=workspace,
        run_git=run_git,
        logs_root=Path(logs_root) if logs_root else workspace,
        run_id=run_id or "unsaved",
    )
    return evaluate_patch(
        row=row,
        workspace=workspace,
        run_git=run_git,
        patch=captured["patch"],
        changed_paths=captured["changed_paths"],
        patch_path=captured["patch_path"],
        runner=runner,
        timeout=timeout,
    )
