#!/usr/bin/env python3
"""Launch unattended Claude Code experiments on local dataset issues.

The launcher runs an Agent SDK session in ``default`` permission mode, so tool
calls that would prompt reach the study's ``can_use_tool`` callback and are
recorded there before being approved. Its only task-specific behavioral inputs
are the selected issue text and the requested model.
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from evaluation import (
    DEFAULT_TIMEOUT_SECONDS,
    capture_agent_patch,
    evaluate_patch,
    evaluate_run,
    parse_node_ids,
    uses_pytest_ids,
)
from sdk_runner import PERMISSION_MODE, load_reference_toolset, run_sdk_session
from study_log import (
    build_run_summary_sdk,
    create_run_manifest,
    default_logs_root,
    load_evaluations,
    load_run_summaries,
    write_evaluation,
    write_report,
    write_run_summary,
)


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "interactive-swe"
CHECKOUTS = Path.cwd() / ".experiment-checkouts"
DEFAULT_MODEL = "claude-opus-4-8"

CONDITION_FIELD = {
    "ambiguous": "problem_statement",
    "full": "original_issue",
}
BATCH_CONDITIONS = (*CONDITION_FIELD, "both")

SCOPES = ("all", "gradable")
DEFAULT_SCOPE = "all"
SCOPE_HELP = (
    "all (default): every dataset instance. gradable: only instances whose "
    "FAIL_TO_PASS/PASS_TO_PASS use pytest node ids and can therefore be scored "
    "(194 of 500; excludes django and sympy, which use their own runners). "
    "Scope is applied before selection, so --count N yields N in-scope instances."
)


@lru_cache(maxsize=1)
def load_rows() -> dict[str, dict[str, Any]]:
    split = load_from_disk(str(DATASET))["test"]
    return {split[i]["instance_id"]: dict(split[i]) for i in range(len(split))}


def is_gradable(row: dict[str, Any]) -> bool:
    """Return whether this instance's pass/fail oracle is pytest-addressable.

    django and sympy record their test ids in their own runners' formats, which
    ``evaluation`` cannot execute; 194 of the 500 instances use pytest node ids.
    """
    node_ids = parse_node_ids(row.get("FAIL_TO_PASS", "")) + parse_node_ids(
        row.get("PASS_TO_PASS", "")
    )
    return uses_pytest_ids(node_ids)


def scoped_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    """Restrict the candidate pool before selection.

    Filtering here rather than after selection is what makes ``--count N`` mean
    "N instances I can actually grade" instead of "N of the first 500, most of
    which are skipped".
    """
    if scope == "gradable":
        return [row for row in rows if is_gradable(row)]
    return rows


def issue_text(row: dict[str, Any], condition: str) -> str:
    """Return exactly the issue field assigned to the requested condition."""
    field = CONDITION_FIELD[condition]
    text = (row.get(field) or "").strip()
    if not text:
        raise ValueError(f"{row['instance_id']} has an empty {field}")
    return text


def build_prompt(row: dict[str, Any], condition: str) -> str:
    return f"Resolve the following issue in this repository:\n\n{issue_text(row, condition)}"


def run_git(
    args: list[str],
    cwd: Path | None = None,
    check: bool = True,
    input: str | None = None,
):
    """Run git. ``input`` feeds stdin, e.g. ``git apply -`` with a patch."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=check,
        input=input,
    )


def workspace_path(row: dict[str, Any], condition: str) -> Path:
    repo_slug = row["repo"].replace("/", "__")
    commit = row["base_commit"][:12]
    return CHECKOUTS / f"{repo_slug}__{commit}__{condition}"


def prepare_workspace(row: dict[str, Any], condition: str) -> Path:
    """Create or validate a normal repository checkout at base_commit."""
    path = workspace_path(row, condition)
    base_commit = row["base_commit"]

    if not path.exists():
        CHECKOUTS.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{row['repo']}.git"
        print(f"Cloning {url} into {path}", flush=True)
        run_git(["clone", url, str(path)])

    if not (path / ".git").exists():
        raise RuntimeError(f"workspace is not a Git checkout: {path}")

    status = run_git(["status", "--porcelain"], cwd=path).stdout
    if status.strip():
        raise RuntimeError(
            f"workspace has existing changes and will not be overwritten:\n{path}\n"
            "Inspect or remove that workspace before running it again."
        )

    have = run_git(["cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=path, check=False)
    if have.returncode != 0:
        print(f"Fetching base commit {base_commit}", flush=True)
        fetched = run_git(["fetch", "origin", base_commit], cwd=path, check=False)
        if fetched.returncode != 0:
            raise RuntimeError(f"could not fetch base commit:\n{fetched.stderr}")

    run_git(["checkout", "--detach", base_commit], cwd=path)
    return path


def command_list(args: argparse.Namespace) -> None:
    rows = scoped_rows(list(load_rows().values()), getattr(args, "scope", DEFAULT_SCOPE))
    if args.repo:
        rows = [row for row in rows if row["repo"] == args.repo]
    print(f"# {len(rows)} instance(s) in scope", flush=True)
    for row in rows[: args.limit]:
        print(f"{row['instance_id']}\t{row['repo']}\t{row['difficulty']}")


def requested_conditions(condition: str) -> tuple[str, ...]:
    """Expand the batch-only ``both`` option into individual experiment conditions."""
    if condition == "both":
        return tuple(CONDITION_FIELD)
    return (condition,)


def completed_run_keys(logs_root: Path) -> set[tuple[str, str, str]]:
    """Return completed (instance, condition, model) runs from immutable summaries."""
    summaries, _ = load_run_summaries(logs_root)
    completed: set[tuple[str, str, str]] = set()
    for summary in summaries:
        task = summary.get("task", {})
        claude = summary.get("claude", {})
        process = summary.get("process", {})
        instance_id = task.get("instance_id")
        condition = task.get("condition")
        model = claude.get("model")
        # A failed process launch never executed an experiment and is retryable.
        if process.get("stop_reason") == "launch_error":
            continue
        if (
            isinstance(instance_id, str)
            and condition in CONDITION_FIELD
            and isinstance(model, str)
        ):
            completed.add((instance_id, condition, model))
    return completed


def select_batch_rows(
    rows: list[dict[str, Any]],
    completed: set[tuple[str, str, str]],
    conditions: tuple[str, ...],
    model: str,
    count: int,
) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    """Select the next incomplete dataset instances in deterministic dataset order."""
    selected: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for row in rows:
        missing = tuple(
            condition
            for condition in conditions
            if (row["instance_id"], condition, model) not in completed
        )
        if missing:
            selected.append((row, missing))
        if len(selected) == count:
            break
    return selected


def command_run(args: argparse.Namespace) -> None:
    rows = load_rows()
    if args.instance_id not in rows:
        raise SystemExit(f"unknown instance_id: {args.instance_id}")

    row = rows[args.instance_id]
    field = CONDITION_FIELD[args.condition]
    prompt = build_prompt(row, args.condition)

    print(f"Instance:  {row['instance_id']}")
    print(f"Repository:{row['repo']}")
    print(f"Condition: {args.condition}")
    print(f"Field:     {field}")
    print(f"Model:     {args.model}")
    print("\n--- EXACT CLAUDE PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---\n")

    if args.dry_run:
        print("Dry run: Claude was not launched.")
        print(
            f"Command shape: Agent SDK session, permission_mode={PERMISSION_MODE}, "
            "tools=<config/reference_toolset.json>, can_use_tool callback registered "
            "(observes and approves prompting tool calls; logs AskUserQuestion in full)."
        )
        return

    workspace = prepare_workspace(row, args.condition)
    print(f"Launching unattended Claude Code (sdk) in {workspace}", flush=True)
    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()

    manifest = create_run_manifest(
        logs_root,
        row=row,
        condition=args.condition,
        model=args.model,
        workspace=workspace,
        prompt=prompt,
        interface="sdk",
    )
    observation = asyncio.run(
        run_sdk_session(
            prompt=prompt,
            workspace=workspace,
            model=args.model,
            tools=load_reference_toolset(),
        )
    )

    if getattr(args, "no_eval", False):
        # Capture only: the patch is saved and the workspace reset, so the run
        # can be graded later with `experiment.py evaluate`.
        captured = capture_agent_patch(
            workspace=workspace,
            run_git=run_git,
            logs_root=logs_root,
            run_id=manifest["run_id"],
        )
        print(
            f"Saved agent patch ({captured['patch_bytes']} bytes); "
            "grading skipped (--no-eval).",
            flush=True,
        )
        evaluation = {"status": "not_evaluated", "resolved": None}
    else:
        print("Grading agent patch against the dataset oracles...", flush=True)
        evaluation = evaluate_run(
            row=row,
            workspace=workspace,
            run_git=run_git,
            logs_root=logs_root,
            run_id=manifest["run_id"],
            timeout=args.eval_timeout,
        )
    summary = build_run_summary_sdk(manifest, observation, evaluation)
    summary_path = write_run_summary(logs_root, summary)
    write_evaluation(logs_root, manifest["run_id"], evaluation)
    print(f"Run log: {summary_path}", flush=True)
    if evaluation.get("status") != "not_evaluated":
        print(
            f"Evaluation: status={evaluation['status']} resolved={evaluation['resolved']} "
            f"F2P={evaluation['f2p_passed']}/{evaluation['f2p_total']} "
            f"P2P={evaluation['p2p_passed']}/{evaluation['p2p_total']} "
            f"localization_hit={evaluation['localization_hit']}",
            flush=True,
        )
        if evaluation["error"]:
            print(f"Evaluation note: {evaluation['error']}", flush=True)
    roster = summary["tool_roster"]
    print(
        f"AskUserQuestion available this run: {roster['askuserquestion_available']}",
        flush=True,
    )
    if roster["matches_reference"] is False:
        print(
            f"WARNING: live tool roster did not match reference_toolset.json — "
            f"missing: {roster['missing_from_actual']}, "
            f"extra: {roster['extra_in_actual']}",
            flush=True,
        )


def command_evaluate(args: argparse.Namespace) -> None:
    """Re-grade stored agent patches without re-running any Claude session.

    Grading is a judgement about a run, not part of it, so changing the rules
    should never cost a batch of sessions. This reads each run's saved patch,
    grades it in a fresh checkout, and overwrites only the evaluation record.
    """
    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()
    summaries, errors = load_run_summaries(logs_root)
    for message in errors:
        print(f"WARNING: unreadable run summary: {message}", flush=True)
    if not summaries:
        raise SystemExit(f"no run summaries found under {logs_root}")

    rows = load_rows()
    stored = load_evaluations(logs_root)
    selected = [
        summary
        for summary in summaries
        if (not args.run_id or summary.get("run_id") == args.run_id)
        and (args.force or summary.get("run_id") not in stored)
    ]
    if not selected:
        raise SystemExit(
            "Nothing to evaluate. Use --force to re-grade runs that already "
            "have a stored evaluation."
        )

    print(f"Evaluating {len(selected)} run(s) from stored patches", flush=True)
    for index, summary in enumerate(selected, start=1):
        run_id = summary["run_id"]
        task = summary.get("task", {})
        instance_id = task.get("instance_id")
        condition = task.get("condition")
        row = rows.get(instance_id)
        if row is None:
            print(f"  [{index}] {run_id}: unknown instance {instance_id}", flush=True)
            continue

        patch_file = logs_root / "patches" / f"{run_id}.patch"
        patch = patch_file.read_text(encoding="utf-8") if patch_file.exists() else ""

        workspace = prepare_workspace(row, condition)
        evaluation = evaluate_patch(
            row=row,
            workspace=workspace,
            run_git=run_git,
            patch=patch,
            patch_path=str(patch_file) if patch_file.exists() else None,
            timeout=args.eval_timeout,
        )
        write_evaluation(logs_root, run_id, evaluation)
        print(
            f"  [{index}/{len(selected)}] {instance_id} ({condition}): "
            f"status={evaluation['status']} resolved={evaluation['resolved']} "
            f"F2P={evaluation['f2p_passed']}/{evaluation['f2p_total']} "
            f"P2P={evaluation['p2p_passed']}/{evaluation['p2p_total']}",
            flush=True,
        )

    print("\nRun `experiment.py report` to regenerate the aggregates.", flush=True)


def command_report(args: argparse.Namespace) -> None:
    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()
    paths = write_report(logs_root)
    print("Wrote AskUserQuestion report:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


def command_batch(args: argparse.Namespace) -> None:
    # Scope narrows the candidate pool before selection, so --count N always
    # means N instances within the scope rather than N of the whole dataset.
    rows = scoped_rows(list(load_rows().values()), args.scope)
    if not rows:
        raise SystemExit(f"no dataset instances match --scope {args.scope}")
    if args.count <= 0 or args.count > len(rows):
        raise SystemExit(
            f"--count must be between 1 and {len(rows)} for --scope {args.scope}"
        )

    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()
    conditions = requested_conditions(args.condition)
    selected = select_batch_rows(
        rows,
        completed_run_keys(logs_root),
        conditions,
        args.model,
        args.count,
    )
    if not selected:
        raise SystemExit(
            "No incomplete instances remain for the requested condition and model."
        )

    session_count = sum(len(missing) for _, missing in selected)
    print(
        f"Batch: {len(selected)} dataset instances, {session_count} Claude session(s), "
        f"condition={args.condition}, model={args.model}, scope={args.scope} "
        f"({len(rows)} instances in scope)",
        flush=True,
    )
    for index, (row, missing) in enumerate(selected, start=1):
        try:
            for condition in missing:
                print(
                    f"\n=== Batch item {index}/{len(selected)}: "
                    f"{row['instance_id']} ({condition}) ===\n",
                    flush=True,
                )
                command_run(
                    argparse.Namespace(
                        instance_id=row["instance_id"],
                        condition=condition,
                        model=args.model,
                        dry_run=args.dry_run,
                        logs_dir=args.logs_dir,
                        eval_timeout=args.eval_timeout,
                        no_eval=args.no_eval,
                    )
                )
        except KeyboardInterrupt:
            raise SystemExit("Batch interrupted; completed runs are resumable.") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list", help="list available dataset instances")
    ls.add_argument("--limit", type=int, default=20)
    ls.add_argument("--repo", help="optional exact owner/name repository filter")
    ls.add_argument(
        "--scope",
        choices=SCOPES,
        default=DEFAULT_SCOPE,
        help=SCOPE_HELP,
    )
    ls.set_defaults(func=command_list)

    run = sub.add_parser("run", help="launch one unattended Claude session")
    run.add_argument("instance_id")
    run.add_argument(
        "--condition",
        choices=tuple(CONDITION_FIELD),
        default="ambiguous",
    )
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--logs-dir",
        help="directory for Git-ignored run manifests, transcripts, and summaries",
    )
    run.add_argument(
        "--eval-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="seconds allowed for the graded pytest run before it is recorded as a timeout",
    )
    run.add_argument(
        "--no-eval",
        action="store_true",
        help="save the agent patch but skip grading; grade later with `evaluate`",
    )
    run.set_defaults(func=command_run)

    batch = sub.add_parser(
        "batch",
        help="run the next incomplete dataset instances sequentially",
    )
    batch.add_argument(
        "--count",
        type=int,
        required=True,
        help="number of instances to run, counted within --scope",
    )
    batch.add_argument(
        "--condition",
        choices=BATCH_CONDITIONS,
        default="ambiguous",
        help="run ambiguous, full, or both conditions for each selected instance",
    )
    batch.add_argument("--model", default=DEFAULT_MODEL)
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument(
        "--logs-dir",
        help="directory whose run summaries define resume state",
    )
    batch.add_argument(
        "--scope",
        choices=SCOPES,
        default=DEFAULT_SCOPE,
        help=SCOPE_HELP,
    )
    batch.add_argument(
        "--eval-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="seconds allowed for each graded pytest run before it is recorded as a timeout",
    )
    batch.add_argument(
        "--no-eval",
        action="store_true",
        help="save agent patches but skip grading; grade later with `evaluate`",
    )
    batch.set_defaults(func=command_batch)

    evaluate = sub.add_parser(
        "evaluate",
        help="re-grade stored agent patches without re-running any session",
    )
    evaluate.add_argument(
        "--logs-dir",
        help="log directory holding the runs and patches to grade",
    )
    evaluate.add_argument(
        "--run-id",
        help="grade only this run (default: every run without a stored evaluation)",
    )
    evaluate.add_argument(
        "--force",
        action="store_true",
        help="re-grade runs that already have a stored evaluation",
    )
    evaluate.add_argument(
        "--eval-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="seconds allowed for each graded pytest run",
    )
    evaluate.set_defaults(func=command_evaluate)

    report = sub.add_parser("report", help="aggregate AskUserQuestion run logs")
    report.add_argument("--logs-dir", help="log directory created by the run command")
    report.set_defaults(func=command_report)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
