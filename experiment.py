#!/usr/bin/env python3
"""Launch unattended Claude Code experiments on local dataset issues.

The launcher uses Claude's non-interactive print mode and bypasses its permission
prompts. Its only task-specific behavioral inputs are the selected issue text and the
requested model.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from sdk_runner import load_reference_toolset, run_sdk_session
from study_log import (
    build_run_summary,
    build_run_summary_sdk,
    create_run_manifest,
    default_logs_root,
    load_run_summaries,
    observe_headless_session,
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


@lru_cache(maxsize=1)
def load_rows() -> dict[str, dict[str, Any]]:
    split = load_from_disk(str(DATASET))["test"]
    return {split[i]["instance_id"]: dict(split[i]) for i in range(len(split))}


def issue_text(row: dict[str, Any], condition: str) -> str:
    """Return exactly the issue field assigned to the requested condition."""
    field = CONDITION_FIELD[condition]
    text = (row.get(field) or "").strip()
    if not text:
        raise ValueError(f"{row['instance_id']} has an empty {field}")
    return text


def build_prompt(row: dict[str, Any], condition: str) -> str:
    return f"Resolve the following issue in this repository:\n\n{issue_text(row, condition)}"


def claude_argv(claude_bin: str, model: str, prompt: str) -> list[str]:
    """The non-interactive Claude command used for unattended experiments."""
    return [
        claude_bin,
        "--model",
        model,
        "-p",
        "--permission-mode",
        "bypassPermissions",
        prompt,
    ]


def run_git(args: list[str], cwd: Path | None = None, check: bool = True):
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=check,
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
    rows = list(load_rows().values())
    if args.repo:
        rows = [row for row in rows if row["repo"] == args.repo]
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
    interface = getattr(args, "interface", "sdk")

    claude_bin = None
    if interface == "cli":
        claude_bin = shutil.which("claude")
        if not claude_bin:
            raise SystemExit("claude CLI not found on PATH")

    print(f"Instance:  {row['instance_id']}")
    print(f"Repository:{row['repo']}")
    print(f"Condition: {args.condition}")
    print(f"Field:     {field}")
    print(f"Model:     {args.model}")
    print(f"Interface: {interface}")
    print("\n--- EXACT CLAUDE PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---\n")

    if args.dry_run:
        print("Dry run: Claude was not launched.")
        if interface == "sdk":
            print(
                "Command shape: Agent SDK session, permission_mode=bypassPermissions, "
                "tools=<config/reference_toolset.json>, can_use_tool callback registered "
                "(required for AskUserQuestion to appear in the roster)."
            )
        else:
            print(
                "Command shape: claude --model",
                args.model,
                "-p --permission-mode bypassPermissions \"<prompt>\"",
            )
        return

    workspace = prepare_workspace(row, args.condition)
    print(f"Launching unattended Claude Code ({interface}) in {workspace}", flush=True)
    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()

    if interface == "sdk":
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
        summary = build_run_summary_sdk(manifest, observation)
        summary_path = write_run_summary(logs_root, summary)
        print(f"Run log: {summary_path}", flush=True)
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
        return

    manifest = create_run_manifest(
        logs_root,
        row=row,
        condition=args.condition,
        model=args.model,
        claude_bin=claude_bin,
        workspace=workspace,
        prompt=prompt,
        interface="cli",
    )
    argv = claude_argv(claude_bin, args.model, prompt)
    try:
        observation = observe_headless_session(
            argv,
            workspace,
            manifest["started_at"],
            output_dir=logs_root / "process-output" / manifest["run_id"],
        )
    except OSError as exc:
        observation = {
            "exit_code": None,
            "stop_reason": "launch_error",
            "operator_interrupted": False,
            "analysis": {
                "paths": [],
                "parse_errors": 0,
                "valid_records": 0,
                "first_direct": None,
                "direct_count": 0,
                "any_agent_count": 0,
            },
        }
        summary = build_run_summary(manifest, observation, logs_root)
        summary_path = write_run_summary(logs_root, summary)
        raise SystemExit(f"could not launch Claude: {exc}\nRun log: {summary_path}") from exc
    summary = build_run_summary(manifest, observation, logs_root)
    summary_path = write_run_summary(logs_root, summary)
    print(f"Run log: {summary_path}", flush=True)
    if observation["operator_interrupted"]:
        raise KeyboardInterrupt


def command_report(args: argparse.Namespace) -> None:
    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()
    paths = write_report(logs_root)
    print("Wrote AskUserQuestion report:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


def command_batch(args: argparse.Namespace) -> None:
    rows = list(load_rows().values())
    if args.count <= 0 or args.count > len(rows):
        raise SystemExit(f"--count must be between 1 and {len(rows)}")

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
        f"condition={args.condition}, model={args.model}",
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
                        interface=args.interface,
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
        "--interface",
        choices=("sdk", "cli"),
        default="sdk",
        help=(
            "sdk (default): Agent SDK session with a can_use_tool callback, "
            "required for AskUserQuestion to appear in the tool roster at all. "
            "cli: subprocess `claude -p`, which has no callback mechanism and "
            "cannot resolve an AskUserQuestion call -- kept only for comparison."
        ),
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
        help="number of dataset instances to run (1 through the dataset size)",
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
        "--interface",
        choices=("sdk", "cli"),
        default="sdk",
        help="see `run --interface` for the difference; applies to every session in the batch",
    )
    batch.set_defaults(func=command_batch)

    report = sub.add_parser("report", help="aggregate AskUserQuestion run logs")
    report.add_argument("--logs-dir", help="log directory created by the run command")
    report.set_defaults(func=command_report)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
