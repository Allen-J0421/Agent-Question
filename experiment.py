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
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import load_from_disk

import swebench_eval
from sdk_runner import PERMISSION_MODE, load_reference_toolset, run_sdk_session
from swebench_eval import capture_agent_patch
from study_log import (
    build_run_summary_sdk,
    create_run_manifest,
    default_logs_root,
    load_evaluations,
    load_run_summaries,
    preserve_session_artifacts,
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
    rows = list(load_rows().values())
    if args.repo:
        rows = [row for row in rows if row["repo"] == args.repo]
    print(f"# {len(rows)} instance(s)", flush=True)
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
        # A session that errored out or never did meaningful work observed
        # nothing about the ask decision (`direct_asked` is None), so it must
        # not block a re-run. Usage-limit rejections land here: they exit in
        # one turn at $0 cost, and treating them as "completed" would bury the
        # instance forever.
        if summary.get("session", {}).get("ran_meaningfully") is False:
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


# The canary must force one real AskUserQuestion round-trip through the same
# SDK configuration the experiment uses. Ten runs of the 2026-07-30 batch all
# recorded zero asks; without this check there is no way to tell "the agent
# chose not to ask" (the study's outcome) from "asking was structurally
# impossible" (a broken measurement channel).
PREFLIGHT_PROMPT = (
    "This is an instrumentation check of the question-asking channel, not a "
    "real task. Call the AskUserQuestion tool exactly once: ask a single "
    "question with exactly two options about which word this check should "
    "print. After you receive an answer, reply with the single word DONE. "
    "Do not use any other tools and do not create any files."
)


def run_preflight(model: str) -> None:
    """Verify a forced AskUserQuestion call is asked *and answered* end-to-end.

    Runs one short SDK session with the experiment's exact toolset and
    permission mode in a throwaway directory, and fails loudly if the ask
    channel does not round-trip. A passing preflight certifies that a zero-ask
    result in the following batch is agent behavior, not harness breakage.
    """
    print(f"Preflight: forcing one AskUserQuestion round-trip on {model}...", flush=True)
    with tempfile.TemporaryDirectory(prefix="ambig-swe-preflight-") as tmp:
        observation = asyncio.run(
            run_sdk_session(
                prompt=PREFLIGHT_PROMPT,
                workspace=Path(tmp),
                model=model,
                tools=load_reference_toolset(),
            )
        )

    result = observation.get("result") or {}
    analysis = observation["analysis"]
    answered = observation.get("answered_questions") or []

    problems: list[str] = []
    if not observation.get("askuserquestion_available"):
        problems.append("AskUserQuestion is missing from the live tool roster")
    if result.get("is_error"):
        problems.append(
            "session errored (subtype="
            f"{result.get('subtype')}): {str(result.get('result'))[:300]}"
        )
    if analysis["direct_count"] < 1:
        problems.append("the model never called AskUserQuestion")
    if not answered:
        problems.append("no synthetic answer was recorded; the ask channel did not round-trip")

    if problems:
        raise SystemExit(
            "Preflight FAILED — ask-rate measurements from this configuration "
            "cannot be trusted:\n  - " + "\n  - ".join(problems)
        )

    first = analysis["first_direct"] or {}
    latency = first.get("latency_seconds")
    cost = result.get("total_cost_usd") or 0
    print(
        "Preflight PASSED: AskUserQuestion was asked and answered "
        f"(latency {latency:.1f}s, {len(answered)} answer(s), cost ${cost:.4f}).",
        flush=True,
    )


def command_preflight(args: argparse.Namespace) -> None:
    run_preflight(args.model)


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

    # Capture only: the patch is saved and the workspace reset. Grading
    # happens later in the official SWE-bench harness (`evaluate`), which
    # provisions the per-instance environment this machine cannot.
    captured = capture_agent_patch(
        workspace=workspace,
        run_git=run_git,
        logs_root=logs_root,
        run_id=manifest["run_id"],
    )
    print(
        f"Saved agent patch ({captured['patch_bytes']} bytes); grade later "
        "with `experiment.py evaluate` (official SWE-bench harness).",
        flush=True,
    )
    # Preserve the raw session alongside the patch: Claude Code prunes its
    # own copies under ~/.claude/projects on a retention schedule, and this
    # run's trace should outlive that. Never let preservation kill a run
    # whose summary has not been written yet.
    try:
        preserved = preserve_session_artifacts(
            logs_root,
            run_id=manifest["run_id"],
            workspace=workspace,
            started_at=manifest["started_at"],
            session_id=(observation.get("result") or {}).get("session_id"),
        )
        if preserved["copied"]:
            print(
                f"Preserved {len(preserved['copied'])} raw session file(s) in "
                f"{preserved['sessions_dir']}; agent-only transcripts in "
                f"{preserved['transcripts_dir']}.",
                flush=True,
            )
        else:
            print("WARNING: found no session files to preserve.", flush=True)
    except OSError as error:
        print(f"WARNING: session preservation failed: {error}", flush=True)
    evaluation = {"status": swebench_eval.STATUS_NOT_EVALUATED, "resolved": None}
    summary = build_run_summary_sdk(manifest, observation, evaluation)
    summary_path = write_run_summary(logs_root, summary)
    write_evaluation(logs_root, manifest["run_id"], evaluation)
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


def command_evaluate(args: argparse.Namespace) -> None:
    """Grade stored agent patches without re-running any Claude session.

    Grading is a judgement about a run, not part of it, so changing the rules
    should never cost a batch of sessions. All grading happens in the official
    SWE-bench harness: Docker, per-instance environments, every repository's
    own runner — all 500 instances are gradable.
    """
    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()
    summaries, errors = load_run_summaries(logs_root)
    for message in errors:
        print(f"WARNING: unreadable run summary: {message}", flush=True)
    if not summaries:
        raise SystemExit(f"no run summaries found under {logs_root}")

    rows = load_rows()
    stored = load_evaluations(logs_root)

    def is_graded(run_id: str | None) -> bool:
        # Runs write a `not_evaluated` placeholder at capture time; only a
        # real grade counts as "already evaluated".
        record = stored.get(run_id)
        return (
            record is not None
            and record.get("status") != swebench_eval.STATUS_NOT_EVALUATED
        )

    selected = [
        summary
        for summary in summaries
        if (not args.run_id or summary.get("run_id") == args.run_id)
        and (args.force or not is_graded(summary.get("run_id")))
    ]
    for summary in selected:
        instance_id = summary.get("task", {}).get("instance_id")
        if instance_id not in rows:
            print(
                f"WARNING: {summary.get('run_id')}: unknown instance "
                f"{instance_id}; skipped",
                flush=True,
            )
    selected = [
        summary
        for summary in selected
        if summary.get("task", {}).get("instance_id") in rows
    ]
    if not selected:
        raise SystemExit(
            "Nothing to evaluate. Use --force to re-grade runs that already "
            "have a stored evaluation."
        )

    ready, reason = swebench_eval.docker_ready()
    if not ready:
        raise SystemExit(
            "The official SWE-bench harness needs a running Docker daemon: "
            f"{reason}\nStart Docker Desktop and re-run."
        )
    print(
        f"Evaluating {len(selected)} run(s) with the official SWE-bench "
        f"harness (dataset {swebench_eval.DATASET_NAME})",
        flush=True,
    )
    evaluations = swebench_eval.evaluate_with_harness(
        logs_root=logs_root,
        rows=rows,
        summaries=selected,
        max_workers=args.max_workers,
        timeout=args.eval_timeout,
        namespace=args.swebench_namespace or None,
    )
    by_run = {summary["run_id"]: summary for summary in selected}
    for run_id, evaluation in evaluations.items():
        write_evaluation(logs_root, run_id, evaluation)
        task = by_run[run_id].get("task", {})
        print(
            f"  {task.get('instance_id')} ({task.get('condition')}): "
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
        f"condition={args.condition}, model={args.model} "
        f"({len(rows)} instances in the dataset)",
        flush=True,
    )
    # Certify the ask channel before spending a batch of sessions on it.
    if not args.dry_run and not args.skip_preflight:
        run_preflight(args.model)
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
        "--skip-preflight",
        action="store_true",
        help=(
            "skip the AskUserQuestion preflight canary that certifies the ask "
            "channel before the batch spends any sessions"
        ),
    )
    batch.set_defaults(func=command_batch)

    evaluate = sub.add_parser(
        "evaluate",
        help="grade stored agent patches without re-running any session",
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
        "--max-workers",
        type=int,
        default=swebench_eval.DEFAULT_MAX_WORKERS,
        help="parallel containers for the swebench harness",
    )
    evaluate.add_argument(
        "--swebench-namespace",
        default=swebench_eval.DEFAULT_NAMESPACE,
        help=(
            "Docker Hub namespace for prebuilt per-instance images; pass '' "
            "to build images locally instead of pulling"
        ),
    )
    evaluate.add_argument(
        "--eval-timeout",
        type=int,
        default=swebench_eval.DEFAULT_TIMEOUT,
        help="seconds allowed per graded instance",
    )
    evaluate.set_defaults(func=command_evaluate)

    preflight = sub.add_parser(
        "preflight",
        help="certify the AskUserQuestion channel with one forced round-trip",
    )
    preflight.add_argument("--model", default=DEFAULT_MODEL)
    preflight.set_defaults(func=command_preflight)

    report = sub.add_parser("report", help="aggregate AskUserQuestion run logs")
    report.add_argument("--logs-dir", help="log directory created by the run command")
    report.set_defaults(func=command_report)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
