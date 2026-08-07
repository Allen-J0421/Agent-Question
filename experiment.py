#!/usr/bin/env python3
"""Launch unattended coding-agent experiments on local dataset issues.

One ``--model`` switch selects the study arm; the runner is inferred from the
model name:

* ``claude-*``  -> Claude Agent SDK session in ``default`` permission mode,
  where tool calls that would prompt reach the study's ``can_use_tool``
  callback and are recorded before being approved.
* ``gpt-*`` / ``codex-*`` -> stock ``codex exec`` CLI session (vanilla
  toolset, user config ignored); the ask channel is the model's own
  final-message turn yield, since vanilla Codex has no question tool.

Either way the only task-specific behavioral inputs are the selected issue
text and the requested model; nothing tells the agent to ask.
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import codex_runner
import datasets_registry
import swebench_eval
from codex_runner import require_supported_cli, run_codex_session
from sdk_runner import PERMISSION_MODE, load_reference_toolset, run_sdk_session
from swebench_eval import capture_agent_patch
from study_log import (
    agent_info,
    dataset_of as study_log_dataset_of,
    build_run_summary_codex,
    build_run_summary_sdk,
    create_run_manifest,
    default_logs_root,
    load_evaluations,
    load_run_summaries,
    preserve_codex_session_artifacts,
    preserve_session_artifacts,
    write_evaluation,
    write_run_summary,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = datasets_registry.DEFAULT_DATASET
DATASETS = datasets_registry.DATASETS
CHECKOUTS = Path.cwd() / ".experiment-checkouts"
DEFAULT_MODEL = "claude-opus-4-8"

RUNNER_CLAUDE = "claude-sdk"
RUNNER_CODEX = "codex-cli"

MODEL_HELP = (
    "model and study arm: claude-* runs through the Claude Agent SDK "
    f"(default: {DEFAULT_MODEL}); gpt-*/codex-* runs through the Codex CLI "
    f"(primary: {codex_runner.PRIMARY_GPT_MODEL}; also verified: "
    f"{', '.join(codex_runner.KNOWN_GPT_MODELS)})"
)

DATASET_HELP = (
    f"issue source (default: {DEFAULT_DATASET}); 'interactive-swe' pairs "
    "ambiguous/full, 'missing-info' pairs mi_ambiguous/mi_full, whose "
    "ambiguous text hides annotated categories of information"
)

CONDITION_HELP = (
    "ambiguous/full read the interactive-swe dataset; mi_ambiguous/mi_full "
    "read the missing-info workbook"
)

# Each condition names exactly one issue field, so a condition can never
# widen what the agent sees. The `mi_*` conditions belong to the
# missing-info workbook, whose ambiguous text was produced by masking
# annotated categories of information; they are named distinctly because both
# datasets cover the same 500 instance_ids, and identical condition names
# would conflate their runs in resume state and every aggregate report.
CONDITION_FIELD = {
    "ambiguous": "problem_statement",
    "full": "original_issue",
    "mi_ambiguous": "rewrite_3",
    "mi_full": "original_issue",
}
BATCH_CONDITIONS = (*CONDITION_FIELD, "both")


def runner_for_model(model: str) -> str:
    """Infer the study arm's runner from the model name.

    Any ``gpt-*``/``codex-*`` slug is accepted (the Codex CLI validates it
    server-side and the preflight canary certifies the ask channel before a
    batch spends sessions), so new GPT models work without a code change;
    ``codex_runner.PRIMARY_GPT_MODEL`` is the arm this study targets first.
    """
    if model.startswith("claude"):
        return RUNNER_CLAUDE
    if model.startswith(("gpt", "codex")):
        return RUNNER_CODEX
    raise SystemExit(
        f"cannot infer a runner from model {model!r}: expected claude-* "
        "(Claude Agent SDK) or gpt-*/codex-* (Codex CLI), e.g. "
        f"{DEFAULT_MODEL} or {codex_runner.PRIMARY_GPT_MODEL}"
    )


def load_rows(dataset: str = DEFAULT_DATASET) -> dict[str, dict[str, Any]]:
    """Return ``{instance_id: row}`` with evaluator-only fields already removed.

    ``datasets_registry`` normalizes both datasets onto one schema and strips
    the masking answer keys, so nothing here can route them into a prompt.
    """
    return datasets_registry.load_dataset_rows(dataset)


def resolve_conditions(dataset: str, conditions: tuple[str, ...]) -> tuple[str, ...]:
    """Reject conditions whose issue field the requested dataset does not have.

    Without this, ``--dataset missing-info --condition ambiguous`` would read a
    ``problem_statement`` that exists in the workbook but is a duplicate of
    ``rewrite_3``, quietly recording the run under the wrong condition name.
    """
    valid = datasets_registry.conditions_for(dataset)
    invalid = [condition for condition in conditions if condition not in valid]
    if invalid:
        raise SystemExit(
            f"condition {', '.join(invalid)} is not available for dataset "
            f"{dataset}; use {' or '.join(valid)}"
        )
    return conditions


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
    rows = list(load_rows(args.dataset).values())
    if args.repo:
        rows = [row for row in rows if row["repo"] == args.repo]
    print(f"# {len(rows)} instance(s) in {args.dataset}", flush=True)
    for row in rows[: args.limit]:
        print(f"{row['instance_id']}\t{row['repo']}\t{row['difficulty']}")


def requested_conditions(condition: str, dataset: str = DEFAULT_DATASET) -> tuple[str, ...]:
    """Expand the batch-only ``both`` option into individual experiment conditions.

    ``both`` means both conditions *of the selected dataset*, never the full
    four-entry table.
    """
    if condition == "both":
        return datasets_registry.conditions_for(dataset)
    return (condition,)


def dataset_of(summary: dict[str, Any]) -> str:
    """Return the dataset a stored run was launched from, for grading.

    ``study_log.dataset_of`` does the record reading (including the fallback
    for runs written before the field existed); this narrows the result to a
    dataset the harness can actually load.
    """
    recorded = study_log_dataset_of(summary)
    if recorded in DATASETS:
        return recorded
    return DEFAULT_DATASET


def completed_run_keys(logs_root: Path) -> set[tuple[str, str, str]]:
    """Return completed (instance, condition, model) runs from immutable summaries.

    The key stays ``(instance, condition, model)`` across both datasets because
    the ``mi_*`` condition names are distinct, so a missing-info run never
    marks the matching interactive-swe run complete.
    """
    summaries, _ = load_run_summaries(logs_root)
    completed: set[tuple[str, str, str]] = set()
    for summary in summaries:
        task = summary.get("task", {})
        process = summary.get("process", {})
        instance_id = task.get("instance_id")
        condition = task.get("condition")
        # agent_info reads the multi-runner `agent` key and falls back to
        # the legacy `claude` key, so pre-existing runs still count as
        # completed for their model while other models remain runnable.
        model = agent_info(summary).get("model")
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


# The canary must force one real ask round-trip through the same
# configuration the experiment uses. Ten runs of the 2026-07-30 batch all
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

# The Codex arm's canary forces the same round-trip through that arm's
# natural ask channel: end the turn with a clarifying question, receive the
# synthetic answer via `codex exec resume`, and finish. A pass certifies the
# CLI version, the ask classifier, and the resume mechanics for the exact
# model the batch will use.
CODEX_PREFLIGHT_PROMPT = (
    "This is an instrumentation check of the question-asking channel, not a "
    "real task. Before doing anything else, ask me exactly one short "
    "clarifying question: should this check print ALPHA or BETA? End your "
    "turn with that question and wait for my answer. Do not use any tools "
    "and do not create any files. After I answer, reply with the single "
    "word DONE."
)


def run_preflight(model: str) -> None:
    """Certify the model's ask channel with one forced round-trip."""
    if runner_for_model(model) == RUNNER_CODEX:
        run_preflight_codex(model)
    else:
        run_preflight_claude(model)


def run_preflight_claude(model: str) -> None:
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


def run_preflight_codex(model: str) -> None:
    """Verify the Codex arm's final-message ask channel round-trips.

    One short ``codex exec`` session in a throwaway directory must (1) end
    its first turn with a question the deterministic classifier detects,
    (2) receive the synthetic neutral answer via ``codex exec resume``, and
    (3) complete a second turn. Failing any step aborts the batch: a zero-ask
    result is only meaningful when asking was demonstrably possible for this
    exact CLI version and model.
    """
    cli_version = require_supported_cli(model)
    print(
        f"Preflight: forcing one final-message ask round-trip on {model} "
        f"({cli_version})...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="ambig-swe-preflight-") as tmp:
        observation = run_codex_session(
            prompt=CODEX_PREFLIGHT_PROMPT,
            workspace=Path(tmp),
            model=model,
            events_dir=Path(tmp) / "events",
        )

    result = observation.get("result") or {}
    analysis = observation["analysis"]
    answered = observation.get("answered_questions") or []

    problems: list[str] = []
    if result.get("is_error"):
        problems.append(
            f"session errored ({result.get('stop_reason')}): "
            f"{str(result.get('result'))[:300]}"
        )
    if not observation.get("thread_id"):
        problems.append("no thread.started event; resume would be impossible")
    if analysis["direct_count"] < 1:
        problems.append(
            "the forced question was not detected on the final-message channel"
        )
    if not answered:
        problems.append("no synthetic answer was recorded; the ask channel did not round-trip")
    if (result.get("num_turns") or 0) < 2:
        problems.append("the session did not continue after the synthetic answer")

    if problems:
        raise SystemExit(
            "Preflight FAILED — ask-rate measurements from this configuration "
            "cannot be trusted:\n  - " + "\n  - ".join(problems)
        )

    first = analysis["first_direct"] or {}
    latency = first.get("latency_seconds")
    tokens = (result.get("usage") or {}).get("output_tokens")
    print(
        "Preflight PASSED: the final-message ask round-tripped "
        f"(latency {latency:.1f}s, {len(answered)} answer(s), "
        f"{result.get('num_turns')} rounds, {tokens} output tokens).",
        flush=True,
    )


def command_preflight(args: argparse.Namespace) -> None:
    run_preflight(args.model)


def command_run(args: argparse.Namespace) -> None:
    resolve_conditions(args.dataset, (args.condition,))
    rows = load_rows(args.dataset)
    if args.instance_id not in rows:
        raise SystemExit(f"unknown instance_id: {args.instance_id}")

    runner = runner_for_model(args.model)
    row = rows[args.instance_id]
    field = CONDITION_FIELD[args.condition]
    prompt = build_prompt(row, args.condition)

    print(f"Instance:  {row['instance_id']}")
    print(f"Repository:{row['repo']}")
    print(f"Dataset:   {args.dataset}")
    print(f"Condition: {args.condition}")
    print(f"Field:     {field}")
    print(f"Model:     {args.model}")
    print(f"Runner:    {runner}")
    print("\n--- EXACT AGENT PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---\n")

    if args.dry_run:
        print("Dry run: the agent was not launched.")
        if runner == RUNNER_CODEX:
            print(
                "Command shape: "
                + " ".join(codex_runner.exec_argv(args.model, "<prompt>"))
                + f"; ask channel = final-message question (classifier v"
                f"{codex_runner.ASK_CLASSIFIER_VERSION}); clarifying questions "
                "answered neutrally via `codex exec resume <thread_id>` "
                f"(max {codex_runner.MAX_ASK_ROUNDS} answers)."
            )
        else:
            print(
                f"Command shape: Agent SDK session, permission_mode={PERMISSION_MODE}, "
                "tools=<config/reference_toolset.json>, can_use_tool callback registered "
                "(observes and approves prompting tool calls; logs AskUserQuestion in full)."
            )
        return

    if runner == RUNNER_CODEX:
        # Fail before any artifact is written: an old CLI is rejected
        # server-side for gpt-5.6 models and would burn a dead run per
        # instance.
        cli_version = require_supported_cli(args.model)

    workspace = prepare_workspace(row, args.condition)
    print(f"Launching unattended agent ({runner}) in {workspace}", flush=True)
    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()

    if runner == RUNNER_CODEX:
        manifest = create_run_manifest(
            logs_root,
            row=row,
            dataset=args.dataset,
            condition=args.condition,
            model=args.model,
            workspace=workspace,
            prompt=prompt,
            runner=RUNNER_CODEX,
            runner_details={
                "cli_version": cli_version,
                "sandbox": codex_runner.SANDBOX_MODE,
                "argv_shape": codex_runner.exec_argv(args.model, "<prompt>"),
            },
        )
        observation = run_codex_session(
            prompt=prompt,
            workspace=workspace,
            model=args.model,
            # Stream every round's raw events straight into the run's
            # sessions/ slot so they exist even if capture dies mid-run.
            events_dir=logs_root / "sessions" / manifest["run_id"],
        )
    else:
        manifest = create_run_manifest(
            logs_root,
            row=row,
            dataset=args.dataset,
            condition=args.condition,
            model=args.model,
            workspace=workspace,
            prompt=prompt,
            runner=RUNNER_CLAUDE,
            runner_details={"interface": "sdk"},
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
    # Preserve the raw session alongside the patch: both harnesses prune
    # their own session records on a retention schedule (~/.claude/projects,
    # $CODEX_HOME/sessions), and this run's trace should outlive that. Never
    # let preservation kill a run whose summary has not been written yet.
    try:
        if runner == RUNNER_CODEX:
            preserved = preserve_codex_session_artifacts(
                logs_root,
                run_id=manifest["run_id"],
                thread_id=observation.get("thread_id"),
                rounds=observation.get("rounds"),
            )
        else:
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
    if runner == RUNNER_CODEX:
        summary = build_run_summary_codex(manifest, observation, evaluation)
    else:
        summary = build_run_summary_sdk(manifest, observation, evaluation)
    summary_path = write_run_summary(logs_root, summary)
    write_evaluation(logs_root, manifest["run_id"], evaluation)
    print(f"Run log: {summary_path}", flush=True)
    if runner == RUNNER_CODEX:
        ask = summary["ask_user_question"]
        print(
            f"Ask channel this run: final-message (classifier v"
            f"{ask.get('classifier_version')}); asked={ask.get('direct_asked')}",
            flush=True,
        )
    else:
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

    # Runs from both datasets land in one log directory and share
    # instance_ids, so each run must be graded against the rows of the dataset
    # it was actually launched from rather than a merged table. The oracle
    # columns agree across datasets, but only the missing-info rows carry the
    # PASS_TO_PASS repaired from Excel truncation.
    by_dataset = {dataset: load_rows(dataset) for dataset in DATASETS}

    def rows_for(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return by_dataset[dataset_of(summary)]

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
        if instance_id not in rows_for(summary):
            print(
                f"WARNING: {summary.get('run_id')}: unknown instance "
                f"{instance_id} in dataset {dataset_of(summary)}; skipped",
                flush=True,
            )
    selected = [
        summary
        for summary in selected
        if summary.get("task", {}).get("instance_id") in rows_for(summary)
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
    # Build the grading table from each selected run's own dataset, so a run
    # is never graded against another dataset's copy of its instance_id.
    rows = {
        summary["task"]["instance_id"]: rows_for(summary)[summary["task"]["instance_id"]]
        for summary in selected
    }
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
    print("\nRun `dashboard.py` to regenerate index.html with these grades.", flush=True)


def command_batch(args: argparse.Namespace) -> None:
    conditions = resolve_conditions(
        args.dataset, requested_conditions(args.condition, args.dataset)
    )
    rows = list(load_rows(args.dataset).values())

    # The missing-info workbook leaves four instances unannotated, so they have
    # no ambiguous rewrite to present. Drop them with a visible count rather
    # than failing the batch partway through.
    runnable = [
        row
        for row in rows
        if all((row.get(CONDITION_FIELD[c]) or "").strip() for c in conditions)
    ]
    if len(runnable) != len(rows):
        print(
            f"Skipping {len(rows) - len(runnable)} instance(s) with no text for "
            f"condition {'/'.join(conditions)} in {args.dataset}",
            flush=True,
        )
    rows = runnable

    if args.count <= 0 or args.count > len(rows):
        raise SystemExit(f"--count must be between 1 and {len(rows)}")

    logs_root = Path(args.logs_dir) if args.logs_dir else default_logs_root()
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
        f"Batch: {len(selected)} dataset instances, {session_count} session(s), "
        f"dataset={args.dataset}, condition={args.condition}, model={args.model} "
        f"(runner={runner_for_model(args.model)}, "
        f"{len(rows)} runnable instances in the dataset)",
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
                        dataset=args.dataset,
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
    ls.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET, help=DATASET_HELP)
    ls.set_defaults(func=command_list)

    run = sub.add_parser("run", help="launch one unattended agent session")
    run.add_argument("instance_id")
    run.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET, help=DATASET_HELP)
    run.add_argument(
        "--condition",
        choices=tuple(CONDITION_FIELD),
        default="ambiguous",
        help=CONDITION_HELP,
    )
    run.add_argument("--model", default=DEFAULT_MODEL, help=MODEL_HELP)
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
        "--dataset", choices=DATASETS, default=DEFAULT_DATASET, help=DATASET_HELP
    )
    batch.add_argument(
        "--condition",
        choices=BATCH_CONDITIONS,
        default="ambiguous",
        help=(
            "condition to run for each selected instance, or 'both' for the "
            f"selected dataset's pair. {CONDITION_HELP}"
        ),
    )
    batch.add_argument("--model", default=DEFAULT_MODEL, help=MODEL_HELP)
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument(
        "--logs-dir",
        help="directory whose run summaries define resume state",
    )
    batch.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "skip the preflight canary that certifies the selected model's "
            "ask channel before the batch spends any sessions"
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
        help=(
            "certify the selected model's ask channel with one forced "
            "round-trip (AskUserQuestion for claude-*, final-message ask "
            "via codex exec resume for gpt-*)"
        ),
    )
    preflight.add_argument("--model", default=DEFAULT_MODEL, help=MODEL_HELP)
    preflight.set_defaults(func=command_preflight)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
