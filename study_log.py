"""Capture spontaneous user-directed questions across study arms.

Manifests, run summaries, session preservation, and the aggregate builder for
both runners: Claude Agent SDK sessions (ask channel: the AskUserQuestion
tool) and Codex CLI sessions (ask channel: a turn-ending clarifying question
in the final message). This module only observes each harness's own records;
it never alters an agent's prompt, tools, permissions, or output mode.

``build_report`` keeps models apart -- each model is its own study arm, and
their ask channels differ by construction, so nothing is ever pooled across
them. It returns the aggregates; rendering them is ``dashboard.py``'s job.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


SCHEMA_VERSION = 1
POLL_SECONDS = 0.25
INTERRUPT_GRACE_SECONDS = 10.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_logs_root() -> Path:
    return Path.cwd() / ".experiment-logs"


def default_projects_dir() -> Path:
    return Path.home() / ".claude" / "projects"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def agent_info(record: dict[str, Any]) -> dict[str, Any]:
    """Return the runner/model identity of a manifest or run summary.

    Multi-runner records store it under ``agent`` (``{"model", "runner",
    ...}``). Records written before the multi-runner change stored it under
    ``claude`` -- implicitly the Claude runner, with ``interface`` saying
    whether it was the transcript-tailing CLI path or the Agent SDK path.
    Every reader goes through this helper so both generations of records
    aggregate together instead of silently splitting the dataset.
    """
    agent = record.get("agent")
    if isinstance(agent, dict):
        return agent
    legacy = record.get("claude")
    if isinstance(legacy, dict):
        runner = "claude-sdk" if legacy.get("interface", "sdk") == "sdk" else "claude-cli"
        return {**legacy, "runner": runner}
    return {}


# Conditions that identify their dataset on records written before the
# `dataset` field existed. Kept in sync with `experiment.CONDITION_FIELD`.
_DATASET_BY_CONDITION = {
    "ambiguous": "interactive-swe",
    "full": "interactive-swe",
    "mi_ambiguous": "missing-info",
    "mi_full": "missing-info",
}


def dataset_of(summary: dict[str, Any]) -> str | None:
    """Return the issue source a manifest or run summary came from.

    The two datasets cover the same 500 ``instance_id``s, so reports must key
    on this to avoid conflating them. Records written before the second
    dataset existed carry no ``dataset`` key; their condition names identify
    the original dataset unambiguously, the same way ``agent_info`` keeps
    pre-multi-runner records readable.
    """
    task = summary.get("task", {})
    recorded = task.get("dataset")
    if isinstance(recorded, str) and recorded:
        return recorded
    return _DATASET_BY_CONDITION.get(task.get("condition"))


def ask_channel_of(summary: dict[str, Any]) -> str | None:
    """The channel a run's ask outcome was observed on.

    Explicit on new summaries (``ask_user_question.channel``); Claude
    summaries written before the field existed could only ever observe the
    AskUserQuestion tool.
    """
    channel = summary.get("ask_user_question", {}).get("channel")
    if channel:
        return channel
    runner = agent_info(summary).get("runner") or ""
    return "askuserquestion_tool" if runner.startswith("claude") else None


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_run_manifest(
    logs_root: Path,
    *,
    row: dict[str, Any],
    condition: str,
    model: str,
    workspace: Path,
    prompt: str,
    dataset: str = "interactive-swe",
    runner: str = "claude-sdk",
    runner_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist immutable pre-launch metadata and return the manifest.

    ``runner`` names the harness the session runs through and therefore
    which ask channel is even reachable: ``"claude-sdk"`` (Agent SDK session
    with a ``can_use_tool`` callback; the AskUserQuestion tool is the ask
    channel) or ``"codex-cli"`` (stock ``codex exec``; the final-message
    turn yield is the ask channel -- vanilla Codex has no question tool).
    ``runner_details`` carries runner-specific facts worth pinning before
    launch (CLI version, sandbox mode, argv shape). Every summary inherits
    this block, so runs from different runners and models are never
    silently conflated in analysis. Records written before the multi-runner
    change carry the same identity under a ``claude`` key; ``agent_info``
    reads both.

    ``dataset`` names the issue source. The two datasets cover the same 500
    ``instance_id``s, so without it a report cannot tell their runs apart;
    records written before the second dataset existed omit the key and are
    read through ``experiment.dataset_of``.
    """
    run_id = str(uuid.uuid4())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": utc_now(),
        "task": {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "difficulty": row.get("difficulty"),
            "dataset": dataset,
            "condition": condition,
            "prompt_sha256": prompt_hash(prompt),
        },
        "agent": {"model": model, "runner": runner, **(runner_details or {})},
        "workspace": str(workspace.resolve()),
    }
    write_new_json(logs_root / "manifests" / f"{run_id}.json", manifest)
    return manifest


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    records: list[tuple[int, dict[str, Any]]] = []
    errors = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue
                if isinstance(value, dict):
                    records.append((line_number, value))
                else:
                    errors += 1
    except OSError:
        errors += 1
    return records, errors


def _record_matches_workspace(record: dict[str, Any], workspace: Path) -> bool:
    cwd = record.get("cwd")
    if not isinstance(cwd, str):
        return False
    try:
        return Path(cwd).resolve() == workspace.resolve()
    except OSError:
        return False


def find_workspace_transcripts(
    projects_dir: Path, workspace: Path, started_at: str
) -> list[Path]:
    """Return session files updated for this workspace after the run began."""
    started = _parse_timestamp(started_at)
    if not projects_dir.exists() or started is None:
        return []

    candidates: list[Path] = []
    for path in projects_dir.rglob("*.jsonl"):
        try:
            if path.stat().st_mtime < started.timestamp() - 2:
                continue
        except OSError:
            continue
        records, _ = _read_jsonl(path)
        if any(_record_matches_workspace(record, workspace) for _, record in records):
            candidates.append(path)
    return sorted(candidates)


def agent_messages_text(records: list[tuple[int, dict[str, Any]]]) -> str:
    """Render only the agent's own words from one session file.

    Assistant records carry a content list; the text blocks are what the
    agent said. Tool calls, tool results, and harness bookkeeping are all
    omitted -- the raw ``.jsonl`` copy under ``sessions/`` keeps those.
    """
    parts: list[str] = []
    index = 0
    for _, record in records:
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            continue
        index += 1
        parts.append(f"[assistant #{index} @ {record.get('timestamp') or '?'}]")
        parts.extend(texts)
        parts.append("")
    return "\n".join(parts).rstrip() + ("\n" if parts else "")


def preserve_session_artifacts(
    logs_root: Path,
    *,
    run_id: str,
    workspace: Path,
    started_at: str,
    session_id: str | None,
    projects_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy this run's raw session files and distill agent-only transcripts.

    Claude Code's own session records live under ``~/.claude/projects`` and
    are pruned on a retention schedule, so a run summary's
    ``sdk_session_id`` would eventually dangle. Copying at capture time
    makes the raw trace a first-class, run_id-keyed study artifact:

    - ``sessions/<run_id>/``    raw ``.jsonl``, everything the harness wrote
      (main session at the top, subagents under ``subagents/``)
    - ``transcripts/<run_id>/`` agent-message-only ``.txt`` renderings of
      the same files
    """
    projects_dir = projects_dir or default_projects_dir()
    candidates = find_workspace_transcripts(projects_dir, workspace, started_at)
    # The main session file is known by name; keep it even if its records
    # somehow failed the cwd match (e.g. a run that died before writing a
    # cwd-stamped record).
    if session_id:
        for path in projects_dir.rglob(f"{session_id}.jsonl"):
            if path not in candidates:
                candidates.append(path)

    copied: list[str] = []
    sessions_dir = logs_root / "sessions" / run_id
    transcripts_dir = logs_root / "transcripts" / run_id
    for path in sorted(candidates):
        is_subagent = "subagents" in path.relative_to(projects_dir).parts
        # A reused workspace can hold sessions from earlier runs; only the
        # named main session (plus subagent files updated during this run,
        # which the mtime filter already scoped) belongs to this run.
        if not is_subagent and session_id and path.stem != session_id:
            continue
        subdir = Path("subagents") if is_subagent else Path()
        raw_target = sessions_dir / subdir / path.name
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, raw_target)
        records, _ = _read_jsonl(path)
        text_target = transcripts_dir / subdir / f"{path.stem}.txt"
        text_target.parent.mkdir(parents=True, exist_ok=True)
        text_target.write_text(agent_messages_text(records), encoding="utf-8")
        copied.append(str(raw_target))
    return {
        "copied": copied,
        "sessions_dir": str(sessions_dir),
        "transcripts_dir": str(transcripts_dir),
    }


def default_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def codex_rounds_text(rounds: list[dict[str, Any]] | None) -> str:
    """Render a Codex run's agent-only transcript, one block per message.

    The last agent message of a round is the final channel -- the message
    that yielded the turn (and, when classified as a question, the ask
    itself) -- so it is marked apart from mid-turn commentary.
    """
    parts: list[str] = []
    for entry in rounds or []:
        index = entry.get("index")
        parts.append(f"[round {index} :: {entry.get('prompt_kind')} prompt]")
        parts.append(str(entry.get("prompt") or "").strip())
        parts.append("")
        messages = entry.get("agent_messages") or []
        for position, message in enumerate(messages):
            marker = "final" if position == len(messages) - 1 else "commentary"
            parts.append(f"[round {index} :: agent {marker}]")
            parts.append(message)
            parts.append("")
    return "\n".join(parts).rstrip() + ("\n" if parts else "")


def preserve_codex_session_artifacts(
    logs_root: Path,
    *,
    run_id: str,
    thread_id: str | None,
    rounds: list[dict[str, Any]] | None,
    codex_home: Path | None = None,
) -> dict[str, Any]:
    """Copy Codex's own rollout files and render an agent-only transcript.

    The launcher already streamed each round's ``--json`` events into
    ``sessions/<run_id>/`` while the session ran; this adds the CLI's own
    rollout ``.jsonl`` from ``$CODEX_HOME/sessions`` (subject to Codex's own
    retention, exactly like Claude's ``~/.claude/projects``) and writes
    ``transcripts/<run_id>/rounds.txt`` -- the same two artifact slots the
    Claude path fills, so ``locate_logs.py`` needs no per-runner cases.
    """
    codex_home = codex_home or default_codex_home()
    sessions_dir = logs_root / "sessions" / run_id
    transcripts_dir = logs_root / "transcripts" / run_id

    copied: list[str] = []
    if thread_id:
        for root_name in ("sessions", "archived_sessions"):
            root = codex_home / root_name
            if not root.exists():
                continue
            for path in sorted(root.rglob(f"*{thread_id}*.jsonl")):
                sessions_dir.mkdir(parents=True, exist_ok=True)
                target = sessions_dir / path.name
                try:
                    shutil.copy2(path, target)
                except OSError:
                    continue
                copied.append(str(target))

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / "rounds.txt").write_text(
        codex_rounds_text(rounds), encoding="utf-8"
    )
    return {
        "copied": copied,
        "sessions_dir": str(sessions_dir),
        "transcripts_dir": str(transcripts_dir),
    }


def _tool_uses(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def analyze_transcripts(
    paths: list[Path], projects_dir: Path, started_at: str
) -> dict[str, Any]:
    """Extract AskUserQuestion events produced during the current run only."""
    started = _parse_timestamp(started_at)
    events: list[dict[str, Any]] = []
    parse_errors = 0
    valid_records = 0

    for path in paths:
        records, errors = _read_jsonl(path)
        parse_errors += errors
        is_subagent = "subagents" in path.relative_to(projects_dir).parts
        for line_number, record in records:
            timestamp = _parse_timestamp(record.get("timestamp"))
            if timestamp is None or started is None or timestamp < started:
                continue
            valid_records += 1
            for block_index, block in enumerate(_tool_uses(record)):
                events.append(
                    {
                        "timestamp": record["timestamp"],
                        "timestamp_value": timestamp,
                        "path": path,
                        "line_number": line_number,
                        "block_index": block_index,
                        "is_subagent": is_subagent,
                        "tool_name": block.get("name"),
                        "tool_use_id": block.get("id"),
                        "caller_type": (
                            block.get("caller", {}).get("type")
                            if isinstance(block.get("caller"), dict)
                            else None
                        ),
                        "input": block.get("input"),
                    }
                )

    events.sort(
        key=lambda event: (
            event["timestamp_value"],
            str(event["path"]),
            event["line_number"],
            event["block_index"],
        )
    )
    main_tool_actions = 0
    first_direct: dict[str, Any] | None = None
    direct_ids: set[str] = set()
    any_ids: set[str] = set()
    direct_count = 0
    any_agent_count = 0

    for event in events:
        is_main = not event["is_subagent"]
        is_ask = event["tool_name"] == "AskUserQuestion"
        if is_main and is_ask and event["caller_type"] == "direct":
            identifier = event["tool_use_id"] or (
                f"{event['path']}:{event['line_number']}:{event['block_index']}"
            )
            if identifier not in direct_ids:
                direct_ids.add(identifier)
                direct_count += 1
                if first_direct is None:
                    first_direct = {
                        "timestamp": event["timestamp"],
                        "tool_use_id": event["tool_use_id"],
                        "input": event["input"],
                        "assistant_tool_actions_before": main_tool_actions,
                        "source_path": str(event["path"]),
                        "source_line": event["line_number"],
                    }
        if is_ask:
            identifier = event["tool_use_id"] or (
                f"{event['path']}:{event['line_number']}:{event['block_index']}"
            )
            if identifier not in any_ids:
                any_ids.add(identifier)
                any_agent_count += 1
        if is_main:
            main_tool_actions += 1

    return {
        "paths": paths,
        "parse_errors": parse_errors,
        "valid_records": valid_records,
        "first_direct": first_direct,
        "direct_count": direct_count,
        "any_agent_count": any_agent_count,
    }


def interrupt_process_group(pid: int, sig: int = signal.SIGINT) -> None:
    """Signal Claude and any children without signalling the launcher itself."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        # Claude can exit in the small interval between poll() and signal delivery.
        pass


def observe_headless_session(
    argv: list[str],
    workspace: Path,
    started_at: str,
    projects_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run headless Claude and stop it once it directly asks a user."""
    projects_dir = projects_dir or default_projects_dir()
    output_dir = output_dir or Path.cwd() / ".experiment-logs" / "process-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "claude.stdout.log"
    stderr_path = output_dir / "claude.stderr.log"
    stdout_handle = stdout_path.open("x", encoding="utf-8")
    stderr_handle = stderr_path.open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(workspace),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
    except BaseException:
        stdout_handle.close()
        stderr_handle.close()
        raise
    stop_reason = "completed"
    interrupted_at: float | None = None
    force_termination_sent = False
    operator_interrupted = False
    analysis: dict[str, Any] = {
        "paths": [],
        "parse_errors": 0,
        "valid_records": 0,
        "first_direct": None,
        "direct_count": 0,
        "any_agent_count": 0,
    }

    try:
        while process.poll() is None:
            paths = find_workspace_transcripts(projects_dir, workspace, started_at)
            analysis = analyze_transcripts(paths, projects_dir, started_at)
            if analysis["first_direct"] is not None and interrupted_at is None:
                interrupt_process_group(process.pid)
                stop_reason = "stopped_on_first_ask"
                interrupted_at = time.monotonic()
            elif (
                interrupted_at is not None
                and time.monotonic() - interrupted_at > INTERRUPT_GRACE_SECONDS
                and not force_termination_sent
            ):
                interrupt_process_group(process.pid, signal.SIGTERM)
                stop_reason = "stopped_on_first_ask_forced"
                force_termination_sent = True
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        operator_interrupted = True
        stop_reason = "interrupted_by_operator"
        if process.poll() is None:
            interrupt_process_group(process.pid)
    finally:
        try:
            exit_code = process.wait(timeout=INTERRUPT_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            interrupt_process_group(process.pid, signal.SIGTERM)
            exit_code = process.wait()
        stdout_handle.close()
        stderr_handle.close()

    # Let Claude flush the final JSONL entry, then perform one complete analysis.
    time.sleep(POLL_SECONDS)
    paths = find_workspace_transcripts(projects_dir, workspace, started_at)
    analysis = analyze_transcripts(paths, projects_dir, started_at)
    return {
        "exit_code": exit_code,
        "stop_reason": stop_reason,
        "operator_interrupted": operator_interrupted,
        "process_output": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        "analysis": analysis,
    }


def copy_transcripts(paths: list[Path], logs_root: Path, run_id: str) -> list[dict[str, str]]:
    destination = logs_root / "transcripts" / run_id
    copied: list[dict[str, str]] = []
    for index, source in enumerate(paths):
        target = destination / f"{index:02d}-{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
        except OSError:
            continue
        copied.append({"source": str(source), "copy": str(target)})
    return copied


def build_run_summary(
    manifest: dict[str, Any], observation: dict[str, Any], logs_root: Path
) -> dict[str, Any]:
    analysis = observation["analysis"]
    first = analysis["first_direct"]
    started = _parse_timestamp(manifest["started_at"])
    first_time = _parse_timestamp(first["timestamp"]) if first else None
    latency = (
        (first_time - started).total_seconds()
        if started is not None and first_time is not None
        else None
    )
    first_summary = dict(first) if first is not None else None
    if first_summary is not None:
        payload = first_summary.get("input")
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if isinstance(questions, list):
            first_summary["question_count"] = len(questions)
            first_summary["option_count"] = sum(
                len(question.get("options", []))
                for question in questions
                if isinstance(question, dict) and isinstance(question.get("options", []), list)
            )
    ended_at = utc_now()
    ended = _parse_timestamp(ended_at)
    duration = (
        (ended - started).total_seconds()
        if started is not None and ended is not None
        else None
    )
    if first is not None:
        direct_asked: bool | None = True
        monitoring_status = "observed_ask"
    elif analysis["paths"] and analysis["parse_errors"] == 0 and analysis["valid_records"]:
        direct_asked = False
        monitoring_status = "complete_no_ask"
    else:
        direct_asked = None
        monitoring_status = "unknown"

    transcript_copies = copy_transcripts(analysis["paths"], logs_root, manifest["run_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "started_at": manifest["started_at"],
        "ended_at": ended_at,
        "task": manifest["task"],
        "agent": agent_info(manifest),
        "workspace": manifest["workspace"],
        "process": {
            "exit_code": observation["exit_code"],
            "stop_reason": observation["stop_reason"],
            "operator_interrupted": observation["operator_interrupted"],
            "duration_seconds": duration,
            "stdout_log": observation.get("process_output", {}).get("stdout"),
            "stderr_log": observation.get("process_output", {}).get("stderr"),
        },
        "transcript": {
            "monitoring_status": monitoring_status,
            "parse_errors": analysis["parse_errors"],
            "source_paths": [str(path) for path in analysis["paths"]],
            "copies": transcript_copies,
        },
        "ask_user_question": {
            "direct_asked": direct_asked,
            "direct_count": analysis["direct_count"],
            "any_agent_count": analysis["any_agent_count"],
            "first_direct": first_summary,
            "first_direct_latency_seconds": latency,
        },
    }


def build_run_summary_sdk(
    manifest: dict[str, Any],
    observation: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run summary from an ``sdk_runner.run_sdk_session`` observation.

    Distinct from ``build_run_summary`` (which analyzes CLI transcript
    files) because the SDK path observes ``AskUserQuestion`` directly
    through its ``can_use_tool`` callback -- there is no transcript to tail,
    and no way for a run to be ``unknown`` the way a missing/malformed CLI
    transcript could be. The top-level schema matches
    ``build_run_summary`` wherever the underlying data is genuinely the
    same, and adds ``tool_roster`` / ``askuserquestion_available`` so every
    run is self-certifying instead of relying on an assumption about
    whether the tool was reachable.

    First-ask latency is measured in-callback by the runner (the SDK has no
    transcript timestamps to difference against), so it is read straight off
    ``first_direct`` rather than recomputed here.
    """
    analysis = observation["analysis"]
    first = analysis["first_direct"]
    first_summary = dict(first) if first is not None else None
    if first_summary is not None:
        payload = first_summary.get("input")
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if isinstance(questions, list):
            first_summary["question_count"] = len(questions)
            first_summary["option_count"] = sum(
                len(question.get("options", []))
                for question in questions
                if isinstance(question, dict) and isinstance(question.get("options", []), list)
            )

    started = _parse_timestamp(manifest["started_at"])
    ended_at = utc_now()
    ended = _parse_timestamp(ended_at)
    duration = (
        (ended - started).total_seconds()
        if started is not None and ended is not None
        else None
    )

    result = observation.get("result") or {}

    # A session that errored, or that ended without the model doing anything,
    # is not evidence that the model declined to ask -- it never got the
    # chance. Recording it as ``False`` would put a non-run in the
    # denominator of the ask rate, so it is ``None`` ("unknown") instead,
    # mirroring the CLI path's handling of an unreadable transcript.
    ran_meaningfully = (
        not result.get("is_error")
        and (result.get("num_turns") or 0) > 1
        and observation.get("permission_prompts", 0) > 0
    )
    if first is not None:
        direct_asked: bool | None = True
    elif ran_meaningfully:
        direct_asked = False
    else:
        direct_asked = None

    actual_tools = observation.get("tool_roster")
    reference_tools = observation.get("reference_toolset")
    if actual_tools is not None and reference_tools is not None:
        actual_set = set(actual_tools)
        reference_set = set(reference_tools)
        missing_from_actual = sorted(reference_set - actual_set)
        extra_in_actual = sorted(actual_set - reference_set)
        matches_reference = not missing_from_actual and not extra_in_actual
    else:
        missing_from_actual = None
        extra_in_actual = None
        matches_reference = None

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "started_at": manifest["started_at"],
        "ended_at": ended_at,
        "task": manifest["task"],
        "agent": agent_info(manifest),
        "workspace": manifest["workspace"],
        "process": {
            "exit_code": 1 if result.get("is_error") else 0,
            "stop_reason": (
                # Same label the Codex arm uses when an ask beyond the
                # synthetic-answer cap ends the session.
                "max_ask_rounds"
                if observation.get("hit_ask_cap")
                else "stopped_on_first_ask"
                if observation.get("stopped_on_first_ask")
                else result.get("stop_reason") or result.get("subtype")
            ),
            "operator_interrupted": False,
            "duration_seconds": duration,
            "sdk_session_id": result.get("session_id"),
            "sdk_num_turns": result.get("num_turns"),
            "sdk_total_cost_usd": result.get("total_cost_usd"),
            # Failure evidence. Without these fields an errored run is
            # indistinguishable from "the agent chose to do nothing", and the
            # actual error text (e.g. a usage-limit rejection) is lost with
            # the observation. See the 2026-07-30 batch: six $0 one-turn runs
            # whose cause had to be reconstructed from timing alone.
            "sdk_is_error": bool(result.get("is_error")),
            "sdk_result_subtype": result.get("subtype"),
            "sdk_error": (
                str(result.get("result"))[:2000]
                if result.get("is_error") and result.get("result")
                else None
            ),
        },
        "permissions": {
            "mode": observation.get("permission_mode"),
            "prompts_reaching_callback": observation.get("permission_prompts"),
        },
        "session": {
            "ran_meaningfully": ran_meaningfully,
            "monitoring_status": (
                "observed_ask"
                if first is not None
                else "complete_no_ask"
                if ran_meaningfully
                else "no_work_performed"
            ),
        },
        # Grades are stored separately (see write_evaluation) so they can be
        # recomputed without discarding the session record. This mirror is
        # convenience only; the report always prefers the stored evaluation.
        "evaluation": evaluation or {"status": "not_evaluated", "resolved": None},
        "tool_roster": {
            "tools": observation.get("tool_roster"),
            "askuserquestion_available": observation.get("askuserquestion_available"),
            "matches_reference": matches_reference,
            "missing_from_actual": missing_from_actual,
            "extra_in_actual": extra_in_actual,
        },
        "ask_user_question": {
            "channel": "askuserquestion_tool",
            "max_ask_rounds": observation.get("max_ask_rounds"),
            "hit_ask_cap": observation.get("hit_ask_cap"),
            "direct_asked": direct_asked,
            "direct_count": analysis["direct_count"],
            "any_agent_count": analysis["any_agent_count"],
            "first_direct": first_summary,
            "first_direct_latency_seconds": (
                first.get("latency_seconds") if first is not None else None
            ),
            "answered_questions": observation.get("answered_questions") or [],
        },
    }


def build_run_summary_codex(
    manifest: dict[str, Any],
    observation: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run summary from a ``codex_runner.run_codex_session`` observation.

    Field-compatible with ``build_run_summary_sdk`` wherever the underlying
    data means the same thing (task, ask outcome, evaluation, duration), and
    explicit where it does not: ``ask_user_question.channel`` is
    ``final_message`` because vanilla Codex has no question tool -- the
    observed ask is a turn that ended with a clarifying question, classified
    by the versioned deterministic rule recorded alongside it.
    ``first_direct`` therefore carries the question's message text rather
    than tool input, plus ``workspace_had_changes`` so pre-work clarifying
    questions can be separated from post-work offers in analysis.

    ``ran_meaningfully`` mirrors the SDK builder's intent with the signals
    this runner has: a session that errored, or that ended without either a
    tool action or an ask, observed nothing about the ask decision and must
    not enter the ask-rate denominator.
    """
    analysis = observation["analysis"]
    first = analysis["first_direct"]

    started = _parse_timestamp(manifest["started_at"])
    ended_at = utc_now()
    ended = _parse_timestamp(ended_at)
    duration = (
        (ended - started).total_seconds()
        if started is not None and ended is not None
        else None
    )

    result = observation.get("result") or {}
    tool_actions = observation.get("tool_actions_total") or 0
    ran_meaningfully = not result.get("is_error") and (
        tool_actions > 0 or analysis["direct_count"] > 0
    )
    if first is not None:
        direct_asked: bool | None = True
    elif ran_meaningfully:
        direct_asked = False
    else:
        direct_asked = None

    # Compact per-round digest: prompts, agent messages, and raw events stay
    # in sessions/ and transcripts/; the summary keeps the shape of the run.
    round_digest = [
        {
            "index": entry.get("index"),
            "prompt_kind": entry.get("prompt_kind"),
            "asked": entry.get("asked"),
            "turn_edited": entry.get("turn_edited"),
            "regex_asked": entry.get("regex_asked"),
            "tool_actions": entry.get("tool_actions"),
            "file_changes": entry.get("file_changes"),
            "exit_code": entry.get("exit_code"),
            "duration_seconds": entry.get("duration_seconds"),
        }
        for entry in observation.get("rounds") or []
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "started_at": manifest["started_at"],
        "ended_at": ended_at,
        "task": manifest["task"],
        "agent": agent_info(manifest),
        "workspace": manifest["workspace"],
        "process": {
            "exit_code": 1 if result.get("is_error") else 0,
            "stop_reason": result.get("stop_reason") or result.get("subtype"),
            "operator_interrupted": False,
            "duration_seconds": duration,
            "codex_thread_id": result.get("session_id"),
            "codex_rounds": result.get("num_turns"),
            "codex_token_usage": result.get("usage"),
            "codex_is_error": bool(result.get("is_error")),
            "codex_error": (
                str(result.get("result"))[:2000]
                if result.get("is_error") and result.get("result")
                else None
            ),
        },
        "sandbox": {
            "mode": observation.get("sandbox"),
            # codex exec has no interactive approval channel; the model is
            # told approvals are unavailable rather than pausing for them.
            "approval_policy": "never",
        },
        "session": {
            "ran_meaningfully": ran_meaningfully,
            "monitoring_status": (
                "observed_ask"
                if first is not None
                else "complete_no_ask"
                if ran_meaningfully
                else "no_work_performed"
            ),
            "rounds": round_digest,
            "tool_actions_total": tool_actions,
        },
        "evaluation": evaluation or {"status": "not_evaluated", "resolved": None},
        "ask_user_question": {
            "channel": observation.get("ask_channel"),
            "classifier_version": observation.get("ask_classifier_version"),
            "gate": observation.get("ask_gate"),
            "max_ask_rounds": observation.get("max_ask_rounds"),
            "neutral_answer": observation.get("neutral_answer"),
            "direct_asked": direct_asked,
            "direct_count": analysis["direct_count"],
            "any_agent_count": analysis["any_agent_count"],
            # Regex fired on an edited turn (gated to not-asked). Zero in
            # all harvested data; non-zero flags the gate assumption.
            "questions_with_edits": analysis.get("questions_with_edits"),
            "first_direct": dict(first) if first is not None else None,
            "first_direct_latency_seconds": (
                first.get("latency_seconds") if first is not None else None
            ),
            "answered_questions": observation.get("answered_questions") or [],
        },
    }


def write_run_summary(logs_root: Path, summary: dict[str, Any]) -> Path:
    path = logs_root / "runs" / f"{summary['run_id']}.json"
    write_new_json(path, summary)
    return path


def write_evaluation(logs_root: Path, run_id: str, evaluation: dict[str, Any]) -> Path:
    """Store a run's grade as its own artifact, overwriting any earlier grade.

    Evaluations live in ``evaluations/`` rather than inside the run summary
    because the two have different lifetimes: a summary records what happened
    during a session and is immutable, while a grade is a *judgement* about
    that session which can legitimately be recomputed when the grading rules
    change. Keeping them apart means re-grading never requires re-running.
    """
    path = logs_root / "evaluations" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**evaluation, "run_id": run_id, "evaluated_at": utc_now()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_evaluations(logs_root: Path) -> dict[str, dict[str, Any]]:
    """Return stored evaluations keyed by ``run_id``."""
    evaluations: dict[str, dict[str, Any]] = {}
    directory = logs_root / "evaluations"
    if not directory.exists():
        return evaluations
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            evaluations[value.get("run_id") or path.stem] = value
    return evaluations


def load_run_summaries(logs_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runs_dir = logs_root / "runs"
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    if not runs_dir.exists():
        return summaries, errors
    for path in sorted(runs_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError("summary is not a JSON object")
            summaries.append(value)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    return summaries, errors


def _numeric_summary(values: list[float | int | None]) -> dict[str, float | None]:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return {"mean": None, "median": None}
    return {"mean": mean(numeric), "median": median(numeric)}


def _resolution_cell(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolution counts over the subset that could actually be scored.

    Runs whose status is not ``scored`` (an unsupported runner, an environment
    the repository will not import in, a timeout) are deliberately excluded
    rather than counted as failures, so a harness limitation can never be read
    as the agent producing a bad patch.
    """
    scored = [
        summary
        for summary in summaries
        if summary.get("evaluation", {}).get("status") == "scored"
    ]
    resolved = sum(
        summary["evaluation"].get("resolved") is True for summary in scored
    )
    localization = [
        summary.get("evaluation", {}).get("localization_hit")
        for summary in summaries
        if isinstance(summary.get("evaluation", {}).get("localization_hit"), bool)
    ]
    return {
        "runs": len(summaries),
        "scored": len(scored),
        "resolved": resolved,
        "resolve_rate": resolved / len(scored) if scored else None,
        "localization_checked": len(localization),
        "localization_hits": sum(localization),
        "localization_rate": (
            sum(localization) / len(localization) if localization else None
        ),
    }


def _ask_cell(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Ask-rate counts over one slice of runs.

    ``valid`` excludes runs whose ask outcome is unobservable (errored or
    no-work sessions record ``direct_asked: null``), matching the top-level
    primary-outcome denominator.
    """
    asks = [summary.get("ask_user_question", {}) for summary in summaries]
    valid = [ask for ask in asks if isinstance(ask.get("direct_asked"), bool)]
    asked = sum(ask.get("direct_asked") is True for ask in valid)
    return {
        "runs": len(summaries),
        "valid": len(valid),
        "asked": asked,
        "ask_rate": asked / len(valid) if valid else None,
    }


def _model_cells(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-model (and per model x condition) slices of ask and resolution.

    Models are separate arms of the study and their ask channels differ
    (Claude: AskUserQuestion tool call; Codex/GPT: final-message question),
    so nothing here is ever pooled across models -- ``ask_channels`` records
    which channel each slice was measured on so the comparison stays honest.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        model = agent_info(summary).get("model") or "unknown"
        by_model.setdefault(model, []).append(summary)

    cells: dict[str, Any] = {}
    for model, rows in sorted(by_model.items()):
        channels: dict[str, int] = {}
        for summary in rows:
            channel = ask_channel_of(summary) or "unknown"
            channels[channel] = channels.get(channel, 0) + 1
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for summary in rows:
            key = summary.get("task", {}).get("condition") or "unknown"
            by_condition.setdefault(key, []).append(summary)
        cells[model] = {
            "runner": agent_info(rows[0]).get("runner"),
            "ask_channels": channels,
            "ask": _ask_cell(rows),
            "resolution": _resolution_cell(rows),
            "by_condition": {
                condition: {
                    "ask": _ask_cell(sub),
                    "resolution": _resolution_cell(sub),
                }
                for condition, sub in sorted(by_condition.items())
            },
        }
    return cells


def build_report(summaries: list[dict[str, Any]], input_errors: list[str]) -> dict[str, Any]:
    asks = [summary.get("ask_user_question", {}) for summary in summaries]
    valid = [ask for ask in asks if isinstance(ask.get("direct_asked"), bool)]
    direct_ask_runs = sum(ask.get("direct_asked") is True for ask in valid)
    any_agent_ask_runs = sum((ask.get("any_agent_count") or 0) > 0 for ask in asks)
    first_questions = [ask.get("first_direct") for ask in asks if ask.get("first_direct")]
    question_counts = []
    option_counts = []
    for first in first_questions:
        payload = first.get("input") if isinstance(first, dict) else None
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if isinstance(questions, list):
            question_counts.append(len(questions))
            option_counts.append(
                sum(
                    len(question.get("options", []))
                    for question in questions
                    if isinstance(question, dict) and isinstance(question.get("options", []), list)
                )
            )
    rosters = [
        summary.get("tool_roster", {})
        for summary in summaries
        if isinstance(summary.get("tool_roster", {}).get("matches_reference"), bool)
    ]
    mismatched_rosters = [roster for roster in rosters if not roster["matches_reference"]]

    evaluated = [
        summary
        for summary in summaries
        if summary.get("evaluation", {}).get("status")
        not in (None, "not_evaluated")
    ]
    status_counts: dict[str, int] = {}
    for summary in evaluated:
        status = summary["evaluation"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    # The study's headline comparison. Asking is self-selected rather than
    # randomized, so these cells are reported raw and stratified by difficulty
    # instead of collapsed into a single number.
    asked_runs = [
        summary
        for summary in evaluated
        if summary.get("ask_user_question", {}).get("direct_asked") is True
    ]
    not_asked_runs = [
        summary
        for summary in evaluated
        if summary.get("ask_user_question", {}).get("direct_asked") is False
    ]
    by_difficulty: dict[str, Any] = {}
    for summary in evaluated:
        key = summary.get("task", {}).get("difficulty") or "unknown"
        by_difficulty.setdefault(key, []).append(summary)

    # The dataset's ambiguous/full split is the study's actual independent
    # variable; by_asked cross-tabs the (self-selected) outcome, this cross-
    # tabs the condition itself, so a two-condition run is legible on its own.
    by_condition: dict[str, Any] = {}
    for summary in evaluated:
        key = summary.get("task", {}).get("condition") or "unknown"
        by_condition.setdefault(key, []).append(summary)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "input_errors": input_errors,
        "runs": {
            "total": len(summaries),
            "valid_for_primary_outcome": len(valid),
            "unknown": len(summaries) - len(valid),
            "direct_ask_runs": direct_ask_runs,
            "direct_ask_rate": direct_ask_runs / len(valid) if valid else None,
            "any_agent_ask_runs": any_agent_ask_runs,
            "any_agent_ask_rate": any_agent_ask_runs / len(summaries) if summaries else None,
        },
        # Each model is its own study arm; nothing in this section pools
        # across models, and the ask channel each arm was measured on is
        # recorded next to its rates.
        "models": _model_cells(summaries),
        "tool_roster": {
            "checked": len(rosters),
            "mismatched": len(mismatched_rosters),
            "mismatch_rate": (
                len(mismatched_rosters) / len(rosters) if rosters else None
            ),
            "mismatched_runs": [
                {
                    "missing": roster.get("missing_from_actual"),
                    "extra": roster.get("extra_in_actual"),
                }
                for roster in mismatched_rosters
            ],
        },
        "evaluation": {
            "evaluated": len(evaluated),
            "status_counts": status_counts,
            **_resolution_cell(evaluated),
            "by_asked": {
                "asked": _resolution_cell(asked_runs),
                "not_asked": _resolution_cell(not_asked_runs),
            },
            "by_difficulty": {
                key: _resolution_cell(rows)
                for key, rows in sorted(by_difficulty.items())
            },
            "by_condition": {
                key: _resolution_cell(rows)
                for key, rows in sorted(by_condition.items())
            },
        },
        "secondary": {
            "first_ask_latency_seconds": _numeric_summary(
                [ask.get("first_direct_latency_seconds") for ask in asks]
            ),
            "tool_actions_before_first_ask": _numeric_summary(
                [
                    first.get("assistant_tool_actions_before")
                    for first in first_questions
                    if isinstance(first, dict)
                ]
            ),
            "questions_per_first_call": _numeric_summary(question_counts),
            "options_per_first_call": _numeric_summary(option_counts),
            "run_duration_seconds": _numeric_summary(
                [summary.get("process", {}).get("duration_seconds") for summary in summaries]
            ),
        },
        "termination_states": {
            state: sum(
                summary.get("process", {}).get("stop_reason") == state
                for summary in summaries
            )
            for state in sorted(
                {
                    summary.get("process", {}).get("stop_reason", "unknown")
                    for summary in summaries
                }
            )
        },
    }


def attach_evaluations(
    summaries: list[dict[str, Any]], evaluations: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join each summary with its stored grade, without mutating the summary.

    A grade stored alongside the run always wins over one embedded in an older
    summary, so re-grading takes effect on the next report.
    """
    joined: list[dict[str, Any]] = []
    for summary in summaries:
        stored = evaluations.get(summary.get("run_id"))
        if stored is None:
            joined.append(summary)
        else:
            joined.append({**summary, "evaluation": stored})
    return joined

