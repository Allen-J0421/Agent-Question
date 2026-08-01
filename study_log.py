"""Capture and report spontaneous AskUserQuestion use in Claude Code sessions.

Claude Code writes its own session transcript beneath ``~/.claude/projects``.  This
module only observes those records; it does not alter Claude's prompt, tools,
permissions, or output mode.
"""
from __future__ import annotations

import csv
import argparse
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
    interface: str = "cli",
    claude_bin: str | None = None,
) -> dict[str, Any]:
    """Persist immutable pre-launch metadata and return the manifest.

    ``interface`` is ``"cli"`` (subprocess ``claude -p``, no callback
    mechanism -- AskUserQuestion is structurally unreachable there) or
    ``"sdk"`` (Agent SDK session with a ``can_use_tool`` callback, required
    for AskUserQuestion to appear in the tool roster at all). Every summary
    is stamped with this so historical CLI-path runs and current SDK-path
    runs are never silently conflated in analysis.
    """
    run_id = str(uuid.uuid4())
    claude_info: dict[str, Any] = {"model": model, "interface": interface}
    if interface == "cli":
        claude_info["binary"] = claude_bin
        claude_info["argv_shape"] = ["claude", "--model", model, "<prompt>"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": utc_now(),
        "task": {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "difficulty": row.get("difficulty"),
            "condition": condition,
            "prompt_sha256": prompt_hash(prompt),
        },
        "claude": claude_info,
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
        "claude": manifest["claude"],
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
        "claude": manifest["claude"],
        "workspace": manifest["workspace"],
        "process": {
            "exit_code": 1 if result.get("is_error") else 0,
            "stop_reason": (
                "stopped_on_first_ask"
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


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _resolution_row(label: str, cell: dict[str, Any]) -> str:
    return (
        f"| {label} | {cell['runs']} | {cell['scored']} | {cell['resolved']} "
        f"| {_pct(cell['resolve_rate'])} | {cell['localization_hits']}/"
        f"{cell['localization_checked']} | {_pct(cell['localization_rate'])} |"
    )


def _run_row(summary: dict[str, Any]) -> str:
    task = summary.get("task", {})
    process = summary.get("process", {})
    ask = summary.get("ask_user_question", {})
    evaluation = summary.get("evaluation", {})
    asked = ask.get("direct_asked")
    asked_text = {True: "yes", False: "no"}.get(asked, "?")
    status = evaluation.get("status") or "not_evaluated"
    if status == "scored":
        grade = "resolved" if evaluation.get("resolved") else "unresolved"
    else:
        grade = status
    duration = process.get("duration_seconds")
    duration_text = f"{duration / 60:.1f}" if isinstance(duration, (int, float)) else "-"
    cost = process.get("sdk_total_cost_usd")
    cost_text = f"{cost:.2f}" if isinstance(cost, (int, float)) else "-"
    return (
        f"| {task.get('instance_id', '?')} | {task.get('condition', '?')} "
        f"| {asked_text} | {grade} | {duration_text} | {cost_text} "
        f"| {summary.get('run_id', '?')[:8]} |"
    )


def render_markdown_report(
    report: dict[str, Any],
    summaries: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> str:
    """Render the full-detail Markdown report.

    The JSON report already carries every number; this renders essentially
    all of it into one readable document instead of a short summary, so
    reading the study's results doesn't require cross-referencing JSON keys.
    Three views, in order: the headline numbers, results sliced the ways that
    matter (condition, difficulty, ask), then every individual run.
    """
    runs = report["runs"]
    roster = report["tool_roster"]
    evaluation = report["evaluation"]
    secondary = report["secondary"]
    generated_at = report.get("generated_at", "?")

    lines: list[str] = []
    lines.append("# ambig-SWE study report")
    lines.append("")
    lines.append(f"Generated {generated_at} from {runs['total']} logged run(s).")
    lines.append(
        f"Full per-run data: [`{csv_path.name}`]({csv_path.name}) · "
        f"raw aggregates: [`{json_path.name}`]({json_path.name})"
    )
    lines.append("")

    # ---- Primary outcome ---------------------------------------------------
    lines.append("## Primary outcome: did the agent ask?")
    lines.append("")
    lines.append(
        f"- Direct `AskUserQuestion` calls: **{runs['direct_ask_runs']}/"
        f"{runs['valid_for_primary_outcome']}** valid runs ({_pct(runs['direct_ask_rate'])})"
    )
    lines.append(
        f"- Any agent (including subagents) asked: {runs['any_agent_ask_runs']}/"
        f"{runs['total']} ({_pct(runs['any_agent_ask_rate'])})"
    )
    if runs["unknown"]:
        lines.append(
            f"- ⚠️ {runs['unknown']} run(s) have no observable ask outcome "
            "(excluded from the rate above)"
        )
    lines.append("")

    # ---- Results by condition ----------------------------------------------
    by_condition = evaluation.get("by_condition") or {}
    if by_condition:
        lines.append("## Results by condition (ambiguous vs. full)")
        lines.append("")
        lines.append("| condition | runs | scored | resolved | resolve rate | localization | loc. rate |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, cell in sorted(by_condition.items()):
            lines.append(_resolution_row(key, cell))
        lines.append("")

    # ---- Results by difficulty ----------------------------------------------
    by_difficulty = evaluation.get("by_difficulty") or {}
    if by_difficulty:
        lines.append("## Results by difficulty")
        lines.append("")
        lines.append("| difficulty | runs | scored | resolved | resolve rate | localization | loc. rate |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, cell in sorted(by_difficulty.items()):
            lines.append(_resolution_row(key, cell))
        lines.append("")

    # ---- Results by whether the agent asked --------------------------------
    by_asked = evaluation.get("by_asked") or {}
    if by_asked and (by_asked["asked"]["runs"] or by_asked["not_asked"]["runs"]):
        lines.append(
            "## Results by whether the agent asked "
            "(self-selected, not randomized)"
        )
        lines.append("")
        lines.append("| asked? | runs | scored | resolved | resolve rate | localization | loc. rate |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(_resolution_row("asked", by_asked["asked"]))
        lines.append(_resolution_row("did not ask", by_asked["not_asked"]))
        lines.append("")

    # ---- Overall patch evaluation -------------------------------------------
    if evaluation["evaluated"]:
        statuses = ", ".join(
            f"{name} {count}" for name, count in sorted(evaluation["status_counts"].items())
        )
        lines.append("## Patch evaluation (overall)")
        lines.append("")
        lines.append(f"- Evaluated runs: {evaluation['evaluated']} ({statuses})")
        lines.append(
            f"- Resolved: {evaluation['resolved']}/{evaluation['scored']} scored "
            f"({_pct(evaluation['resolve_rate'])})"
        )
        lines.append(
            f"- Localization hits: {evaluation['localization_hits']}/"
            f"{evaluation['localization_checked']} ({_pct(evaluation['localization_rate'])})"
        )
        lines.append("")
    else:
        lines.append("## Patch evaluation (overall)")
        lines.append("")
        lines.append("No runs are graded yet — run `experiment.py evaluate`.")
        lines.append("")

    # ---- Tool roster / measurement channel ----------------------------------
    lines.append("## Measurement channel")
    lines.append("")
    if roster["checked"]:
        lines.append(
            f"- Tool roster mismatches: {roster['mismatched']}/{roster['checked']} "
            f"SDK runs ({_pct(roster['mismatch_rate'])})"
        )
        if roster["mismatched_runs"]:
            for entry in roster["mismatched_runs"]:
                lines.append(f"  - missing: {entry['missing']}, extra: {entry['extra']}")
    else:
        lines.append("- No runs had a checkable tool roster.")
    termination = report.get("termination_states") or {}
    if termination:
        states = ", ".join(f"{name}: {count}" for name, count in sorted(termination.items()))
        lines.append(f"- Termination states: {states}")
    lines.append("")

    # ---- Secondary stats -----------------------------------------------------
    lines.append("## Secondary stats")
    lines.append("")
    lines.append("| metric | mean | median |")
    lines.append("|---|---|---|")
    metric_labels = {
        "first_ask_latency_seconds": "first-ask latency (s)",
        "tool_actions_before_first_ask": "tool actions before first ask",
        "questions_per_first_call": "questions per first ask call",
        "options_per_first_call": "options per first ask call",
        "run_duration_seconds": "run duration (s)",
    }
    for key, label in metric_labels.items():
        cell = secondary.get(key) or {}
        mean_v, median_v = cell.get("mean"), cell.get("median")
        mean_text = f"{mean_v:.1f}" if isinstance(mean_v, (int, float)) else "n/a"
        median_text = f"{median_v:.1f}" if isinstance(median_v, (int, float)) else "n/a"
        lines.append(f"| {label} | {mean_text} | {median_text} |")
    lines.append("")

    # ---- Data quality ---------------------------------------------------------
    input_errors = report.get("input_errors") or []
    if input_errors:
        lines.append("## Data quality")
        lines.append("")
        lines.append(f"- {len(input_errors)} unreadable run summary file(s):")
        for error in input_errors:
            lines.append(f"  - {error}")
        lines.append("")

    # ---- Every run -------------------------------------------------------------
    lines.append("## All runs")
    lines.append("")
    lines.append(f"{len(summaries)} run(s), most recent first.")
    lines.append("")
    lines.append("| instance | condition | asked | grade | min | cost$ | run_id |")
    lines.append("|---|---|---|---|---|---|---|")
    ordered = sorted(summaries, key=lambda s: s.get("started_at") or "", reverse=True)
    for summary in ordered:
        lines.append(_run_row(summary))
    lines.append("")

    return "\n".join(lines) + "\n"


def write_report(logs_root: Path) -> dict[str, Path]:
    summaries, input_errors = load_run_summaries(logs_root)
    summaries = attach_evaluations(summaries, load_evaluations(logs_root))
    report = build_report(summaries, input_errors)
    reports_dir = logs_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "askuserquestion-report.json"
    csv_path = reports_dir / "run-summary.csv"
    markdown_path = reports_dir / "askuserquestion-report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id", "instance_id", "condition", "model", "started_at", "ended_at",
                "direct_asked", "direct_count", "any_agent_count", "first_ask_latency_seconds",
                "tool_actions_before_first_ask", "monitoring_status", "stop_reason", "exit_code",
                "duration_seconds",
                "eval_status", "resolved", "f2p_passed", "f2p_total", "p2p_passed",
                "p2p_total", "localization_hit", "empty_patch",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            ask = summary.get("ask_user_question", {})
            first = ask.get("first_direct") or {}
            evaluation = summary.get("evaluation", {})
            writer.writerow(
                {
                    "run_id": summary.get("run_id"),
                    "instance_id": summary.get("task", {}).get("instance_id"),
                    "condition": summary.get("task", {}).get("condition"),
                    "model": summary.get("claude", {}).get("model"),
                    "started_at": summary.get("started_at"),
                    "ended_at": summary.get("ended_at"),
                    "direct_asked": ask.get("direct_asked"),
                    "direct_count": ask.get("direct_count"),
                    "any_agent_count": ask.get("any_agent_count"),
                    "first_ask_latency_seconds": ask.get("first_direct_latency_seconds"),
                    "tool_actions_before_first_ask": first.get("assistant_tool_actions_before"),
                    "monitoring_status": (
                        summary.get("transcript", {}).get("monitoring_status")
                        or summary.get("session", {}).get("monitoring_status")
                    ),
                    "stop_reason": summary.get("process", {}).get("stop_reason"),
                    "exit_code": summary.get("process", {}).get("exit_code"),
                    "duration_seconds": summary.get("process", {}).get("duration_seconds"),
                    "eval_status": evaluation.get("status"),
                    "resolved": evaluation.get("resolved"),
                    "f2p_passed": evaluation.get("f2p_passed"),
                    "f2p_total": evaluation.get("f2p_total"),
                    "p2p_passed": evaluation.get("p2p_passed"),
                    "p2p_total": evaluation.get("p2p_total"),
                    "localization_hit": evaluation.get("localization_hit"),
                    "empty_patch": evaluation.get("empty_patch"),
                }
            )

    markdown_path.write_text(
        render_markdown_report(report, summaries, csv_path, json_path),
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=default_logs_root(),
        help="directory containing experiment run logs",
    )
    args = parser.parse_args()
    paths = write_report(args.logs_dir)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
