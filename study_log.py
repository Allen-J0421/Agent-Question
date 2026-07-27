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
    claude_bin: str,
    workspace: Path,
    prompt: str,
) -> dict[str, Any]:
    """Persist immutable pre-launch metadata and return the manifest."""
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
            "condition": condition,
            "prompt_sha256": prompt_hash(prompt),
        },
        "claude": {
            "binary": claude_bin,
            "model": model,
            "argv_shape": ["claude", "--model", model, "<prompt>"],
        },
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


def write_run_summary(logs_root: Path, summary: dict[str, Any]) -> Path:
    path = logs_root / "runs" / f"{summary['run_id']}.json"
    write_new_json(path, summary)
    return path


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


def write_report(logs_root: Path) -> dict[str, Path]:
    summaries, input_errors = load_run_summaries(logs_root)
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
            ],
        )
        writer.writeheader()
        for summary in summaries:
            ask = summary.get("ask_user_question", {})
            first = ask.get("first_direct") or {}
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
                    "monitoring_status": summary.get("transcript", {}).get("monitoring_status"),
                    "stop_reason": summary.get("process", {}).get("stop_reason"),
                    "exit_code": summary.get("process", {}).get("exit_code"),
                    "duration_seconds": summary.get("process", {}).get("duration_seconds"),
                }
            )

    runs = report["runs"]
    rate = runs["direct_ask_rate"]
    rate_text = "n/a" if rate is None else f"{rate:.1%}"
    markdown_path.write_text(
        "# AskUserQuestion report\n\n"
        f"- Runs logged: {runs['total']}\n"
        f"- Valid primary-outcome runs: {runs['valid_for_primary_outcome']}\n"
        f"- Direct AskUserQuestion runs: {runs['direct_ask_runs']} ({rate_text})\n"
        f"- Unknown runs: {runs['unknown']}\n",
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
