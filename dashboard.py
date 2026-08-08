#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from the experiment logs.

Run it with no arguments::

    python dashboard.py

It reads ``.experiment-logs/`` and rewrites ``index.html`` at the repo root --
a single file with all data inlined, so it opens by double-click with no
server and no network access. Every run is a full rebuild; there is no
incremental state to go stale.

Writing it at the repo root rather than inside the (git-ignored) log directory
keeps ``.gitignore`` simple and lets GitHub Pages serve it directly. Because
that makes the page shareable, the build scrubs local machine paths out of the
payload; see ``scrub_paths``.

The dashboard is a superset of the Markdown report: the same aggregates (taken
from ``study_log.build_report``, never recomputed), plus per-run drill-down into
the tool trace, the agent's messages, the verbatim prompt it was given, the
patch it produced, and the grader's verdict.

The Prompt panel also carries the datasets' own context for each task: both
conditions' issue text, and for ``missing-info`` the masking answer key --
which information categories were withheld from the ambiguous rewrite, the
probe question each stands for, and the spans that were cut (see
``build_keys``). That key is included on purpose: the study measures whether
an agent asks for missing information, and judging that means reading its
question next to what was actually missing. The runs are finished and
immutable, so reading the key cannot change what any agent did.

The key lives in one top-level ``keys`` block and is never merged into a run
record, so the invariant worth checking is provenance: answer-key text appears
only under ``keys``, and every rebuilt prompt still matches the hash recorded
when the agent ran.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import study_log
import locate_logs

try:  # experiment imports sdk_runner -> claude_agent_sdk, absent outside the venv.
    import experiment

    HAVE_EXPERIMENT = True
except Exception:  # pragma: no cover - depends on the interpreter used
    HAVE_EXPERIMENT = False

try:
    import codex_runner

    HAVE_CODEX = True
except Exception:  # pragma: no cover
    HAVE_CODEX = False


ROOT = Path(__file__).resolve().parent
LOGS_ROOT = ROOT / ".experiment-logs"
# Written at the repo root as index.html: it keeps .gitignore simple (the whole
# .experiment-logs/ directory stays ignored) and lets GitHub Pages serve the
# dashboard as-is. The page is therefore shareable, so the build scrubs local
# machine paths out of it -- see `scrub_paths`.
OUTPUT = ROOT / "index.html"
TEMPLATE = ROOT / "dashboard_template.html"

# Per-field cap for captured payloads. Measured: tool results have a 405-byte
# median and a 2.9KB p90, but a single Codex command emitted 189KB. The cap
# keeps the page a few MB while leaving the artifact path for the full text.
CAP_BYTES = 2000
# Commands are evidence of what the agent did, so they get a far looser cap.
COMMAND_CAP = 4000

EDIT_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}
SEARCH_TOOLS = {"WebSearch", "WebFetch", "ToolSearch"}
ASK_TOOL = "AskUserQuestion"

# Claude session records that carry no trace signal.
SKIP_RECORD_TYPES = {"queue-operation", "attachment", "ai-title", "last-prompt"}


# --------------------------------------------------------------------------
# Text handling
# --------------------------------------------------------------------------


def truncate(text: str | None, cap: int = CAP_BYTES) -> tuple[str | None, int]:
    """Cap ``text`` keeping both ends, and report how many bytes were dropped.

    Head-only truncation would systematically hide the one line a researcher
    wants: pytest, git and build output all put the verdict at the *end*. So a
    long value keeps a large head and a smaller tail with an explicit marker
    between them. Cuts are moved back to a newline where one is close by, to
    avoid splitting a line mid-token.
    """
    if text is None:
        return None, 0
    # Scrub before cutting -- a path split by the cut would no longer match the
    # scrubber and would leak a fragment into the published page.
    text = scrub_paths(text)
    raw = text.encode("utf-8")
    if len(raw) <= cap:
        return text, 0

    head_cap = int(cap * 0.7)
    tail_cap = cap - head_cap
    head = raw[:head_cap].decode("utf-8", "ignore")
    tail = raw[-tail_cap:].decode("utf-8", "ignore")
    # Prefer a line boundary when one sits near the cut.
    newline = head.rfind("\n")
    if newline > head_cap * 0.6:
        head = head[:newline]
    newline = tail.find("\n")
    if -1 < newline < len(tail) * 0.4:
        tail = tail[newline + 1 :]

    dropped = len(raw) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    marker = f"\n\n… [{dropped:,} bytes elided] …\n\n"
    return head + marker + tail, dropped


def first_line(text: str | None, limit: int = 140) -> str:
    """A one-line summary for a collapsed row.

    Scrubs before truncating: a path cut mid-string would no longer match the
    scrubber's search pattern and would survive into the published page as a
    fragment like ``/Users/allenj…``.
    """
    if not text:
        return ""
    line = " ".join(scrub_paths(str(text)).split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def normalize_content(content: Any) -> str | None:
    """Flatten a polymorphic ``tool_result.content`` into text.

    Measured across the stored sessions: 1231 plain strings, 4 lists of
    ``tool_reference`` blocks, and 1 list of ``text`` blocks. Anything else is
    JSON-encoded rather than dropped.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
            elif block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif block.get("type") == "tool_reference":
                parts.append(f"<tool_reference: {block.get('name') or '?'}>")
            else:
                parts.append(json.dumps(block))
        return "\n".join(p for p in parts if p)
    return json.dumps(content)


# --------------------------------------------------------------------------
# Trace extraction -- the one genuinely new parsing layer
# --------------------------------------------------------------------------


def scrub_paths(text: str) -> str:
    """Replace this machine's absolute paths with stable placeholders.

    The dashboard is written to the repo root so it can be shared or served by
    GitHub Pages, and the captured trace is full of absolute paths from the
    machine that ran the experiment (~1.7k of them). They identify a home
    directory and carry no analytical value, so the checkout root becomes
    ``<repo>`` and any remaining home directory becomes ``~``.
    """
    return text.replace(str(ROOT), "<repo>").replace(str(Path.home()), "~")


def repo_relative(path: str | None, workspace: str | None) -> str | None:
    """Strip the checkout prefix so a path reads as the repo sees it.

    Agents work in ``.experiment-checkouts/<repo>__<commit>__<condition>/``, so
    every edited path is absolute and the part that identifies the file is at
    the far end -- exactly what gets truncated away in a narrow column.
    """
    if not path:
        return path
    if workspace and path.startswith(workspace):
        return path[len(workspace):].lstrip("/")
    marker = ".experiment-checkouts/"
    index = path.find(marker)
    if index >= 0:
        rest = path[index + len(marker):]
        slash = rest.find("/")
        return rest[slash + 1:] if slash >= 0 else rest
    return path


def _event(index: int, kind: str, **fields: Any) -> dict[str, Any]:
    event = {"i": index, "k": kind}
    event.update({key: value for key, value in fields.items() if value not in (None, {}, [])})
    return event


def extract_claude_trace(
    session_dir: Path, started_at: str | None, workspace: str | None = None
) -> tuple[list[dict], int]:
    """Turn preserved Claude session JSONL into unified trace events.

    ``tool_use`` and its matching ``tool_result`` are merged into ONE event,
    joined on ``tool_use_id``. Codex already emits a call and its output as a
    single item, so merging is what makes the two arms structurally comparable
    -- otherwise Claude reports twice the events for identical work.
    """
    files = sorted(
        path
        for path in session_dir.rglob("*.jsonl")
        if not path.name.startswith(("codex-", "rollout"))
    )
    records: list[tuple[Path, dict]] = []
    parse_errors = 0
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append((path, json.loads(line)))
            except json.JSONDecodeError:
                parse_errors += 1

    # Pass 1: collect results so a call can carry its own output.
    results: dict[str, dict] = {}
    for _, record in records:
        if record.get("type") != "user":
            continue
        content = (record.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue  # bare-string content is an injected turn, handled below
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = block

    start = study_log._parse_timestamp(started_at) if started_at else None
    events: list[dict] = []
    subagent_files = {path for path in files if "subagents" in path.parts}

    for path, record in records:
        rtype = record.get("type")
        if rtype in SKIP_RECORD_TYPES or rtype not in {"assistant", "user"}:
            continue
        timestamp = record.get("timestamp")
        moment = study_log._parse_timestamp(timestamp)
        delta = (moment - start).total_seconds() if (moment and start) else None
        is_sub = path in subagent_files
        content = (record.get("message") or {}).get("content")

        if isinstance(content, str):
            # The injected prompt / ask-answer turns arrive as bare strings.
            if rtype == "user" and content.strip():
                events.append(
                    _event(
                        0,
                        "msg",
                        t=timestamp,
                        dt=delta,
                        name="user",
                        title=first_line(content),
                        body=truncate(content)[0],
                        meta={"role": "user", "subagent": is_sub or None},
                    )
                )
            continue
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text") or ""
                if not text.strip():
                    continue
                body, dropped = truncate(text)
                events.append(
                    _event(
                        0, "msg", t=timestamp, dt=delta, name="assistant",
                        title=first_line(text), body=body,
                        trunc={"body": dropped} if dropped else None,
                        meta={"role": "assistant", "subagent": is_sub or None},
                    )
                )
            elif btype == "thinking":
                # Measured: all 683 stored thinking blocks are signature-only
                # with empty text -- the reasoning itself was never persisted.
                # Kept for the day it is, but it emits nothing today.
                text = block.get("thinking") or ""
                if not text.strip():
                    continue
                body, dropped = truncate(text)
                events.append(
                    _event(
                        0, "think", t=timestamp, dt=delta,
                        title=first_line(text), body=body,
                        trunc={"body": dropped} if dropped else None,
                        meta={"subagent": is_sub or None},
                    )
                )
            elif btype == "tool_use":
                name = block.get("name") or "?"
                caller = block.get("caller")
                caller_type = caller.get("type") if isinstance(caller, dict) else None
                result = results.get(block.get("id")) or {}
                out, out_dropped = truncate(normalize_content(result.get("content")))
                payload = block.get("input") or {}
                # A command reads better raw than as JSON.
                if name == "Bash" and isinstance(payload.get("command"), str):
                    body, body_dropped = truncate(payload["command"], COMMAND_CAP)
                    title = first_line(payload.get("description") or payload["command"])
                else:
                    body, body_dropped = truncate(json.dumps(payload, indent=1))
                    title = first_line(
                        repo_relative(payload.get("file_path"), workspace)
                        or payload.get("pattern")
                        or payload.get("query")
                        or payload.get("description")
                        or ""
                    )
                if name == ASK_TOOL:
                    kind, flag = "ask", "ask"
                    title = f"Asked {len(payload.get('questions') or [])} question(s)"
                elif name in EDIT_TOOLS:
                    kind, flag = "edit", "edit"
                    title = first_line(
                        repo_relative(payload.get("file_path"), workspace) or title
                    )
                elif name in SEARCH_TOOLS:
                    kind, flag = "search", None
                else:
                    kind, flag = "tool", None
                is_error = bool(result.get("is_error"))
                if is_error:
                    flag = "error"
                events.append(
                    _event(
                        0, kind, t=timestamp, dt=delta, name=name,
                        title=title or name, body=body, out=out,
                        ok=(not is_error) if result else None,
                        flag=flag,
                        trunc={
                            key: value
                            for key, value in
                            (("body", body_dropped), ("out", out_dropped))
                            if value
                        } or None,
                        meta={
                            "tool_use_id": block.get("id"),
                            "caller": caller_type,
                            "subagent": is_sub or None,
                            "questions": payload.get("questions") if kind == "ask" else None,
                        },
                    )
                )

    events.sort(key=lambda event: (event.get("t") or "", event["i"]))
    for index, event in enumerate(events):
        event["i"] = index
    return events, parse_errors


def _round_index(path: Path) -> int:
    match = re.search(r"round-(\d+)", path.name)
    return int(match.group(1)) if match else 0


def extract_codex_trace(
    session_dir: Path, workspace: str | None = None
) -> tuple[list[dict], int]:
    """Turn Codex round event files into unified trace events.

    Codex items carry no timestamps, so these events are ordinal-only; the UI
    says so rather than interpolating a fake clock.
    """
    events: list[dict] = []
    parse_errors = 0
    files = sorted(session_dir.glob("codex-events-round-*.jsonl"), key=_round_index)
    for path in files:
        rnd = _round_index(path)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if record.get("type") != "item.completed":
                continue
            item = record.get("item") or {}
            itype = item.get("type")

            if itype == "command_execution":
                command = item.get("command") or ""
                body, body_dropped = truncate(command, COMMAND_CAP)
                out, out_dropped = truncate(item.get("aggregated_output"))
                exit_code = item.get("exit_code")
                ok = exit_code == 0 if exit_code is not None else None
                events.append(
                    _event(
                        0, "tool", r=rnd, name="shell", title=first_line(command),
                        body=body, out=out, ok=ok,
                        flag=None if ok is not False else "error",
                        trunc={
                            key: value for key, value in
                            (("body", body_dropped), ("out", out_dropped)) if value
                        } or None,
                        meta={"exit_code": exit_code, "status": item.get("status")},
                    )
                )
            elif itype == "agent_message":
                text = item.get("text") or ""
                body, dropped = truncate(text)
                events.append(
                    _event(
                        0, "msg", r=rnd, name="assistant", title=first_line(text),
                        body=body, trunc={"body": dropped} if dropped else None,
                        meta={"role": "assistant"},
                    )
                )
            elif itype == "file_change":
                changes = item.get("changes") or []
                paths = [
                    repo_relative(change.get("path"), workspace)
                    for change in changes if isinstance(change, dict)
                ]
                events.append(
                    _event(
                        0, "edit", r=rnd, name="apply_patch",
                        title=first_line(", ".join(p for p in paths if p)) or "file change",
                        body=truncate("\n".join(
                            f"{c.get('kind','?'):8s} {repo_relative(c.get('path'), workspace)}"
                            for c in changes if isinstance(c, dict)
                        ))[0],
                        flag="edit", meta={"paths": paths},
                    )
                )
            elif itype == "web_search":
                query = item.get("query") or ""
                events.append(
                    _event(
                        0, "search", r=rnd, name="web_search", title=first_line(query),
                        body=truncate(query)[0],
                        out=truncate(json.dumps(item.get("action") or {}, indent=1))[0],
                        meta={"query": query},
                    )
                )
    for index, event in enumerate(events):
        event["i"] = index
    return events, parse_errors


# --------------------------------------------------------------------------
# Patches, prompts, swebench reports
# --------------------------------------------------------------------------


TEST_HINTS = ("test_", "_test.py", "/tests/", "tests/")


def parse_patch(text: str, gold: list[str] | None = None) -> list[dict]:
    """Per-file added/removed counts from a unified diff."""
    gold_set = set(gold or [])
    files: list[dict] = []
    current: dict | None = None
    for line in (text or "").splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            path = match.group(2) if match else line[len("diff --git ") :]
            current = {
                "path": path,
                "added": 0,
                "removed": 0,
                "is_test": any(hint in path for hint in TEST_HINTS),
                "in_gold": path in gold_set,
            }
            files.append(current)
        elif current is None:
            continue
        elif line.startswith("+") and not line.startswith("+++"):
            current["added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            current["removed"] += 1
    return files


def load_swebench_tests(logs: Path, evaluation: dict, instance_id: str) -> dict | None:
    """Per-test F2P/P2P names from the harness's own report for this instance."""
    run_id = evaluation.get("swebench_run_id")
    if not run_id:
        return None
    path = (
        logs / "swebench" / "logs" / "run_evaluation" / run_id
        / "ambig-swe" / instance_id / "report.json"
    )
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    entry = data.get(instance_id) or {}
    status = entry.get("tests_status") or {}

    def summarize(bucket: dict) -> dict:
        """Keep every failure; cap the passes.

        Failures are what a researcher reads. Passing PASS_TO_PASS lists run to
        732 names on a single instance -- a megabyte of test IDs nobody
        scrolls -- so they are counted and sampled instead of inlined whole.
        """
        success = bucket.get("success") or []
        failure = bucket.get("failure") or []
        return {
            "passed": len(success),
            "failed": len(failure),
            "failures": failure,
            "sample": success[:20],
            "more": max(0, len(success) - 20),
        }

    return {
        "f2p": summarize(status.get("FAIL_TO_PASS") or {}),
        "p2p": summarize(status.get("PASS_TO_PASS") or {}),
        "applied": entry.get("patch_successfully_applied"),
        "path": str(path),
    }


def relative_artifacts(artifacts: dict, logs: Path) -> dict:
    """Shorten artifact paths against the logs root so they stay readable."""
    root = str(logs.resolve())
    trimmed = {}
    for slot, info in artifacts.items():
        entry = dict(info)
        path = entry.get("path") or ""
        if path.startswith(root):
            entry["path"] = ".experiment-logs" + path[len(root):]
        trimmed[slot] = entry
    return trimmed


def rebuild_prompt(summary: dict, cache: dict[str, dict]) -> dict:
    """Reconstruct the verbatim prompt and prove it against the stored hash.

    Prompts are never stored in the logs -- only ``task.prompt_sha256`` -- so
    the text is rebuilt from the dataset and verified. A mismatch means the
    dataset changed under the run and the text shown would be a lie, so it is
    surfaced rather than silently displayed.
    """
    task = summary.get("task", {})
    expected = task.get("prompt_sha256")
    if not HAVE_EXPERIMENT:
        return {"source": "unavailable", "expected_sha256": expected, "verified": False}
    dataset = study_log.dataset_of(summary) or "interactive-swe"
    condition = task.get("condition")
    try:
        if dataset not in cache:
            cache[dataset] = experiment.load_rows(dataset)
        row = cache[dataset][task["instance_id"]]
        text = experiment.build_prompt(row, condition)
    except Exception as exc:  # missing row, unknown condition, unreadable dataset
        return {
            "source": "unavailable",
            "expected_sha256": expected,
            "verified": False,
            "error": str(exc),
        }
    actual = study_log.prompt_hash(text)
    return {
        "source": "rebuilt",
        "text": text,
        "sha256": actual,
        "expected_sha256": expected,
        "verified": actual == expected,
        "chars": len(text),
        "condition": condition,
        "dataset": dataset,
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_run(summary: dict, logs: Path, prompt_cache: dict, warnings: list[str]) -> dict:
    run_id = summary["run_id"]
    task = summary.get("task", {})
    agent = study_log.agent_info(summary)
    runner = agent.get("runner") or "claude-sdk"
    model = agent.get("model") or "unknown"
    ask = summary.get("ask_user_question", {})
    evaluation = summary.get("evaluation", {}) or {}
    process = summary.get("process", {})
    session = summary.get("session", {})
    condition = task.get("condition")
    instance_id = task.get("instance_id")

    session_dir = logs / "sessions" / run_id
    workspace = summary.get("workspace")
    if runner.startswith("codex"):
        trace, parse_errors = (
            extract_codex_trace(session_dir, workspace) if session_dir.is_dir() else ([], 0)
        )
    else:
        trace, parse_errors = (
            extract_claude_trace(session_dir, summary.get("started_at"), workspace)
            if session_dir.is_dir() else ([], 0)
        )
    if parse_errors:
        warnings.append(f"{run_id[:8]}: {parse_errors} unparsed session line(s)")
    if not trace and session_dir.is_dir():
        warnings.append(f"{run_id[:8]}: session files present but no trace events extracted")

    # Cross-check the Codex tool count against the runner's own parser.
    if HAVE_CODEX and runner.startswith("codex") and session_dir.is_dir():
        try:
            lines: list[str] = []
            for path in sorted(session_dir.glob("codex-events-round-*.jsonl"), key=_round_index):
                lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
            parsed = codex_runner.parse_events(lines)
            mine = sum(1 for event in trace if event["k"] in {"tool", "search", "edit"})
            theirs = parsed.get("tool_actions", 0)
            if mine != theirs:
                warnings.append(
                    f"{run_id[:8]}: tool count {mine} != parse_events {theirs}"
                )
        except Exception as exc:
            warnings.append(f"{run_id[:8]}: parse_events cross-check failed: {exc}")

    direct_asked = ask.get("direct_asked")
    if direct_asked is not None and not isinstance(direct_asked, bool):
        warnings.append(f"{run_id[:8]}: direct_asked is {type(direct_asked).__name__}")

    patch_path = logs / "patches" / f"{run_id}.patch"
    patch_text = (
        patch_path.read_text(encoding="utf-8", errors="replace")
        if patch_path.is_file() else ""
    )
    gold = evaluation.get("gold_files") or []

    prompt = rebuild_prompt(summary, prompt_cache)
    if prompt.get("source") == "rebuilt" and not prompt.get("verified"):
        warnings.append(f"{run_id[:8]}: prompt hash MISMATCH for {instance_id}")

    first = ask.get("first_direct") or {}
    counts = {
        "events": len(trace),
        "tools": sum(1 for e in trace if e["k"] in {"tool", "search"}),
        "edits": sum(1 for e in trace if e["k"] == "edit"),
        "messages": sum(1 for e in trace if e["k"] == "msg"),
        "errors": sum(1 for e in trace if e.get("flag") == "error"),
    }
    search_blob = " ".join(
        filter(None, [
            instance_id, run_id, task.get("repo"), condition, model, runner,
            " ".join(e.get("title") or "" for e in trace),
            " ".join(e.get("name") or "" for e in trace),
            " ".join(f["path"] for f in parse_patch(patch_text, gold)),
        ])
    ).lower()

    return {
        "run_id": run_id,
        "short_id": run_id[:8],
        "dataset": study_log.dataset_of(summary) or "interactive-swe",
        "instance_id": instance_id,
        "repo": task.get("repo"),
        "condition": condition,
        "condition_kind": "full" if (condition or "").endswith("full") else "ambiguous",
        "model": model,
        "runner": runner,
        "arm": f"{runner}::{model}",
        "ask_channel": study_log.ask_channel_of(summary),
        "difficulty": task.get("difficulty"),
        "base_commit": task.get("base_commit"),
        "started_at": summary.get("started_at"),
        "duration_seconds": process.get("duration_seconds"),
        "ask": {
            "direct_asked": direct_asked,
            "direct_count": ask.get("direct_count"),
            "any_agent_count": ask.get("any_agent_count"),
            "latency_seconds": ask.get("first_direct_latency_seconds"),
            "tool_actions_before": first.get("assistant_tool_actions_before"),
            "questions": ask.get("answered_questions") or [],
            "channel": study_log.ask_channel_of(summary),
            "valid": isinstance(direct_asked, bool),
        },
        "eval": {
            "status": evaluation.get("status"),
            "resolved": evaluation.get("resolved"),
            "f2p_passed": evaluation.get("f2p_passed"),
            "f2p_total": evaluation.get("f2p_total"),
            "p2p_passed": evaluation.get("p2p_passed"),
            "p2p_total": evaluation.get("p2p_total"),
            "localization_hit": evaluation.get("localization_hit"),
            "empty_patch": evaluation.get("empty_patch"),
            "gold_files": gold,
            "agent_source_files": evaluation.get("agent_source_files") or [],
            "agent_test_files": evaluation.get("agent_test_files") or [],
            "swebench_run_id": evaluation.get("swebench_run_id"),
            "error": evaluation.get("error"),
            "valid": evaluation.get("status") == "scored",
            "tests": load_swebench_tests(logs, evaluation, instance_id or ""),
        },
        "process": {
            "exit_code": process.get("exit_code"),
            "stop_reason": process.get("stop_reason"),
            "num_turns": process.get("sdk_num_turns") or process.get("num_turns"),
            "cost_usd": process.get("sdk_total_cost_usd"),
            "session_id": process.get("sdk_session_id") or process.get("codex_thread_id"),
            "operator_interrupted": process.get("operator_interrupted"),
            "monitoring_status": session.get("monitoring_status"),
            "ran_meaningfully": session.get("ran_meaningfully"),
        },
        "counts": counts,
        "prompt": prompt,
        "trace": trace,
        "patch": {
            "text": patch_text,
            "bytes": len(patch_text.encode("utf-8")),
            "files": parse_patch(patch_text, gold),
        },
        "artifacts": relative_artifacts(locate_logs.artifact_map(logs, run_id), logs),
        "_search": search_blob,
    }


def build_pairs(runs: list[dict]) -> list[dict]:
    """Group runs that differ only by condition, within one dataset and model."""
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        groups.setdefault(
            (run["dataset"], run["model"], run["instance_id"]), []
        ).append(run)
    pairs = []
    for (dataset, model, instance_id), members in sorted(groups.items()):
        ambiguous = next((r for r in members if r["condition_kind"] == "ambiguous"), None)
        full = next((r for r in members if r["condition_kind"] == "full"), None)
        pairs.append({
            "pair_id": f"{dataset}::{model}::{instance_id}",
            "dataset": dataset,
            "model": model,
            "instance_id": instance_id,
            "ambiguous_run": ambiguous["run_id"] if ambiguous else None,
            "full_run": full["run_id"] if full else None,
            "complete": bool(ambiguous and full),
        })
    return pairs


def build_keys(instance_ids: set[str], warnings: list[str]) -> dict[str, dict]:
    """Per-instance dataset context for the Prompt panel's sub-views.

    Keyed by ``instance_id``, never by ``run_id``: the runs share far fewer
    instances than there are runs (58 runs over 14 instances today), so a
    per-run copy would multiply this severalfold for no gain.

    This block carries the masking answer key -- what was withheld from each
    ambiguous rewrite. That is deliberate: the study measures whether an agent
    asks for the missing information, and judging that requires the question
    and the key side by side. The runs are finished and immutable, so reading
    the key cannot change what any agent did.

    ``clarification_questions_gpt5_nano_3`` is dropped (674KB at 500 instances,
    and the GPT-5 baseline dominates it on every metric in the workbook's own
    table). ``category_mapping`` and ``hints_text`` are dropped too -- both are
    larger and riskier than what they add, and everything scoreable in them is
    already in ``hidden``.
    """
    if not HAVE_EXPERIMENT:
        return {}
    try:
        import datasets_registry

        swe = experiment.load_rows("interactive-swe")
        mi = experiment.load_rows("missing-info")
        answer_keys = datasets_registry.load_answer_keys()
    except Exception as exc:
        warnings.append(f"dataset context unavailable: {exc}")
        return {}

    keys: dict[str, dict] = {}
    repaired_ids: list[str] = []
    for instance_id in sorted(instance_ids):
        swe_row = swe.get(instance_id) or {}
        mi_row = mi.get(instance_id) or {}
        key = answer_keys.get(instance_id) or {}
        if key.get("repaired"):
            repaired_ids.append(instance_id)
        baselines = key.get("baselines") or {}
        keys[instance_id] = {
            # original_issue is byte-identical across both datasets (verified
            # 500/500), so one copy serves either arm.
            "issue_full": swe_row.get("original_issue") or mi_row.get("original_issue"),
            "amb_swe": swe_row.get("problem_statement"),
            "amb_mi": mi_row.get("rewrite_3"),
            "hidden": key.get("hidden") or [],
            "hidden_cats": key.get("hidden_categories") or [],
            "present": key.get("present_categories") or [],
            "repaired": bool(key.get("repaired")),
            "baselines": {
                "gpt5": baselines.get("gpt5"),
                "grpo": baselines.get("grpo"),
            },
        }
    if repaired_ids:
        warnings.append(
            {
                "text": (
                    f"{len(repaired_ids)} instance(s) needed the Implementation "
                    "Details label repair (the probe lost its category name "
                    "upstream); category counts read from hidden_categories_3 "
                    "alone would undercount them"
                ),
                # The ids are what turns this from "something is broken
                # somewhere" into a list the reader can click through.
                "instances": repaired_ids,
            }
        )
    return keys


def check_masking(runs: list[dict], keys: dict[str, dict]) -> list[str]:
    """Assert the masking held: no withheld span survives in an ambiguous prompt.

    The hidden spans are verbatim excerpts of the *original* issue, so finding
    them in a ``full`` prompt is the expected case -- that text is the control
    condition. Finding one in an ambiguous prompt would mean the rewrite failed
    to remove what its answer key claims was removed, which would invalidate
    that instance's ask measurement.
    """
    problems: list[str] = []
    for run in runs:
        if run.get("condition_kind") != "ambiguous":
            continue
        key = keys.get(run["instance_id"])
        if not key:
            continue
        text = " ".join((run["prompt"].get("text") or "").split())
        for segment in key["hidden"]:
            for example in segment["examples"]:
                span = " ".join(example.split())
                if len(span) > 40 and span in text:
                    problems.append(
                        f"{run['short_id']}: {segment['category']} span survives "
                        f"in the {run['condition']} prompt it was masked from"
                    )
    return problems


def normalize_warning(warning: Any, runs: list[dict]) -> dict:
    """Give every build warning the same shape: ``{text, instances, runs}``.

    Most warnings are bare strings raised far from here, so the page would have
    to special-case two shapes to render them. Normalising once keeps the
    template's job to display-only. Warnings that name instances get the
    matching runs resolved to ``short_id``/``condition``/``model`` so the reader
    can locate the affected rows instead of guessing what the warning covers.
    """
    if isinstance(warning, str):
        return {"text": warning, "instances": [], "runs": []}

    instances = list(warning.get("instances") or [])
    affected = set(instances)
    return {
        "text": warning.get("text", ""),
        "instances": instances,
        "runs": [
            {
                "short_id": run.get("short_id"),
                "instance_id": run["instance_id"],
                "condition": run.get("condition"),
                "model": run.get("model"),
            }
            for run in runs
            if run["instance_id"] in affected
        ],
    }


def build_payload(logs: Path) -> dict:
    summaries, errors = study_log.load_run_summaries(logs)
    # attach_evaluations is pure -- it returns a NEW list. Ignoring the return
    # value silently drops nearly every grade.
    summaries = study_log.attach_evaluations(summaries, study_log.load_evaluations(logs))
    summaries.sort(key=lambda s: s.get("started_at") or "")

    warnings = list(errors)
    prompt_cache: dict[str, dict] = {}
    runs = [build_run(summary, logs, prompt_cache, warnings) for summary in summaries]
    report = study_log.build_report(summaries, errors)

    if not HAVE_EXPERIMENT:
        warnings.append(
            "experiment module unavailable - prompts could not be rebuilt "
            "(run with the project venv interpreter)"
        )

    total_dropped = sum(
        value
        for run in runs for event in run["trace"]
        for value in (event.get("trunc") or {}).values()
    )
    # Built before the literal below so its warnings are in `warnings` by the
    # time the meta block reads it, rather than depending on evaluation order.
    keys = build_keys({r["instance_id"] for r in runs}, warnings)
    warnings.extend(check_masking(runs, keys))
    warnings = [normalize_warning(w, runs) for w in warnings]
    return {
        "meta": {
            "generated_at": study_log.utc_now(),
            "logs_root": str(logs),
            "git_commit": git_commit(),
            "cap_bytes": CAP_BYTES,
            "counts": {
                "runs": len(runs),
                "instances": len({r["instance_id"] for r in runs}),
                "events": sum(len(r["trace"]) for r in runs),
                "pairs": sum(1 for p in build_pairs(runs) if p["complete"]),
                "bytes_dropped": total_dropped,
            },
            "build_warnings": warnings,
        },
        "report": report,
        "runs": runs,
        "pairs": build_pairs(runs),
        "keys": keys,
    }


def render_html(payload: dict) -> str:
    """Inline the payload into the template.

    The data is full of shell commands, diffs and HTML fixtures, so a literal
    ``</script>`` inside any captured output would end the script tag early and
    silently truncate the page. Escaping the two sequences that can close it is
    what makes inlining safe.
    """
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    blob = scrub_paths(blob)
    blob = blob.replace("</", "<\\/").replace("<!--", "<\\!--")
    template = TEMPLATE.read_text(encoding="utf-8")
    return template.replace("/*__DATA__*/", blob)


def main() -> int:
    if not LOGS_ROOT.is_dir():
        print(f"no logs directory at {LOGS_ROOT}", file=sys.stderr)
        return 1
    payload = build_payload(LOGS_ROOT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = render_html(payload)
    OUTPUT.write_text(document, encoding="utf-8")

    counts = payload["meta"]["counts"]
    # Measure what was written rather than stat-ing the path: `Path` caches the
    # stat taken by the mkdir above, which reports a stale size here.
    size_mb = len(document.encode("utf-8")) / 1e6
    print(
        f"Wrote {OUTPUT} ({size_mb:.1f} MB)\n"
        f"  {counts['runs']} runs · {counts['instances']} instances · "
        f"{counts['events']} events · {counts['pairs']} complete pairs"
    )
    warnings = payload["meta"]["build_warnings"]
    if warnings:
        print(f"  {len(warnings)} build warning(s):")
        for warning in warnings[:10]:
            print(f"    - {warning['text']}")
            for instance_id in warning["instances"]:
                print(f"        · {instance_id}")
        if len(warnings) > 10:
            print(f"    … and {len(warnings) - 10} more (shown in the page)")
    else:
        print("  no build warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
