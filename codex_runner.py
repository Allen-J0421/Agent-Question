"""Run one unattended Codex CLI session (GPT models) on a workspace.

The GPT arm of the study runs through the stock ``codex exec`` CLI, not an
SDK, and deliberately does **not** copy the Claude arm's tool configuration:
Codex has no AskUserQuestion-style tool in its vanilla toolset (the
``default_mode_request_user_input`` feature exists but ships disabled), and
injecting one would hand the model a signpost saying "asking is expected
here" -- exactly the bias the study must avoid. In Codex's natural setting
the only way for the agent to ask the user anything is to end its turn with
a question in its final message. That turn yield *is* the ask channel, so it
is what this runner observes.

Session shape:

* ``codex exec --json`` with ``--ignore-user-config`` and ``--ignore-rules``
  so the operator's personal config (personality, temperature, plugins,
  notify hooks, execpolicy rules) never leaks into the experiment;
  authentication still comes from ``$CODEX_HOME``. No extra instructions, no
  MCP servers, no feature flags: the model decides on its own judgement.
* ``--sandbox danger-full-access`` for parity with the Claude arm, whose
  ``can_use_tool`` callback approves every tool call. A workspace-write
  sandbox would tell the model "network restricted, approvals unavailable"
  -- an environmental input the Claude arm never sees, which would bias the
  comparison. Same warning as the Claude arm: only run this on machines
  where unrestricted tool execution is acceptable.
* When a turn ends with a clarifying question (deterministic classifier
  below), the question is recorded as the primary outcome and the session is
  resumed (``codex exec resume <thread_id>``) with the same tie-break the
  Claude arm applies -- take the first option offered -- so the run can
  still produce a gradable patch. Halting at the first ask would leave
  asking runs patchless and bias the asked-vs-not-asked comparison. Both
  arms answer up to the same number of asks per run (``MAX_ASK_ROUNDS``,
  symmetric with ``sdk_runner``); an ask beyond the cap ends the session
  with its state captured.

Ask detection is a versioned, deterministic two-layer classifier. Layer 1
is a **zero-edit turn gate** enforced here: empirically (36/36 harvested
real turns) asking and editing are mutually exclusive per turn, so a turn
that edited (file_change items, or a git fingerprint delta) is never an
ask regardless of its text -- ``question_with_edits`` is recorded if the
regex would have fired, keeping the assumption auditable. Layer 2
(``ask_detection`` module + ``config/ask_detection.json``) then only
separates zero-edit asks from zero-edit reports. The raw message text,
classifier version, gate evidence, and matched categories are all
recorded, and every event stream is preserved on disk, so any future rule
can be re-benchmarked over past runs with ``reclassify_asks.py`` without
re-running a session.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import ask_detection

RUNNER = "codex-cli"
ASK_CHANNEL = "final_message"
# The classifier lives in ask_detection.py + config/ask_detection.json
# (versioned pattern dictionary; architecture adopted from the
# sycophancy-ACE refusal detector). The version stamped on every run is the
# config's, so a pattern change is always visible in the data.
ASK_CLASSIFIER_VERSION = ask_detection.CLASSIFIER_VERSION

SANDBOX_MODE = "danger-full-access"
# Up to three synthetic answers per run; the same cap applies to the Claude
# arm (sdk_runner.MAX_ASK_ROUNDS), so neither arm can out-ask the other by
# harness construction. An ask beyond the cap ends the run with
# ``stop_reason: "max_ask_rounds"`` (recorded, patch captured as-is).
MAX_ASK_ROUNDS = 3
ROUND_TIMEOUT_SECONDS = 3600

# gpt-5.6-sol is the study's primary GPT target; every gpt-*/codex-* slug the
# CLI accepts will run, and the preflight canary certifies the ask channel
# for whichever model a batch selects before any session is spent on it.
PRIMARY_GPT_MODEL = "gpt-5.6-sol"
KNOWN_GPT_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra")

# codex-cli 0.139 was rejected server-side for the gpt-5.6 family ("requires
# a newer version of Codex", HTTP 400); 0.146.0 is the earliest version
# verified to run gpt-5.6-sol end-to-end.
MIN_CLI_VERSION = (0, 146, 0)

# The synthetic reply to a clarifying question: the same first-option
# tie-break the Claude arm's AskUserQuestion callback applies, phrased for a
# free-text channel (Codex questions carry no machine-readable option list,
# so the rule is stated instead of a label being selected). Deliberately
# content-free -- it adds no task information -- and recorded verbatim in
# every summary so its influence on the final patch stays auditable.
NEUTRAL_ANSWER = "Go with the first option you presented."

# Item types that represent the agent acting on the world, used for the
# "tool actions before first ask" measure (agent_message and reasoning items
# are words, not actions).
TOOL_ITEM_TYPES = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
    "patch_apply",
}

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


# --------------------------------------------------------------------------
# CLI availability
# --------------------------------------------------------------------------


def codex_cli_version(codex_bin: str = "codex") -> str | None:
    """Return the ``codex --version`` string, or None if the CLI is unusable."""
    try:
        probe = subprocess.run(
            [codex_bin, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (probe.stdout or "").strip()
    return text or None


def parse_cli_version(text: str | None) -> tuple[int, int, int] | None:
    match = _VERSION.search(text or "")
    return tuple(int(part) for part in match.groups()) if match else None


def require_supported_cli(model: str, codex_bin: str = "codex") -> str:
    """Fail fast, before any manifest is written, if the CLI cannot run ``model``.

    The failure mode this guards against is silent and expensive: an old CLI
    starts a session, the server rejects the model with HTTP 400, and the
    batch records a dead run per instance. Verified live: codex-cli 0.139.0
    is rejected for gpt-5.6-sol; 0.146.0 works.
    """
    version_text = codex_cli_version(codex_bin)
    if version_text is None:
        raise RuntimeError(
            "the `codex` CLI is not available on PATH; install or upgrade it "
            "with: npm install -g @openai/codex@latest"
        )
    version = parse_cli_version(version_text)
    if version is not None and version < MIN_CLI_VERSION:
        wanted = ".".join(str(part) for part in MIN_CLI_VERSION)
        raise RuntimeError(
            f"{version_text} is too old for {model}: the server rejects "
            f"gpt-5.6 models below codex-cli {wanted}. Upgrade with: "
            "npm install -g @openai/codex@latest"
        )
    return version_text


# --------------------------------------------------------------------------
# Pure helpers (offline-testable)
# --------------------------------------------------------------------------


def is_clarifying_question(text: str | None) -> bool:
    """Deterministic ask classifier, applied to the turn's *final* message.

    The message that yields the turn back to the user is the only place a
    vanilla Codex agent can ask anything; commentary messages emitted while
    the agent keeps working are never blocking questions. The rule itself
    lives in ``ask_detection`` (layer 2: question units + blockers +
    no-question-mark requests; see ``config/ask_detection.json``). Callers
    own layer 1, the zero-edit turn gate.
    """
    return ask_detection.is_clarifying_question(text)


def _base_flags() -> list[str]:
    # --skip-git-repo-check is passed unconditionally so the same argv shape
    # runs in real checkouts and in the preflight's throwaway directory.
    return ["--json", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"]


def exec_argv(model: str, prompt: str, codex_bin: str = "codex") -> list[str]:
    return [
        codex_bin,
        "exec",
        *_base_flags(),
        "--sandbox",
        SANDBOX_MODE,
        "-m",
        model,
        prompt,
    ]


def resume_argv(
    model: str, thread_id: str, prompt: str, codex_bin: str = "codex"
) -> list[str]:
    # `resume` has no --sandbox flag; the -c override pins the same policy
    # instead of relying on session inheritance.
    return [
        codex_bin,
        "exec",
        "resume",
        thread_id,
        *_base_flags(),
        "-c",
        f'sandbox_mode="{SANDBOX_MODE}"',
        "-m",
        model,
        prompt,
    ]


def parse_events(lines: list[str]) -> dict[str, Any]:
    """Digest one round's ``--json`` event stream (JSONL, one event per line).

    Event shapes verified live against codex-cli 0.146.0:
    ``thread.started`` (thread_id), ``item.started/updated/completed`` with
    item types like ``agent_message`` / ``command_execution``,
    ``turn.completed`` (token usage), ``turn.failed`` / ``error`` on failure.
    """
    thread_id: str | None = None
    items: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    turn_failed: str | None = None
    fatal_error: str | None = None
    parse_errors = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(event, dict):
            parse_errors += 1
            continue
        kind = event.get("type")
        if kind == "thread.started":
            thread_id = event.get("thread_id")
        elif kind == "item.completed":
            item = event.get("item")
            if isinstance(item, dict):
                items.append(item)
        elif kind == "turn.completed":
            usage = event.get("usage")
        elif kind == "turn.failed":
            error = event.get("error")
            turn_failed = (
                error.get("message") if isinstance(error, dict) else str(error)
            )
        elif kind == "error":
            fatal_error = str(event.get("message"))

    agent_messages = [
        item.get("text")
        for item in items
        if item.get("type") == "agent_message" and isinstance(item.get("text"), str)
    ]
    tool_actions = sum(1 for item in items if item.get("type") in TOOL_ITEM_TYPES)
    file_changes = sum(1 for item in items if item.get("type") == "file_change")

    return {
        "thread_id": thread_id,
        "agent_messages": agent_messages,
        "final_message": agent_messages[-1] if agent_messages else None,
        "tool_actions": tool_actions,
        # Codex emits one file_change item per apply_patch edit; this is the
        # primary evidence for the zero-edit turn gate (verified: fired in
        # 6/6 real editing turns, 0/30 real asking turns).
        "file_changes": file_changes,
        "usage": usage,
        "turn_failed": turn_failed,
        "fatal_error": fatal_error,
        "parse_errors": parse_errors,
        "turn_completed": usage is not None,
    }


def _accumulate_usage(total: dict[str, int], usage: dict[str, Any] | None) -> None:
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + int(value)


def _workspace_has_changes(workspace: Path) -> bool | None:
    """Best-effort ``git status`` probe; None when the answer is unknowable."""
    try:
        probe = subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    return bool(probe.stdout.strip())


def _workspace_fingerprint(workspace: Path) -> str | None:
    """Hash of the workspace's git-visible state; None outside a git repo.

    Backstop for the zero-edit gate: comparing fingerprints before and after
    a turn catches shell-based edits (``sed -i``, redirects) that produce no
    ``file_change`` item. Covers tracked content (``git diff HEAD``) and
    untracked names (``status --porcelain``); side-effect files git ignores
    (``__pycache__`` etc.) deliberately do not count as edits.
    """
    try:
        status = subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain"],
            capture_output=True, text=True, timeout=60,
        )
        diff = subprocess.run(
            ["git", "-C", str(workspace), "diff", "HEAD"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if status.returncode != 0 or diff.returncode != 0:
        return None
    payload = status.stdout + "\0" + diff.stdout
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


# --------------------------------------------------------------------------
# Process boundary
# --------------------------------------------------------------------------


def _run_exec(
    argv: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool]:
    """Run one codex invocation; stream events to disk. Returns (rc, timed_out).

    stdin is closed so the CLI never blocks waiting for piped input, and the
    process gets its own group so a timeout kill cannot touch the launcher.
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as err_handle:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=out_handle,
            stderr=err_handle,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_seconds), False
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
            return process.poll() if process.poll() is not None else -1, True


# --------------------------------------------------------------------------
# Session loop
# --------------------------------------------------------------------------


def run_codex_session(
    *,
    prompt: str,
    workspace: Path,
    model: str,
    events_dir: Path,
    max_ask_rounds: int = MAX_ASK_ROUNDS,
    timeout_seconds: int = ROUND_TIMEOUT_SECONDS,
    codex_bin: str = "codex",
    run_exec: Callable[..., tuple[int, bool]] | None = None,
) -> dict[str, Any]:
    """Run one unattended Codex session and return a structured observation.

    Mirrors the observation contract of ``sdk_runner.run_sdk_session``
    (``analysis.first_direct`` / ``direct_count`` / ``any_agent_count``,
    ``result``, ``answered_questions``) so the downstream summary, report,
    and evaluation stages treat both runners uniformly.

    A "round" is one user turn: the task prompt first, then one synthetic
    neutral answer per clarifying question, up to ``max_ask_rounds`` answers.
    Every round's raw event stream lands in ``events_dir`` before any
    interpretation happens, so the ask classification is re-derivable
    offline.
    """
    run_exec = run_exec or _run_exec
    events_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    rounds: list[dict[str, Any]] = []
    answered_questions: list[dict[str, Any]] = []
    first_direct: dict[str, Any] | None = None
    direct_count = 0
    tool_actions_total = 0
    usage_total: dict[str, int] = {}
    thread_id: str | None = None

    stop_reason = "completed"
    is_error = False
    error_text: str | None = None

    round_prompt = prompt
    prompt_kind = "task"
    questions_with_edits = 0
    fingerprint_before = _workspace_fingerprint(workspace)

    while True:
        round_index = len(rounds) + 1
        if round_index == 1:
            argv = exec_argv(model, round_prompt, codex_bin)
        else:
            argv = resume_argv(model, thread_id or "", round_prompt, codex_bin)

        events_path = events_dir / f"codex-events-round-{round_index}.jsonl"
        stderr_path = events_dir / f"codex-stderr-round-{round_index}.log"
        round_started = time.monotonic()
        returncode, timed_out = run_exec(
            argv, workspace, events_path, stderr_path, timeout_seconds
        )
        round_duration = time.monotonic() - round_started

        try:
            lines = events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        parsed = parse_events(lines)

        thread_id = thread_id or parsed["thread_id"]
        tool_actions_total += parsed["tool_actions"]
        _accumulate_usage(usage_total, parsed["usage"])

        # Layer 1 of ask detection: the zero-edit turn gate. Empirically
        # (36/36 harvested real turns) asking and editing are mutually
        # exclusive per turn, so a turn that edited is never an ask no
        # matter what its final message says. Evidence is file_change items
        # plus a git fingerprint delta (catches shell-based edits).
        fingerprint_after = _workspace_fingerprint(workspace)
        git_delta = (
            fingerprint_before is not None
            and fingerprint_after is not None
            and fingerprint_before != fingerprint_after
        )
        turn_edited = parsed["file_changes"] > 0 or git_delta
        fingerprint_before = fingerprint_after

        final_message = parsed["final_message"]
        verdict = ask_detection.classify(final_message)
        regex_asked = verdict["asked"]
        asked = regex_asked and not turn_edited
        if regex_asked and turn_edited:
            # Should not happen per the harvested data; recorded so the
            # gate's core assumption stays auditable on every future run.
            questions_with_edits += 1
        rounds.append(
            {
                "index": round_index,
                "prompt_kind": prompt_kind,
                "prompt": round_prompt,
                "events_path": str(events_path),
                "exit_code": returncode,
                "timed_out": timed_out,
                "duration_seconds": round_duration,
                "agent_messages": parsed["agent_messages"],
                "final_message": final_message,
                "asked": asked,
                "turn_edited": turn_edited,
                "regex_asked": regex_asked,
                # Full classifier audit trail: which signal patterns fired
                # and which blockers suppressed a question unit, per round,
                # so pattern-level ablation can run over stored outcomes.
                "ask_signals": verdict["signals"],
                "ask_blockers": verdict["blockers"],
                "tool_actions": parsed["tool_actions"],
                "file_changes": parsed["file_changes"],
                "usage": parsed["usage"],
                "parse_errors": parsed["parse_errors"],
            }
        )

        if timed_out:
            stop_reason = "timeout"
            is_error = True
            error_text = f"round {round_index} exceeded {timeout_seconds}s and was killed"
            break

        failure = parsed["fatal_error"] or parsed["turn_failed"]
        if failure or returncode != 0:
            stop_reason = "error"
            is_error = True
            error_text = failure or f"codex exited {returncode} without a turn failure event"
            break

        if asked:
            elapsed = time.monotonic() - started_monotonic
            direct_count += 1
            if first_direct is None:
                first_direct = {
                    "round": round_index,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    # Measured at round end: the moment the question reached
                    # the (synthetic) user. Codex events carry no timestamps.
                    "latency_seconds": elapsed,
                    "message_text": final_message,
                    "matched_signals": verdict["signals"],
                    "assistant_tool_actions_before": tool_actions_total,
                    "workspace_had_changes": _workspace_has_changes(workspace),
                }
            if len(answered_questions) >= max_ask_rounds:
                stop_reason = "max_ask_rounds"
                break
            answered_questions.append(
                {
                    "round": round_index,
                    "question_text": final_message,
                    "reply": NEUTRAL_ANSWER,
                    "latency_seconds": elapsed,
                }
            )
            if thread_id is None:
                stop_reason = "error"
                is_error = True
                error_text = "cannot resume: no thread.started event was emitted"
                break
            round_prompt = NEUTRAL_ANSWER
            prompt_kind = "synthetic_answer"
            continue

        break

    ended_at = datetime.now(timezone.utc)
    duration_ms = int((time.monotonic() - started_monotonic) * 1000)

    return {
        "runner": RUNNER,
        "model": model,
        "sandbox": SANDBOX_MODE,
        "ask_channel": ASK_CHANNEL,
        "ask_classifier_version": ASK_CLASSIFIER_VERSION,
        "ask_gate": "zero_edit_turn",
        "max_ask_rounds": max_ask_rounds,
        "neutral_answer": NEUTRAL_ANSWER,
        "thread_id": thread_id,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "rounds": rounds,
        "tool_actions_total": tool_actions_total,
        "answered_questions": answered_questions,
        "result": {
            "subtype": stop_reason,
            "is_error": is_error,
            "num_turns": len(rounds),
            "duration_ms": duration_ms,
            "session_id": thread_id,
            "stop_reason": stop_reason,
            "total_cost_usd": None,  # codex bills the signed-in account; no per-run figure
            "usage": usage_total,
            "result": error_text,
        },
        "analysis": {
            "first_direct": first_direct,
            "direct_count": direct_count,
            # Codex exec exposes no per-subagent ask events; the main
            # thread's final messages are the only observable ask channel.
            "any_agent_count": direct_count,
            # Turns where the regex fired but the turn edited (gated to
            # not-asked). Zero in all harvested data; non-zero would mean
            # the zero-edit gate's core assumption needs revisiting.
            "questions_with_edits": questions_with_edits,
        },
    }
