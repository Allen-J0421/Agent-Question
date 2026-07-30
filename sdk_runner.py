"""Run one unattended Claude Code session via the Agent SDK.

The session runs in ``default`` permission mode, so the ``can_use_tool``
callback is a genuine permission prompt: every tool call the CLI's rules
evaluate to "ask" reaches it, and the agent experiences the same friction
points an attended session has. This matters for the study's outcome --
under ``bypassPermissions`` the callback is shadowed for ordinary tools
(the SDK emits ``CanUseToolShadowedWarning``), the agent never pauses,
and it can resolve any ambiguity by reading the repository instead of
asking about it.

The callback approves every ordinary tool automatically -- no run is
gated on a human -- while recording the approval, so "the agent had to
stop here" is observable rather than removed. ``AskUserQuestion`` calls
are logged in full (question/options/timing) and answered with a neutral
first-option tie-break so a headless run can complete.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import PermissionResultAllow


ROOT = Path(__file__).resolve().parent
REFERENCE_TOOLSET_PATH = ROOT / "config" / "reference_toolset.json"

PERMISSION_MODE = "default"


def load_reference_toolset() -> list[str]:
    data = json.loads(REFERENCE_TOOLSET_PATH.read_text(encoding="utf-8"))
    return list(data["tools"])


def _first_option_answers(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Neutral tie-break: always the first offered option per question."""
    answers: dict[str, Any] = {}
    for question in questions:
        options = question.get("options") or []
        if not options:
            continue
        label = options[0].get("label")
        if question.get("multiSelect"):
            answers[question["question"]] = [label]
        else:
            answers[question["question"]] = label
    return answers


async def run_sdk_session(
    *,
    prompt: str,
    workspace: Path,
    model: str,
    tools: list[str] | None = None,
    stop_on_first_ask: bool = False,
) -> dict[str, Any]:
    """Run one headless SDK session and return a structured observation.

    Mirrors the shape ``study_log.observe_headless_session`` returns from
    CLI transcript-tailing, so ``build_run_summary`` can stay largely
    unchanged: ``analysis.first_direct``, ``analysis.direct_count``,
    ``analysis.any_agent_count``, plus ``tool_roster`` /
    ``askuserquestion_available`` so every run is self-certifying.

    ``stop_on_first_ask`` defaults to ``False`` so the agent runs to
    completion and leaves a patch that can be graded. Halting at the first
    ask would make every asking run ungradeable while non-asking runs stayed
    gradeable, which biases the asked-vs-not-asked comparison in exactly the
    direction the study is trying to measure. The first ask is still the
    primary outcome and is recorded in full before any synthetic answer is
    produced; ``answered_questions`` keeps the rest auditable.
    """
    tools = tools if tools is not None else load_reference_toolset()

    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)

    tool_roster: list[str] | None = None
    main_tool_actions = 0
    direct_ids: set[str] = set()
    any_ids: set[str] = set()
    first_direct: dict[str, Any] | None = None
    direct_count = 0
    any_agent_count = 0
    permission_prompts = 0
    answered_questions: list[dict[str, Any]] = []
    result_message: dict[str, Any] | None = None
    stopped_on_first_ask = False

    async def can_use_tool(tool_name, input_data, context):
        nonlocal main_tool_actions, first_direct, direct_count
        nonlocal any_agent_count, permission_prompts

        is_subagent = context.agent_id is not None
        is_ask = tool_name == "AskUserQuestion"

        if is_ask:
            identifier = context.tool_use_id
            if identifier not in any_ids:
                any_ids.add(identifier)
                any_agent_count += 1
            if not is_subagent:
                if identifier not in direct_ids:
                    direct_ids.add(identifier)
                    direct_count += 1
                    if first_direct is None:
                        now = time.monotonic()
                        first_direct = {
                            "tool_use_id": identifier,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "latency_seconds": now - started_monotonic,
                            "input": input_data,
                            "assistant_tool_actions_before": main_tool_actions,
                            "permission_prompts_before": permission_prompts,
                        }
            questions = input_data.get("questions") or []
            answers = _first_option_answers(questions)
            # Every synthetic answer is recorded, not just the first, so the
            # tie-break's influence on the final patch stays auditable.
            answered_questions.append(
                {
                    "tool_use_id": identifier,
                    "is_subagent": is_subagent,
                    "latency_seconds": time.monotonic() - started_monotonic,
                    "questions": questions,
                    "answers": answers,
                }
            )
            return PermissionResultAllow(
                updated_input={"questions": questions, "answers": answers}
            )

        # Reaching here means the CLI's permission rules evaluated to "ask"
        # for an ordinary tool: a real friction point in default mode.
        permission_prompts += 1
        if not is_subagent:
            main_tool_actions += 1
        return PermissionResultAllow(updated_input=input_data)

    options = ClaudeAgentOptions(
        cwd=str(workspace),
        model=model,
        permission_mode=PERMISSION_MODE,
        tools=tools,
        strict_mcp_config=True,
        mcp_servers={},
        can_use_tool=can_use_tool,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            msg_type = type(message).__name__
            if msg_type == "SystemMessage" and message.subtype in (
                "init",
                "initialize",
            ):
                tool_roster = list(message.data.get("tools", []))
            elif msg_type == "ResultMessage":
                result_message = {
                    "subtype": message.subtype,
                    "is_error": message.is_error,
                    "num_turns": message.num_turns,
                    "duration_ms": message.duration_ms,
                    "session_id": message.session_id,
                    "stop_reason": message.stop_reason,
                    "total_cost_usd": message.total_cost_usd,
                    "result": message.result,
                }
            if stop_on_first_ask and first_direct is not None:
                # The primary outcome is already recorded; everything after
                # the first ask is shaped by the synthetic answer, so stop.
                await client.interrupt()
                stopped_on_first_ask = True
                break

    ended_at = datetime.now(timezone.utc)

    return {
        "tool_roster": tool_roster,
        "reference_toolset": tools,
        "permission_mode": PERMISSION_MODE,
        "askuserquestion_available": bool(
            tool_roster and "AskUserQuestion" in tool_roster
        ),
        "result": result_message,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "stopped_on_first_ask": stopped_on_first_ask,
        "permission_prompts": permission_prompts,
        "answered_questions": answered_questions,
        "analysis": {
            "first_direct": first_direct,
            "direct_count": direct_count,
            "any_agent_count": any_agent_count,
        },
    }
