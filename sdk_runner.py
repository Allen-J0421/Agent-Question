"""Run one unattended Claude Code session via the Agent SDK.

Unlike the CLI's ``-p`` mode (which has no callback mechanism at all),
this module registers a ``can_use_tool`` callback -- required for
``AskUserQuestion`` to appear in the session's tool roster in the first
place, confirmed by direct testing against the live backend. The callback
observes every ``AskUserQuestion`` call (full question/options/timing) and
answers with a neutral first-option tie-break so a headless run can
complete; every other tool is auto-approved unchanged, matching
``bypassPermissions`` semantics as closely as the SDK allows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import PermissionResultAllow


ROOT = Path(__file__).resolve().parent
REFERENCE_TOOLSET_PATH = ROOT / "config" / "reference_toolset.json"


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
) -> dict[str, Any]:
    """Run one headless SDK session and return a structured observation.

    Mirrors the shape ``study_log.observe_headless_session`` used to return
    from CLI transcript-tailing, so ``build_run_summary`` can stay largely
    unchanged: ``analysis.first_direct``, ``analysis.direct_count``,
    ``analysis.any_agent_count``, plus a new ``tool_roster`` /
    ``askuserquestion_available`` pair that makes every run self-certifying.
    """
    tools = tools if tools is not None else load_reference_toolset()

    events: list[dict[str, Any]] = []
    tool_roster: list[str] | None = None
    main_tool_actions = 0
    direct_ids: set[str] = set()
    any_ids: set[str] = set()
    first_direct: dict[str, Any] | None = None
    direct_count = 0
    any_agent_count = 0
    result_message: dict[str, Any] | None = None

    async def can_use_tool(tool_name, input_data, context):
        nonlocal main_tool_actions, first_direct, direct_count, any_agent_count

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
                        first_direct = {
                            "tool_use_id": identifier,
                            "input": input_data,
                            "assistant_tool_actions_before": main_tool_actions,
                        }
            questions = input_data.get("questions") or []
            answers = _first_option_answers(questions)
            return PermissionResultAllow(
                updated_input={"questions": questions, "answers": answers}
            )

        if not is_subagent:
            main_tool_actions += 1
        return PermissionResultAllow(updated_input=input_data)

    options = ClaudeAgentOptions(
        cwd=str(workspace),
        model=model,
        permission_mode="bypassPermissions",
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

    return {
        "tool_roster": tool_roster,
        "askuserquestion_available": bool(
            tool_roster and "AskUserQuestion" in tool_roster
        ),
        "result": result_message,
        "analysis": {
            "first_direct": first_direct,
            "direct_count": direct_count,
            "any_agent_count": any_agent_count,
        },
    }
