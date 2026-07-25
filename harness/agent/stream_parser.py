"""Parse a Claude-CLI `--output-format stream-json` transcript (NDJSON) into typed
events. Tolerant of partial/truncated final lines (a killed subprocess may leave a
half-written line). Anchored to the real event structure verified against fixtures in
tests/fixtures/ (see memory: claude-cli-headless-ask-behavior).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]
    turn: int  # index among assistant messages (0-based), assigned by the parser


@dataclass
class AssistantMsg:
    turn: int
    texts: list[str] = field(default_factory=list)       # 'text' blocks
    thinking: list[str] = field(default_factory=list)     # 'thinking' blocks
    tool_uses: list[ToolUse] = field(default_factory=list)
    usage: dict[str, Any] | None = None


@dataclass
class ResultEvent:
    subtype: str | None
    is_error: bool
    stop_reason: str | None
    num_turns: int | None
    total_cost_usd: float | None
    usage: dict[str, Any]
    permission_denials: list[dict[str, Any]]


@dataclass
class ParsedTranscript:
    system_init: dict[str, Any] | None
    assistant_msgs: list[AssistantMsg]
    tool_results: list[dict[str, Any]]     # raw tool_result blocks (role=user)
    result: ResultEvent | None
    raw_line_count: int
    parse_errors: int                       # lines that failed json.loads


def _iter_json_lines(text: str) -> Iterator[tuple[dict[str, Any] | None, bool]]:
    """Yield (obj, ok). A trailing truncated line yields (None, False) instead of raising."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line), True
        except json.JSONDecodeError:
            yield None, False


def parse_transcript(text: str) -> ParsedTranscript:
    system_init: dict[str, Any] | None = None
    assistant_msgs: list[AssistantMsg] = []
    tool_results: list[dict[str, Any]] = []
    result: ResultEvent | None = None
    raw = 0
    errs = 0
    turn = 0

    for obj, ok in _iter_json_lines(text):
        raw += 1
        if not ok or obj is None:
            errs += 1
            continue
        etype = obj.get("type")

        if etype == "system":
            if obj.get("subtype") == "init" and system_init is None:
                system_init = obj

        elif etype == "assistant":
            msg = obj.get("message", {})
            am = AssistantMsg(turn=turn, usage=msg.get("usage"))
            for block in msg.get("content", []):
                btype = block.get("type")
                if btype == "text":
                    am.texts.append(block.get("text", ""))
                elif btype == "thinking":
                    am.thinking.append(block.get("thinking", ""))
                elif btype == "tool_use":
                    am.tool_uses.append(ToolUse(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}) or {},
                        turn=turn,
                    ))
            assistant_msgs.append(am)
            turn += 1

        elif etype == "user":
            for block in obj.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    tool_results.append(block)

        elif etype == "result":
            result = ResultEvent(
                subtype=obj.get("subtype"),
                is_error=bool(obj.get("is_error")),
                stop_reason=obj.get("stop_reason"),
                num_turns=obj.get("num_turns"),
                total_cost_usd=obj.get("total_cost_usd"),
                usage=obj.get("usage", {}) or {},
                permission_denials=obj.get("permission_denials", []) or [],
            )
        # rate_limit_event / system/thinking_tokens etc. are ignored.

    return ParsedTranscript(
        system_init=system_init,
        assistant_msgs=assistant_msgs,
        tool_results=tool_results,
        result=result,
        raw_line_count=raw,
        parse_errors=errs,
    )


def parse_file(path: str | Path) -> ParsedTranscript:
    return parse_transcript(Path(path).read_text())
