"""Detect whether the agent asked a clarifying question, and extract the questions.

Two ground-truth signals for an ask (verified against fixtures):
  1. an `AskUserQuestion` tool_use in the assistant stream, and
  2. an entry in `result.permission_denials` (the CLI auto-denies it headless).
We union both and dedupe by tool_use_id, so we catch the ask even if one signal is
missing (e.g. a truncated transcript with no final result event).
"""
from __future__ import annotations

from harness.agent.stream_parser import ParsedTranscript
from harness.constants import TOOL_ASK_USER_QUESTION
from harness.record.schema import AskInfo, Question


def _questions_from_input(inp: dict, turn: int) -> list[Question]:
    out: list[Question] = []
    for q in inp.get("questions", []) or []:
        out.append(Question(
            turn=turn,
            header=q.get("header", "") or "",
            question=q.get("question", "") or "",
            options=[o.get("label", "") for o in (q.get("options", []) or [])],
            multi_select=bool(q.get("multiSelect", False)),
        ))
    return out


def detect_ask(parsed: ParsedTranscript) -> AskInfo:
    # tool_use_id -> (turn, questions)
    by_id: dict[str, tuple[int, list[Question]]] = {}
    order: list[str] = []  # preserve first-seen order and track earliest turn

    # signal 1: tool_use blocks in the assistant stream
    for am in parsed.assistant_msgs:
        for tu in am.tool_uses:
            if tu.name == TOOL_ASK_USER_QUESTION:
                if tu.id not in by_id:
                    order.append(tu.id)
                by_id[tu.id] = (tu.turn, _questions_from_input(tu.input, tu.turn))

    # signal 2: permission_denials on the result (turn unknown -> use a large sentinel
    # so it never becomes the "first" ask if a stream tool_use exists)
    if parsed.result:
        for d in parsed.result.permission_denials:
            if d.get("tool_name") != TOOL_ASK_USER_QUESTION:
                continue
            tuid = d.get("tool_use_id", "") or f"denial_{len(order)}"
            if tuid not in by_id:
                order.append(tuid)
                by_id[tuid] = (10**9, _questions_from_input(d.get("tool_input", {}) or {}, -1))

    if not by_id:
        return AskInfo(asked=False)

    all_questions: list[Question] = []
    first_turn: int | None = None
    for tuid in order:
        turn, qs = by_id[tuid]
        all_questions.extend(qs)
        if turn < 10**9:
            first_turn = turn if first_turn is None else min(first_turn, turn)

    return AskInfo(
        asked=True,
        n_questions=len(all_questions),
        first_ask_turn=first_turn,
        questions=all_questions,
    )
