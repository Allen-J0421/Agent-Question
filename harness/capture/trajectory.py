"""Fold a ParsedTranscript into the Trajectory record: turn/tool/token/cost tallies.
Token totals come from result.usage (authoritative); falls back to summing per-message
usage if the result event is missing (truncated transcript).
"""
from __future__ import annotations

from harness.agent.stream_parser import ParsedTranscript
from harness.constants import READ_TOOLS
from harness.record.schema import Tokens, Trajectory


def _tokens_from_result_usage(usage: dict) -> Tokens:
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    cc = usage.get("cache_creation_input_tokens", 0) or 0
    return Tokens(input=inp, output=out, cache_read=cr, cache_creation=cc,
                  total=inp + out + cr + cc)


def _tokens_from_messages(parsed: ParsedTranscript) -> Tokens:
    inp = out = cr = cc = 0
    for am in parsed.assistant_msgs:
        u = am.usage or {}
        inp += u.get("input_tokens", 0) or 0
        out += u.get("output_tokens", 0) or 0
        cr += u.get("cache_read_input_tokens", 0) or 0
        cc += u.get("cache_creation_input_tokens", 0) or 0
    return Tokens(input=inp, output=out, cache_read=cr, cache_creation=cc,
                  total=inp + out + cr + cc)


def _file_arg(inp: dict) -> str | None:
    for key in ("file_path", "path", "notebook_path"):
        if inp.get(key):
            return inp[key]
    return None


def fold_trajectory(parsed: ParsedTranscript) -> Trajectory:
    tools_used: dict[str, int] = {}
    files_read: list[str] = []
    n_tool_calls = 0

    for am in parsed.assistant_msgs:
        for tu in am.tool_uses:
            n_tool_calls += 1
            tools_used[tu.name] = tools_used.get(tu.name, 0) + 1
            if tu.name in READ_TOOLS:
                fp = _file_arg(tu.input)
                if fp and fp not in files_read:
                    files_read.append(fp)

    if parsed.result and parsed.result.usage:
        tokens = _tokens_from_result_usage(parsed.result.usage)
        cost = parsed.result.total_cost_usd or 0.0
        num_turns_reported = parsed.result.num_turns
    else:
        tokens = _tokens_from_messages(parsed)
        cost = 0.0
        num_turns_reported = None

    return Trajectory(
        n_turns=len(parsed.assistant_msgs),
        n_assistant_msgs=len(parsed.assistant_msgs),
        n_tool_calls=n_tool_calls,
        tools_used=tools_used,
        files_read=files_read,
        tokens=tokens,
        cost_usd=cost,
        num_turns_reported=num_turns_reported,
    )
