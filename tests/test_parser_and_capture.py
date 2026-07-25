"""Unit tests for the transcript parser, ask detector, and trajectory folder against
recorded real-CLI fixtures. These are the modules every behavioral metric depends on,
so they get pinned tests.

Run: .venv/bin/python -m pytest tests/ -q
"""
from pathlib import Path

from harness.agent.stream_parser import parse_file, parse_transcript
from harness.capture.ask_detector import detect_ask
from harness.capture.trajectory import fold_trajectory

FIX = Path(__file__).parent / "fixtures"


def test_read_only_transcript_does_not_ask():
    p = parse_file(FIX / "transcript_read_only.jsonl")
    assert p.parse_errors == 0
    assert p.system_init and p.system_init.get("model")
    ask = detect_ask(p)
    assert ask.asked is False
    assert ask.n_questions == 0


def test_read_only_trajectory_tokens_and_tools():
    p = parse_file(FIX / "transcript_read_only.jsonl")
    traj = fold_trajectory(p)
    assert "Read" in traj.tools_used
    assert traj.tokens.total > 0
    assert traj.cost_usd > 0
    assert traj.num_turns_reported is not None


def test_asked_transcript_detected_with_question_fields():
    p = parse_file(FIX / "transcript_asked.jsonl")
    ask = detect_ask(p)
    assert ask.asked is True
    assert ask.n_questions >= 1
    q = ask.questions[0]
    assert q.header == "Return type"
    assert q.options == ["String", "Integer"]
    assert q.multi_select is False
    assert q.question  # non-empty text


def test_asked_dedupes_tooluse_and_denial():
    # The asked fixture has BOTH a tool_use and a permission_denial for the same
    # AskUserQuestion; detector must count the questions once, not twice.
    p = parse_file(FIX / "transcript_asked.jsonl")
    ask = detect_ask(p)
    # one question was posed; dedup by tool_use_id keeps it at 1
    assert ask.n_questions == 1


def test_parser_tolerates_truncated_final_line():
    text = (FIX / "transcript_read_only.jsonl").read_text()
    truncated = text[: len(text) - 30]  # chop the last line mid-JSON
    p = parse_transcript(truncated)
    # should not raise; parse_errors counts the broken tail (0 or 1)
    assert p.assistant_msgs  # earlier events still parsed


def test_empty_transcript_is_safe():
    p = parse_transcript("")
    assert p.assistant_msgs == []
    assert detect_ask(p).asked is False
    traj = fold_trajectory(p)
    assert traj.n_turns == 0
