"""Offline tests for the Codex CLI runner: event parsing, ask
classification, and the exec/resume session loop -- no test shells out to
the real codex binary or the network."""
import json

import pytest

import codex_runner
from codex_runner import (
    MIN_CLI_VERSION,
    NEUTRAL_ANSWER,
    SANDBOX_MODE,
    exec_argv,
    is_clarifying_question,
    parse_cli_version,
    parse_events,
    resume_argv,
    run_codex_session,
)

THREAD = "0000-thread-id"


# --------------------------------------------------------------------------
# Ask classifier
# --------------------------------------------------------------------------


def test_classifier_detects_a_turn_ending_question():
    assert is_clarifying_question("Should this return None or raise ValueError?")
    assert is_clarifying_question(
        "I found two interpretations.\n\nWhich one do you want?"
    )


def test_classifier_ignores_statements_and_empty_messages():
    assert not is_clarifying_question("Done. All FAIL_TO_PASS tests now pass.")
    assert not is_clarifying_question("")
    assert not is_clarifying_question(None)


def test_classifier_ignores_question_marks_inside_code_spans():
    # A '?' in a diff hunk, regex, or shell snippet is not the agent asking.
    assert not is_clarifying_question("Applied the fix:\n```py\nx = a if b else c  # why?\n```")
    assert not is_clarifying_question("I updated the regex `[a-z]?` and reran the tests.")
    # ...but prose questions still count even when code is present.
    assert is_clarifying_question(
        "I can change `foo()` or `bar()`.\n```py\nfoo()\n```\nWhich should it be?"
    )


# --------------------------------------------------------------------------
# CLI version gate and argv shapes
# --------------------------------------------------------------------------


def test_cli_version_parsing_and_minimum():
    # 0.139.0 was rejected server-side for gpt-5.6-sol; 0.146.0 verified live.
    assert parse_cli_version("codex-cli 0.146.0") == (0, 146, 0)
    assert parse_cli_version("nonsense") is None
    assert parse_cli_version("codex-cli 0.139.0") < MIN_CLI_VERSION


def test_exec_argv_is_vanilla_and_isolated_from_user_config():
    argv = exec_argv("gpt-5.6-sol", "PROMPT")
    assert argv[0:2] == ["codex", "exec"]
    # Operator config must never leak into a study session.
    assert "--ignore-user-config" in argv and "--ignore-rules" in argv
    assert "--json" in argv
    assert argv[argv.index("--sandbox") + 1] == SANDBOX_MODE
    assert argv[argv.index("-m") + 1] == "gpt-5.6-sol"
    assert argv[-1] == "PROMPT"
    # No injected instructions, tools, or feature flags: the prompt is the
    # only task-specific input.
    assert not any(flag in argv for flag in ("--enable", "--disable", "--output-schema"))


def test_resume_argv_targets_the_thread_and_pins_the_sandbox():
    argv = resume_argv("gpt-5.6-sol", THREAD, "ANSWER")
    assert argv[0:4] == ["codex", "exec", "resume", THREAD]
    assert f'sandbox_mode="{SANDBOX_MODE}"' in argv
    assert argv[-1] == "ANSWER"


# --------------------------------------------------------------------------
# Event parsing (shapes verified live against codex-cli 0.146.0)
# --------------------------------------------------------------------------


def _round_events(messages, commands=0, file_changes=0, thread=THREAD, usage=None, tail=None):
    lines = [
        json.dumps({"type": "thread.started", "thread_id": thread}),
        json.dumps({"type": "turn.started"}),
    ]
    for index in range(commands):
        lines.append(json.dumps({
            "type": "item.completed",
            "item": {"id": f"cmd{index}", "type": "command_execution",
                     "command": "echo x", "exit_code": 0, "status": "completed"},
        }))
    for index in range(file_changes):
        lines.append(json.dumps({
            "type": "item.completed",
            "item": {"id": f"fc{index}", "type": "file_change",
                     "changes": [{"path": f"file{index}.py", "kind": "update"}],
                     "status": "completed"},
        }))
    for message in messages:
        lines.append(json.dumps({
            "type": "item.completed",
            "item": {"id": "msg", "type": "agent_message", "text": message},
        }))
    if tail is None:
        lines.append(json.dumps({
            "type": "turn.completed",
            "usage": usage or {"input_tokens": 100, "output_tokens": 10},
        }))
    else:
        lines.extend(tail)
    return lines


def test_parse_events_extracts_thread_final_message_and_tool_actions():
    parsed = parse_events(_round_events(["working on it", "All done."], commands=2, file_changes=1))
    assert parsed["thread_id"] == THREAD
    # Only the *final* message yields the turn; commentary is not the ask.
    assert parsed["final_message"] == "All done."
    assert parsed["tool_actions"] == 3  # commands + file_change edits
    assert parsed["file_changes"] == 1  # the zero-edit gate's primary evidence
    assert parsed["usage"]["output_tokens"] == 10
    assert parsed["turn_failed"] is None and parsed["fatal_error"] is None


def test_parse_events_captures_turn_failures_and_tolerates_junk():
    lines = [
        json.dumps({"type": "thread.started", "thread_id": THREAD}),
        "not json at all",
        json.dumps({"type": "error", "message": "server said no"}),
        json.dumps({"type": "turn.failed", "error": {"message": "server said no"}}),
    ]
    parsed = parse_events(lines)
    assert parsed["turn_failed"] == "server said no"
    assert parsed["fatal_error"] == "server said no"
    assert parsed["parse_errors"] == 1
    assert parsed["turn_completed"] is False


# --------------------------------------------------------------------------
# Session loop
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_real_git(monkeypatch):
    """Keep unit tests off the real git binary; the fingerprint side of the
    zero-edit gate is exercised via scripted values in the gate tests."""
    monkeypatch.setattr(codex_runner, "_workspace_fingerprint", lambda workspace: None)
    monkeypatch.setattr(codex_runner, "_workspace_has_changes", lambda workspace: None)


class FakeExec:
    """Scripted stand-in for the codex subprocess boundary."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = []

    def __call__(self, argv, cwd, stdout_path, stderr_path, timeout_seconds):
        self.calls.append(argv)
        lines, returncode, timed_out = self.rounds.pop(0)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return returncode, timed_out


def _run(tmp_path, fake, **kwargs):
    return run_codex_session(
        prompt="Resolve the issue.",
        workspace=tmp_path / "ws",
        model="gpt-5.6-sol",
        events_dir=tmp_path / "events",
        run_exec=fake,
        **kwargs,
    )


def test_session_without_ask_is_one_round(tmp_path):
    (tmp_path / "ws").mkdir()
    fake = FakeExec([(_round_events(["Fixed it. Tests pass."], commands=3), 0, False)])

    observation = _run(tmp_path, fake)

    assert observation["analysis"]["direct_count"] == 0
    assert observation["analysis"]["first_direct"] is None
    assert observation["answered_questions"] == []
    assert observation["result"]["num_turns"] == 1
    assert observation["result"]["is_error"] is False
    assert observation["result"]["stop_reason"] == "completed"
    assert observation["tool_actions_total"] == 3
    assert observation["thread_id"] == THREAD
    # The raw event stream is on disk for offline reclassification.
    assert (tmp_path / "events" / "codex-events-round-1.jsonl").exists()


def test_ask_is_recorded_then_answered_neutrally_via_resume(tmp_path):
    (tmp_path / "ws").mkdir()
    fake = FakeExec([
        (_round_events(["Looking around.", "Should I cap the value or raise?"], commands=2), 0, False),
        (_round_events(["DONE"], commands=4), 0, False),
    ])

    observation = _run(tmp_path, fake)

    analysis = observation["analysis"]
    assert analysis["direct_count"] == 1
    first = analysis["first_direct"]
    assert first["round"] == 1
    assert first["message_text"] == "Should I cap the value or raise?"
    assert first["assistant_tool_actions_before"] == 2
    # The primary outcome is recorded before the synthetic answer is sent.
    answered = observation["answered_questions"]
    assert len(answered) == 1
    assert answered[0]["question_text"] == "Should I cap the value or raise?"
    assert answered[0]["reply"] == NEUTRAL_ANSWER
    # Round 2 is a resume of the same thread carrying the neutral answer.
    resume_call = fake.calls[1]
    assert resume_call[2] == "resume" and THREAD in resume_call
    assert resume_call[-1] == NEUTRAL_ANSWER
    # The session ran to completion, so the run is still gradable.
    assert observation["result"]["num_turns"] == 2
    assert observation["result"]["stop_reason"] == "completed"
    assert observation["tool_actions_total"] == 6


def test_the_answer_mirrors_claude_first_option_tie_break():
    # Parity with the Claude arm's AskUserQuestion callback: the synthetic
    # answer always takes the first option offered, stated as a rule because
    # a free-text question carries no machine-readable option list.
    assert NEUTRAL_ANSWER == "Go with the first option you presented."


def test_ask_cap_is_symmetric_across_both_arms(tmp_path):
    # Both arms answer up to the same number of asks per run, so neither
    # model can out-ask the other by harness construction.
    import inspect

    import sdk_runner

    assert codex_runner.MAX_ASK_ROUNDS == 3
    assert sdk_runner.MAX_ASK_ROUNDS == codex_runner.MAX_ASK_ROUNDS
    sdk_default = inspect.signature(sdk_runner.run_sdk_session).parameters[
        "max_ask_rounds"
    ].default
    assert sdk_default == sdk_runner.MAX_ASK_ROUNDS
    # Behavioral check at the default cap: a fourth ask ends the run with
    # its state captured; the first three are answered.
    (tmp_path / "ws").mkdir()
    ask = _round_events(["Which option do you prefer?"])
    fake = FakeExec([(ask, 0, False)] * 4)

    observation = _run(tmp_path, fake)

    assert observation["result"]["stop_reason"] == "max_ask_rounds"
    assert observation["result"]["is_error"] is False
    assert observation["analysis"]["direct_count"] == 4
    assert len(observation["answered_questions"]) == 3
    assert observation["result"]["num_turns"] == 4


def test_a_persistent_asker_is_capped_not_looped_forever(tmp_path):
    (tmp_path / "ws").mkdir()
    ask = _round_events(["Which option do you prefer?"])
    fake = FakeExec([(ask, 0, False)] * 3)

    observation = _run(tmp_path, fake, max_ask_rounds=2)

    assert observation["result"]["stop_reason"] == "max_ask_rounds"
    assert observation["result"]["is_error"] is False
    assert observation["analysis"]["direct_count"] == 3
    assert len(observation["answered_questions"]) == 2  # cap on synthetic answers
    assert observation["result"]["num_turns"] == 3


def test_turn_failure_is_persisted_as_error_evidence(tmp_path):
    # e.g. codex-cli 0.139 + gpt-5.6-sol: the server rejects the model with
    # HTTP 400 and the events stream carries turn.failed. Without this text
    # a dead run is indistinguishable from "the agent chose to do nothing".
    (tmp_path / "ws").mkdir()
    failure_tail = [
        json.dumps({"type": "error", "message": "model requires a newer Codex"}),
        json.dumps({"type": "turn.failed", "error": {"message": "model requires a newer Codex"}}),
    ]
    fake = FakeExec([(_round_events([], tail=failure_tail), 1, False)])

    observation = _run(tmp_path, fake)

    result = observation["result"]
    assert result["is_error"] is True
    assert result["stop_reason"] == "error"
    assert "newer Codex" in result["result"]
    assert observation["analysis"]["direct_count"] == 0


def test_timeout_kills_the_round_and_marks_the_run(tmp_path):
    (tmp_path / "ws").mkdir()
    fake = FakeExec([(_round_events(["still going"]), -1, True)])

    observation = _run(tmp_path, fake, timeout_seconds=5)

    assert observation["result"]["stop_reason"] == "timeout"
    assert observation["result"]["is_error"] is True


def test_usage_is_summed_across_rounds(tmp_path):
    (tmp_path / "ws").mkdir()
    fake = FakeExec([
        (_round_events(["Cap it or raise it?"], usage={"input_tokens": 7, "output_tokens": 3}), 0, False),
        (_round_events(["DONE"], usage={"input_tokens": 5, "output_tokens": 2}), 0, False),
    ])

    observation = _run(tmp_path, fake)

    assert observation["result"]["usage"] == {"input_tokens": 12, "output_tokens": 5}


def test_observation_records_the_measurement_channel_metadata(tmp_path):
    # Every run must be self-certifying about *how* asking was measured.
    (tmp_path / "ws").mkdir()
    fake = FakeExec([(_round_events(["Done."]), 0, False)])

    observation = _run(tmp_path, fake)

    assert observation["runner"] == "codex-cli"
    assert observation["ask_channel"] == "final_message"
    assert observation["ask_classifier_version"] == codex_runner.ASK_CLASSIFIER_VERSION
    assert observation["ask_gate"] == "zero_edit_turn"
    assert observation["sandbox"] == SANDBOX_MODE
    assert observation["neutral_answer"] == NEUTRAL_ANSWER


# --------------------------------------------------------------------------
# Layer 1: the zero-edit turn gate
# --------------------------------------------------------------------------


def test_question_on_edited_turn_is_gated_not_answered(tmp_path):
    # Empirical basis: in 36/36 harvested real turns, asking and editing
    # never co-occurred. A turn that edited and ended with a question (e.g.
    # a post-work offer) is therefore not an ask: no synthetic answer, no
    # resume, but the contradiction is recorded for audit.
    (tmp_path / "ws").mkdir()
    fake = FakeExec([
        (_round_events(["Fixed it. Want me to also add a changelog entry?"],
                       commands=2, file_changes=1), 0, False),
    ])

    observation = _run(tmp_path, fake)

    assert observation["analysis"]["direct_count"] == 0
    assert observation["answered_questions"] == []
    assert observation["result"]["num_turns"] == 1
    round_record = observation["rounds"][0]
    assert round_record["turn_edited"] is True
    assert round_record["regex_asked"] is True
    assert round_record["asked"] is False
    assert observation["analysis"]["questions_with_edits"] == 1


def test_git_fingerprint_delta_gates_shell_based_edits(tmp_path, monkeypatch):
    # Edits made through the shell (sed -i, redirects) emit no file_change
    # item; the before/after git fingerprint delta must catch them.
    fingerprints = iter(["before", "after"])
    monkeypatch.setattr(
        codex_runner, "_workspace_fingerprint", lambda workspace: next(fingerprints)
    )
    (tmp_path / "ws").mkdir()
    fake = FakeExec([
        (_round_events(["Should I also update the docs?"], commands=1), 0, False),
    ])

    observation = _run(tmp_path, fake)

    assert observation["rounds"][0]["turn_edited"] is True
    assert observation["rounds"][0]["asked"] is False
    assert observation["analysis"]["direct_count"] == 0


def test_zero_edit_asking_turn_passes_the_gate(tmp_path):
    # The converse: a turn with commands but no edits (exploration only) is
    # exactly the shape of all 30 harvested real asks, and must be answered.
    (tmp_path / "ws").mkdir()
    fake = FakeExec([
        (_round_events(["Should I cap the value or raise?"], commands=3), 0, False),
        (_round_events(["DONE"], commands=1, file_changes=1), 0, False),
    ])

    observation = _run(tmp_path, fake)

    assert observation["rounds"][0]["turn_edited"] is False
    assert observation["rounds"][0]["asked"] is True
    assert observation["analysis"]["direct_count"] == 1
    assert len(observation["answered_questions"]) == 1
    # Round 2 (after the answer) edited and completed: not an ask, no gate
    # contradiction (its final message is not a question).
    assert observation["rounds"][1]["turn_edited"] is True
    assert observation["analysis"]["questions_with_edits"] == 0
