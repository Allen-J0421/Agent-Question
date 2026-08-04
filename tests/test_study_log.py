import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import study_log


def _timestamp(offset: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat()


def _tool_record(cwd: Path, timestamp: str, name: str, tool_id: str, caller="direct"):
    return {
        "timestamp": timestamp,
        "cwd": str(cwd),
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": name,
                    "id": tool_id,
                    "caller": {"type": caller},
                    "input": {
                        "questions": [
                            {
                                "question": "What behavior should this have?",
                                "options": [{"label": "A"}, {"label": "B"}],
                            }
                        ]
                    }
                    if name == "AskUserQuestion"
                    else {},
                }
            ]
        },
    }


def _write_jsonl(path: Path, records: list[dict], malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) for record in records]
    if malformed:
        lines.append("{not json}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_transcript_analysis_counts_direct_and_subagent_asks_separately(tmp_path):
    projects = tmp_path / "projects"
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    started = _timestamp(-1)
    root = projects / "project" / "root.jsonl"
    subagent = projects / "project" / "subagents" / "worker.jsonl"
    _write_jsonl(
        root,
        [
            _tool_record(workspace, _timestamp(0), "Read", "read-1"),
            _tool_record(workspace, _timestamp(1), "AskUserQuestion", "ask-1"),
            _tool_record(workspace, _timestamp(2), "AskUserQuestion", "ask-1"),
        ],
        malformed=True,
    )
    _write_jsonl(
        subagent,
        [_tool_record(workspace, _timestamp(3), "AskUserQuestion", "ask-sub")],
    )

    analysis = study_log.analyze_transcripts([root, subagent], projects, started)

    assert analysis["direct_count"] == 1
    assert analysis["any_agent_count"] == 2
    assert analysis["parse_errors"] == 1
    assert analysis["first_direct"]["assistant_tool_actions_before"] == 1
    assert analysis["first_direct"]["tool_use_id"] == "ask-1"


def test_find_workspace_transcripts_uses_recorded_cwd(tmp_path):
    projects = tmp_path / "projects"
    workspace = tmp_path / "checkout"
    other_workspace = tmp_path / "other"
    workspace.mkdir()
    other_workspace.mkdir()
    started = _timestamp(-1)
    matching = projects / "one" / "match.jsonl"
    other = projects / "two" / "other.jsonl"
    _write_jsonl(matching, [_tool_record(workspace, _timestamp(0), "Read", "one")])
    _write_jsonl(other, [_tool_record(other_workspace, _timestamp(0), "Read", "two")])

    assert study_log.find_workspace_transcripts(projects, workspace, started) == [matching]


def test_observer_interrupts_once_on_first_direct_ask(monkeypatch, tmp_path):
    class Process:
        pid = 123
        done = False

        def poll(self):
            return 130 if self.done else None

        def wait(self, timeout=None):
            self.done = True
            return 130

    process = Process()
    launches = []
    interrupts = []
    analysis = {
        "paths": [],
        "parse_errors": 0,
        "valid_records": 1,
        "first_direct": {"tool_use_id": "ask-1"},
        "direct_count": 1,
        "any_agent_count": 1,
    }
    monkeypatch.setattr(
        study_log.subprocess,
        "Popen",
        lambda *args, **kwargs: launches.append((args, kwargs)) or process,
    )
    monkeypatch.setattr(study_log, "find_workspace_transcripts", lambda *args: [])
    monkeypatch.setattr(study_log, "analyze_transcripts", lambda *args: analysis)
    monkeypatch.setattr(study_log.time, "sleep", lambda _: None)

    def stop(pid, sig=study_log.signal.SIGINT):
        interrupts.append((pid, sig))
        process.done = True

    monkeypatch.setattr(study_log, "interrupt_process_group", stop)
    result = study_log.observe_headless_session(
        ["claude", "--model", "model", "PROMPT"],
        tmp_path,
        _timestamp(-1),
        tmp_path / "projects",
        tmp_path / "output",
    )

    assert launches[0][0][0] == ["claude", "--model", "model", "PROMPT"]
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["stdin"] is study_log.subprocess.DEVNULL
    assert interrupts == [(123, study_log.signal.SIGINT)]
    assert result["stop_reason"] == "stopped_on_first_ask"
    assert Path(result["process_output"]["stdout"]).exists()


def test_interrupt_process_group_targets_only_child_group(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))

    study_log.interrupt_process_group(42)

    assert calls == [(42, study_log.signal.SIGINT)]


def test_unmatched_or_malformed_transcript_is_unknown(tmp_path):
    logs_root = tmp_path / "logs"
    manifest = {
        "run_id": "unknown-run",
        "started_at": _timestamp(-1),
        "task": {"instance_id": "one", "condition": "ambiguous"},
        "claude": {"model": "model"},
        "workspace": str(tmp_path),
    }
    observation = {
        "exit_code": 0,
        "stop_reason": "completed",
        "operator_interrupted": False,
        "analysis": {
            "paths": [],
            "parse_errors": 1,
            "valid_records": 0,
            "first_direct": None,
            "direct_count": 0,
            "any_agent_count": 0,
        },
    }

    summary = study_log.build_run_summary(manifest, observation, logs_root)

    assert summary["ask_user_question"]["direct_asked"] is None
    assert summary["transcript"]["monitoring_status"] == "unknown"


def _summary(run_id: str, asked: bool | None) -> dict:
    first = None
    if asked:
        first = {
            "assistant_tool_actions_before": 2,
            "input": {"questions": [{"options": [{}, {}]}]},
        }
    return {
        "run_id": run_id,
        "task": {"instance_id": run_id, "condition": "ambiguous"},
        "claude": {"model": "claude-opus-4-8"},
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:01:00+00:00",
        "process": {"exit_code": 0, "stop_reason": "completed"},
        "transcript": {"monitoring_status": "complete_no_ask"},
        "ask_user_question": {
            "direct_asked": asked,
            "direct_count": int(asked is True),
            "any_agent_count": int(asked is True),
            "first_direct": first,
            "first_direct_latency_seconds": 3.0 if asked else None,
        },
    }


def test_report_writes_csv_json_and_markdown(tmp_path):
    logs_root = tmp_path / "logs"
    study_log.write_run_summary(logs_root, _summary("asked", True))
    study_log.write_run_summary(logs_root, _summary("not-asked", False))
    study_log.write_run_summary(logs_root, _summary("unknown", None))

    paths = study_log.write_report(logs_root)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert report["runs"]["total"] == 3
    assert report["runs"]["valid_for_primary_outcome"] == 2
    assert report["runs"]["direct_ask_rate"] == 0.5
    assert "direct_asked" in paths["csv"].read_text(encoding="utf-8")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    # Headline rate, a per-run table row for each logged run, and pointers
    # to the sibling CSV/JSON so the Markdown view is self-sufficient.
    assert "**1/2** valid runs (50.0%)" in markdown
    assert "asked" in markdown and "not-aske" in markdown and "unknown" in markdown
    assert "run-summary.csv" in markdown and "askuserquestion-report.json" in markdown


def test_build_report_breaks_out_results_by_condition():
    # by_condition is the study's actual independent variable (ambiguous vs
    # full); it must not collapse across conditions the way an aggregate
    # resolve rate alone would.
    ambiguous_resolved = {
        **_summary("amb-1", False),
        "task": {"instance_id": "amb-1", "condition": "ambiguous"},
        "evaluation": {"status": "scored", "resolved": True, "localization_hit": True},
    }
    full_unresolved = {
        **_summary("full-1", False),
        "task": {"instance_id": "full-1", "condition": "full"},
        "evaluation": {"status": "scored", "resolved": False, "localization_hit": True},
    }
    report = study_log.build_report([ambiguous_resolved, full_unresolved], [])
    by_condition = report["evaluation"]["by_condition"]

    assert by_condition["ambiguous"]["scored"] == 1
    assert by_condition["ambiguous"]["resolved"] == 1
    assert by_condition["ambiguous"]["resolve_rate"] == 1.0
    assert by_condition["full"]["scored"] == 1
    assert by_condition["full"]["resolved"] == 0
    assert by_condition["full"]["resolve_rate"] == 0.0
    # The aggregate cell (not sliced by condition) still covers both.
    assert report["evaluation"]["resolved"] == 1
    assert report["evaluation"]["scored"] == 2


def _sdk_manifest():
    return {
        "run_id": "sdk-run",
        "started_at": _timestamp(),
        "task": {"instance_id": "owner__repo-1", "condition": "ambiguous"},
        "claude": {"model": "claude-opus-4-8", "interface": "sdk"},
        "workspace": "/tmp/workspace",
    }


def _sdk_observation(first_direct=None, **overrides):
    observation = {
        "tool_roster": ["AskUserQuestion", "Read"],
        "reference_toolset": ["AskUserQuestion", "Read"],
        "permission_mode": "default",
        "permission_prompts": 3,
        "askuserquestion_available": True,
        "stopped_on_first_ask": first_direct is not None,
        "result": {"is_error": False, "stop_reason": "end_turn", "num_turns": 5},
        "analysis": {
            "first_direct": first_direct,
            "direct_count": 1 if first_direct else 0,
            "any_agent_count": 1 if first_direct else 0,
        },
    }
    observation.update(overrides)
    return observation


def test_sdk_summary_reports_callback_measured_first_ask_latency():
    first = {
        "tool_use_id": "tu-1",
        "timestamp": _timestamp(),
        "latency_seconds": 12.5,
        "assistant_tool_actions_before": 4,
        "input": {"questions": [{"question": "Which?", "options": [{"label": "a"}, {"label": "b"}]}]},
    }
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), _sdk_observation(first))

    ask = summary["ask_user_question"]
    assert ask["direct_asked"] is True
    assert ask["first_direct_latency_seconds"] == 12.5
    assert ask["first_direct"]["question_count"] == 1
    assert ask["first_direct"]["option_count"] == 2


def test_sdk_summary_records_stop_on_first_ask_and_permission_mode():
    first = {"tool_use_id": "tu-1", "latency_seconds": 1.0, "input": {"questions": []}}
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), _sdk_observation(first))

    assert summary["process"]["stop_reason"] == "stopped_on_first_ask"
    assert summary["permissions"]["mode"] == "default"
    assert summary["permissions"]["prompts_reaching_callback"] == 3


def test_sdk_summary_labels_an_ask_cap_stop_like_the_codex_arm():
    # Both arms cap synthetic answers per run; the run summary uses the same
    # stop_reason label so capped runs from either arm sort together.
    first = {"tool_use_id": "tu-1", "latency_seconds": 1.0, "input": {"questions": []}}
    observation = _sdk_observation(first, hit_ask_cap=True, max_ask_rounds=3)
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), observation)

    assert summary["process"]["stop_reason"] == "max_ask_rounds"
    assert summary["ask_user_question"]["hit_ask_cap"] is True
    assert summary["ask_user_question"]["max_ask_rounds"] == 3
    assert summary["ask_user_question"]["direct_asked"] is True


def test_sdk_summary_without_ask_keeps_model_stop_reason_and_null_latency():
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), _sdk_observation(None))

    assert summary["process"]["stop_reason"] == "end_turn"
    assert summary["ask_user_question"]["direct_asked"] is False
    assert summary["ask_user_question"]["first_direct_latency_seconds"] is None


def test_sdk_summary_marks_a_session_that_never_ran_as_unknown():
    # Observed live: stop_reason=stop_sequence, 1 turn, $0 cost, 0 permission
    # prompts -- the model never acted. Counting this as "did not ask" would
    # put a non-run in the denominator of the ask rate.
    observation = _sdk_observation(
        None,
        permission_prompts=0,
        result={"is_error": False, "stop_reason": "stop_sequence", "num_turns": 1},
    )
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), observation)

    assert summary["ask_user_question"]["direct_asked"] is None
    assert summary["session"]["ran_meaningfully"] is False
    assert summary["session"]["monitoring_status"] == "no_work_performed"


def test_sdk_summary_marks_an_errored_session_as_unknown():
    observation = _sdk_observation(
        None,
        permission_prompts=5,
        result={"is_error": True, "stop_reason": "error", "num_turns": 9},
    )
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), observation)

    assert summary["ask_user_question"]["direct_asked"] is None


def test_sdk_summary_counts_a_real_working_session_as_a_no_ask_observation():
    observation = _sdk_observation(
        None,
        permission_prompts=19,
        result={"is_error": False, "stop_reason": "end_turn", "num_turns": 30},
    )
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), observation)

    assert summary["ask_user_question"]["direct_asked"] is False
    assert summary["session"]["monitoring_status"] == "complete_no_ask"


def _eval(status="scored", resolved=True, localization_hit=True):
    return {
        "status": status,
        "resolved": resolved,
        "f2p_total": 1, "f2p_passed": 1 if resolved else 0,
        "p2p_total": 1, "p2p_passed": 1 if resolved else 0,
        "missing_node_ids": [],
        "localization_hit": localization_hit,
        "gold_files": ["src/a.py"], "agent_source_files": ["src/a.py"],
        "agent_test_files": [], "patch_path": None, "patch_bytes": 10,
        "empty_patch": False, "duration_seconds": 1.0, "error": None,
    }


def _eval_summary(run_id, asked, evaluation, difficulty="15 min - 1 hour"):
    summary = _summary(run_id, asked)
    summary["task"]["difficulty"] = difficulty
    summary["evaluation"] = evaluation
    return summary


def test_sdk_summary_embeds_the_evaluation_block():
    summary = study_log.build_run_summary_sdk(
        _sdk_manifest(), _sdk_observation(None), _eval()
    )
    assert summary["evaluation"]["status"] == "scored"
    assert summary["evaluation"]["resolved"] is True


def test_sdk_summary_without_an_evaluation_records_not_evaluated():
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), _sdk_observation(None))
    assert summary["evaluation"] == {"status": "not_evaluated", "resolved": None}


def test_report_compares_resolution_between_asking_and_non_asking_runs(tmp_path):
    logs_root = tmp_path / "logs"
    study_log.write_run_summary(logs_root, _eval_summary("a", True, _eval(resolved=True)))
    study_log.write_run_summary(logs_root, _eval_summary("b", False, _eval(resolved=False)))
    study_log.write_run_summary(logs_root, _eval_summary("c", False, _eval(resolved=True)))

    report = json.loads(study_log.write_report(logs_root)["json"].read_text())
    block = report["evaluation"]

    assert block["evaluated"] == 3
    assert block["scored"] == 3
    assert block["resolved"] == 2
    assert block["by_asked"]["asked"]["resolved"] == 1
    assert block["by_asked"]["asked"]["resolve_rate"] == 1.0
    assert block["by_asked"]["not_asked"]["resolved"] == 1
    assert block["by_asked"]["not_asked"]["resolve_rate"] == 0.5


def test_report_excludes_ungradable_runs_from_the_resolve_rate(tmp_path):
    # An unsupported runner or a missing dependency must never be counted as
    # the agent producing a bad patch.
    logs_root = tmp_path / "logs"
    study_log.write_run_summary(logs_root, _eval_summary("a", False, _eval(resolved=True)))
    study_log.write_run_summary(
        logs_root,
        _eval_summary("b", False, _eval(status="unsupported_runner", resolved=None)),
    )
    study_log.write_run_summary(
        logs_root,
        _eval_summary("c", False, _eval(status="env_unavailable", resolved=None)),
    )

    report = json.loads(study_log.write_report(logs_root)["json"].read_text())
    block = report["evaluation"]

    assert block["evaluated"] == 3
    assert block["scored"] == 1
    assert block["resolve_rate"] == 1.0
    assert block["status_counts"]["unsupported_runner"] == 1
    assert block["status_counts"]["env_unavailable"] == 1


def test_report_csv_includes_the_evaluation_columns(tmp_path):
    logs_root = tmp_path / "logs"
    study_log.write_run_summary(logs_root, _eval_summary("a", False, _eval()))

    text = study_log.write_report(logs_root)["csv"].read_text(encoding="utf-8")

    assert "eval_status" in text and "resolved" in text
    assert "localization_hit" in text
    assert "scored" in text


def test_report_markdown_reports_the_asked_versus_not_asked_split(tmp_path):
    logs_root = tmp_path / "logs"
    study_log.write_run_summary(logs_root, _eval_summary("a", True, _eval(resolved=True)))

    text = study_log.write_report(logs_root)["markdown"].read_text(encoding="utf-8")

    assert "Patch evaluation" in text
    assert "asked" in text


def test_evaluations_are_stored_separately_and_can_be_overwritten(tmp_path):
    # Grades live outside the write-once run summary precisely so they can be
    # recomputed when the grading rules change.
    logs_root = tmp_path / "logs"
    study_log.write_evaluation(logs_root, "run-1", _eval(resolved=False))
    study_log.write_evaluation(logs_root, "run-1", _eval(resolved=True))

    stored = study_log.load_evaluations(logs_root)
    assert stored["run-1"]["resolved"] is True
    assert "evaluated_at" in stored["run-1"]


def test_run_summaries_stay_write_once_while_evaluations_do_not(tmp_path):
    logs_root = tmp_path / "logs"
    study_log.write_run_summary(logs_root, _summary("run-1", False))

    try:
        study_log.write_run_summary(logs_root, _summary("run-1", True))
    except FileExistsError:
        pass
    else:
        raise AssertionError("run summaries must remain immutable")

    study_log.write_evaluation(logs_root, "run-1", _eval())
    study_log.write_evaluation(logs_root, "run-1", _eval(resolved=False))


def test_a_stored_evaluation_overrides_one_embedded_in_an_older_summary(tmp_path):
    logs_root = tmp_path / "logs"
    summary = _eval_summary("run-1", False, _eval(resolved=False))
    study_log.write_run_summary(logs_root, summary)
    study_log.write_evaluation(logs_root, "run-1", _eval(resolved=True))

    report = json.loads(study_log.write_report(logs_root)["json"].read_text())

    assert report["evaluation"]["resolved"] == 1
    assert report["evaluation"]["resolve_rate"] == 1.0


def test_sdk_summary_persists_error_evidence():
    # Without these fields an errored run (e.g. a usage-limit rejection: one
    # turn, $0) is indistinguishable from "the agent chose to do nothing",
    # and the actual error text dies with the observation.
    observation = _sdk_observation(
        result={
            "is_error": True,
            "subtype": "error_during_execution",
            "stop_reason": None,
            "num_turns": 1,
            "total_cost_usd": 0,
            "result": "usage limit reached — resets at 22:00",
        }
    )
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), observation)

    process = summary["process"]
    assert process["sdk_is_error"] is True
    assert process["sdk_result_subtype"] == "error_during_execution"
    assert "usage limit" in process["sdk_error"]
    assert summary["session"]["ran_meaningfully"] is False


def test_sdk_summary_records_no_error_fields_on_success():
    summary = study_log.build_run_summary_sdk(_sdk_manifest(), _sdk_observation(None))
    process = summary["process"]
    assert process["sdk_is_error"] is False
    assert process["sdk_error"] is None


def test_agent_messages_text_keeps_only_assistant_prose():
    # The transcripts/ rendering is the agent's words alone: tool calls,
    # tool results, and non-assistant records stay in the raw sessions/ copy.
    records = [
        (1, {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}),
        (2, {
            "type": "assistant",
            "timestamp": "2026-07-31T00:00:01Z",
            "message": {"content": [
                {"type": "text", "text": "Looking at the repo."},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ]},
        }),
        (3, {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {}},
        ]}}),
        (4, {"type": "assistant", "timestamp": "2026-07-31T00:00:09Z",
             "message": {"content": [{"type": "text", "text": "Done; fix applied."}]}}),
    ]
    text = study_log.agent_messages_text(records)
    assert "Looking at the repo." in text and "Done; fix applied." in text
    assert "hi" not in text          # user words excluded
    assert "Bash" not in text        # tool calls excluded
    assert "[assistant #1 @ 2026-07-31T00:00:01Z]" in text
    assert "[assistant #2 @ 2026-07-31T00:00:09Z]" in text  # tool-only record not numbered


def test_agent_info_reads_both_record_generations():
    legacy = {"claude": {"model": "claude-opus-4-8", "interface": "sdk"}}
    modern = {"agent": {"model": "gpt-5.6-sol", "runner": "codex-cli"}}
    assert study_log.agent_info(legacy) == {
        "model": "claude-opus-4-8", "interface": "sdk", "runner": "claude-sdk",
    }
    assert study_log.agent_info(modern)["runner"] == "codex-cli"
    assert study_log.agent_info({}) == {}


def _codex_manifest():
    return {
        "run_id": "codex-run",
        "started_at": _timestamp(),
        "task": {"instance_id": "owner__repo-1", "condition": "ambiguous"},
        "agent": {
            "model": "gpt-5.6-sol",
            "runner": "codex-cli",
            "cli_version": "codex-cli 0.146.0",
            "sandbox": "danger-full-access",
        },
        "workspace": "/tmp/workspace",
    }


def _codex_observation(first_direct=None, **overrides):
    asked = first_direct is not None
    observation = {
        "runner": "codex-cli",
        "model": "gpt-5.6-sol",
        "sandbox": "danger-full-access",
        "ask_channel": "final_message",
        "ask_classifier_version": 6,
        "max_ask_rounds": 3,
        "neutral_answer": "Up to you.",
        "thread_id": "thread-1",
        "started_at": _timestamp(),
        "ended_at": _timestamp(1),
        "rounds": [
            {"index": 1, "prompt_kind": "task", "prompt": "Resolve...", "asked": asked,
             "tool_actions": 5, "exit_code": 0, "duration_seconds": 60.0,
             "agent_messages": ["working", "final words"]},
        ],
        "tool_actions_total": 5,
        "answered_questions": (
            [{"round": 1, "question_text": "A or B?", "reply": "Up to you."}] if asked else []
        ),
        "result": {
            "subtype": "completed", "is_error": False, "num_turns": 1,
            "duration_ms": 60000, "session_id": "thread-1",
            "stop_reason": "completed", "total_cost_usd": None,
            "usage": {"input_tokens": 10, "output_tokens": 4}, "result": None,
        },
        "analysis": {
            "first_direct": first_direct,
            "direct_count": 1 if asked else 0,
            "any_agent_count": 1 if asked else 0,
        },
    }
    observation.update(overrides)
    return observation


def test_codex_summary_records_the_final_message_ask_channel():
    first = {
        "round": 1,
        "timestamp": _timestamp(),
        "latency_seconds": 42.0,
        "message_text": "Should I cap the value or raise?",
        "assistant_tool_actions_before": 5,
        "workspace_had_changes": False,
    }
    summary = study_log.build_run_summary_codex(_codex_manifest(), _codex_observation(first))

    ask = summary["ask_user_question"]
    assert ask["channel"] == "final_message"
    assert ask["classifier_version"] == 6
    assert ask["direct_asked"] is True
    assert ask["first_direct"]["message_text"] == "Should I cap the value or raise?"
    assert ask["first_direct_latency_seconds"] == 42.0
    assert ask["answered_questions"][0]["reply"] == "Up to you."
    assert summary["agent"]["model"] == "gpt-5.6-sol"
    assert summary["agent"]["runner"] == "codex-cli"
    assert summary["process"]["codex_thread_id"] == "thread-1"
    assert summary["session"]["ran_meaningfully"] is True
    assert summary["session"]["monitoring_status"] == "observed_ask"


def test_codex_summary_counts_a_working_no_ask_session_as_an_observation():
    summary = study_log.build_run_summary_codex(_codex_manifest(), _codex_observation(None))
    assert summary["ask_user_question"]["direct_asked"] is False
    assert summary["session"]["monitoring_status"] == "complete_no_ask"


def test_codex_summary_marks_a_session_that_never_worked_as_unknown():
    # No tool actions and no ask: the model observed nothing about the ask
    # decision, so it must not enter the ask-rate denominator.
    observation = _codex_observation(None, tool_actions_total=0)
    summary = study_log.build_run_summary_codex(_codex_manifest(), observation)
    assert summary["ask_user_question"]["direct_asked"] is None
    assert summary["session"]["ran_meaningfully"] is False
    assert summary["session"]["monitoring_status"] == "no_work_performed"


def test_codex_summary_persists_error_evidence():
    observation = _codex_observation(
        None,
        result={
            "subtype": "error", "is_error": True, "num_turns": 1,
            "duration_ms": 100, "session_id": "thread-1", "stop_reason": "error",
            "total_cost_usd": None, "usage": {},
            "result": "model requires a newer version of Codex",
        },
    )
    summary = study_log.build_run_summary_codex(_codex_manifest(), observation)
    process = summary["process"]
    assert process["codex_is_error"] is True
    assert "newer version of Codex" in process["codex_error"]
    assert summary["ask_user_question"]["direct_asked"] is None


def _codex_report_summary(run_id, asked, condition="ambiguous", evaluation=None):
    summary = {
        "run_id": run_id,
        "task": {"instance_id": run_id, "condition": condition},
        "agent": {"model": "gpt-5.6-sol", "runner": "codex-cli"},
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:01:00+00:00",
        "process": {"exit_code": 0, "stop_reason": "completed"},
        "session": {"monitoring_status": "complete_no_ask"},
        "ask_user_question": {
            "channel": "final_message",
            "direct_asked": asked,
            "direct_count": int(asked is True),
            "any_agent_count": int(asked is True),
            "first_direct": {"message_text": "A or B?"} if asked else None,
            "first_direct_latency_seconds": 5.0 if asked else None,
        },
    }
    if evaluation:
        summary["evaluation"] = evaluation
    return summary


def test_report_keeps_models_apart_and_supports_comparison(tmp_path):
    # Two arms over the same instances: aggregates must never pool models,
    # and legacy claude-keyed summaries must land in their own arm.
    logs_root = tmp_path / "logs"
    study_log.write_run_summary(
        logs_root, {**_eval_summary("claude-amb", True, _eval(resolved=True)),
                    "task": {"instance_id": "i1", "condition": "ambiguous"}},
    )
    study_log.write_run_summary(
        logs_root, {**_codex_report_summary("gpt-amb", False, evaluation=_eval(resolved=False)),
                    "task": {"instance_id": "i1", "condition": "ambiguous"}},
    )
    study_log.write_run_summary(
        logs_root, {**_codex_report_summary("gpt-full", False, evaluation=_eval(resolved=True)),
                    "task": {"instance_id": "i1", "condition": "full"}},
    )

    paths = study_log.write_report(logs_root)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    models = report["models"]

    assert set(models) == {"claude-opus-4-8", "gpt-5.6-sol"}
    claude_cell = models["claude-opus-4-8"]
    gpt_cell = models["gpt-5.6-sol"]
    assert claude_cell["ask"] == {"runs": 1, "valid": 1, "asked": 1, "ask_rate": 1.0}
    assert gpt_cell["ask"] == {"runs": 2, "valid": 2, "asked": 0, "ask_rate": 0.0}
    # Channel provenance stays attached to each arm.
    assert claude_cell["ask_channels"] == {"askuserquestion_tool": 1}
    assert gpt_cell["ask_channels"] == {"final_message": 2}
    # Per-model condition split supports the ambiguous-vs-full comparison.
    assert gpt_cell["by_condition"]["ambiguous"]["resolution"]["resolved"] == 0
    assert gpt_cell["by_condition"]["full"]["resolution"]["resolved"] == 1
    assert claude_cell["by_condition"]["ambiguous"]["ask"]["ask_rate"] == 1.0

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "runner" in csv_text and "ask_channel" in csv_text
    assert "codex-cli" in csv_text and "final_message" in csv_text

    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "## Model comparison" in markdown
    assert "gpt-5.6-sol" in markdown and "claude-opus-4-8" in markdown


def test_codex_transcript_marks_the_final_message_of_each_round():
    rounds = [
        {"index": 1, "prompt_kind": "task", "prompt": "Fix it.",
         "agent_messages": ["digging in", "Should I do A or B?"]},
        {"index": 2, "prompt_kind": "synthetic_answer", "prompt": "Up to you.",
         "agent_messages": ["DONE"]},
    ]
    text = study_log.codex_rounds_text(rounds)
    assert "[round 1 :: agent commentary]" in text and "digging in" in text
    assert "[round 1 :: agent final]" in text and "Should I do A or B?" in text
    assert "[round 2 :: synthetic_answer prompt]" in text
    assert "[round 2 :: agent final]" in text and "DONE" in text


def test_preserve_codex_session_artifacts_copies_rollouts_and_transcript(tmp_path):
    codex_home = tmp_path / "codex-home"
    rollout = codex_home / "sessions" / "2026" / "08" / "01" / "rollout-x-thread-9.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text("{}\n", encoding="utf-8")
    logs = tmp_path / "logs"

    result = study_log.preserve_codex_session_artifacts(
        logs,
        run_id="run-1",
        thread_id="thread-9",
        rounds=[{"index": 1, "prompt_kind": "task", "prompt": "p",
                 "agent_messages": ["words"]}],
        codex_home=codex_home,
    )

    assert (logs / "sessions/run-1/rollout-x-thread-9.jsonl").exists()
    assert "words" in (logs / "transcripts/run-1/rounds.txt").read_text(encoding="utf-8")
    assert len(result["copied"]) == 1


def test_preserve_session_artifacts_copies_main_and_subagent_files(tmp_path):
    projects = tmp_path / "projects"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = projects / "some-munged-name"
    (project / "subagents").mkdir(parents=True)
    record = json.dumps({
        "type": "assistant", "cwd": str(workspace), "timestamp": _timestamp(),
        "message": {"content": [{"type": "text", "text": "agent words"}]},
    })
    (project / "sess-1.jsonl").write_text(record + "\n")
    (project / "sess-0.jsonl").write_text(record + "\n")  # older/other session
    (project / "subagents" / "agent-x.jsonl").write_text(record + "\n")
    logs = tmp_path / "logs"

    result = study_log.preserve_session_artifacts(
        logs,
        run_id="run-1",
        workspace=workspace,
        started_at=_timestamp(-60),
        session_id="sess-1",
        projects_dir=projects,
    )

    assert (logs / "sessions/run-1/sess-1.jsonl").exists()
    assert not (logs / "sessions/run-1/sess-0.jsonl").exists()  # other session excluded
    assert (logs / "sessions/run-1/subagents/agent-x.jsonl").exists()
    transcript = (logs / "transcripts/run-1/sess-1.txt").read_text()
    assert "agent words" in transcript
    assert len(result["copied"]) == 2
