import json
from pathlib import Path

import pytest

import datasets_registry
from experiment import (
    CHECKOUTS,
    CONDITION_FIELD,
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    build_prompt,
    dataset_of,
    load_rows,
    requested_conditions,
    resolve_conditions,
    runner_for_model,
    select_batch_rows,
    issue_text,
    workspace_path,
)
from sdk_runner import PERMISSION_MODE, load_reference_toolset


ROW = {
    "instance_id": "owner__repo-1",
    "repo": "owner/repo",
    "base_commit": "0123456789abcdef",
    "problem_statement": "AMBIGUOUS TEXT",
    "original_issue": "FULL TEXT",
}

# One sentinel per evaluator-only column, so a leak names the guilty field.
ANSWER_KEY_SENTINELS = {
    field: f"SENTINEL_{field.upper()}"
    for field in datasets_registry.ANSWER_KEY_FIELDS
}

MISSING_INFO_ROW = {
    **ROW,
    "rewrite_3": "MASKED TEXT",
    **ANSWER_KEY_SENTINELS,
}


def test_condition_mapping_is_explicit():
    # Every condition names exactly one issue field, so no condition can widen
    # what the agent sees. mi_* read the missing-info workbook.
    assert CONDITION_FIELD == {
        "ambiguous": "problem_statement",
        "full": "original_issue",
        "mi_ambiguous": "rewrite_3",
        "mi_full": "original_issue",
    }
    assert issue_text(ROW, "ambiguous") == "AMBIGUOUS TEXT"
    assert issue_text(ROW, "full") == "FULL TEXT"
    assert issue_text(MISSING_INFO_ROW, "mi_ambiguous") == "MASKED TEXT"
    assert issue_text(MISSING_INFO_ROW, "mi_full") == "FULL TEXT"


def test_prompt_contains_only_selected_issue_text():
    ambiguous = build_prompt(ROW, "ambiguous")
    full = build_prompt(ROW, "full")
    assert "AMBIGUOUS TEXT" in ambiguous
    assert "FULL TEXT" not in ambiguous
    assert "FULL TEXT" in full
    assert "AMBIGUOUS TEXT" not in full


def test_prompt_never_carries_the_missing_info_answer_key():
    # The workbook ships the answer key to the exact question this study asks:
    # which categories of information were hidden from the prompt. A prompt
    # that carried any of it would invalidate the ask measurement.
    for condition in ("mi_ambiguous", "mi_full"):
        prompt = build_prompt(MISSING_INFO_ROW, condition)
        for field, sentinel in ANSWER_KEY_SENTINELS.items():
            assert sentinel not in prompt, f"{field} leaked into the {condition} prompt"


def test_loaded_rows_do_not_carry_evaluator_only_fields():
    # Defense in depth: the answer keys are stripped at load time, so they are
    # not merely unused by the prompt -- they are absent from the row.
    for dataset in datasets_registry.DATASETS:
        rows = load_rows(dataset)
        present = {key for row in rows.values() for key in row}
        assert not (present & datasets_registry.ANSWER_KEY_FIELDS)


def test_conditions_are_rejected_for_the_wrong_dataset():
    # Both datasets share all 500 instance_ids, so a mismatched condition would
    # silently record a run under the wrong name instead of failing.
    with pytest.raises(SystemExit):
        resolve_conditions("missing-info", ("ambiguous",))
    with pytest.raises(SystemExit):
        resolve_conditions("interactive-swe", ("mi_ambiguous",))
    assert resolve_conditions("missing-info", ("mi_ambiguous",)) == ("mi_ambiguous",)


def test_batch_both_expands_within_the_selected_dataset():
    assert requested_conditions("both", "interactive-swe") == ("ambiguous", "full")
    assert requested_conditions("both", "missing-info") == ("mi_ambiguous", "mi_full")


def test_dataset_of_reads_runs_written_before_the_field_existed():
    assert dataset_of({"task": {"dataset": "missing-info"}}) == "missing-info"
    # Legacy runs carry no `dataset`; their condition identifies the source.
    assert dataset_of({"task": {"condition": "ambiguous"}}) == "interactive-swe"
    assert dataset_of({"task": {"condition": "mi_ambiguous"}}) == "missing-info"


def test_session_does_not_bypass_permission_prompts():
    # bypassPermissions shadows can_use_tool for ordinary tools, so the agent
    # never pauses and can always resolve ambiguity by reading the repo
    # instead of asking. The study depends on that friction being present.
    assert PERMISSION_MODE == "default"


def test_reference_toolset_exposes_askuserquestion():
    assert "AskUserQuestion" in load_reference_toolset()


def test_reference_toolset_was_not_captured_under_claude_code():
    # A shell running under Claude Code exports CLAUDE_CODE_ENABLE_TASKS=0,
    # which swaps the Task* tools for the legacy TodoWrite. A reference
    # captured there does not match what an ordinary run receives, and every
    # run logs matches_reference: false against it.
    tools = load_reference_toolset()
    assert "TodoWrite" not in tools
    assert {"TaskCreate", "TaskGet", "TaskList", "TaskUpdate"} <= set(tools)


def test_checkout_is_stored_in_a_gitignored_subdirectory_of_cwd():
    assert CHECKOUTS.name == ".experiment-checkouts"
    assert CHECKOUTS.parent == Path.cwd()
    assert workspace_path(ROW, "ambiguous") == CHECKOUTS / (
        "owner__repo__0123456789ab__ambiguous"
    )


def test_batch_selection_resumes_from_the_next_incomplete_instance():
    rows = [
        {**ROW, "instance_id": "one"},
        {**ROW, "instance_id": "two"},
        {**ROW, "instance_id": "three"},
    ]
    completed = {
        ("one", "ambiguous", "model"),
        ("two", "ambiguous", "model"),
    }

    selected = select_batch_rows(
        rows, completed, requested_conditions("ambiguous"), "model", 2
    )

    assert [(row["instance_id"], missing) for row, missing in selected] == [
        ("three", ("ambiguous",)),
    ]


def test_batch_both_runs_only_the_missing_condition_for_resumed_instances():
    rows = [{**ROW, "instance_id": "one"}, {**ROW, "instance_id": "two"}]
    completed = {("one", "ambiguous", "model")}

    selected = select_batch_rows(
        rows, completed, requested_conditions("both"), "model", 2
    )

    assert [(row["instance_id"], missing) for row, missing in selected] == [
        ("one", ("full",)),
        ("two", ("ambiguous", "full")),
    ]


def test_the_real_dataset_has_500_instances():
    assert len(load_rows()) == 500


def test_both_datasets_cover_the_same_500_instances():
    assert set(load_rows("interactive-swe")) == set(load_rows("missing-info"))


def test_missing_info_test_lists_are_valid_json():
    # 26 PASS_TO_PASS cells hit Excel's 32,767-character limit and are
    # truncated to invalid JSON. swebench_eval.parse_node_ids swallows that
    # and returns [], which would grade those instances against an empty
    # regression suite and report them resolved. They are repaired from the
    # Arrow dataset at load time; this guards the repair.
    for instance_id, row in load_rows("missing-info").items():
        for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            value = json.loads(row[field])
            assert isinstance(value, list), f"{instance_id}: {field} is not a list"


def test_missing_info_ambiguous_text_is_a_distinct_stimulus():
    # rewrite_3 is an independent masked rewrite, not a copy of the
    # interactive-swe ambiguous text. If these ever coincide the two datasets
    # would be measuring the same prompt under different condition names.
    interactive = load_rows("interactive-swe")
    overlap = [
        instance_id
        for instance_id, row in load_rows("missing-info").items()
        if (row.get("rewrite_3") or "").strip()
        and (row.get("rewrite_3") or "").strip()
        == (interactive[instance_id].get("problem_statement") or "").strip()
    ]
    assert not overlap


def test_missing_info_runnable_instance_count():
    # Four instances are unannotated and have no ambiguous rewrite to present.
    rows = load_rows("missing-info")
    runnable = [row for row in rows.values() if (row.get("rewrite_3") or "").strip()]
    assert len(runnable) == 496


def test_sessions_run_to_completion_so_asking_runs_are_still_gradable():
    # Halting at the first ask would leave asking runs without a patch while
    # non-asking runs kept theirs, biasing the comparison the study exists for.
    import inspect

    import sdk_runner

    default = inspect.signature(sdk_runner.run_sdk_session).parameters[
        "stop_on_first_ask"
    ].default
    assert default is False


def test_dead_runs_do_not_block_a_retry(tmp_path):
    # A run that errored or never did meaningful work observed nothing about
    # the ask decision, so the batch must pick its instance up again. The
    # 2026-07-30 batch buried six usage-limit rejections without this.
    import json

    from experiment import completed_run_keys

    runs = tmp_path / "runs"
    runs.mkdir()

    def write(run_id, instance_id, ran_meaningfully, stop_reason="end_turn"):
        (runs / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "task": {"instance_id": instance_id, "condition": "ambiguous"},
                    "claude": {"model": "model"},
                    "process": {"stop_reason": stop_reason},
                    "session": {"ran_meaningfully": ran_meaningfully},
                }
            )
        )

    write("good", "one", True)
    write("dead", "two", False)          # e.g. usage-limit rejection
    write("launch", "three", True, stop_reason="launch_error")

    assert completed_run_keys(tmp_path) == {("one", "ambiguous", "model")}


def test_one_model_switch_routes_to_the_right_runner():
    # A single --model flag selects the study arm; the runner follows from
    # the model name, so no separate runner flag can drift out of sync.
    assert runner_for_model("claude-opus-4-8") == RUNNER_CLAUDE
    assert runner_for_model("claude-sonnet-5") == RUNNER_CLAUDE
    assert runner_for_model("gpt-5.6-sol") == RUNNER_CODEX
    assert runner_for_model("gpt-5.6-terra") == RUNNER_CODEX
    assert runner_for_model("codex-mini") == RUNNER_CODEX
    with pytest.raises(SystemExit):
        runner_for_model("gemini-3")


def test_resume_state_is_keyed_per_model_across_both_record_generations(tmp_path):
    # The same instance+condition must stay runnable for a *different* model,
    # and legacy summaries (claude key) must count for their own model just
    # like multi-runner summaries (agent key).
    import json

    from experiment import completed_run_keys

    runs = tmp_path / "runs"
    runs.mkdir()
    legacy = {
        "run_id": "legacy",
        "task": {"instance_id": "one", "condition": "ambiguous"},
        "claude": {"model": "claude-opus-4-8", "interface": "sdk"},
        "process": {"stop_reason": "end_turn"},
        "session": {"ran_meaningfully": True},
    }
    codex = {
        "run_id": "codex",
        "task": {"instance_id": "one", "condition": "ambiguous"},
        "agent": {"model": "gpt-5.6-sol", "runner": "codex-cli"},
        "process": {"stop_reason": "completed"},
        "session": {"ran_meaningfully": True},
    }
    (runs / "legacy.json").write_text(json.dumps(legacy))
    (runs / "codex.json").write_text(json.dumps(codex))

    assert completed_run_keys(tmp_path) == {
        ("one", "ambiguous", "claude-opus-4-8"),
        ("one", "ambiguous", "gpt-5.6-sol"),
    }


def test_full_conditions_share_one_equivalence_key_but_ambiguous_do_not():
    # full and mi_full present the same original issue at the same commit, so
    # a run of one IS a run of the other. The ambiguous conditions are
    # independently written rewrites and must never satisfy each other.
    from experiment import equivalence_key

    isw = {
        "instance_id": "owner__repo-1", "repo": "owner/repo",
        "base_commit": "0123456789abcdef",
        "problem_statement": "SWE REWRITE", "original_issue": "FULL TEXT",
    }
    mi = {**isw, "rewrite_3": "MI REWRITE"}

    assert equivalence_key(isw, "full") == equivalence_key(mi, "mi_full")
    assert equivalence_key(isw, "ambiguous") != equivalence_key(mi, "mi_ambiguous")


def test_full_equivalence_ignores_whitespace_only_differences():
    # The two datasets round-tripped the same issues through different tooling:
    # 252 of 500 full prompts differ only in line endings and trailing spaces.
    # Keying on raw text would call those distinct tasks and re-run them.
    from experiment import equivalence_key

    base = {"instance_id": "owner__repo-1", "repo": "owner/repo",
            "base_commit": "0123456789abcdef"}
    crlf = {**base, "original_issue": "line one  \r\nline two\r\n"}
    lf = {**base, "original_issue": "line one\nline two"}

    assert equivalence_key(crlf, "full") == equivalence_key(lf, "mi_full")


def test_a_finished_full_run_satisfies_the_other_datasets_full_condition(tmp_path):
    # The reported bug: a completed interactive-swe/full run did not stop
    # `--dataset missing-info --condition both` from spending a duplicate
    # mi_full session that measured only sampling noise.
    import json

    from experiment import completed_run_keys, select_batch_rows

    isw = {"instance_id": "owner__repo-1", "repo": "owner/repo",
           "base_commit": "0123456789abcdef",
           "problem_statement": "SWE REWRITE", "original_issue": "FULL TEXT"}
    mi = {**isw, "rewrite_3": "MI REWRITE"}
    rows = {"interactive-swe": {isw["instance_id"]: isw},
            "missing-info": {mi["instance_id"]: mi}}

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "done.json").write_text(json.dumps({
        "run_id": "done",
        "task": {"instance_id": "owner__repo-1", "condition": "full",
                 "dataset": "interactive-swe"},
        "agent": {"model": "gpt-5.6-sol", "runner": "codex-cli"},
        "process": {"stop_reason": "completed"},
        "session": {"ran_meaningfully": True},
    }))
    completed = completed_run_keys(tmp_path, rows)

    selected = select_batch_rows(
        [mi], completed, ("mi_ambiguous", "mi_full"), "gpt-5.6-sol", 1
    )
    assert [missing for _, missing in selected] == [("mi_ambiguous",)]

    # A different model is untouched by the reuse.
    other = select_batch_rows([mi], completed, ("mi_full",), "claude-opus-4-8", 1)
    assert [missing for _, missing in other] == [("mi_full",)]
