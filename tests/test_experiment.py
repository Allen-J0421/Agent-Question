from pathlib import Path

from experiment import (
    CHECKOUTS,
    CONDITION_FIELD,
    build_prompt,
    load_rows,
    requested_conditions,
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


def test_condition_mapping_is_explicit():
    assert CONDITION_FIELD == {
        "ambiguous": "problem_statement",
        "full": "original_issue",
    }
    assert issue_text(ROW, "ambiguous") == "AMBIGUOUS TEXT"
    assert issue_text(ROW, "full") == "FULL TEXT"


def test_prompt_contains_only_selected_issue_text():
    ambiguous = build_prompt(ROW, "ambiguous")
    full = build_prompt(ROW, "full")
    assert "AMBIGUOUS TEXT" in ambiguous
    assert "FULL TEXT" not in ambiguous
    assert "FULL TEXT" in full
    assert "AMBIGUOUS TEXT" not in full


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
