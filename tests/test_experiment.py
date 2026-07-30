from pathlib import Path

from experiment import (
    CHECKOUTS,
    CONDITION_FIELD,
    build_prompt,
    is_gradable,
    load_rows,
    requested_conditions,
    scoped_rows,
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


GRADABLE_ROW = {
    **ROW,
    "instance_id": "astropy__astropy-1",
    "FAIL_TO_PASS": '["astropy/x/tests/test_a.py::test_b"]',
    "PASS_TO_PASS": "[]",
}
UNGRADABLE_ROW = {
    **ROW,
    "instance_id": "django__django-1",
    "FAIL_TO_PASS": '["test_x (auth_tests.test_validators.Tests)"]',
    "PASS_TO_PASS": "[]",
}


def test_gradable_scope_recognizes_pytest_node_ids_only():
    assert is_gradable(GRADABLE_ROW)
    assert not is_gradable(UNGRADABLE_ROW)


def test_scope_filters_the_candidate_pool_before_selection():
    # --count N must mean "N instances I can grade", not "N of the first 500
    # most of which get skipped".
    rows = [UNGRADABLE_ROW, UNGRADABLE_ROW, GRADABLE_ROW]

    assert len(scoped_rows(rows, "all")) == 3
    assert scoped_rows(rows, "gradable") == [GRADABLE_ROW]

    selected = select_batch_rows(
        scoped_rows(rows, "gradable"), set(), ("ambiguous",), "model", 1
    )
    assert [row["instance_id"] for row, _ in selected] == ["astropy__astropy-1"]


def test_the_real_dataset_has_the_expected_gradable_subset():
    rows = list(load_rows().values())
    gradable = scoped_rows(rows, "gradable")
    assert len(rows) == 500
    assert len(gradable) == 194
    assert not any(row["repo"] in ("django/django", "sympy/sympy") for row in gradable)


def test_sessions_run_to_completion_so_asking_runs_are_still_gradable():
    # Halting at the first ask would leave asking runs without a patch while
    # non-asking runs kept theirs, biasing the comparison the study exists for.
    import inspect

    import sdk_runner

    default = inspect.signature(sdk_runner.run_sdk_session).parameters[
        "stop_on_first_ask"
    ].default
    assert default is False
