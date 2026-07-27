from pathlib import Path

from experiment import (
    CHECKOUTS,
    CONDITION_FIELD,
    build_prompt,
    claude_argv,
    requested_conditions,
    select_batch_rows,
    issue_text,
    workspace_path,
)


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


def test_claude_command_is_headless_and_bypasses_permission_prompts():
    argv = claude_argv("/usr/bin/claude", "claude-opus-4-8", "PROMPT")
    assert argv == [
        "/usr/bin/claude",
        "--model",
        "claude-opus-4-8",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "PROMPT",
    ]


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
