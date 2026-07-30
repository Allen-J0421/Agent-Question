import json
from pathlib import Path

import evaluation


ROW = {
    "instance_id": "owner__repo-1",
    "repo": "owner/repo",
    "files": "src/thing.py",
    "patch": "diff --git a/src/thing.py b/src/thing.py\n@@ -1 +1 @@\n-a\n+b\n",
    "test_patch": "diff --git a/tests/test_thing.py b/tests/test_thing.py\n",
    "FAIL_TO_PASS": json.dumps(["tests/test_thing.py::test_new"]),
    "PASS_TO_PASS": json.dumps(["tests/test_thing.py::test_old"]),
}


class FakeGit:
    """Record git invocations and replay canned results.

    No test in this repository shells out to real git; this is the seam the
    existing suite uses for subprocess-shaped behavior.
    """

    def __init__(self, status="", diff="", apply_returncode=0):
        self.calls: list[list[str]] = []
        self.applied: list[str] = []
        self.status = status
        self.diff = diff
        self.apply_returncode = apply_returncode

    def __call__(self, args, cwd=None, check=True, input=None):
        self.calls.append(list(args))
        if args[0] == "apply":
            self.applied.append(input or "")

        class Result:
            pass

        result = Result()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if args[0] == "status":
            result.stdout = self.status
        elif args[0] == "diff":
            result.stdout = self.diff
        elif args[0] == "apply":
            result.returncode = self.apply_returncode
            result.stderr = "" if self.apply_returncode == 0 else "does not apply"
        return result


def test_is_test_path_matches_every_repository_test_convention():
    assert evaluation.is_test_path("tests/test_thing.py")
    assert evaluation.is_test_path("astropy/modeling/tests/test_separable.py")
    assert evaluation.is_test_path("testing/test_capture.py")
    assert evaluation.is_test_path("pkg/foo_test.py")
    assert not evaluation.is_test_path("astropy/modeling/separable.py")
    assert not evaluation.is_test_path("src/thing.py")


def test_gold_files_falls_back_to_patch_headers_when_files_is_empty():
    # 26 of the 500 instances have an empty `files` column.
    row = {**ROW, "files": ""}
    assert evaluation.gold_files(row) == ["src/thing.py"]
    assert evaluation.gold_files(ROW) == ["src/thing.py"]


def test_uses_pytest_ids_rejects_the_django_and_sympy_id_formats():
    assert evaluation.uses_pytest_ids(["tests/test_a.py::test_b"])
    assert not evaluation.uses_pytest_ids(
        ["test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)"]
    )
    assert not evaluation.uses_pytest_ids(["test_issue_11617"])
    assert not evaluation.uses_pytest_ids([])


def test_test_files_for_deduplicates_and_preserves_order():
    ids = [
        "a/test_x.py::test_1",
        "a/test_x.py::test_2",
        "b/test_y.py::test_3",
        "not-a-node-id",
    ]
    assert evaluation.test_files_for(ids) == ["a/test_x.py", "b/test_y.py"]


def test_parse_pytest_outcomes_reads_short_summary_lines():
    output = (
        "PASSED tests/test_thing.py::test_old\n"
        "FAILED tests/test_thing.py::test_new\n"
        "ERROR tests/test_thing.py::test_broken\n"
        "some other line\n"
    )
    assert evaluation.parse_pytest_outcomes(output) == {
        "tests/test_thing.py::test_old": "PASSED",
        "tests/test_thing.py::test_new": "FAILED",
        "tests/test_thing.py::test_broken": "ERROR",
    }


def test_pytest_is_given_whole_files_never_node_ids(monkeypatch, tmp_path):
    # A node id that no longer exists makes pytest report "no tests ran" and
    # discard every other id in the same invocation, so the grader must pass
    # files. This is the regression guard for that behavior.
    seen: dict[str, list[str]] = {}

    class Completed:
        returncode = 0
        stdout = "PASSED tests/test_thing.py::test_old\nPASSED tests/test_thing.py::test_new\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return Completed()

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    outcomes, error = evaluation.run_pytest_files(
        tmp_path, ["tests/test_thing.py"], ["python"], 60
    )

    assert error is None
    assert outcomes["tests/test_thing.py::test_old"] == "PASSED"
    assert not any("::" in arg for arg in seen["argv"])
    assert "tests/test_thing.py" in seen["argv"]


def test_unsupported_runner_is_recorded_without_a_resolved_verdict(tmp_path):
    row = {
        **ROW,
        "FAIL_TO_PASS": json.dumps(
            ["test_x (auth_tests.test_validators.UsernameValidatorsTests)"]
        ),
        "PASS_TO_PASS": json.dumps([]),
    }
    git = FakeGit(status=" M src/thing.py\n", diff="diff\n")

    result = evaluation.evaluate_run(row=row, workspace=tmp_path, run_git=git)

    assert result["status"] == "unsupported_runner"
    assert result["resolved"] is None
    assert result["localization_hit"] is True


def test_env_unavailable_is_recorded_without_a_resolved_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluation, "probe_environment", lambda *a, **k: (False, "ModuleNotFoundError")
    )
    git = FakeGit(status=" M src/thing.py\n", diff="diff\n")

    result = evaluation.evaluate_run(row=ROW, workspace=tmp_path, run_git=git)

    assert result["status"] == "env_unavailable"
    assert result["resolved"] is None
    assert "ModuleNotFoundError" in result["error"]


def test_test_patch_failure_is_recorded_without_a_resolved_verdict(tmp_path):
    git = FakeGit(status=" M src/thing.py\n", diff="diff\n", apply_returncode=1)

    result = evaluation.evaluate_run(row=ROW, workspace=tmp_path, run_git=git)

    assert result["status"] == "test_patch_failed"
    assert result["resolved"] is None


def test_resolved_requires_every_f2p_and_p2p_test_to_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "probe_environment", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        evaluation,
        "run_pytest_files",
        lambda *a, **k: (
            {
                "tests/test_thing.py::test_new": "PASSED",
                "tests/test_thing.py::test_old": "PASSED",
            },
            None,
        ),
    )
    git = FakeGit(status=" M src/thing.py\n", diff="diff\n")

    result = evaluation.evaluate_run(row=ROW, workspace=tmp_path, run_git=git)

    assert result["status"] == "scored"
    assert result["resolved"] is True
    assert (result["f2p_passed"], result["f2p_total"]) == (1, 1)
    assert (result["p2p_passed"], result["p2p_total"]) == (1, 1)


def test_a_regression_in_p2p_makes_the_run_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "probe_environment", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        evaluation,
        "run_pytest_files",
        lambda *a, **k: (
            {
                "tests/test_thing.py::test_new": "PASSED",
                "tests/test_thing.py::test_old": "FAILED",
            },
            None,
        ),
    )
    git = FakeGit(status=" M src/thing.py\n", diff="diff\n")

    result = evaluation.evaluate_run(row=ROW, workspace=tmp_path, run_git=git)

    assert result["resolved"] is False
    assert result["p2p_passed"] == 0


def test_node_ids_absent_from_pytest_output_are_reported_as_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "probe_environment", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        evaluation,
        "run_pytest_files",
        lambda *a, **k: ({"tests/test_thing.py::test_old": "PASSED"}, None),
    )
    git = FakeGit(status=" M src/thing.py\n", diff="diff\n")

    result = evaluation.evaluate_run(row=ROW, workspace=tmp_path, run_git=git)

    assert result["missing_node_ids"] == ["tests/test_thing.py::test_new"]
    assert result["resolved"] is False


def test_agent_authored_tests_are_never_applied_to_the_graded_checkout():
    # The agent's tests must not become the oracle it is judged by, and would
    # also collide with the gold test_patch. They stay in the stored patch as
    # evidence, but are filtered out of what gets applied for grading.
    patch = (
        "diff --git a/src/thing.py b/src/thing.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/tests/test_thing.py b/tests/test_thing.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
    )
    source_only = evaluation._source_only_patch(patch)

    assert "src/thing.py" in source_only
    assert "tests/test_thing.py" not in source_only


def test_grading_applies_only_the_source_half_of_the_stored_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "probe_environment", lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluation, "run_pytest_files", lambda *a, **k: ({}, None))
    git = FakeGit()
    patch = (
        "diff --git a/src/thing.py b/src/thing.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/tests/test_thing.py b/tests/test_thing.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
    )

    result = evaluation.evaluate_patch(
        row=ROW, workspace=tmp_path, run_git=git, patch=patch
    )

    applied = [c for c in git.calls if c[0] == "apply"]
    assert len(applied) == 2, "the agent's source patch, then the gold test patch"
    assert result["agent_test_files"] == ["tests/test_thing.py"]
    assert result["agent_source_files"] == ["src/thing.py"]


def test_workspace_is_always_reset_even_when_grading_raises(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(evaluation, "probe_environment", explode)
    git = FakeGit(status=" M src/thing.py\n", diff="diff\n")

    result = evaluation.evaluate_run(row=ROW, workspace=tmp_path, run_git=git)

    assert result["status"] == "error"
    assert "boom" in result["error"]
    assert ["checkout", "--", "."] in git.calls
    assert ["clean", "-fd"] in git.calls


def test_the_agent_patch_is_persisted_for_later_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "probe_environment", lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluation, "run_pytest_files", lambda *a, **k: ({}, None))
    git = FakeGit(status=" M src/thing.py\n", diff="THE PATCH\n")
    logs_root = tmp_path / "logs"

    result = evaluation.evaluate_run(
        row=ROW,
        workspace=tmp_path,
        run_git=git,
        logs_root=logs_root,
        run_id="run-1",
    )

    saved = Path(result["patch_path"])
    assert saved.read_text(encoding="utf-8") == "THE PATCH\n"
    assert result["patch_bytes"] == len("THE PATCH\n")
    assert result["empty_patch"] is False


def test_a_run_that_changed_nothing_is_marked_as_an_empty_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "probe_environment", lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluation, "run_pytest_files", lambda *a, **k: ({}, None))
    git = FakeGit(status="", diff="")

    result = evaluation.evaluate_run(row=ROW, workspace=tmp_path, run_git=git)

    assert result["empty_patch"] is True
    assert result["localization_hit"] is False


def test_capture_saves_the_patch_and_resets_the_workspace(tmp_path):
    # Capture is the only step that needs the live workspace; everything after
    # it works from the saved patch, which is what lets grading be redone.
    git = FakeGit(status=" M src/thing.py\n", diff="THE PATCH\n")
    logs_root = tmp_path / "logs"

    captured = evaluation.capture_agent_patch(
        workspace=tmp_path, run_git=git, logs_root=logs_root, run_id="run-1"
    )

    assert Path(captured["patch_path"]).read_text(encoding="utf-8") == "THE PATCH\n"
    assert captured["changed_paths"] == ["src/thing.py"]
    assert ["checkout", "--", "."] in git.calls
    assert ["clean", "-fd"] in git.calls


def test_a_stored_patch_can_be_graded_without_the_original_workspace(tmp_path, monkeypatch):
    # The whole point of the split: re-grading reads a patch off disk and needs
    # no access to the session that produced it.
    monkeypatch.setattr(evaluation, "probe_environment", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        evaluation,
        "run_pytest_files",
        lambda *a, **k: (
            {
                "tests/test_thing.py::test_new": "PASSED",
                "tests/test_thing.py::test_old": "PASSED",
            },
            None,
        ),
    )
    stored = "diff --git a/src/thing.py b/src/thing.py\n@@ -1 +1 @@\n-a\n+b\n"

    result = evaluation.evaluate_patch(
        row=ROW, workspace=tmp_path, run_git=FakeGit(), patch=stored
    )

    assert result["status"] == "scored"
    assert result["resolved"] is True
    assert result["agent_source_files"] == ["src/thing.py"]


def test_regrading_leaves_the_checkout_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "probe_environment", lambda *a, **k: (True, None))
    monkeypatch.setattr(evaluation, "run_pytest_files", lambda *a, **k: ({}, None))
    git = FakeGit()

    evaluation.evaluate_patch(
        row=ROW, workspace=tmp_path, run_git=git, patch="diff --git a/src/thing.py b/src/thing.py\n"
    )

    assert ["checkout", "--", "."] in git.calls
    assert ["clean", "-fd"] in git.calls
