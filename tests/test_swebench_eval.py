import json
from pathlib import Path

import swebench_eval
from swebench_eval import (
    MODEL_NAME,
    bucket_by_instance,
    capture_agent_patch,
    empty_patch_evaluation,
    evaluate_with_harness,
    evaluation_from_report,
    gold_files,
    is_test_path,
    localization_fields,
    predictions_for,
    source_only_patch,
)


ROW = {
    "instance_id": "astropy__astropy-1",
    "FAIL_TO_PASS": '["astropy/x/tests/test_a.py::test_b", "astropy/x/tests/test_a.py::test_c"]',
    "PASS_TO_PASS": '["astropy/x/tests/test_a.py::test_d"]',
    "files": "astropy/x/core.py",
    "patch": "",
}

SOURCE_PATCH = (
    "diff --git a/astropy/x/core.py b/astropy/x/core.py\n"
    "--- a/astropy/x/core.py\n"
    "+++ b/astropy/x/core.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)

TEST_PATCH = (
    "diff --git a/astropy/x/tests/test_a.py b/astropy/x/tests/test_a.py\n"
    "--- a/astropy/x/tests/test_a.py\n"
    "+++ b/astropy/x/tests/test_a.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)


def _summary(run_id: str, instance_id: str) -> dict:
    return {"run_id": run_id, "task": {"instance_id": instance_id}}


class FakeGit:
    """Record git invocations and replay canned results.

    No test in this repository shells out to real git; this is the seam the
    suite uses for subprocess-shaped behavior.
    """

    def __init__(self, status="", diff=""):
        self.calls: list[list[str]] = []
        self.status = status
        self.diff = diff

    def __call__(self, args, cwd=None, check=True, input=None):
        self.calls.append(list(args))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        result = Result()
        if args[0] == "status":
            result.stdout = self.status
        elif args[0] == "diff":
            result.stdout = self.diff
        return result


def test_is_test_path_matches_every_repository_test_convention():
    assert is_test_path("tests/test_thing.py")
    assert is_test_path("astropy/modeling/tests/test_separable.py")
    assert is_test_path("testing/test_capture.py")
    assert is_test_path("pkg/foo_test.py")
    assert not is_test_path("astropy/modeling/separable.py")
    assert not is_test_path("src/thing.py")


def test_gold_files_falls_back_to_patch_headers_when_files_is_empty():
    # 26 of the 500 instances have an empty `files` column.
    row = {
        "files": "",
        "patch": "diff --git a/src/thing.py b/src/thing.py\n@@ -1 +1 @@\n-a\n+b\n",
    }
    assert gold_files(row) == ["src/thing.py"]
    assert gold_files({**row, "files": "src/thing.py"}) == ["src/thing.py"]


def test_agent_authored_tests_are_stripped_from_the_graded_patch():
    # The agent's tests must not become the oracle it is judged by, and would
    # also collide with the gold test_patch. They stay in the stored patch as
    # evidence, but are filtered out of what gets graded.
    stripped = source_only_patch(SOURCE_PATCH + TEST_PATCH)
    assert "core.py" in stripped
    assert "test_a.py" not in stripped


def test_capture_saves_the_patch_and_resets_the_workspace(tmp_path):
    # Capture is the only step that needs the live workspace; everything after
    # it works from the saved patch, which is what lets grading be redone.
    git = FakeGit(status=" M src/thing.py\n", diff="THE PATCH\n")
    logs_root = tmp_path / "logs"

    captured = capture_agent_patch(
        workspace=tmp_path, run_git=git, logs_root=logs_root, run_id="run-1"
    )

    assert Path(captured["patch_path"]).read_text(encoding="utf-8") == "THE PATCH\n"
    assert captured["changed_paths"] == ["src/thing.py"]
    assert captured["patch_bytes"] == len("THE PATCH\n")
    assert ["checkout", "--", "."] in git.calls
    assert ["clean", "-fd"] in git.calls


def test_duplicate_instances_are_split_across_buckets():
    # The harness keys predictions by instance_id, so the same instance run
    # under both conditions must land in different harness invocations.
    summaries = [
        _summary("r1", "a"),
        _summary("r2", "a"),
        _summary("r3", "b"),
    ]
    buckets = bucket_by_instance(summaries)
    assert [[s["run_id"] for s in bucket] for bucket in buckets] == [
        ["r1", "r3"],
        ["r2"],
    ]


def test_predictions_grade_source_edits_only():
    bucket = [_summary("r1", ROW["instance_id"])]
    patches = {"r1": SOURCE_PATCH + TEST_PATCH}
    (prediction,) = predictions_for(bucket, patches)
    assert prediction["instance_id"] == ROW["instance_id"]
    assert prediction["model_name_or_path"] == MODEL_NAME
    assert "core.py" in prediction["model_patch"]
    assert "test_a.py" not in prediction["model_patch"]


def test_localization_uses_source_files_against_gold():
    fields = localization_fields(ROW, SOURCE_PATCH + TEST_PATCH, "/p/r1.patch")
    assert fields["localization_hit"] is True
    assert fields["agent_source_files"] == ["astropy/x/core.py"]
    assert fields["agent_test_files"] == ["astropy/x/tests/test_a.py"]
    assert fields["empty_patch"] is False


def test_empty_source_patch_is_unresolved_by_definition():
    # A test-only patch counts as empty for grading purposes.
    evaluation = empty_patch_evaluation(ROW, TEST_PATCH, None)
    assert evaluation["status"] == "scored"
    assert evaluation["resolved"] is False
    assert evaluation["empty_patch"] is True
    assert evaluation["f2p_total"] == 2
    assert evaluation["f2p_passed"] == 0
    assert "unresolved by definition" in evaluation["error"]


def test_report_maps_onto_the_study_schema():
    report = {
        "resolved": True,
        "patch_successfully_applied": True,
        "tests_status": {
            "FAIL_TO_PASS": {
                "success": [
                    "astropy/x/tests/test_a.py::test_b",
                    "astropy/x/tests/test_a.py::test_c",
                ],
                "failure": [],
            },
            "PASS_TO_PASS": {
                "success": ["astropy/x/tests/test_a.py::test_d"],
                "failure": [],
            },
        },
    }
    evaluation = evaluation_from_report(ROW, report, SOURCE_PATCH, "/p/r1.patch", "run-0")
    assert evaluation["status"] == "scored"
    assert evaluation["resolved"] is True
    assert evaluation["f2p_passed"] == 2
    assert evaluation["p2p_passed"] == 1
    assert evaluation["missing_node_ids"] == []
    assert evaluation["harness"] == "swebench"
    assert evaluation["swebench_run_id"] == "run-0"


def test_node_ids_absent_from_the_report_are_recorded_as_missing():
    report = {
        "resolved": False,
        "tests_status": {
            "FAIL_TO_PASS": {
                "success": ["astropy/x/tests/test_a.py::test_b"],
                "failure": [],
            },
            "PASS_TO_PASS": {"success": [], "failure": []},
        },
    }
    evaluation = evaluation_from_report(ROW, report, SOURCE_PATCH, None, "run-0")
    assert evaluation["resolved"] is False
    assert set(evaluation["missing_node_ids"]) == {
        "astropy/x/tests/test_a.py::test_c",
        "astropy/x/tests/test_a.py::test_d",
    }


def test_a_missing_report_is_an_error_never_a_failed_patch():
    evaluation = evaluation_from_report(ROW, None, SOURCE_PATCH, None, "run-0")
    assert evaluation["status"] == "error"
    assert evaluation["resolved"] is None
    assert "no report.json" in evaluation["error"]


def test_evaluate_with_harness_short_circuits_empty_patches(tmp_path):
    # An empty patch must be graded without any harness invocation, and a
    # non-empty one must round-trip through the (injected) harness report.
    logs_root = tmp_path
    (logs_root / "patches").mkdir()
    (logs_root / "patches" / "full.patch").write_text(SOURCE_PATCH, encoding="utf-8")

    rows = {ROW["instance_id"]: ROW}
    summaries = [
        _summary("full", ROW["instance_id"]),
        _summary("empty", ROW["instance_id"]),
    ]

    invocations = []

    def fake_harness(*, root, predictions_path, swebench_run_id, **_):
        invocations.append(swebench_run_id)
        (prediction,) = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
        ]
        report_dir = (
            root
            / "logs"
            / "run_evaluation"
            / swebench_run_id
            / MODEL_NAME
            / prediction["instance_id"]
        )
        report_dir.mkdir(parents=True)
        (report_dir / "report.json").write_text(
            json.dumps(
                {
                    prediction["instance_id"]: {
                        "resolved": True,
                        "patch_successfully_applied": True,
                        "tests_status": {
                            "FAIL_TO_PASS": {
                                "success": [
                                    "astropy/x/tests/test_a.py::test_b",
                                    "astropy/x/tests/test_a.py::test_c",
                                ],
                                "failure": [],
                            },
                            "PASS_TO_PASS": {
                                "success": ["astropy/x/tests/test_a.py::test_d"],
                                "failure": [],
                            },
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return 0

    evaluations = evaluate_with_harness(
        logs_root=logs_root,
        rows=rows,
        summaries=summaries,
        harness=fake_harness,
    )

    assert len(invocations) == 1  # only the non-empty run reached the harness
    assert evaluations["full"]["resolved"] is True
    assert evaluations["full"]["status"] == "scored"
    assert evaluations["empty"]["resolved"] is False
    assert evaluations["empty"]["empty_patch"] is True


def test_harness_root_stays_inside_the_logs_directory(tmp_path):
    assert swebench_eval.harness_root(tmp_path) == tmp_path / "swebench"
