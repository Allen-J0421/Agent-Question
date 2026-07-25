"""Read a swebench per-instance report.json and fold it into a run record's Evaluation.

Report shape (from grading.get_eval_report):
  { "<instance_id>": {
       "patch_is_None", "patch_exists", "patch_successfully_applied", "resolved",
       "tests_status": {  # present when include_tests_status
          "FAIL_TO_PASS": {"success": [...], "failure": [...]},
          "PASS_TO_PASS": {"success": [...], "failure": [...]}, ... } } }
"""
from __future__ import annotations

import json
from pathlib import Path

from harness.capture.localization import compute_localization
from harness.constants import EVAL_ERROR, EVAL_EVALUATED
from harness.record.schema import Evaluation, RunRecord, TestBreakdown


def _breakdown(status: dict | None) -> TestBreakdown:
    if not status:
        return TestBreakdown()
    succ = status.get("success", []) or []
    fail = status.get("failure", []) or []
    return TestBreakdown(
        total=len(succ) + len(fail),
        passed=len(succ),
        failed=len(fail),
        unresolved=len(fail),
    )


def merge_report(record: RunRecord, report_path: Path, gold_files: list[str]) -> Evaluation:
    ev = record.evaluation
    ev.localization = compute_localization(record.patch.files_touched, gold_files)
    ev.swebench_report_path = str(report_path)

    if not report_path.exists():
        ev.eval_status = EVAL_ERROR
        ev.eval_error_text = "report.json not found (image build / apply failure?)"
        return ev

    try:
        data = json.loads(report_path.read_text())
    except json.JSONDecodeError as e:
        ev.eval_status = EVAL_ERROR
        ev.eval_error_text = f"unparseable report: {e}"
        return ev

    inst = data.get(record.instance_id, {})
    ev.resolved = bool(inst.get("resolved", False))

    tests = inst.get("tests_status", {})
    ev.fail_to_pass = _breakdown(tests.get("FAIL_TO_PASS"))
    ev.pass_to_pass = _breakdown(tests.get("PASS_TO_PASS"))
    ev.regression = (ev.pass_to_pass.failed or 0) > 0

    # If the patch didn't apply, resolved is False and tests_status may be empty.
    if not inst.get("patch_successfully_applied", False):
        ev.eval_error_text = "patch did not apply cleanly"
    ev.eval_status = EVAL_EVALUATED
    return ev
