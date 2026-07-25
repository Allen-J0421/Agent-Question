"""On-disk layout + atomic persistence + resumability checks for run records.

Layout:
    runs/<instance_id>/<condition>/r<NN>/
        result.json        <- source of truth (this module owns it)
        transcript.jsonl    <- raw CLI stream-json (written by the runner)
        agent.patch         <- extracted diff
        stderr.log
        eval/report.json    <- raw swebench report for this run
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from harness.config import PathsConfig
from harness.constants import EVAL_EVALUATED, EVAL_ERROR
from harness.record.schema import RunRecord, make_run_id


class RunStore:
    def __init__(self, paths: PathsConfig | None = None):
        self.paths = (paths or PathsConfig()).ensure()

    # ---- path helpers ----
    def run_dir(self, instance_id: str, condition: str, repeat_index: int) -> Path:
        return self.paths.runs_dir / instance_id / condition / f"r{repeat_index:02d}"

    def result_path(self, instance_id: str, condition: str, repeat_index: int) -> Path:
        return self.run_dir(instance_id, condition, repeat_index) / "result.json"

    def transcript_path(self, instance_id: str, condition: str, repeat_index: int) -> Path:
        return self.run_dir(instance_id, condition, repeat_index) / "transcript.jsonl"

    def diff_path(self, instance_id: str, condition: str, repeat_index: int) -> Path:
        return self.run_dir(instance_id, condition, repeat_index) / "agent.patch"

    def stderr_path(self, instance_id: str, condition: str, repeat_index: int) -> Path:
        return self.run_dir(instance_id, condition, repeat_index) / "stderr.log"

    def eval_report_path(self, instance_id: str, condition: str, repeat_index: int) -> Path:
        return self.run_dir(instance_id, condition, repeat_index) / "eval" / "report.json"

    # ---- write ----
    def write_atomic(self, record: RunRecord) -> Path:
        """Write result.json via temp-file + os.replace so a crash never leaves a
        half-written file that would falsely read as complete."""
        rd = self.run_dir(record.instance_id, record.condition, record.repeat_index)
        rd.mkdir(parents=True, exist_ok=True)
        target = rd / "result.json"
        fd, tmp = tempfile.mkstemp(dir=str(rd), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return target

    # ---- read ----
    def read(self, instance_id: str, condition: str, repeat_index: int) -> RunRecord | None:
        p = self.result_path(instance_id, condition, repeat_index)
        if not p.exists():
            return None
        try:
            return RunRecord.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, KeyError):
            return None  # malformed -> treat as absent (will be re-run)

    # ---- resumability ----
    def is_complete(self, instance_id: str, condition: str, repeat_index: int) -> bool:
        """A run is complete iff result.json exists, parses, and has an exit reason."""
        rec = self.read(instance_id, condition, repeat_index)
        return bool(rec and rec.exit and rec.exit.reason)

    def needs_eval(self, record: RunRecord) -> bool:
        """True iff this run produced a patch but hasn't been (successfully) evaluated."""
        return (
            record.patch.produced_patch
            and record.evaluation.eval_status not in (EVAL_EVALUATED, EVAL_ERROR)
        )

    def iter_records(self):
        """Yield every persisted RunRecord (for aggregation / eval passes)."""
        for result_file in self.paths.runs_dir.rglob("result.json"):
            try:
                yield RunRecord.from_dict(json.loads(result_file.read_text()))
            except (json.JSONDecodeError, KeyError):
                continue


__all__ = ["RunStore", "make_run_id"]
