"""Write a SWE-bench predictions JSONL from produced-patch run records.

Each prediction uses the run_id as `model_name_or_path` so that ambiguous/full/repeat
predictions for the same instance stay distinct within one eval pass and map back 1:1.
"""
from __future__ import annotations

import json
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION

from harness.record.schema import RunRecord


def prediction_for(record: RunRecord) -> dict:
    return {
        KEY_INSTANCE_ID: record.instance_id,
        KEY_MODEL: record.run_id,           # unique per run
        KEY_PREDICTION: record.patch.diff or "",
    }


def write_predictions(records: list[RunRecord], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            if rec.patch.produced_patch and rec.patch.diff:
                f.write(json.dumps(prediction_for(rec)) + "\n")
    return out_path
