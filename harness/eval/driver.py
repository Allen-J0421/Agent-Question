"""The decoupled evaluation pass: find produced-patch records that need eval, run the
swebench harness once over them, then merge each per-instance report back into its
result.json. Resumable and idempotent (skips already-evaluated runs).

Note: multiple runs can share one instance_id (ambiguous r0/r1/r2, full r0). Each run is
a distinct swebench "model" (its run_id), so we run a separate eval invocation per run
to keep report paths 1:1. Runs are grouped so identical instance images are reused.
"""
from __future__ import annotations

from pathlib import Path

from harness.config import HarnessConfig
from harness.data.loader import instances_by_id
from harness.eval.predictions import write_predictions
from harness.eval.result_reader import merge_report
from harness.eval.swebench_adapter import per_instance_report_path, run_swebench_eval
from harness.record.store import RunStore


def _eval_run_id(record) -> str:
    # unique per run; swebench uses it as the log subdir
    return record.run_id.replace("/", "_")


def run_eval_pass(cfg: HarnessConfig, store: RunStore | None = None,
                  only_instance_ids: list[str] | None = None) -> dict:
    store = store or RunStore(cfg.paths)
    by_id = instances_by_id()
    to_eval = [
        r for r in store.iter_records()
        if store.needs_eval(r)
        and (only_instance_ids is None or r.instance_id in only_instance_ids)
    ]
    stats = {"evaluated": 0, "errors": 0, "resolved": 0, "skipped": 0}

    for rec in to_eval:
        eval_run_id = _eval_run_id(rec)
        preds_path = cfg.paths.eval_log_dir / f"{eval_run_id}__predictions.jsonl"
        write_predictions([rec], preds_path)

        try:
            run_swebench_eval(
                predictions_path=preds_path,
                instance_ids=[rec.instance_id],
                eval_run_id=eval_run_id,
                paths=cfg.paths,
                cfg=cfg.eval,
            )
        except Exception as e:  # image build / harness failure -> eval_error record
            from harness.constants import EVAL_ERROR
            rec.evaluation.eval_status = EVAL_ERROR
            rec.evaluation.eval_error_text = str(e)[:500]
            store.write_atomic(rec)
            stats["errors"] += 1
            continue

        report_path = per_instance_report_path(
            eval_run_id, rec.run_id, rec.instance_id, cwd=Path.cwd())
        gold_files = by_id[rec.instance_id].gold_files
        rec.evaluation = merge_report(rec, report_path, gold_files)
        store.write_atomic(rec)

        stats["evaluated"] += 1
        if rec.evaluation.eval_status == "eval_error":
            stats["errors"] += 1
        if rec.evaluation.resolved:
            stats["resolved"] += 1

    return stats
