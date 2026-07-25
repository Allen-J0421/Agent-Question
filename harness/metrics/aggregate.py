"""Flatten all persisted RunRecords into one long-format table (one row per run),
written as both parquet and csv. Every downstream metric is a groupby over this table.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from harness.config import PathsConfig
from harness.record.store import RunStore


def _row(rec) -> dict:
    ev = rec.evaluation
    return {
        "run_id": rec.run_id,
        "instance_id": rec.instance_id,
        "repo": rec.repo,
        "difficulty": rec.difficulty,
        "condition": rec.condition,
        "repeat_index": rec.repeat_index,
        "exit_reason": rec.exit.reason,
        "asked": rec.ask.asked,
        "n_questions": rec.ask.n_questions,
        "first_ask_turn": rec.ask.first_ask_turn,
        "produced_patch": rec.patch.produced_patch,
        "n_files_touched": rec.patch.n_files_touched,
        "loc_added": rec.patch.loc_added,
        "loc_removed": rec.patch.loc_removed,
        "n_turns": rec.trajectory.n_turns,
        "n_tool_calls": rec.trajectory.n_tool_calls,
        "tokens_total": rec.trajectory.tokens.total,
        "cost_usd": rec.trajectory.cost_usd,
        "wall_time_s": rec.run_meta.wall_time_s,
        "eval_status": ev.eval_status,
        "resolved": ev.resolved,
        "regression": ev.regression,
        "loc_hit_any": ev.localization.hit_any,
        "loc_hit_all": ev.localization.hit_all,
        "loc_recall": ev.localization.recall,
        "loc_precision": ev.localization.precision,
        "loc_jaccard": ev.localization.jaccard,
        "f2p_total": ev.fail_to_pass.total,
        "f2p_passed": ev.fail_to_pass.passed,
        "p2p_total": ev.pass_to_pass.total,
        "p2p_failed": ev.pass_to_pass.failed,
    }


def build_table(paths: PathsConfig | None = None) -> pd.DataFrame:
    store = RunStore(paths or PathsConfig())
    rows = [_row(r) for r in store.iter_records()]
    return pd.DataFrame(rows)


def aggregate_to_table(paths: PathsConfig | None = None) -> Path:
    paths = (paths or PathsConfig()).ensure()
    df = build_table(paths)
    out_csv = paths.runs_dir.parent / "runs_table.csv"
    out_parquet = paths.runs_dir.parent / "runs_table.parquet"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception:
        pass  # parquet engine optional
    return out_csv
