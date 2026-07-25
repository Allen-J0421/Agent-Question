"""Bridge to the official swebench evaluation harness.

Our local dataset loads directly via swebench's own loader (verified:
`load_swebench_dataset("data/interactive-swe", "test")`), so NO dataset conversion is
needed. This module writes a predictions file, invokes `run_evaluation.main`, and
locates the per-instance report each run produces.

Docker daemon must be running before calling run_swebench_eval.
"""
from __future__ import annotations

from pathlib import Path

from harness.config import EvalConfig, PathsConfig


def per_instance_report_path(eval_run_id: str, model_name: str, instance_id: str,
                             cwd: Path | None = None) -> Path:
    """logs/run_evaluation/<eval_run_id>/<model_name>/<instance_id>/report.json"""
    base = (cwd or Path.cwd()) / "logs" / "run_evaluation" / eval_run_id / model_name / instance_id
    return base / "report.json"


def run_swebench_eval(predictions_path: Path, instance_ids: list[str], eval_run_id: str,
                      paths: PathsConfig | None = None,
                      cfg: EvalConfig | None = None) -> None:
    """Run the swebench evaluation for the given predictions. Blocks until done.
    Reports are written under ./logs/run_evaluation/<eval_run_id>/... (swebench's
    fixed layout, relative to cwd)."""
    from swebench.harness.run_evaluation import main as run_evaluation_main

    paths = paths or PathsConfig()
    cfg = cfg or EvalConfig()

    run_evaluation_main(
        dataset_name=str(paths.dataset_dir),
        split=paths.dataset_split,
        instance_ids=instance_ids,
        predictions_path=str(predictions_path),
        max_workers=cfg.max_workers,
        force_rebuild=cfg.force_rebuild,
        cache_level=cfg.cache_level,
        clean=False,
        open_file_limit=4096,
        run_id=eval_run_id,
        timeout=cfg.timeout_s,
        namespace=cfg.namespace,
        rewrite_reports=False,
        modal=False,
        instance_image_tag="latest",
        env_image_tag="latest",
        report_dir=".",
    )
