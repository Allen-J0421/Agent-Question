"""Execute a single run unit end-to-end and persist its record.

Control flow (record & stop):
  build_prompt -> create workspace @ base_commit -> run CLI -> parse transcript
    -> if the agent ASKED: exit=asked, no diff, no eval  (record & stop is OUR policy)
    -> else: extract diff; exit = produced_patch | no_patch (| max_turns | timeout)
  -> write result.json atomically ; always tear down the workspace

Any exception becomes an `error` record, not a crash.
"""
from __future__ import annotations

import datetime as _dt
import traceback

from harness.agent.runner import RunOutcome, run_agent
from harness.agent.stream_parser import parse_file
from harness.agent.workspace import create_workspace, teardown_workspace
from harness.capture.ask_detector import detect_ask
from harness.capture.annotations import empty_annotations
from harness.capture.diff_extractor import extract_diff
from harness.capture.localization import compute_localization
from harness.capture.trajectory import fold_trajectory
from harness.config import HarnessConfig
from harness.constants import (
    EVAL_SKIPPED_NO_PATCH,
    EXIT_ASKED,
    EXIT_ERROR,
    EXIT_MAX_TURNS,
    EXIT_NO_PATCH,
    EXIT_PRODUCED_PATCH,
    EXIT_TIMEOUT,
)
from harness.data.loader import Instance
from harness.prompt.builder import build_prompt
from harness.record.schema import (
    Evaluation,
    ExitInfo,
    RunMeta,
    RunRecord,
    make_run_id,
)
from harness.record.store import RunStore
from harness.orchestrator.plan import RunUnit


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _classify_no_ask_exit(outcome: RunOutcome, parsed, produced: bool) -> str:
    if produced:
        return EXIT_PRODUCED_PATCH
    if outcome.timed_out:
        return EXIT_TIMEOUT
    # CLI reported it stopped on max turns?
    if parsed.result and (parsed.result.subtype or "").startswith("error_max_turns"):
        return EXIT_MAX_TURNS
    if outcome.exit_code not in (0, None):
        return EXIT_ERROR
    return EXIT_NO_PATCH


def execute_unit(unit: RunUnit, instance: Instance, cfg: HarnessConfig,
                 store: RunStore) -> RunRecord:
    run_id = make_run_id(unit.instance_id, unit.condition, unit.repeat_index)
    prompt, prompt_info = build_prompt(instance, unit.condition)
    transcript_path = store.transcript_path(unit.instance_id, unit.condition, unit.repeat_index)
    stderr_path = store.stderr_path(unit.instance_id, unit.condition, unit.repeat_index)
    diff_path = store.diff_path(unit.instance_id, unit.condition, unit.repeat_index)

    started = _utcnow()
    t0 = _dt.datetime.now()
    ws = None
    try:
        ws = create_workspace(unit.instance_id, instance.repo, instance.base_commit,
                              cfg.paths)
        outcome = run_agent(prompt, ws.path, transcript_path, stderr_path, cfg.run)
        parsed = parse_file(transcript_path)
        ask = detect_ask(parsed)
        traj = fold_trajectory(parsed)

        if ask.asked:
            exit_reason = EXIT_ASKED
            patch = _empty_patch()
        else:
            patch = extract_diff(ws.path, instance.base_commit)
            if patch.produced_patch and patch.diff:
                diff_path.write_text(patch.diff)
            exit_reason = _classify_no_ask_exit(outcome, parsed,
                                                patch.produced_patch)

        cli_subtype = parsed.result.subtype if parsed.result else None
        error_text = None
        if exit_reason == EXIT_ERROR:
            error_text = _tail(stderr_path)

        record = RunRecord(
            run_id=run_id,
            instance_id=unit.instance_id,
            repo=instance.repo,
            difficulty=instance.difficulty,
            condition=unit.condition,
            repeat_index=unit.repeat_index,
            prompt=prompt_info,
            run_meta=RunMeta(
                model=cfg.run.model,
                permission_mode=cfg.run.permission_mode,
                max_turns=cfg.run.max_turns,
                cli_version=outcome.cli_version,
                started_at=started,
                ended_at=_utcnow(),
                wall_time_s=(_dt.datetime.now() - t0).total_seconds(),
            ),
            exit=ExitInfo(reason=exit_reason, cli_exit_code=outcome.exit_code,
                          cli_subtype=cli_subtype, error_text=error_text),
            ask=ask,
            patch=patch,
            trajectory=traj,
            evaluation=_eval_scaffold(patch, instance.gold_files),
            annotations=empty_annotations(),
        )
        _set_artifacts(record, store, unit)
        store.write_atomic(record)
        return record

    except Exception as e:  # failure = a record, not a crash
        record = _error_record(run_id, unit, instance, prompt_info, started, t0,
                               f"{e}\n{traceback.format_exc()[:2000]}")
        _set_artifacts(record, store, unit)
        store.write_atomic(record)
        return record
    finally:
        if ws is not None:
            teardown_workspace(ws, cfg.paths)


# ---- helpers ----
def _empty_patch():
    from harness.record.schema import PatchInfo
    return PatchInfo(produced_patch=False)


def _eval_scaffold(patch, gold_files: list[str]) -> Evaluation:
    ev = Evaluation()
    ev.localization = compute_localization(patch.files_touched, gold_files)
    if not patch.produced_patch:
        ev.eval_status = EVAL_SKIPPED_NO_PATCH
    return ev


def _set_artifacts(record: RunRecord, store: RunStore, unit: RunUnit) -> None:
    record.artifacts.transcript_path = str(
        store.transcript_path(unit.instance_id, unit.condition, unit.repeat_index))
    record.artifacts.workspace_diff_path = str(
        store.diff_path(unit.instance_id, unit.condition, unit.repeat_index))
    record.artifacts.stderr_path = str(
        store.stderr_path(unit.instance_id, unit.condition, unit.repeat_index))


def _tail(path, n: int = 1500) -> str:
    try:
        return path.read_text()[-n:]
    except Exception:
        return ""


def _error_record(run_id, unit, instance, prompt_info, started, t0, err) -> RunRecord:
    return RunRecord(
        run_id=run_id, instance_id=unit.instance_id, repo=instance.repo,
        difficulty=instance.difficulty, condition=unit.condition,
        repeat_index=unit.repeat_index, prompt=prompt_info,
        run_meta=RunMeta(model="", permission_mode="", max_turns=0, cli_version="",
                         started_at=started, ended_at=_utcnow(),
                         wall_time_s=(_dt.datetime.now() - t0).total_seconds()),
        exit=ExitInfo(reason=EXIT_ERROR, error_text=err),
        evaluation=Evaluation(eval_status=EVAL_SKIPPED_NO_PATCH),
        annotations=empty_annotations(),
    )
