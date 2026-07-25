"""Unit tests for the data loader (field parsing, None-files fallback) and the record
schema round-trip.
"""
import tempfile
from pathlib import Path

from harness.config import PathsConfig
from harness.data.loader import instances_by_id, load_instances
from harness.record.schema import (
    AskInfo,
    ExitInfo,
    PromptInfo,
    Question,
    RunMeta,
    RunRecord,
    Tokens,
    Trajectory,
    make_run_id,
)
from harness.record.store import RunStore


def test_all_500_load():
    insts = load_instances()
    assert len(insts) == 500


def test_json_test_lists_parsed():
    by_id = instances_by_id()
    a = by_id["astropy__astropy-12907"]
    assert isinstance(a.fail_to_pass, list) and len(a.fail_to_pass) == 2
    assert isinstance(a.pass_to_pass, list) and len(a.pass_to_pass) > 0


def test_none_files_fallback_from_patch():
    by_id = instances_by_id()
    a = by_id["astropy__astropy-13236"]  # files field is None in the dataset
    assert a.gold_files == ["astropy/table/table.py"]


def test_every_instance_has_gold_files():
    insts = load_instances()
    assert all(i.gold_files for i in insts)


def test_record_round_trip():
    rec = RunRecord(
        run_id=make_run_id("django__django-11477", "ambiguous", 2),
        instance_id="django__django-11477", repo="django/django",
        difficulty="15 min - 1 hour", condition="ambiguous", repeat_index=2,
        prompt=PromptInfo("problem_statement", 254, "abc", False),
        run_meta=RunMeta("opus", "acceptEdits", 40, "2.1.179", "t0", "t1", 1.0),
        exit=ExitInfo("asked", 0, "success", None),
        ask=AskInfo(asked=True, n_questions=1, first_ask_turn=6,
                    questions=[Question(6, "Scope", "q?", ["Yes", "No"], False)]),
        trajectory=Trajectory(n_turns=6, tools_used={"Read": 7}, tokens=Tokens(1, 2, 3, 4, 10)),
    )
    d = rec.to_dict()
    assert RunRecord.from_dict(d).to_dict() == d


def test_store_write_read_resumability():
    with tempfile.TemporaryDirectory() as tmp:
        paths = PathsConfig(runs_dir=Path(tmp) / "runs")
        store = RunStore(paths)
        rec = RunRecord(
            run_id=make_run_id("x__y-1", "full", 0), instance_id="x__y-1",
            repo="x/y", difficulty="<15 min fix", condition="full", repeat_index=0,
            prompt=PromptInfo("original_issue", 10, "h", False),
            run_meta=RunMeta("opus", "acceptEdits", 40, "2.1.179", "t0", "t1", 1.0),
            exit=ExitInfo("produced_patch", 0, "success", None),
        )
        assert not store.is_complete("x__y-1", "full", 0)
        store.write_atomic(rec)
        assert store.is_complete("x__y-1", "full", 0)
        assert store.read("x__y-1", "full", 0).exit.reason == "produced_patch"
