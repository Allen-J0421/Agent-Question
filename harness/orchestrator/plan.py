"""Enumerate the set of run units for a sweep and diff against what's already complete.

A run unit = (instance_id, condition, repeat_index). Ambiguous gets N repeats; full
gets 1. The orchestrator skips units whose result.json is already complete (resumability).
"""
from __future__ import annotations

from dataclasses import dataclass

from harness.config import RunConfig
from harness.constants import CONDITION_AMBIGUOUS, CONDITION_FULL
from harness.data.loader import Instance
from harness.record.store import RunStore


@dataclass(frozen=True)
class RunUnit:
    instance_id: str
    condition: str
    repeat_index: int


def enumerate_units(instances: list[Instance], cfg: RunConfig) -> list[RunUnit]:
    units: list[RunUnit] = []
    for inst in instances:
        for r in range(cfg.n_repeats_ambiguous):
            units.append(RunUnit(inst.instance_id, CONDITION_AMBIGUOUS, r))
        for r in range(cfg.n_repeats_full):
            units.append(RunUnit(inst.instance_id, CONDITION_FULL, r))
    return units


def pending_units(units: list[RunUnit], store: RunStore) -> list[RunUnit]:
    return [
        u for u in units
        if not store.is_complete(u.instance_id, u.condition, u.repeat_index)
    ]
