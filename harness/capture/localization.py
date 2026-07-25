"""Compute file-level localization accuracy: how well the files the agent touched
match the gold files the fix should edit. Pure function, available even without running
the test suite (needs only the diff + gold_files).
"""
from __future__ import annotations

from harness.record.schema import Localization


def compute_localization(files_touched: list[str], gold_files: list[str]) -> Localization:
    gold = set(gold_files)
    touched = set(files_touched)
    if not gold:
        return Localization(gold_files=gold_files)

    inter = gold & touched
    union = gold | touched
    precision = len(inter) / len(touched) if touched else 0.0
    recall = len(inter) / len(gold) if gold else 0.0
    jaccard = len(inter) / len(union) if union else 0.0

    return Localization(
        gold_files=gold_files,
        hit_any=bool(inter),
        hit_all=gold.issubset(touched),
        precision=precision,
        recall=recall,
        jaccard=jaccard,
    )
