"""Deterministic stratified sampler for the M1 pilot.

Goals (from the plan): de-bias Django's ~46% dominance and the ≤1hr difficulty skew.
We stratify on repo × difficulty, cap any single repo's share, and deliberately
over-sample the rare 1-4hr / >4hr tail so hard-case behavior is observable. The exact
instance_ids are frozen to a manifest for reproducibility.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from harness.config import PathsConfig
from harness.data.loader import Instance, load_instances


def stratified_pilot(instances: list[Instance] | None = None, n: int = 60,
                     max_repo_frac: float = 0.30, seed: int = 42,
                     oversample_hard: bool = True) -> list[str]:
    """Return a frozen list of instance_ids for the pilot.

    Strategy: allocate a per-repo cap = ceil(n * max_repo_frac). Within each repo, draw
    proportionally across difficulty buckets but guarantee at least one from each rare
    (1-4hr, >4hr) bucket present. Fill remaining slots round-robin across under-filled
    repos. Deterministic given seed.
    """
    insts = list(instances if instances is not None else load_instances())
    rng = random.Random(seed)

    by_repo_diff: dict[tuple[str, str], list[Instance]] = defaultdict(list)
    for inst in insts:
        by_repo_diff[(inst.repo, inst.difficulty)].append(inst)
    for v in by_repo_diff.values():
        rng.shuffle(v)

    repos = sorted({i.repo for i in insts})
    cap = max(1, int(n * max_repo_frac + 0.999))

    selected: list[str] = []
    used: set[str] = set()
    per_repo_count: dict[str, int] = defaultdict(int)

    hard_buckets = ("1-4 hours", ">4 hours")

    def take(repo: str, difficulty: str) -> bool:
        pool = by_repo_diff.get((repo, difficulty), [])
        for inst in pool:
            if inst.instance_id in used:
                continue
            if per_repo_count[repo] >= cap:
                return False
            used.add(inst.instance_id)
            selected.append(inst.instance_id)
            per_repo_count[repo] += 1
            return True
        return False

    # Pass 1: seed each repo with its hard-tail instances (observability of hard cases).
    if oversample_hard:
        for repo in repos:
            for d in hard_buckets:
                if len(selected) >= n:
                    break
                take(repo, d)

    # Pass 2: round-robin across repo × difficulty until we hit n or exhaust the pool.
    all_diffs = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
    progressed = True
    while len(selected) < n and progressed:
        progressed = False
        for repo in repos:
            if len(selected) >= n:
                break
            for d in all_diffs:
                if len(selected) >= n:
                    break
                if take(repo, d):
                    progressed = True
                    break  # one draw per repo per round -> spreads across repos

    return selected


def write_manifest(instance_ids: list[str], name: str = "pilot_60",
                   paths: PathsConfig | None = None, meta: dict | None = None) -> Path:
    paths = (paths or PathsConfig()).ensure()
    path = paths.manifest_dir / f"{name}.json"
    payload = {"name": name, "n": len(instance_ids),
               "instance_ids": instance_ids, "meta": meta or {}}
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_manifest(name: str, paths: PathsConfig | None = None) -> list[str]:
    paths = paths or PathsConfig()
    path = paths.manifest_dir / f"{name}.json"
    return json.loads(path.read_text())["instance_ids"]
