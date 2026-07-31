#!/usr/bin/env python3
"""Locate every stored artifact for a dataset instance.

Experiment artifacts are keyed by ``run_id``, but analysis starts from a
dataset entry: "show me everything for astropy__astropy-13398, ambiguous vs
full". This maps instance -> runs -> artifact paths, so no one has to grep
run summaries by hand to find the patch, grade, raw session, or transcript
belonging to a data point.

Stdlib only, read-only.

Usage:
    python3 locate_logs.py                            # index: every instance that has runs
    python3 locate_logs.py 13398                      # substring of an instance_id is fine
    python3 locate_logs.py astropy__astropy-13398 --condition ambiguous
    python3 locate_logs.py 13398 --json               # machine-readable, same information
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_runs(logs: Path) -> list[dict]:
    summaries = []
    for path in sorted((logs / "runs").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"WARNING: unreadable run summary: {path}", file=sys.stderr)
            continue
        if isinstance(payload, dict):
            summaries.append(payload)
    summaries.sort(key=lambda s: s.get("started_at") or "")
    return summaries


def artifact_map(logs: Path, run_id: str) -> dict[str, dict]:
    """Every artifact slot for one run: its path, whether it exists, size."""
    slots = {
        "run_summary": logs / "runs" / f"{run_id}.json",
        "manifest": logs / "manifests" / f"{run_id}.json",
        "patch": logs / "patches" / f"{run_id}.patch",
        "evaluation": logs / "evaluations" / f"{run_id}.json",
        "sessions": logs / "sessions" / run_id,
        "transcripts": logs / "transcripts" / run_id,
    }
    out = {}
    for name, path in slots.items():
        if path.is_dir():
            files = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
            out[name] = {"path": str(path), "exists": bool(files), "files": files}
        else:
            out[name] = {"path": str(path), "exists": path.is_file()}
            if path.is_file():
                out[name]["bytes"] = path.stat().st_size
    return out


def grade_label(logs: Path, run_id: str, summary: dict) -> str:
    grade_path = logs / "evaluations" / f"{run_id}.json"
    grade = None
    if grade_path.is_file():
        try:
            grade = json.loads(grade_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    grade = grade or summary.get("evaluation") or {}
    status = grade.get("status") or "-"
    if status == "scored":
        return "scored:resolved" if grade.get("resolved") else "scored:unresolved"
    return status


def describe(summary: dict, logs: Path) -> dict:
    run_id = summary.get("run_id", "?")
    task = summary.get("task", {})
    process = summary.get("process", {})
    return {
        "run_id": run_id,
        "instance_id": task.get("instance_id"),
        "condition": task.get("condition"),
        "model": summary.get("claude", {}).get("model"),
        "started_at": summary.get("started_at"),
        "clean": summary.get("session", {}).get("ran_meaningfully") is not False,
        "asked": summary.get("ask_user_question", {}).get("direct_asked"),
        "sdk_session_id": process.get("sdk_session_id"),
        "grade": grade_label(logs, run_id, summary),
        "artifacts": artifact_map(logs, run_id),
    }


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("instance", nargs="?", help="instance_id or a substring of one")
    cli.add_argument("--condition", choices=["ambiguous", "full"])
    cli.add_argument("--logs-dir", default=".experiment-logs")
    cli.add_argument("--json", action="store_true", dest="as_json")
    args = cli.parse_args()

    logs = Path(args.logs_dir)
    summaries = load_runs(logs)
    if not summaries:
        print(f"no run summaries under {logs}", file=sys.stderr)
        return 1

    by_instance: dict[str, list[dict]] = {}
    for summary in summaries:
        by_instance.setdefault(summary.get("task", {}).get("instance_id") or "?", []).append(summary)

    # No argument: print the index so there is something to copy-paste from.
    if not args.instance:
        if args.as_json:
            print(json.dumps(
                {iid: [s.get("run_id") for s in runs] for iid, runs in sorted(by_instance.items())},
                indent=2,
            ))
            return 0
        print(f"{len(summaries)} run(s) across {len(by_instance)} instance(s):")
        for iid, runs in sorted(by_instance.items()):
            conditions = ", ".join(
                f"{s.get('task', {}).get('condition')}:{s.get('run_id', '?')[:8]}" for s in runs
            )
            print(f"  {iid:30s} {conditions}")
        return 0

    matches = {
        iid: runs for iid, runs in by_instance.items()
        if iid == args.instance or args.instance in iid
    }
    if not matches:
        print(f"no runs match {args.instance!r}; run without arguments for the index", file=sys.stderr)
        return 1

    selected = []
    for iid, runs in sorted(matches.items()):
        for summary in runs:
            if args.condition and summary.get("task", {}).get("condition") != args.condition:
                continue
            selected.append(describe(summary, logs))
    if not selected:
        print(f"runs exist for {', '.join(matches)} but none in condition {args.condition!r}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(selected, indent=2))
        return 0

    for iid in sorted({entry["instance_id"] for entry in selected}):
        entries = [e for e in selected if e["instance_id"] == iid]
        print(f"{iid}  ({len(entries)} run(s))")
        for entry in entries:
            print(
                f"  [{entry['condition']}] {entry['run_id']}\n"
                f"      model={entry['model']}  started={entry['started_at']}  "
                f"clean={entry['clean']}  asked={entry['asked']}  grade={entry['grade']}\n"
                f"      sdk_session_id={entry['sdk_session_id']}"
            )
            for name, info in entry["artifacts"].items():
                if not info["exists"]:
                    detail = "(missing)"
                elif "files" in info:
                    detail = f"({len(info['files'])} file(s): {', '.join(info['files'])})"
                else:
                    detail = f"({info['bytes']} bytes)"
                print(f"      {name:12s} {info['path']} {detail}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
