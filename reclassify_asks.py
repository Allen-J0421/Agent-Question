#!/usr/bin/env python3
"""Re-run the ask classifier over stored Codex runs and audit the patterns.

The offline half of the ask-detection design (architecture adopted from
sycophancy-ACE's ``benchmark_keyword_regex.py`` / ``ablate_patterns.py``):
every Codex run preserves its raw ``--json`` event streams under
``sessions/<run_id>/``, so any classifier config can be re-applied to every
final message ever produced -- without spending a session -- and compared
against the verdicts recorded at run time.

Reports, per stored Codex run and round:

* the recorded verdict (classifier version stamped at run time) vs. the
  current config's verdict, with disagreements listed alongside the final
  message text -- the review queue after any pattern change;
* per-pattern firing counts across all final messages, split by current
  verdict (the ablation precursor: a signal pattern that fires mostly on
  non-asking messages is a false-positive source; once hand labels exist,
  these counts become TP/FP/precision per pattern);
* a structural cross-check: asked verdicts vs. ``workspace_had_changes``,
  separating clarifying-before-work asks from post-work offers.

Read-only. Exit code 0 always: disagreements after a config change are the
expected output, not an error.

Usage:
    python3 reclassify_asks.py [--logs-dir DIR] [--config PATH] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import ask_detection
from codex_runner import parse_events


def load_codex_runs(logs: Path) -> list[dict]:
    """Every stored run summary whose runner is codex-cli, oldest first."""
    runs = []
    runs_dir = logs / "runs"
    if not runs_dir.exists():
        return runs
    for path in sorted(runs_dir.glob("*.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"WARNING: unreadable run summary: {path}", file=sys.stderr)
            continue
        agent = summary.get("agent") or {}
        if agent.get("runner") == "codex-cli":
            runs.append(summary)
    runs.sort(key=lambda s: s.get("started_at") or "")
    return runs


def rounds_for(logs: Path, run_id: str) -> list[tuple[int, str | None, int]]:
    """(round_index, final_message, file_changes) from preserved event streams.

    file_changes drives the offline zero-edit gate. Events are the only
    gate evidence available offline (the runner's git fingerprint delta is
    not reconstructible after the workspace was reset), so a shell-based
    edit with no file_change item would pass this gate; the recorded
    verdict in the summary keeps the runner's full-evidence gate.
    """
    out: list[tuple[int, str | None, int]] = []
    sessions = logs / "sessions" / run_id
    for path in sorted(sessions.glob("codex-events-round-*.jsonl")):
        try:
            round_index = int(path.stem.rsplit("-", 1)[1])
        except ValueError:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        parsed = parse_events(lines)
        out.append((round_index, parsed["final_message"], parsed["file_changes"]))
    return out


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--logs-dir", default=".experiment-logs")
    cli.add_argument(
        "--config",
        default=None,
        help="alternate ask_detection config to evaluate (default: the live one)",
    )
    cli.add_argument("--json", action="store_true", dest="as_json")
    args = cli.parse_args()

    logs = Path(args.logs_dir)
    config = ask_detection.load_config(args.config) if args.config else ask_detection.default_config()
    runs = load_codex_runs(logs)
    if not runs:
        print(f"no codex-cli run summaries under {logs}", file=sys.stderr)
        return 0

    rows = []
    for summary in runs:
        run_id = summary.get("run_id", "?")
        recorded_rounds = {
            entry.get("index"): entry
            for entry in summary.get("session", {}).get("rounds", [])
        }
        recorded_version = summary.get("ask_user_question", {}).get("classifier_version")
        first = summary.get("ask_user_question", {}).get("first_direct") or {}
        for round_index, message, file_changes in rounds_for(logs, run_id):
            verdict = ask_detection.classify(message, config)
            # Offline two-layer verdict: zero-edit gate (events evidence)
            # then the regex layer, matching the runner's live rule.
            gated = verdict["asked"] and file_changes == 0
            recorded = recorded_rounds.get(round_index, {}).get("asked")
            rows.append(
                {
                    "run_id": run_id,
                    "instance_id": summary.get("task", {}).get("instance_id"),
                    "round": round_index,
                    "recorded_version": recorded_version,
                    "recorded_asked": recorded,
                    "reclassified_asked": gated,
                    "regex_asked": verdict["asked"],
                    "file_changes": file_changes,
                    "agrees": (recorded is None) or (recorded == gated),
                    "signals": verdict["signals"],
                    "blockers": verdict["blockers"],
                    "question_units": verdict["question_units"],
                    "workspace_had_changes": (
                        first.get("workspace_had_changes")
                        if first.get("round") == round_index
                        else None
                    ),
                    "final_message": message,
                }
            )

    if args.as_json:
        print(json.dumps(rows, indent=1))
        return 0

    total = len(rows)
    asked_now = [r for r in rows if r["reclassified_asked"]]
    gated_out = [r for r in rows if r["regex_asked"] and not r["reclassified_asked"]]
    disagreements = [r for r in rows if not r["agrees"]]
    print(
        f"reclassified {total} round(s) from {len(runs)} codex run(s) "
        f"with config v{config['version']} (zero-edit gate + regex)"
    )
    print(
        f"  asked (current config): {len(asked_now)}   "
        f"regex fired but turn edited (gated): {len(gated_out)}   "
        f"disagreements vs recorded verdicts: {len(disagreements)}"
    )
    if gated_out:
        print(
        "  NOTE: gated rounds contradict the 36/36 zero-edit observation — "
        "review them before trusting the gate."
        )

    if disagreements:
        print("\nDISAGREEMENTS (recorded -> reclassified) — the review queue:")
        for r in disagreements:
            snippet = " ".join((r["final_message"] or "").split())[:160]
            print(
                f"  {r['run_id'][:8]} round {r['round']} "
                f"[v{r['recorded_version']}] {r['recorded_asked']} -> "
                f"{r['reclassified_asked']}  ({r['instance_id']})\n"
                f"      {snippet!r}"
            )

    # Per-pattern firing counts, split by verdict (ablation precursor).
    fired: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        for section in ("signals", "blockers"):
            for category, patterns in r[section].items():
                for pattern in patterns:
                    fired[(category, pattern)][0 if r["reclassified_asked"] else 1] += 1
    if fired:
        print("\nPER-PATTERN FIRING (rounds classified asked / not-asked):")
        for (category, pattern), (on_asked, on_other) in sorted(
            fired.items(), key=lambda kv: -(kv[1][0] + kv[1][1])
        ):
            print(f"  {on_asked:4d} / {on_other:4d}  [{category}] {pattern}")

    pre_work = [r for r in asked_now if r["workspace_had_changes"] is False]
    post_work = [r for r in asked_now if r["workspace_had_changes"] is True]
    if asked_now:
        print(
            f"\nSTRUCTURAL SPLIT of asked rounds: {len(pre_work)} before any "
            f"workspace change, {len(post_work)} after changes, "
            f"{len(asked_now) - len(pre_work) - len(post_work)} unknown"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
