#!/usr/bin/env python3
"""Read-only health report for recent experiment runs.

Answers "how good was the recent run?" in one terminal screen: how many
sessions finished clean, which failed and why (with the persisted SDK error
text), whether the ask-measurement channel stayed trustworthy, whether the
captured patches are self-consistent (apply cleanly, no dangling imports),
what is graded vs pending, and whether the next batch can launch safely.

Stdlib only, touches nothing on disk, and never guesses: every check reads
the same records the experiment itself wrote. Exit code is 0 when nothing
needs attention and 1 otherwise, so it can gate automation.

Usage:
    python3 sanity_check.py                 # all runs, most recent first
    python3 sanity_check.py --last 10       # only the 10 most recent runs
    python3 sanity_check.py --logs-dir DIR  # e.g. an archive folder
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CHECKOUTS = Path.cwd() / ".experiment-checkouts"

OK = "✅"
WARN = "⚠️ "
BAD = "❌"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_json_dir(directory: Path) -> tuple[dict[str, dict], list[str]]:
    """Return {stem: payload} for every readable JSON file, plus error names."""
    records: dict[str, dict] = {}
    unreadable: list[str] = []
    if not directory.exists():
        return records, unreadable
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("not a JSON object")
            records[path.stem] = payload
        except (OSError, ValueError, json.JSONDecodeError):
            unreadable.append(str(path))
    return records, unreadable


def parse_time(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Per-run classification
# --------------------------------------------------------------------------


def classify(summary: dict) -> str:
    """clean | dead — mirrors the experiment's own ran_meaningfully signal."""
    if summary.get("session", {}).get("ran_meaningfully") is False:
        return "dead"
    return "clean"


def is_limit_shaped(summary: dict) -> bool:
    """The usage-limit signature: errored with (almost) no work performed."""
    process = summary.get("process", {})
    return bool(
        classify(summary) == "dead"
        and (process.get("sdk_num_turns") or 0) <= 1
        and not process.get("sdk_total_cost_usd")
    )


def short(value, width):
    text = "-" if value is None else str(value)
    return text[:width].ljust(width)


# --------------------------------------------------------------------------
# Patch self-consistency
# --------------------------------------------------------------------------


_RELATIVE_IMPORT = re.compile(r"\s*from\s+(\.+)([\w.]*)\s+import\s+(.+)")


def patched_paths(patch_text: str) -> list[str]:
    """Every path the patch names, from its diff headers."""
    return [
        line.split(" b/", 1)[1].strip()
        for line in patch_text.splitlines()
        if line.startswith("diff --git ") and " b/" in line
    ]


def added_lines(patch_text: str) -> dict[str, list[str]]:
    """{path: added source lines} for every file the patch touches."""
    current = None
    out: dict[str, list[str]] = {}
    for line in patch_text.splitlines():
        if line.startswith("diff --git ") and " b/" in line:
            current = line.split(" b/", 1)[1].strip()
            out.setdefault(current, [])
        elif current and line.startswith("+") and not line.startswith("+++"):
            out[current].append(line[1:])
    return out


def _bound_in_init(
    workspace: Path, package: Path, name: str, added: dict[str, list[str]]
) -> bool:
    """Whether the package __init__ binds ``name`` without a module file.

    ``from . import name`` is also satisfied by a plain attribute of the
    package (an assignment, def/class, or absolute-import alias), so those
    bindings are not dangling. Purely relative import lines do not count as
    bindings here: they are exactly the pattern under suspicion.
    """
    init_rel = str(package / "__init__.py")
    text = "\n".join(added.get(init_rel, []))
    init_file = workspace / init_rel
    if init_file.exists():
        try:
            text = init_file.read_text(encoding="utf-8") + "\n" + text
        except OSError:
            pass
    pattern = re.compile(
        rf"^\s*(?:{name}\s*=|def\s+{name}\b|class\s+{name}\b"
        rf"|import\s+[\w.]+\s+as\s+{name}\b"
        rf"|from\s+\w[\w.]*\s+import\s+[^\n]*\b{name}\b)",
        re.MULTILINE,
    )
    return bool(pattern.search(text))


def dangling_relative_imports(patch_text: str, workspace: Path) -> list[str]:
    """Added relative imports whose target module does not exist.

    A target counts as existing when its file is in the checkout, when the
    patch itself creates it, or when the package __init__ binds the name.
    A miss is the signature of lost work: the patch imports a module nothing
    provides, so every graded test would die at import time. Only relative
    imports are checked -- absolute ones can come from other distributions,
    which a static scan cannot rule in or out.
    """
    touched = set(patched_paths(patch_text))
    added = added_lines(patch_text)
    problems: list[str] = []
    for path, lines in added.items():
        if not path.endswith(".py"):
            continue
        for text in lines:
            match = _RELATIVE_IMPORT.match(text)
            if not match:
                continue
            dots, trail, names = match.groups()
            package = Path(path).parent
            for _ in range(len(dots) - 1):
                package = package.parent
            if trail:
                targets = [(package / trail.replace(".", "/"), None)]
            else:
                targets = [
                    (package / name, name)
                    for name in (
                        raw.replace("(", "").split(" as ")[0].strip()
                        for raw in names.split(",")
                    )
                    if name.isidentifier()
                ]
            for module, name in targets:
                candidates = (f"{module}.py", f"{module}/__init__.py")
                if any((workspace / c).exists() or c in touched for c in candidates):
                    continue
                if name and _bound_in_init(workspace, package, name, added):
                    continue
                problems.append(
                    f"{path}: '{text.strip()}' resolves to nothing "
                    f"({candidates[0]} missing) — lost-file capture signature"
                )
    return problems


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--logs-dir", default=".experiment-logs")
    cli.add_argument("--last", type=int, help="only the N most recent runs")
    args = cli.parse_args()

    logs = Path(args.logs_dir)
    problems: list[str] = []
    notes: list[str] = []

    runs, unreadable = load_json_dir(logs / "runs")
    manifests, unreadable_manifests = load_json_dir(logs / "manifests")
    evaluations, _ = load_json_dir(logs / "evaluations")
    unreadable += unreadable_manifests

    summaries = sorted(
        runs.values(),
        key=lambda s: s.get("started_at") or "",
        reverse=True,
    )
    if args.last:
        summaries = summaries[: args.last]

    print(f"=== ambig-SWE sanity check ===  logs: {logs}")
    if not summaries:
        print("No run summaries found. Nothing to check.")
        return 0

    # ---- 1. Run table -----------------------------------------------------
    print(f"\nRUNS ({len(summaries)} shown, most recent first)")
    header = (
        f"  {'started':16} {'instance':26} {'cond':9} {'turns':>5} "
        f"{'min':>5} {'cost$':>6} {'asked':5} {'eval':14} verdict"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for summary in summaries:
        run_id = summary.get("run_id", "?")
        task = summary.get("task", {})
        process = summary.get("process", {})
        started = parse_time(summary.get("started_at"))
        duration = process.get("duration_seconds")
        asked = summary.get("ask_user_question", {}).get("direct_asked")
        grade = evaluations.get(run_id, summary.get("evaluation") or {})
        status = grade.get("status") or "-"
        eval_text = status
        if status == "scored":
            eval_text = f"scored:{'R' if grade.get('resolved') else 'unres'}"
        verdict = classify(summary)
        flag = OK if verdict == "clean" else BAD
        turns = process.get("sdk_num_turns")
        cost = process.get("sdk_total_cost_usd")
        columns = [
            short(started.strftime("%m-%d %H:%M") if started else None, 16),
            short(task.get("instance_id"), 26),
            short(task.get("condition"), 9),
            f"{turns if turns is not None else '-':>5}",
            f"{duration / 60:>5.1f}" if isinstance(duration, (int, float)) else f"{'-':>5}",
            f"{cost:>6.2f}" if isinstance(cost, (int, float)) else f"{'-':>6}",
            f"{ {True: 'yes', False: 'no'}.get(asked, '-'):5}",
            short(eval_text, 14),
            f"{flag} {verdict}",
        ]
        print("  " + " ".join(columns))

    # ---- 2. Failure diagnosis ---------------------------------------------
    dead = [s for s in summaries if classify(s) == "dead"]
    print(f"\nFAILURES ({len(dead)} dead run(s))")
    if not dead:
        print(f"  {OK} every run finished clean")
    else:
        problems.append(f"{len(dead)} dead run(s) — instances remain retryable")
        limit_shaped = [s for s in dead if is_limit_shaped(s)]
        if len(limit_shaped) >= 2:
            first = min(
                (parse_time(s.get("started_at")) for s in limit_shaped if parse_time(s.get("started_at"))),
                default=None,
            )
            print(
                f"  {BAD} {len(limit_shaped)} usage-limit-shaped failures "
                f"(<=1 turn, $0) starting {first.strftime('%m-%d %H:%M') if first else '?'} "
                "— likely one exhausted-limit event"
            )
        for summary in dead:
            process = summary.get("process", {})
            error = process.get("sdk_error") or process.get("stop_reason") or "no error text recorded"
            print(
                f"  {BAD} {summary.get('run_id', '?')[:8]}  "
                f"{summary.get('task', {}).get('instance_id')}  "
                f"[{process.get('sdk_result_subtype') or '-'}] {str(error)[:110]}"
            )

    # Interrupted sessions: a manifest with no matching run summary.
    orphans = [rid for rid in manifests if rid not in runs]
    if orphans:
        problems.append(f"{len(orphans)} interrupted session(s) (manifest, no summary)")
        for rid in orphans:
            task = manifests[rid].get("task", {})
            print(f"  {BAD} interrupted: {rid[:8]}  {task.get('instance_id')} — force-stopped mid-session?")
    if unreadable:
        problems.append(f"{len(unreadable)} unreadable JSON file(s)")
        for path in unreadable:
            print(f"  {BAD} unreadable: {path}")

    # ---- 3. Measurement channel -------------------------------------------
    print("\nCHANNEL (ask-measurement integrity)")
    roster_bad = [
        s for s in summaries
        if s.get("tool_roster", {}).get("matches_reference") is False
        or s.get("tool_roster", {}).get("askuserquestion_available") is False
    ]
    if roster_bad:
        problems.append(f"{len(roster_bad)} run(s) with roster mismatch / AskUserQuestion missing")
        for summary in roster_bad:
            print(f"  {BAD} roster problem on {summary.get('run_id', '?')[:8]}")
    else:
        print(f"  {OK} tool roster matched reference on all runs; AskUserQuestion available")
    clean = [s for s in summaries if classify(s) == "clean"]
    no_prompts = [
        s for s in clean if not s.get("permissions", {}).get("prompts_reaching_callback")
    ]
    if no_prompts:
        problems.append(f"{len(no_prompts)} clean run(s) saw zero permission prompts")
        for summary in no_prompts:
            print(f"  {BAD} zero prompts reached callback: {summary.get('run_id', '?')[:8]}")
    else:
        print(f"  {OK} permission prompts reached the callback on every clean run")
    asked_counts = {
        "asked": sum(s.get("ask_user_question", {}).get("direct_asked") is True for s in summaries),
        "did not ask": sum(s.get("ask_user_question", {}).get("direct_asked") is False for s in summaries),
        "unobservable": sum(not isinstance(s.get("ask_user_question", {}).get("direct_asked"), bool) for s in summaries),
    }
    print("  ask outcomes: " + ", ".join(f"{v} {k}" for k, v in asked_counts.items()))

    # ---- 4. Grading --------------------------------------------------------
    print("\nGRADING")
    pending, contaminated = [], []
    for summary in summaries:
        rid = summary.get("run_id")
        grade = evaluations.get(rid)
        graded = grade is not None and grade.get("status") != "not_evaluated"
        if classify(summary) == "dead" and graded:
            contaminated.append(summary)
        elif classify(summary) == "clean" and not graded:
            pending.append(summary)
    scored = [
        rid for rid, grade in evaluations.items()
        if rid in runs and grade.get("status") == "scored"
    ]
    print(f"  scored: {len(scored)}   pending: {len(pending)}")
    if pending:
        notes.append(
            f"{len(pending)} clean run(s) await grading — run: .venv/bin/python experiment.py evaluate"
        )
    if contaminated:
        problems.append(
            f"{len(contaminated)} DEAD run(s) have stored grades — these pollute resolve-rate aggregates"
        )
        for summary in contaminated:
            print(f"  {BAD} contaminated grade: {summary.get('run_id', '?')[:8]}")
    seen: dict[tuple, list[str]] = {}
    for summary in summaries:
        task = summary.get("task", {})
        key = (task.get("instance_id"), task.get("condition"), summary.get("claude", {}).get("model"))
        seen.setdefault(key, []).append(f"{summary.get('run_id', '?')[:8]}:{classify(summary)}")
    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate (instance, condition) record(s)")
        for key, rids in duplicates.items():
            print(f"  {WARN} duplicate {key[0]} ({key[1]}): {', '.join(rids)} — quarantine the dead one before reporting")

    # ---- 5. Patch self-consistency ----------------------------------------
    print("\nPATCHES (self-consistency)")
    apply_failures: list[tuple[str, str, str]] = []
    dangling: list[tuple[str, str, str]] = []
    unverifiable: list[str] = []
    checked = 0
    for summary in summaries:
        if classify(summary) != "clean":
            continue
        rid = summary.get("run_id", "?")
        patch_file = logs / "patches" / f"{rid}.patch"
        if not patch_file.exists():
            continue
        workspace = Path(summary.get("workspace") or "")
        if not (workspace / ".git").exists():
            unverifiable.append(rid)
            continue
        checked += 1
        instance = summary.get("task", {}).get("instance_id") or "?"
        patch_text = patch_file.read_text(encoding="utf-8")
        probe = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--check", "-"],
            input=patch_text, capture_output=True, text=True,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout).strip().splitlines()
            apply_failures.append((rid, instance, detail[0] if detail else "?"))
        for issue in dangling_relative_imports(patch_text, workspace):
            dangling.append((rid, instance, issue))
    if apply_failures:
        problems.append(f"{len(apply_failures)} patch(es) fail git apply --check")
        for rid, instance, detail in apply_failures:
            print(f"  {BAD} does not apply: {rid[:8]}  {instance}  {detail[:90]}")
    if dangling:
        problems.append(
            f"{len(dangling)} patch(es) with dangling imports — created file lost at capture?"
        )
        for rid, instance, issue in dangling:
            print(f"  {BAD} dangling import: {rid[:8]}  {instance}  {issue[:110]}")
    if not apply_failures and not dangling:
        print(
            f"  {OK} {checked} patch(es) apply cleanly to their checkouts; "
            "every added relative import resolves"
        )
    if unverifiable:
        notes.append(
            f"{len(unverifiable)} patch(es) not checked — workspace checkout missing"
        )

    # ---- 6. Rerun hygiene ---------------------------------------------------
    print("\nHYGIENE (can the next batch launch?)")
    dirty = []
    if CHECKOUTS.exists():
        for workdir in sorted(CHECKOUTS.iterdir()):
            if not (workdir / ".git").exists():
                continue
            status = subprocess.run(
                ["git", "-C", str(workdir), "status", "--porcelain"],
                capture_output=True, text=True,
            ).stdout.strip()
            if status:
                dirty.append(workdir.name)
    if dirty:
        problems.append(f"{len(dirty)} dirty checkout(s) — prepare_workspace will refuse them")
        for name in dirty:
            print(f"  {BAD} dirty checkout: {name}")
    else:
        print(f"  {OK} all checkouts clean")
    stray = subprocess.run(
        ["pgrep", "-fl", "experiment.py"], capture_output=True, text=True
    ).stdout.strip()
    if stray:
        problems.append("experiment.py process still running")
        print(f"  {BAD} still running: {stray.splitlines()[0]}")
    else:
        print(f"  {OK} no stray experiment processes")

    # ---- 7. Bottom line -----------------------------------------------------
    total_cost = sum(
        s.get("process", {}).get("sdk_total_cost_usd") or 0 for s in summaries
    )
    total_minutes = sum(
        s.get("process", {}).get("duration_seconds") or 0 for s in summaries
    ) / 60
    print("\nBOTTOM LINE")
    print(
        f"  {len(clean)}/{len(summaries)} finished clean | {len(dead)} dead | "
        f"{len(orphans)} interrupted | ${total_cost:.2f} spent | {total_minutes:.0f} min of sessions"
    )
    for note in notes:
        print(f"  {WARN} {note}")
    if problems:
        for problem in problems:
            print(f"  {BAD} {problem}")
        print(f"  {BAD} verdict: NEEDS ATTENTION")
        return 1
    print(f"  {OK} verdict: healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
