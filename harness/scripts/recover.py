"""Usage limit recovery script.

When a Claude Opus run hits the account usage limit mid-sweep, the CLI process
dies with a non-zero exit and leaves a partial/empty transcript.jsonl. The
result.json is either absent (run was in-flight) or has exit=error. This script:

  1. SCAN  — finds all run directories that are incomplete or errored due to
             usage-limit signals (empty transcript, 'usage' in stderr, no result.json)
  2. CLEAN — removes only those corrupted artifacts (transcript.jsonl, agent.patch,
             stderr.log) so the harness will retry them on the next `run` invocation.
             Leaves any *complete* result.json untouched.
  3. WAIT  — polls the CLI every 60 s until a test invocation succeeds (usage reset),
             then prints the command to resume the sweep.

Run from the project root:
    python harness/scripts/recover.py [--manifest MANIFEST] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

# Strings in stderr that indicate a usage-limit termination (not a code bug).
_USAGE_SIGNALS = [
    "Claude AI usage limit reached",
    "usage limit",
    "rate limit",
    "overloaded",
    "exceeded",
    "too many requests",
    "529",
    "429",
]

_CLAUDE_BIN_CANDIDATES = [
    os.environ.get("HARNESS_CLAUDE_BIN", ""),
    "/Users/allenjiang/.nvm/versions/node/v24.12.0/bin/claude",
]


def _claude_bin() -> str | None:
    import shutil
    for c in _CLAUDE_BIN_CANDIDATES:
        if c and Path(c).exists():
            return c
    found = shutil.which("claude")
    return found


def _is_usage_signal(text: str) -> bool:
    t = text.lower()
    return any(s.lower() in t for s in _USAGE_SIGNALS)


def _run_dir_is_corrupted(run_dir: Path) -> tuple[bool, str]:
    """Return (corrupted, reason). Corrupted means: in-progress or usage-killed."""
    result_json = run_dir / "result.json"
    transcript = run_dir / "transcript.jsonl"
    stderr_log = run_dir / "stderr.log"

    # No result.json at all — run was killed before we could write it.
    if not result_json.exists():
        # Only flag it if there's evidence it actually started (transcript or stderr present).
        if transcript.exists() or stderr_log.exists():
            stderr_text = stderr_log.read_text() if stderr_log.exists() else ""
            if _is_usage_signal(stderr_text) or transcript.stat().st_size == 0 if transcript.exists() else True:
                return True, "no result.json + partial artifacts"
        return False, ""

    # result.json exists — check if it's an error record caused by usage limit.
    try:
        rec = json.loads(result_json.read_text())
        exit_reason = rec.get("exit", {}).get("reason", "")
        error_text = rec.get("exit", {}).get("error_text", "") or ""
        if exit_reason == "error" and _is_usage_signal(error_text):
            return True, f"error record with usage signal: {error_text[:120]}"
        # Complete good record — never touch it.
        if exit_reason and exit_reason != "error":
            return False, ""
        # error record without usage signal — might be a code bug, leave it.
        if exit_reason == "error":
            return False, ""
    except (json.JSONDecodeError, KeyError):
        # Malformed result.json — treat as corrupted.
        return True, "malformed result.json"

    return False, ""


def _clean_run_dir(run_dir: Path, dry_run: bool) -> list[str]:
    """Remove partial artifacts but never result.json (store.py re-checks it)."""
    removed = []
    # Remove result.json only if it's an error/malformed record (we detected that above).
    result_json = run_dir / "result.json"
    if result_json.exists():
        try:
            rec = json.loads(result_json.read_text())
            exit_reason = rec.get("exit", {}).get("reason", "")
            error_text = rec.get("exit", {}).get("error_text", "") or ""
            if exit_reason == "error" and _is_usage_signal(error_text):
                if not dry_run:
                    result_json.unlink()
                removed.append(str(result_json))
        except (json.JSONDecodeError, KeyError):
            if not dry_run:
                result_json.unlink()
            removed.append(str(result_json))

    for fname in ("transcript.jsonl", "agent.patch", "stderr.log"):
        p = run_dir / fname
        if p.exists():
            if not dry_run:
                p.unlink()
            removed.append(str(p))
    return removed


def scan(manifest: str | None) -> list[tuple[Path, str]]:
    """Return list of (run_dir, reason) for corrupted runs."""
    instance_ids: set[str] | None = None
    if manifest:
        mfile = MANIFESTS_DIR / f"{manifest}.json"
        if not mfile.exists():
            sys.exit(f"manifest not found: {mfile}")
        instance_ids = set(json.loads(mfile.read_text())["instance_ids"])

    corrupted = []
    if not RUNS_DIR.exists():
        return corrupted

    for instance_dir in sorted(RUNS_DIR.iterdir()):
        if not instance_dir.is_dir():
            continue
        if instance_ids and instance_dir.name not in instance_ids:
            continue
        for condition_dir in sorted(instance_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            for repeat_dir in sorted(condition_dir.iterdir()):
                if not repeat_dir.is_dir():
                    continue
                is_bad, reason = _run_dir_is_corrupted(repeat_dir)
                if is_bad:
                    corrupted.append((repeat_dir, reason))
    return corrupted


def wait_for_reset(poll_interval_s: int = 60) -> None:
    """Poll until a minimal CLI health-check succeeds (usage reset)."""
    claude = _claude_bin()
    if not claude:
        print("  [wait] claude binary not found — cannot poll. Check manually.")
        return

    print(f"\n[wait] Polling every {poll_interval_s}s for usage reset "
          f"(Ctrl-C to abort)...")
    attempt = 0
    while True:
        attempt += 1
        try:
            result = subprocess.run(
                [claude, "-p", "Reply with the single word: ready",
                 "--output-format", "json", "--max-turns", "1",
                 "--model", "claude-opus-4-8"],
                capture_output=True, text=True, timeout=30,
            )
            combined = (result.stdout + result.stderr).lower()
            if result.returncode == 0 and not _is_usage_signal(combined):
                print(f"  [wait] attempt {attempt}: CLI responded OK — usage reset!")
                return
            else:
                signal = next((s for s in _USAGE_SIGNALS
                               if s.lower() in combined), "non-zero exit")
                print(f"  [wait] attempt {attempt}: still limited ({signal}) — "
                      f"retrying in {poll_interval_s}s")
        except subprocess.TimeoutExpired:
            print(f"  [wait] attempt {attempt}: timeout — retrying in {poll_interval_s}s")
        except Exception as e:
            print(f"  [wait] attempt {attempt}: error ({e}) — "
                  f"retrying in {poll_interval_s}s")
        time.sleep(poll_interval_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=None,
                   help="restrict scan to instances in this manifest (e.g. pilot_15)")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be cleaned without deleting anything")
    p.add_argument("--no-wait", action="store_true",
                   help="clean only, do not poll for usage reset")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="seconds between usage-reset polls (default: 60)")
    args = p.parse_args()

    print("=" * 60)
    print("ambig-SWE recovery script")
    print("=" * 60)

    # 1. SCAN
    corrupted = scan(args.manifest)
    if not corrupted:
        print("\n[scan] No corrupted runs found — nothing to clean.")
        if not args.no_wait:
            print("[scan] Checking if CLI is available...")
            wait_for_reset(args.poll_interval)
        return

    print(f"\n[scan] Found {len(corrupted)} corrupted run(s):")
    for run_dir, reason in corrupted:
        print(f"  {run_dir.relative_to(PROJECT_ROOT)}  ({reason})")

    # 2. CLEAN
    if args.dry_run:
        print("\n[clean] DRY RUN — would remove:")
        for run_dir, _ in corrupted:
            files = _clean_run_dir(run_dir, dry_run=True)
            for f in files:
                print(f"  - {Path(f).relative_to(PROJECT_ROOT)}")
        print("\nRe-run without --dry-run to actually delete.")
        return

    print(f"\n[clean] Removing partial artifacts...")
    total_removed = 0
    for run_dir, _ in corrupted:
        removed = _clean_run_dir(run_dir, dry_run=False)
        for f in removed:
            print(f"  removed: {Path(f).relative_to(PROJECT_ROOT)}")
        total_removed += len(removed)
    print(f"[clean] Done — {total_removed} file(s) removed. "
          f"{len(corrupted)} run(s) will be retried on next sweep.")

    # 3. WAIT
    if args.no_wait:
        print("\n[wait] Skipped (--no-wait). Resume sweep when ready:")
    else:
        wait_for_reset(args.poll_interval)
        print("\n[resume] Usage reset confirmed. Resume the sweep with:")

    manifest_flag = f"--manifest {args.manifest}" if args.manifest else "--manifest pilot_15"
    print(f"\n  export PATH=\"/Users/allenjiang/.nvm/versions/node/v24.12.0/bin:$PATH\"")
    print(f"  export PYTHONPATH=\"$PWD\"")
    print(f"  .venv/bin/python -m harness.orchestrator.cli run {manifest_flag}")


if __name__ == "__main__":
    main()
