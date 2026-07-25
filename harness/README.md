# Phase-0 Evidence Harness

Runs the Claude CLI (Opus) as a coding agent on the `interactive-swe` ambiguous SWE
tasks and captures **everything** about its behavior — whether it recognizes ambiguity
and asks, or proceeds silently; what it edits; whether it resolves the task — for later
analysis. No simulated user: when the agent asks a clarifying question we **record & stop**.

See the design doc: `~/.claude/plans/lets-zoom-out-first-eager-stream.md`.

## Setup

```bash
bash harness/scripts/provision.sh     # installs swebench (+cbor2 pin), checks Docker/CLI
```

Requirements already present: Docker Desktop, git, Claude CLI, `.venv` (Python 3.12).
**Start Docker Desktop before running the eval pass.** The CLI is resolved from
`HARNESS_CLAUDE_BIN`, then PATH, then the nvm fallback.

## Usage

```bash
PY=.venv/bin/python

# 1. freeze the stratified pilot manifest (already done -> manifests/pilot_60.json)
$PY -m harness.orchestrator.cli sample --n 60 --seed 42 --name pilot_60

# 2. run the agent sweep (ambiguous x3 + full x1 per instance; resumable)
$PY -m harness.orchestrator.cli run --manifest pilot_60
#   or a single instance / condition:
$PY -m harness.orchestrator.cli run --instances psf__requests-1142 --conditions ambiguous

# 3. evaluate produced patches with swebench (NEEDS DOCKER)
$PY -m harness.orchestrator.cli eval --manifest pilot_60

# 4. aggregate -> table, then print stratified Q1-Q8 metrics
$PY -m harness.orchestrator.cli aggregate
$PY -m harness.orchestrator.cli report
```

## What each run captures

One immutable `runs/<instance_id>/<condition>/r<NN>/result.json` (+ `transcript.jsonl`,
`agent.patch`, `stderr.log`). Fields: ask/no-ask + questions, produced patch + diff +
files touched, turns/tools/tokens/cost, localization vs gold files, and (after the eval
pass) resolved / regression / F2P / P2P. Schema: `harness/record/schema.py`.

## Conditions & repeats

- **ambiguous** (`problem_statement`) × 3 repeats — the behavioral signal + variance
- **full** (`original_issue`) × 1 — control / upper bound
- Model: Opus. Neutral prompt (no "ask if unclear" language) so the ask decision is endogenous.

## Layout

`config` `constants` · `data/{loader,sampling}` · `prompt/builder` ·
`agent/{workspace,runner,stream_parser}` · `capture/{trajectory,ask_detector,diff_extractor,localization,annotations}` ·
`eval/{predictions,swebench_adapter,result_reader,driver}` · `record/{schema,store}` ·
`metrics/{aggregate,report}` · `orchestrator/{plan,executor,cli}`.

Tests: `.venv/bin/python -m pytest tests/ -q` (parser/detector/loader/schema, pinned to
real-CLI fixtures in `tests/fixtures/`).

## Notes

- macOS has no GNU `timeout`; the runner enforces wall-clock timeouts via Python subprocess.
- Bare mirror clones live in `repos/`; per-run worktrees in `.worktrees/` (git
  `safe.bareRepository=all` is set per-command for the mirrors).
- swebench loads the local dataset directly — no conversion needed.
