# ambig-SWE — What Do Coding Agents Do With Ambiguous Tasks?

A behavioral study of how LLM coding agents handle **under-specified** software
engineering tasks. When a GitHub issue is vague — missing the error, the repro steps,
the expected behavior — does the agent **recognize** the ambiguity and **ask** a
clarifying question, or does it silently proceed on assumptions? And how does that
decision affect whether it actually resolves the task?

This repo contains **Phase 0**: an evidence-gathering harness that runs a real coding
agent (Claude CLI, Opus) over 500 ambiguous SWE-bench tasks and records everything it
does, then evaluates its patches against real test suites.

## Research framing

Prior work (e.g. CLARITI, Vijayvargiya et al. 2026) *forces* a clarification step and
studies question *quality* with a simulated user. We study the prior, unforced question:
**the endogenous ask-vs-proceed decision** a deployed agent actually makes. No simulated
user, no oracle answering — when the agent asks, we **record the question and stop**
("record & stop").

The evidence is organized around eight questions:

| | Evidence question |
|---|---|
| **Q1** | How often does the agent spontaneously recognize ambiguity and ask? |
| **Q2** | When and how much does it ask (timing, count, grounding before asking)? |
| **Q3** | What does ambiguity cost in resolve rate (ambiguous vs. full-info control)? |
| **Q4** | How accurately does it localize the fix (touched vs. gold files)? |
| **Q5** | How often does its patch break other tests (regression)? |
| **Q6** | How stable is the ask/resolve behavior across repeated runs? |
| **Q7** | What effort does it spend (turns, tokens, cost)? |
| **Q8** | How does it fail (ask / no-patch / max-turns / timeout / error)? |

## Data — `data/interactive-swe/`

500 SWE-bench Verified instances, each with an **ambiguous** `problem_statement`
(repro/errors/expected-output stripped) and the full `original_issue`, plus standard
eval fields (`FAIL_TO_PASS`, `PASS_TO_PASS`, `patch`, `base_commit`, …). See
[`data/README.md`](data/README.md) for the full schema and caveats. Django is ~46% of
instances and 91% are ≤1 hour — **all rates are reported stratified by repo and
difficulty.**

## Experimental design

Three conditions per instance, driven through the Claude CLI headless with a **neutral**
prompt (no "ask if unclear" language, so the ask decision stays endogenous):

- **ambiguous** (`problem_statement`) × **3 repeats** — the behavioral signal + variance
- **full** (`original_issue`) × 1 — control / upper bound
- Agent model: **Opus**. Downstream resolution via the official `swebench` harness.

## Quick start

```bash
bash harness/scripts/provision.sh          # install swebench (+ deps), check Docker/CLI
PY=.venv/bin/python

# stratified pilot manifest (already frozen -> manifests/pilot_60.json)
$PY -m harness.orchestrator.cli sample --n 60 --seed 42 --name pilot_60

# run the agent sweep (resumable), evaluate, report
$PY -m harness.orchestrator.cli run    --manifest pilot_60
$PY -m harness.orchestrator.cli eval   --manifest pilot_60      # needs Docker running
$PY -m harness.orchestrator.cli aggregate
$PY -m harness.orchestrator.cli report
```

Each run writes an immutable `runs/<instance_id>/<condition>/r<NN>/result.json` plus the
raw `transcript.jsonl`, `agent.patch`, and `stderr.log`. `aggregate` flattens all
records into `runs_table.csv` (one row per run); `report` prints the stratified Q1–Q8
metrics.

## Repository layout

```
data/interactive-swe/     the 500-instance ambiguous SWE-bench dataset (+ README)
harness/                  the evidence harness (see harness/README.md)
  config, constants       run parameters + shared literals
  data/                   loader (field parsing, gold-file fallback), pilot sampler
  prompt/                 neutral prompt builder
  agent/                  git-worktree isolation, CLI runner, stream-json parser
  capture/                trajectory, ask detector, diff extractor, localization
  eval/                   swebench adapter + predictions + report reader + driver
  record/                 the result.json schema + atomic store (resumability)
  metrics/                aggregate table + stratified Q1–Q8 report
  orchestrator/           run-unit plan, per-run executor, CLI entrypoint
manifests/                frozen sample manifests (pilot_60.json)
tests/                    unit tests + recorded real-CLI fixtures
runs/  repos/  eval_logs/  run outputs, cached mirror clones, eval logs (git-ignored)
```

## Status

The harness is code-complete and validated end-to-end on real data: 12 unit tests pass
against recorded real-CLI fixtures, and a live single-instance Opus run produced a
complete `result.json` (correct ask detection, patch capture, localization). Milestones:
**M0** smoke → **M1** stratified ~60-instance pilot → **M2** full 500. See the design
doc at `~/.claude/plans/lets-zoom-out-first-eager-stream.md`.

## Cost note

A single Opus run costs roughly **$0.6**. The pilot (≈240 runs) is ≈$150; the full sweep
(≈2000 runs) is ≈$1,250 in API alone, plus Docker eval compute. Budget accordingly.
