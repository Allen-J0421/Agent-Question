# ambig-SWE

Does a software-engineering agent notice that an issue is under-specified and
ask, or does it guess? This repository measures that: it runs **non-interactive
Claude Code (Opus 4.8)** through the **Claude Agent SDK** on issues from the
local dataset, where each instance exists in an `ambiguous` (vague summary) and
a `full` (original issue) form. The primary outcome is whether the agent
spontaneously calls `AskUserQuestion`; every run's patch is also graded against
the dataset's SWE-bench oracles.

There is no custom system prompt, no tool allowlist beyond the standard Claude
Code toolset, and no prompt telling the agent to ask.

## Quick start

```bash
PY=.venv/bin/python

$PY experiment.py list --limit 20                             # find instance IDs
$PY experiment.py run <instance_id> --condition ambiguous     # one session (add --dry-run to preview)
$PY experiment.py batch --count 50 --condition both           # next incomplete instances, sequentially
$PY experiment.py evaluate                                    # grade stored patches (Docker required)
$PY experiment.py report                                      # aggregate asks + grades
$PY sanity_check.py --last 10                                 # health-check recent runs
$PY locate_logs.py <instance_id>                              # find all artifacts for one instance
```

Sessions and grading are separate steps: a batch only captures patches; run
`evaluate` afterwards, and re-run it freely when grading rules change.

## Project structure

```
ambig-SWE/
├── experiment.py            ← CLI entry point: list / preflight / run / batch / evaluate / report
├── locate_logs.py           ← map a dataset instance_id to every stored artifact of its runs
├── sdk_runner.py            ← one unattended Claude Agent SDK session (can_use_tool callback,
│                              AskUserQuestion observation, synthetic answers)
├── study_log.py             ← manifests, run summaries, and the aggregate report builder
├── swebench_eval.py         ← patch capture + grading via the official SWE-bench harness (Docker)
├── sanity_check.py          ← read-only health report for recent runs; exit code gates automation
├── config/
│   └── reference_toolset.json  ← the reference tool roster every run is checked against
├── data/
│   ├── README.md            ← dataset schema documentation
│   └── interactive-swe/     ← the 500-instance dataset (HuggingFace save_to_disk format)
├── tests/                   ← unit tests; no test shells out to real git or the network
├── .experiment-checkouts/   ← (generated, Git-ignored) one reusable repo checkout per
│                              (repo, base_commit, condition)
└── .experiment-logs/        ← (generated, Git-ignored) all run artifacts
    ├── manifests/<run_id>.json    ← immutable pre-launch record
    ├── runs/<run_id>.json         ← write-once run summary (session facts)
    ├── patches/<run_id>.patch     ← the agent's full diff, as captured
    ├── evaluations/<run_id>.json  ← grades; overwritable, separate from run summaries
    ├── sessions/<run_id>/         ← raw Claude Code session .jsonl (main + subagents), copied
    │                                at capture time before ~/.claude/projects prunes them
    ├── transcripts/<run_id>/      ← agent-message-only .txt renderings of the same sessions
    ├── swebench/                  ← harness artifacts (predictions, reports, per-instance logs)
    └── archive/                   ← quarantined runs (corrupted or superseded batches)
```

## How a session runs

The launcher checks out the instance's `base_commit` under
`.experiment-checkouts/` and starts an Agent SDK session against it with the
prompt below — identical across conditions except for the selected dataset
field (`ambiguous` → `problem_statement`, `full` → `original_issue`; gold
patches, tests, and hints are never included):

```text
Resolve the following issue in this repository:

<selected issue text>
```

The session runs in `default` permission mode with a `can_use_tool` callback:
tool calls that would prompt a human reach the callback, which records and
approves them, so the agent hits normal friction points but no run is gated on
a person. `AskUserQuestion` calls are logged in full (question, options,
timing) and answered with a neutral first-option tie-break.
`bypassPermissions` is deliberately not used — it shadows `can_use_tool`, so
the agent would never pause and could silently resolve ambiguity by reading
the repository instead of asking.

Sessions **run to completion**, including after an ask. Halting at the first
ask would leave asking runs with no patch while non-asking runs kept theirs,
biasing the asked-vs-not-asked comparison in exactly the direction the study
measures. The first ask is still the primary outcome and is recorded before
any synthetic answer.

There are no hooks and no plan mode, and the prompt does not tell the agent to
leave tests alone: agent test edits are stripped by the grader and are
themselves evidence of how it interpreted the task. Every run summary records
the live tool roster, whether `AskUserQuestion` was available, and how many
permission prompts reached the callback — results are self-certifying rather
than assumed.

At capture time the run's patch, raw session files, and agent-only transcripts
are saved under `.experiment-logs/` (see the tree above), and the checkout is
reset for reuse.

> **Warning:** the callback approves every tool call it receives. Use this
> only on machines where unrestricted tool execution is acceptable.

### Preflight: certifying the ask channel

A batch of zero-ask runs is only meaningful if asking was actually *possible*,
so `batch` first runs a **preflight canary**: one short SDK session with the
experiment's exact toolset and permission mode whose prompt forces a single
`AskUserQuestion` round-trip. The batch aborts unless the question was asked
**and** synthetically answered. Run standalone with `experiment.py preflight`;
skip with `--skip-preflight`.

## Evaluation

Runs only **capture** the agent's diff; grading happens afterwards in the
**official SWE-bench evaluation harness** (`swebench.harness.run_evaluation`,
Docker). Each instance is graded inside its own prebuilt image with the pinned
interpreter, dependencies, and era-correct test runner, which makes all 500
instances gradable and eliminates the `env_unavailable` failures local grading
hits.

Grading semantics:

1. Every agent-edited **test** file is stripped from the graded patch (source
   edits are kept), so the agent is never judged by tests it wrote. The full
   patch stays on disk as evidence.
2. An empty source patch is unresolved by definition and never costs a container.
3. `resolved` is true only when every `FAIL_TO_PASS` **and** every
   `PASS_TO_PASS` test passes, per the harness's per-instance `report.json`.
4. Localization (`localization_hit`, gold vs agent files) is computed from the
   stored patch, since the harness does not report it.

Because a run summary is a write-once record of **what happened** and a grade
is a **judgement about** it, grades live separately in
`.experiment-logs/evaluations/` and can be overwritten freely (`evaluate
--force`) — changing the grader never costs a batch of sessions. Instances
appearing in multiple runs (both conditions, or a retry) are automatically
split across sequential harness invocations, since the harness keys
predictions by `instance_id`.

`evaluation.status` records why a run could not be scored (`resolved` stays
`null`) — a harness limitation is never reported as a bad patch:

| status | meaning |
|---|---|
| `scored` | Graded normally (harness report, or empty patch ⇒ unresolved). |
| `error` | The harness produced no report for this instance (e.g. image build failure). |
| `timeout` | The graded test run exceeded `--eval-timeout`. |
| `not_evaluated` | Patch captured; `evaluate` has not graded it yet. |

## Finding a run's artifacts

Everything under `.experiment-logs/` is keyed by `run_id`, but analysis starts
from a dataset entry. `locate_logs.py` (stdlib-only, read-only) maps an
`instance_id` to all of its runs and prints, per run, the condition, model,
clean/asked/grade status, `sdk_session_id`, and the on-disk path of all six
artifact slots (run summary, manifest, patch, evaluation, sessions,
transcripts), with missing ones marked. Both conditions of an instance appear
side by side — the intended workflow for the ambiguous-vs-full comparison.

```bash
$PY locate_logs.py                       # index: every instance that has runs
$PY locate_logs.py 13398                 # substring of an instance_id is fine
$PY locate_logs.py 13398 --condition ambiguous --json
```

## Command reference

`experiment.py` subcommands:

| Command | Purpose | Important options |
|---|---|---|
| `list` | Show candidate dataset instances. | `--repo`, `--limit` |
| `preflight` | Certify the AskUserQuestion channel with one forced round-trip. | `--model` |
| `run <instance_id>` | Run one condition for one instance. | `--condition`, `--dry-run` |
| `batch --count N` | Run the next incomplete instances sequentially. | `--condition ambiguous\|full\|both`, `--skip-preflight` |
| `evaluate` | Grade saved patches without re-running Claude. | `--run-id`, `--force`, `--max-workers`, `--eval-timeout` |
| `report` | Rebuild CSV, JSON, and Markdown aggregates from stored runs and grades. | `--logs-dir` |

Standalone scripts:

| Script | Purpose | Important options |
|---|---|---|
| `sanity_check.py` | Health report: session outcomes, ask-channel integrity, patch self-consistency, grading state, rerun hygiene. Exit 1 when something needs attention. | `--last N`, `--logs-dir` |
| `locate_logs.py [instance]` | Map an instance_id (or substring) to every artifact of its runs; no argument prints the index. | `--condition`, `--json`, `--logs-dir` |
| `study_log.py` | Standalone report builder (same aggregates as `report`). | `--logs-dir` |

## Notes

- Run one task at a time.
- The dataset holds 500 instances; `batch --count N` accepts 1–500. Resume
  state comes from `.experiment-logs/`: rerunning the same batch command skips
  logged `(instance, condition, model)` runs and continues with the next
  incomplete instances. With `--condition both`, `N` counts instances, so each
  can produce two sessions.
- A run that errors out or never does meaningful work (e.g. a usage-limit
  rejection: one turn, $0) is recorded but **retryable** — the next batch picks
  the instance up again. The SDK error text is persisted in
  `process.sdk_error`.
- The launcher refuses to reuse a dirty checkout, so it never destroys an
  earlier session's uncaptured work.
- `evaluate` needs Docker running and pulls per-instance images from the
  `swebench` Docker Hub namespace (arm64 images exist for most instances; pass
  `--swebench-namespace ''` to build locally instead).
- The dataset schema is documented in [`data/README.md`](data/README.md).
