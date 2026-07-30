# ambig-SWE

This repository runs **non-interactive Claude Code with Opus 4.8** on an issue from
the local dataset.

There is no tool allowlist beyond the standard Claude Code toolset
(`config/reference_toolset.json`), no custom system prompt, and no Docker
integration. Each normal run saves its patch and grades it against the dataset's
SWE-bench oracles.

Sessions run through the **Claude Agent SDK** in `default` permission mode with a
`can_use_tool` callback registered. Tool calls that would prompt reach the callback,
which records them and approves them, so the agent hits the same friction points an
attended session has but no run is gated on a human. `AskUserQuestion` calls are
logged in full (question, options, timing) and answered with a neutral first-option
tie-break.

Sessions **run to completion**, including after an ask. Halting at the first ask
would leave asking runs with no patch while non-asking runs kept theirs, which
biases the asked-vs-not-asked comparison in exactly the direction the study
measures. The first ask is still the primary outcome and is recorded before any
synthetic answer; `ask_user_question.answered_questions` keeps the rest auditable.

`bypassPermissions` is deliberately not used: it shadows `can_use_tool` for ordinary
tools, so the agent never pauses and can resolve an under-specified issue by reading
the repository instead of asking.

## Evaluation

Every run is graded against the dataset's own SWE-bench oracles — no test is
invented here. Runs **capture** the agent's diff
(`.experiment-logs/patches/<run_id>.patch`) and grading happens afterwards in the
**official SWE-bench evaluation harness** (`swebench.harness.run_evaluation`,
Docker): each instance is graded inside its own prebuilt image with the pinned
interpreter, dependencies, compiled extensions, and era-correct test runner. That
makes all 500 instances gradable (django's `runtests.py` and sympy's `bin/test`
included) and eliminates the `env_unavailable` failures that local grading hits
(e.g. astropy's logger refusing to initialize under a modern pytest).

Grading semantics:

1. Every agent-edited **test** file is stripped from the graded patch (source
   edits are kept), so the agent is never judged by tests it wrote. The full
   patch stays on disk as evidence.
2. An empty source patch is unresolved by definition and never costs a container.
3. `resolved` is true only when every `FAIL_TO_PASS` **and** every `PASS_TO_PASS`
   test passes, as reported by the harness's per-instance `report.json`.
4. Localization (`localization_hit`, gold vs agent files) is computed from the
   stored patch, since the harness does not report it.

### Grading is separate from running

A run summary records **what happened** during a session and is write-once. A grade
is a **judgement about** that session, and grading rules change — so grades are
stored separately in `.experiment-logs/evaluations/<run_id>.json` and can be
overwritten freely. Changing the grader never costs you a batch of sessions.

```bash
# Sessions only (the default): patches saved, grading deferred.
$PY experiment.py batch --count 20 --condition ambiguous

# Grade (or re-grade) stored patches — needs Docker running; no Claude session is re-run.
$PY experiment.py evaluate                 # runs missing an evaluation
$PY experiment.py evaluate --force         # re-grade everything
$PY experiment.py evaluate --run-id <uuid> # one run
$PY experiment.py report                   # aggregates always read the stored grades
```

Harness artifacts (predictions files, per-instance logs and reports) live under
`.experiment-logs/swebench/`. Instances that appear in multiple runs (both
conditions, or a retry) are automatically split across sequential harness
invocations, since the harness keys predictions by `instance_id`.

`evaluation.status` records why a run could not be scored, and `resolved` is `null`
for all of them — a harness limitation is never reported as a bad patch:

| status | meaning |
|---|---|
| `scored` | Graded normally (harness report, or empty patch ⇒ unresolved). |
| `error` | The harness produced no report for this instance (e.g. image build failure). |
| `timeout` | The graded test run exceeded `--eval-timeout`. |
| `not_evaluated` | Patch captured; `evaluate` has not graded it yet. |

## Preflight: certifying the ask channel

The study's primary outcome is whether the agent calls `AskUserQuestion`. A batch
of zero-ask runs is only meaningful if asking was actually *possible*, so `batch`
first runs a **preflight canary**: one short SDK session, with the experiment's
exact toolset and permission mode, whose prompt forces a single AskUserQuestion
round-trip. The batch aborts unless the question was asked **and** synthetically
answered. Skip with `--skip-preflight`; run standalone with:

```bash
$PY experiment.py preflight
```


## Dataset fields

- `ambiguous` uses only `problem_statement`.
- `full` uses only `original_issue`.

The default is `ambiguous`. Gold patches, tests, hints, and the other issue condition
are never included in the prompt.

## Usage

```bash
PY=.venv/bin/python

# Find instance IDs.
$PY experiment.py list --limit 20

# Preview the exact selected field and Claude command without launching anything.
$PY experiment.py run astropy__astropy-13579 --condition ambiguous --dry-run

# Certify the AskUserQuestion channel (also runs automatically before each batch).
$PY experiment.py preflight

# Run one unattended Claude Agent SDK experiment (patch captured; grading deferred).
$PY experiment.py run astropy__astropy-13579 --condition ambiguous

# Run the next 50 incomplete instances, one unattended session at a time.
$PY experiment.py batch --count 50 --condition ambiguous

# Run both conditions for the next 50 incomplete dataset instances (up to 100 sessions).
$PY experiment.py batch --count 50 --condition both

# Grade every ungraded run in the official SWE-bench harness (Docker must be running).
$PY experiment.py evaluate

# Aggregate ask behavior, tool-roster, and stored evaluation results.
$PY experiment.py report

# Equivalent standalone report builder.
$PY study_log.py
```

The launcher prepares a normal GitHub checkout under the Git-ignored
`.experiment-checkouts/` subdirectory of the directory where you invoke it, checks
out the instance's `base_commit`, and then launches a session against it. It
saves the agent patch, resets the checkout for reuse, and defers grading to
the `evaluate` command (official SWE-bench harness).

Each session is an Agent SDK session: `permission_mode` `default`, `tools` set to
`config/reference_toolset.json`, and a `can_use_tool` callback that records and
approves prompting tool calls and observes `AskUserQuestion`. No hooks, no plan
mode, no prompt telling Claude to ask — in particular the prompt does not tell the
agent to leave tests alone, because agent test edits are handled by the grader and
are themselves evidence of how it interpreted the task. Every run's summary records
the exact live tool roster, whether `AskUserQuestion` was available, and how many
permission prompts reached the callback, so results are self-certifying rather than
assumed.

The callback approves every tool call it receives, so use this only on repositories
and machines where unrestricted tool execution is acceptable.

The prompt is identical across conditions except for the selected dataset field:

```text
Resolve the following issue in this repository:

<selected issue text>
```

## Command reference

| Command | Purpose | Important options |
|---|---|---|
| `list` | Show candidate dataset instances. | `--repo`, `--limit` |
| `preflight` | Certify the AskUserQuestion channel with one forced round-trip. | `--model` |
| `run <instance_id>` | Run one condition for one instance. | `--condition`, `--dry-run` |
| `batch --count N` | Run the next incomplete instances sequentially. | `--condition ambiguous\|full\|both`, `--skip-preflight` |
| `evaluate` | Grade saved patches without re-running Claude. | `--run-id`, `--force`, `--max-workers`, `--eval-timeout` |
| `report` | Rebuild CSV, JSON, and Markdown aggregates from stored runs and grades. | `--logs-dir` |


## Notes

- Run one task at a time.
- The dataset currently contains 500 instances. `batch --count N` accepts 1 through 500 and runs sessions
  sequentially. Resume state comes from `.experiment-logs/`:
  rerunning the same batch command skips logged `(instance, condition, model)` runs
  and continues with the next incomplete instances. With `--condition both`, `N`
  means N instances, so each can produce two sessions.
- The script refuses to reuse a dirty checkout so it never destroys an earlier
  Claude change.
- `.experiment-logs/` is Git-ignored. It contains an immutable pre-launch manifest
  and a normalized result for each run, including the live tool roster
  (`tool_roster`) and the permission mode and prompt count (`permissions`).
- A run that errors out or never does meaningful work (e.g. a usage-limit
  rejection: one turn, $0) is recorded but treated as **retryable** — rerunning
  the batch picks the instance up again instead of burying it. The SDK error
  subtype and message are persisted in the run summary (`process.sdk_error`).
- The default `evaluate` harness needs Docker running and pulls per-instance
  images from the `swebench` Docker Hub namespace (arm64 images exist for most
  instances; pass `--swebench-namespace ''` to build locally instead).
- The local dataset schema is documented in [`data/README.md`](data/README.md).
