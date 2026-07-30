# ambig-SWE

This repository runs **non-interactive Claude Code with Opus 4.8** on an issue from
the local dataset.

There is no tool allowlist beyond the standard Claude Code toolset
(`config/reference_toolset.json`), no custom system prompt, no patch evaluator, and
no Docker integration.

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
invented here:

1. The agent's diff is saved to `.experiment-logs/patches/<run_id>.patch`.
2. Every agent-edited **test** file is reverted (source edits are kept), so the
   agent is never judged by tests it wrote. No gold source patch in the dataset
   touches a test path, so this cannot discard a real fix.
3. The gold `test_patch` is applied.
4. The test files owning the `FAIL_TO_PASS` / `PASS_TO_PASS` node ids are run with
   `pytest -rA`, and each id is looked up in the output. Whole files are used
   deliberately: one stale node id on the command line makes pytest report
   `no tests ran` and discard every other result in the same invocation.
5. `resolved` is true only when every `FAIL_TO_PASS` **and** every `PASS_TO_PASS`
   test passes. The workspace is then reset so the next run can reuse it.

### Grading is separate from running

A run summary records **what happened** during a session and is write-once. A grade
is a **judgement about** that session, and grading rules change — so grades are
stored separately in `.experiment-logs/evaluations/<run_id>.json` and can be
overwritten freely. Changing the grader never costs you a batch of sessions.

```bash
# Sessions only; patches saved, grading skipped entirely.
$PY experiment.py batch --count 20 --scope gradable --no-eval

# Grade (or re-grade) stored patches — no Claude session is re-run.
$PY experiment.py evaluate                 # runs missing an evaluation
$PY experiment.py evaluate --force         # re-grade everything
$PY experiment.py evaluate --run-id <uuid> # one run
$PY experiment.py report                   # aggregates always read the stored grades
```

Each run's diff is saved to `.experiment-logs/patches/<run_id>.patch`. Grading
re-applies that patch to a fresh checkout of `base_commit`, so it never needs the
original workspace. The agent's own test edits stay in the stored patch as evidence
but are filtered out before grading, since they are not an oracle.

`evaluation.status` records why a run could not be scored, and `resolved` is `null`
for all of them — a harness limitation is never reported as a bad patch:

| status | meaning |
|---|---|
| `scored` | Graded normally. |
| `unsupported_runner` | django/sympy ids are not pytest node ids (306 of 500). |
| `env_unavailable` | The repository does not import in this virtualenv. |
| `test_patch_failed` | The gold `test_patch` would not apply. |
| `timeout` | The graded pytest run exceeded `--eval-timeout`. |

## Scope

Only **194 of 500** instances use pytest node ids; django (231) and sympy (75) use
their own runners. `--scope gradable` restricts the candidate pool *before*
selection, so `--count 50` gives 50 gradable instances rather than 50 of the first
500 with most skipped.

```bash
$PY experiment.py batch --count 50 --condition ambiguous --scope gradable
$PY experiment.py list --scope gradable --limit 20
```

Ask-behavior can still be measured on all 500 with the default `--scope all`.

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

# Run one unattended Claude experiment.
$PY experiment.py run astropy__astropy-13579 --condition ambiguous

# Run the next 50 incomplete dataset instances, one unattended session at a time.
$PY experiment.py batch --count 50 --condition ambiguous

# Run both conditions for the next 50 incomplete dataset instances (up to 100 sessions).
$PY experiment.py batch --count 50 --condition both

# Aggregate the run logs after one or more sessions.
$PY experiment.py report

# Equivalent standalone report builder.
$PY study_log.py
```

The launcher prepares a normal GitHub checkout under the Git-ignored
`.experiment-checkouts/` subdirectory of the directory where you invoke it, checks
out the instance's `base_commit`, and then launches a session against it.

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

## Notes

- Run one task at a time.
- The dataset currently contains 500 instances. `batch --count N` accepts 1 through
  500 and runs sessions sequentially. Resume state comes from `.experiment-logs/`:
  rerunning the same batch command skips logged `(instance, condition, model)` runs
  and continues with the next incomplete instances. With `--condition both`, `N`
  means N instances, so each can produce two sessions.
- The script refuses to reuse a dirty checkout so it never destroys an earlier
  Claude change.
- `.experiment-logs/` is Git-ignored. It contains an immutable pre-launch manifest
  and a normalized result for each run, including the live tool roster
  (`tool_roster`) and the permission mode and prompt count (`permissions`).
- This is an observational launcher, not an automated evaluation harness.
- The local dataset schema is documented in [`data/README.md`](data/README.md).
