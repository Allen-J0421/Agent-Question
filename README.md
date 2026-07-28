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
tie-break; the session stops at the first main-thread ask.

`bypassPermissions` is deliberately not used: it shadows `can_use_tool` for ordinary
tools, so the agent never pauses and can resolve an under-specified issue by reading
the repository instead of asking. See `PREREGISTRATION.md` for the full rationale,
scope limitations, and outcome definitions.

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
approves prompting tool calls and observes `AskUserQuestion` (see
`PREREGISTRATION.md`). No hooks, no plan mode, no prompt telling Claude to ask.
Every run's summary records the exact live tool roster, whether `AskUserQuestion`
was available, and how many permission prompts reached the callback, so results
are self-certifying rather than assumed.

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
