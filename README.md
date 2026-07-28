# ambig-SWE

This repository runs **non-interactive Claude Code with Opus 4.8** on an issue from
the local dataset.

There is no tool allowlist beyond the standard Claude Code toolset
(`config/reference_toolset.json`), no custom system prompt, no patch evaluator, and
no Docker integration.

**Default interface: the Claude Agent SDK** (`--interface sdk`, the default for both
`run` and `batch`). A `can_use_tool` callback is registered — this is required for
`AskUserQuestion` to appear in the session's tool roster at all, confirmed by direct
testing; without a callback, the tool is silently absent even when named explicitly.
The callback logs every `AskUserQuestion` call in full (question, options, timing)
and answers with a neutral first-option tie-break so the run can complete headlessly.
Every other tool is auto-approved unchanged (`bypassPermissions`). See
`PREREGISTRATION.md` for the full rationale, scope limitations, and outcome
definitions.

A legacy CLI path (`--interface cli`, plain `claude -p --permission-mode
bypassPermissions`) is kept for comparison. It has no callback mechanism, so
`AskUserQuestion` is structurally unreachable there — this is *why* the SDK path
exists, not an equivalent alternative.

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

By default (`--interface sdk`) this is an Agent SDK session: `permission_mode`
`bypassPermissions`, `tools` set to `config/reference_toolset.json`, and a
`can_use_tool` callback that observes and answers `AskUserQuestion` calls (see
`PREREGISTRATION.md`). No hooks, no plan mode, no prompt telling Claude to ask.
Every run's summary records the exact live tool roster and whether
`AskUserQuestion` was available, so results are self-certifying rather than
assumed.

With `--interface cli`, the session runs as plain:

```bash
claude --model claude-opus-4-8 -p --permission-mode bypassPermissions "<prompt>"
```

`-p` mode has no callback mechanism, so `AskUserQuestion` cannot resolve there —
this path is kept only for reference/comparison, not as an equivalent alternative.
Use only on repositories and machines where unrestricted tool execution is
acceptable.

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
  and a normalized result for each run. SDK-path summaries also carry the live tool
  roster (`tool_roster`). CLI-path summaries additionally include transcript copies
  and `process-output/` stdout/stderr; a missing or malformed matching transcript is
  logged as `unknown` there, never as a no-question result.
- This is an observational launcher, not an automated evaluation harness.
- The local dataset schema is documented in [`data/README.md`](data/README.md).
