# ambig-SWE

This repository runs **non-interactive Claude Code with Opus 4.8** on an issue from
the local dataset.

There is no hook, tool allowlist, custom system prompt, patch evaluator, or Docker
integration. Claude runs in print mode with permission checks bypassed. The launcher
observes Claude's own local session transcript and stops the run at its first direct
`AskUserQuestion` call.

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
out the instance's `base_commit`, and then replaces itself with:

```bash
claude --model claude-opus-4-8 "<prompt>"
```

Each session runs as:

```bash
claude --model claude-opus-4-8 -p --permission-mode bypassPermissions "<prompt>"
```

This removes the interactive terminal and permission prompts. The launcher still does
not pass an allowlist, hooks, plan mode, or a prompt telling Claude to ask. It observes
the matching JSONL record under `~/.claude/projects/`; when the primary session calls
`AskUserQuestion`, it sends Claude an interrupt and records that call. Use only on
repositories and machines where unrestricted tool execution is acceptable.

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
- `.experiment-logs/` is Git-ignored. It contains an immutable pre-launch manifest,
  a normalized result for each run, copies of the source transcripts, and report
  outputs. Each run's Claude stdout and stderr are in `process-output/`. A missing or
  malformed matching transcript is logged as `unknown`, never as a no-question result.
- This is an observational launcher, not an automated evaluation harness.
- The local dataset schema is documented in [`data/README.md`](data/README.md).
