# Pre-registration note: AskUserQuestion under ambiguity

Written before running the SDK-path batch on the 500-instance dataset, so
the environment's neutrality claim can be checked against a record made in
advance rather than asserted after seeing results.

## Research question

When Claude Code resolves an ambiguous SWE-bench issue unattended, does it
call `AskUserQuestion`, and under what conditions (latency, question count,
option count)?

## Why the environment changed from a plain CLI run

The original design ran `claude -p --permission-mode bypassPermissions`.
`-p` mode has no mechanism to resolve a paused `AskUserQuestion` call (no
callback, no terminal), so the tool cannot function there by construction.
Separately, live testing found that even the Agent SDK drops
`AskUserQuestion` from the tool roster unless a `can_use_tool` callback is
registered — this is a real, reproducible client-construction requirement
(see `sdk_runner.py`), not a workaround invented for this study. The SDK
path below is the minimum change that makes the tool functionally
reachable at all; everything else about the environment is held as close
to the CLI baseline as the SDK allows.

## Tool roster

`config/reference_toolset.json`. Captured live from
`ClaudeAgentOptions(tools={"type": "preset", "preset": "claude_code"},
strict_mcp_config=True, mcp_servers={})` — the SDK's own "give me the
standard set" preset, with this machine's personal MCP servers (Gmail,
Calendar, Drive) excluded so the roster reflects a stock install rather
than this developer's account-specific extras. No tool was added beyond
what the preset returns, except `AskUserQuestion` itself, which the preset
capture omits only because a `can_use_tool` callback wasn't registered for
that specific capture — it is otherwise a standard, undocumented-as-gated
tool (confirmed against the official Tools Reference: no plan-tier,
version, or beta annotation, unlike e.g. `Artifact` or `EndConversation`
which carry explicit gating notes in the same table). No tool was narrowed
or removed to make asking more or less likely.

## Permission mode

`bypassPermissions`, unchanged from the original design. This is a **named
scope limitation**, not an oversight: `bypassPermissions` removes the
natural interruption points that, in an attended session, sometimes double
as a moment where a human clarifies intent. The study therefore measures
*"does Claude ask when it doesn't have to stop for permission anyway"* —
not *"does Claude ask in a normal permission-prompting session."* Any
reported ask-rate should be read as a lower-bound-flavored estimate
relative to a fully attended session, not generalized to "Claude's asking
behavior" unqualified.

No neutral auto-responder for ordinary permission requests was built, to
avoid introducing a second new variable alongside the tool-roster fix in
the same change.

## can_use_tool callback: answering policy

`AskUserQuestion` is a documented, verified exception to
`bypassPermissions` shadowing — it reaches the callback even in bypass
mode (confirmed live; ordinary tools like `Write` do not reach the
callback under bypass, `AskUserQuestion` does). The callback:

- **Logs every call in full**: question text, options, per-question
  `multiSelect` flag, timestamp, and the count of main-thread tool actions
  taken before this call (`assistant_tool_actions_before`).
- **Answers with the first offered option per question** (or `[first
  option]` for multi-select), so a headless run can complete instead of
  stalling forever on a question no human is present to answer.

This answering policy is an explicit modeling assumption, not a neutral
non-choice — there is no truly neutral way to auto-answer a question
addressed to a human. It is documented here so it can be evaluated,
challenged, or varied in a follow-up study. It affects whether Claude asks
*again* later in the same session; it does not affect the *first* ask,
which is this study's primary outcome (see below), since the first call
is fully logged before any synthetic answer is generated.

## Primary and secondary outcomes

Computed by the existing `study_log.build_report` / `build_run_summary_sdk`
pipeline, unchanged in definition from the original design:

- **Primary**: `direct_asked` — whether the main thread (not a subagent)
  called `AskUserQuestion` at least once during the run.
- **Secondary**: `direct_count`, `any_agent_count` (including subagent
  calls), first-call question count and option count, and tool actions
  taken before the first ask.

## Self-certification

Every run summary now records `tool_roster.tools` (the exact live roster
the model had) and `tool_roster.askuserquestion_available` (whether
`AskUserQuestion` was actually present that run). This makes each run
verifiable after the fact instead of relying on an assumption about
whether the tool was reachable — the failure mode that produced the
uninterpretable 0/23 result under the original CLI-only design.

## Scope not covered by this study

- Attended/interactive sessions (see permission-mode note above).
- Post-first-ask behavior under a different answering policy than
  first-option tie-break.
- Any account/organization-level variation in tool availability beyond
  this one account, on 2026-07-28.
