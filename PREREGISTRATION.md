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
The Agent SDK path replaces it: a registered `can_use_tool` callback is
the SDK's documented substitute for the interactive permission prompt, and
it makes `AskUserQuestion` both reachable and resolvable. The CLI path has
been removed rather than retained as a comparison arm, because a path where
the primary outcome is structurally impossible cannot produce a comparable
measurement.

## Tool roster

`config/reference_toolset.json`. Captured live on 2026-07-28 from
`ClaudeAgentOptions(tools={"type": "preset", "preset": "claude_code"},
permission_mode="default", strict_mcp_config=True, mcp_servers={},
can_use_tool=<callback>)` — the SDK's own "give me the standard set"
preset, with this machine's personal MCP servers (Gmail, Calendar, Drive)
excluded so the roster reflects a stock install rather than this
developer's account-specific extras. Every one of the 28 entries came from
that live capture. No tool was added by hand, narrowed, or removed.
`AskUserQuestion` appears in the preset on its own once a callback is
registered; it does not need to be named explicitly.

## Permission mode

`default`. Tool calls whose permission rules evaluate to "ask" reach the
study's `can_use_tool` callback, which records them and approves them, so
no run is gated on a human being present.

This replaces the original `bypassPermissions` design, which was not a
neutral choice. Under `bypassPermissions` the SDK does not invoke
`can_use_tool` for calls the mode already permits — it emits
`CanUseToolShadowedWarning` — so ordinary tools never reached the callback
and the agent never paused. An agent that can read, edit, and run tests
without interruption can resolve an under-specified issue by inspecting the
repository, which removes the very occasion on which a human would
otherwise be asked. Measuring clarification behavior in that environment
measures the absence of the opportunity, not the absence of the behavior.

`permissions.prompts_reaching_callback` is recorded per run so the presence
of that friction is verifiable rather than assumed.

## can_use_tool callback: answering policy

The callback receives ordinary tool calls (recorded, then approved) and
`AskUserQuestion` calls. For `AskUserQuestion` it:

- **Logs every call in full**: question text, options, per-question
  `multiSelect` flag, timestamp, and the count of main-thread tool actions
  taken before this call (`assistant_tool_actions_before`).
- **Answers with the first offered option per question** (or `[first
  option]` for multi-select), so a headless run can complete instead of
  stalling forever on a question no human is present to answer.

This answering policy is an explicit modeling assumption, not a neutral
non-choice — there is no truly neutral way to auto-answer a question
addressed to a human. It is documented here so it can be evaluated,
challenged, or varied in a follow-up study. It does not affect the *first*
ask, which is this study's primary outcome (see below), since the first
call is fully logged before any synthetic answer is generated.

**Stopping rule.** The session ends at the first main-thread
`AskUserQuestion`. Everything after that point is shaped by the synthetic
first-option answer rather than by the agent's own reading of the task, so
it is not evidence about clarification behavior. This matches the stopping
behavior the CLI path used and makes `direct_count > 1` rare by
construction — `any_agent_count` remains the measure of subagent asks
occurring before the stop.

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

- Fully attended sessions, where a human answers permission prompts
  instead of the callback approving them automatically.
- Post-first-ask behavior: the run stops at the first main-thread ask.
- Any account/organization-level variation in tool availability beyond
  this one account, on 2026-07-28.
