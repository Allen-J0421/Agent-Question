"""Build the agent-facing prompt for each condition.

RESEARCH-CRITICAL: the framing must be NEUTRAL about clarification. We neither tell the
agent to "ask if anything is unclear" (which would inflate the ask rate) nor to "just
implement it" (which would suppress asking). We give a realistic SWE task framing — the
kind a coding agent sees in deployment — and let the ask/proceed decision be endogenous.
The ONLY difference between the ambiguous and full conditions is the issue text
(problem_statement vs original_issue); the surrounding instructions are identical.
"""
from __future__ import annotations

import hashlib

from harness.constants import CONDITION_SOURCE_FIELD
from harness.data.loader import Instance
from harness.record.schema import PromptInfo

# Identical across conditions. Deliberately does not mention clarifying questions.
_TASK_TEMPLATE = """\
You are working in a checkout of the {repo} repository. Resolve the issue described below. Do not modify test files.

--- ISSUE ---
{issue_text}
--- END ISSUE ---
"""


def build_prompt(instance: Instance, condition: str) -> tuple[str, PromptInfo]:
    field = CONDITION_SOURCE_FIELD[condition]
    issue_text = getattr(instance, field)
    prompt = _TASK_TEMPLATE.format(repo=instance.repo, issue_text=issue_text.strip())
    info = PromptInfo(
        condition_source_field=field,
        prompt_chars=len(prompt),
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        hints_included=False,  # hints_text is never shown in Phase 0
    )
    return prompt, info
