"""Literal constants shared across the harness. Single source of truth for
condition names, exit reasons, event/tool literals, and the on-disk schema version.
"""

# --- schema ---
SCHEMA_VERSION = "0.1"

# --- experimental conditions ---
CONDITION_AMBIGUOUS = "ambiguous"  # feed problem_statement (underspecified)
CONDITION_FULL = "full"            # feed original_issue (control / upper bound)
CONDITIONS = (CONDITION_AMBIGUOUS, CONDITION_FULL)

# Which dataset field sources each condition's prompt.
CONDITION_SOURCE_FIELD = {
    CONDITION_AMBIGUOUS: "problem_statement",
    CONDITION_FULL: "original_issue",
}

# --- run exit reasons (record/schema: exit.reason) ---
EXIT_ASKED = "asked"                    # agent invoked AskUserQuestion -> record & stop
EXIT_PRODUCED_PATCH = "produced_patch"  # finished with a non-empty diff
EXIT_NO_PATCH = "no_patch"              # finished clean but empty diff
EXIT_MAX_TURNS = "max_turns"            # CLI hit --max-turns without patch or ask
EXIT_TIMEOUT = "timeout"                # wall-clock timeout killed the subprocess
EXIT_ERROR = "error"                    # CLI crashed / non-zero / infra failure
EXIT_REASONS = (
    EXIT_ASKED, EXIT_PRODUCED_PATCH, EXIT_NO_PATCH,
    EXIT_MAX_TURNS, EXIT_TIMEOUT, EXIT_ERROR,
)

# --- stream-json event types (as emitted by `claude -p --output-format stream-json`) ---
EVENT_SYSTEM = "system"
EVENT_ASSISTANT = "assistant"
EVENT_USER = "user"          # tool_result messages arrive as role=user
EVENT_RESULT = "result"      # final event: num_turns, cost, usage
EVENT_STREAM = "stream_event"

# --- tool names we care about ---
TOOL_ASK_USER_QUESTION = "AskUserQuestion"
TOOL_EDIT = "Edit"
TOOL_WRITE = "Write"
TOOL_MULTI_EDIT = "MultiEdit"
TOOL_NOTEBOOK_EDIT = "NotebookEdit"
EDIT_TOOLS = (TOOL_EDIT, TOOL_WRITE, TOOL_MULTI_EDIT, TOOL_NOTEBOOK_EDIT)
READ_TOOLS = ("Read", "Grep", "Glob")

# --- evaluation status (record/schema: evaluation.eval_status) ---
EVAL_PENDING = "pending"                  # produced a patch, not yet evaluated
EVAL_EVALUATED = "evaluated"              # swebench report merged in
EVAL_SKIPPED_NO_PATCH = "skipped_no_patch"  # asked / no_patch -> nothing to evaluate
EVAL_ERROR = "eval_error"                 # image build / harness failure (data-quality)

# --- difficulty buckets (from the dataset) ---
DIFFICULTY_BUCKETS = (
    "<15 min fix",
    "15 min - 1 hour",
    "1-4 hours",
    ">4 hours",
)
