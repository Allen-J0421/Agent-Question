"""The per-run record — the immutable on-disk contract every stage reads/writes.
One `result.json` per (instance, condition, repeat). Nested dataclasses mirror the
JSON exactly; `to_dict`/`from_dict` are the only (de)serialization path so the shape
can never drift between modules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from harness.constants import (
    EVAL_PENDING,
    SCHEMA_VERSION,
)


@dataclass
class Question:
    turn: int
    header: str = ""
    question: str = ""
    options: list[str] = field(default_factory=list)
    multi_select: bool = False


@dataclass
class PromptInfo:
    condition_source_field: str
    prompt_chars: int
    prompt_sha256: str
    hints_included: bool = False


@dataclass
class RunMeta:
    model: str
    permission_mode: str
    max_turns: int
    cli_version: str
    started_at: str
    ended_at: str
    wall_time_s: float
    workspace_strategy: str = "git-worktree"


@dataclass
class ExitInfo:
    reason: str                       # one of constants.EXIT_*
    cli_exit_code: int | None = None
    cli_subtype: str | None = None    # from the final `result` stream event
    error_text: str | None = None


@dataclass
class AskInfo:
    asked: bool = False
    n_questions: int = 0
    first_ask_turn: int | None = None
    questions: list[Question] = field(default_factory=list)


@dataclass
class PatchInfo:
    produced_patch: bool = False
    diff: str | None = None
    diff_sha256: str | None = None
    files_touched: list[str] = field(default_factory=list)
    n_files_touched: int = 0
    loc_added: int = 0
    loc_removed: int = 0


@dataclass
class Tokens:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    total: int = 0


@dataclass
class Trajectory:
    n_turns: int = 0
    n_assistant_msgs: int = 0
    n_tool_calls: int = 0
    tools_used: dict[str, int] = field(default_factory=dict)
    files_read: list[str] = field(default_factory=list)
    tokens: Tokens = field(default_factory=Tokens)
    cost_usd: float = 0.0
    num_turns_reported: int | None = None


@dataclass
class TestBreakdown:
    total: int = 0
    passed: int | None = None
    failed: int | None = None
    unresolved: int | None = None


@dataclass
class Localization:
    gold_files: list[str] = field(default_factory=list)
    hit_any: bool | None = None
    hit_all: bool | None = None
    precision: float | None = None
    recall: float | None = None
    jaccard: float | None = None


@dataclass
class Evaluation:
    eval_status: str = EVAL_PENDING
    resolved: bool | None = None
    fail_to_pass: TestBreakdown = field(default_factory=TestBreakdown)
    pass_to_pass: TestBreakdown = field(default_factory=TestBreakdown)
    regression: bool | None = None
    localization: Localization = field(default_factory=Localization)
    swebench_report_path: str | None = None
    eval_error_text: str | None = None


@dataclass
class Annotations:
    """Empty scaffold Phase 0 leaves for later post-hoc coding."""
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    question_categories: list[str] = field(default_factory=list)
    hidden_ambiguity_category: str | None = None
    annotator: str | None = None
    annotated_at: str | None = None


@dataclass
class Artifacts:
    transcript_path: str | None = None
    workspace_diff_path: str | None = None
    stderr_path: str | None = None


@dataclass
class RunRecord:
    run_id: str
    instance_id: str
    repo: str
    difficulty: str
    condition: str
    repeat_index: int
    prompt: PromptInfo
    run_meta: RunMeta
    exit: ExitInfo
    ask: AskInfo = field(default_factory=AskInfo)
    patch: PatchInfo = field(default_factory=PatchInfo)
    trajectory: Trajectory = field(default_factory=Trajectory)
    evaluation: Evaluation = field(default_factory=Evaluation)
    annotations: Annotations = field(default_factory=Annotations)
    artifacts: Artifacts = field(default_factory=Artifacts)
    schema_version: str = SCHEMA_VERSION

    # ---- (de)serialization ----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RunRecord":
        return RunRecord(
            run_id=d["run_id"],
            instance_id=d["instance_id"],
            repo=d["repo"],
            difficulty=d["difficulty"],
            condition=d["condition"],
            repeat_index=d["repeat_index"],
            prompt=PromptInfo(**d["prompt"]),
            run_meta=RunMeta(**d["run_meta"]),
            exit=ExitInfo(**d["exit"]),
            ask=AskInfo(
                asked=d["ask"]["asked"],
                n_questions=d["ask"]["n_questions"],
                first_ask_turn=d["ask"]["first_ask_turn"],
                questions=[Question(**q) for q in d["ask"]["questions"]],
            ),
            patch=PatchInfo(**d["patch"]),
            trajectory=Trajectory(
                **{k: v for k, v in d["trajectory"].items() if k != "tokens"},
                tokens=Tokens(**d["trajectory"]["tokens"]),
            ),
            evaluation=Evaluation(
                **{k: v for k, v in d["evaluation"].items()
                   if k not in ("fail_to_pass", "pass_to_pass", "localization")},
                fail_to_pass=TestBreakdown(**d["evaluation"]["fail_to_pass"]),
                pass_to_pass=TestBreakdown(**d["evaluation"]["pass_to_pass"]),
                localization=Localization(**d["evaluation"]["localization"]),
            ),
            annotations=Annotations(**d["annotations"]),
            artifacts=Artifacts(**d["artifacts"]),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


def make_run_id(instance_id: str, condition: str, repeat_index: int) -> str:
    return f"{instance_id}__{condition}__r{repeat_index:02d}"
