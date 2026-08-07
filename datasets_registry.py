"""Dataset loading and schema normalization for the two study datasets.

The study runs over two datasets that describe the same 500 SWE-bench Verified
tasks but store them differently:

``interactive-swe``
    A Hugging Face/Arrow dataset. ``problem_statement`` is a short ambiguous
    rewrite; ``original_issue`` is the real GitHub issue.

``missing-info``
    A 28-column Excel workbook. Its ambiguous text (``rewrite_3``) was built by
    annotating six categories of issue information, deliberately hiding one to
    three of them, and rewriting the issue so no redaction gap shows. Crucially
    it also ships the *answer key* for what was hidden, which makes it a richer
    ask-evaluation instrument -- and a leakage hazard the Arrow dataset does not
    have.

This module normalizes both into one canonical row schema so the rest of the
harness stays dataset-agnostic. It is the dataset-side analogue of
``study_log.agent_info``, which normalizes runner identity the same way.

Nothing here changes what the agent is given: the prompt is still exactly one
issue field (``experiment.build_prompt``). The firewall below is defense in
depth so that a future edit cannot accidentally route an answer key into a
prompt.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

INTERACTIVE_SWE = "interactive-swe"
MISSING_INFO = "missing-info"
DEFAULT_DATASET = INTERACTIVE_SWE
DATASETS = (INTERACTIVE_SWE, MISSING_INFO)

ARROW_PATH = ROOT / "data" / INTERACTIVE_SWE
WORKBOOK_PATH = ROOT / "data" / MISSING_INFO / "data.xlsx"

# The workbook stores the full issue under a column name containing a space.
# Renaming it to the Arrow spelling is what lets one `CONDITION_FIELD` table
# serve both datasets.
WORKBOOK_FULL_ISSUE_COLUMN = "original issue"

# The one prompt-bearing field each (dataset, condition) pair reads. Keep this
# in sync with ``experiment.CONDITION_FIELD``.
DATASET_CONDITIONS = {
    INTERACTIVE_SWE: ("ambiguous", "full"),
    MISSING_INFO: ("mi_ambiguous", "mi_full"),
}

# Fields that may legitimately reach a prompt. Everything an agent ever sees
# comes from exactly one of these.
AGENT_VISIBLE_FIELDS = frozenset({"problem_statement", "original_issue", "rewrite_3"})

# Fields the grader needs *after* the session has ended and the workspace has
# been reset. They are kept on the row but are never prompt-reachable.
GRADER_FIELDS = frozenset({"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"})

# Provenance needed to check out the task and stratify results.
PROVENANCE_FIELDS = frozenset(
    {
        "repo",
        "instance_id",
        "base_commit",
        "environment_setup_commit",
        "version",
        "created_at",
        "difficulty",
        "files",
    }
)

# The masking answer key. These columns say precisely which information was
# withheld from ``rewrite_3`` -- i.e. the answer to the exact question this
# study asks the agent. They are stripped from every row that
# ``load_dataset_rows`` returns and are reachable only through
# ``load_answer_keys``, which nothing in the run path may import.
ANSWER_KEY_FIELDS = frozenset(
    {
        "category_mapping",
        "present_categories",
        "hidden_categories_1",
        "hidden_categories_2",
        "hidden_categories_3",
        "hidden_info_1",
        "hidden_info_2",
        "hidden_info_3",
        "clarification_questions_grpo_3",
        "clarification_questions_gpt5_nano_3",
        "clarification_questions_gpt5_3",
        # Unused ambiguous variants. The study uses variant 3; leaving 1 and 2
        # on the row would put alternate phrasings of the same task -- each
        # revealing different withheld details -- one attribute access away
        # from a prompt.
        "rewrite_1",
        "rewrite_2",
        # Maintainer discussion. Never read by any consumer and documented as
        # able to leak withheld details.
        "hints_text",
    }
)

KEPT_FIELDS = AGENT_VISIBLE_FIELDS | GRADER_FIELDS | PROVENANCE_FIELDS

# The six information categories the workbook annotates.
INFORMATION_CATEGORIES = (
    "Error Information",
    "Reproduction Steps",
    "Expected Behavior",
    "Version/Environment Information",
    "External References",
    "Implementation Details",
)


def conditions_for(dataset: str) -> tuple[str, ...]:
    """Return the conditions that are meaningful for ``dataset``."""
    try:
        return DATASET_CONDITIONS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}: expected one of {', '.join(DATASETS)}"
        ) from None


def dataset_for_condition(condition: str) -> str | None:
    """Return the dataset a condition belongs to, or ``None`` if unknown."""
    for dataset, conditions in DATASET_CONDITIONS.items():
        if condition in conditions:
            return dataset
    return None


def _strip_answer_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Drop every evaluator-only column, keeping the canonical schema."""
    return {key: value for key, value in row.items() if key in KEPT_FIELDS}


@lru_cache(maxsize=1)
def _load_arrow_rows() -> dict[str, dict[str, Any]]:
    from datasets import load_from_disk

    split = load_from_disk(str(ARROW_PATH))["test"]
    return {split[i]["instance_id"]: dict(split[i]) for i in range(len(split))}


@lru_cache(maxsize=1)
def _load_workbook() -> list[dict[str, Any]]:
    """Read the workbook verbatim, as strings, with no normalization.

    Blank cells must become ``None`` rather than ``float('nan')``: downstream
    code treats issue text as a string, and a bare ``nan`` would survive an
    ``or ""`` guard and reach ``.strip()``.
    """
    import pandas as pd

    frame = pd.read_excel(WORKBOOK_PATH, dtype=str)
    records = frame.to_dict(orient="records")
    return [
        {key: (value if isinstance(value, str) else None) for key, value in record.items()}
        for record in records
    ]


def _normalize_workbook_row(
    raw: dict[str, Any], arrow: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Map one workbook record onto the canonical Arrow-style schema.

    Two columns arrive damaged by Excel and are repaired from the Arrow
    dataset, which covers the same 500 ``instance_id``s and agrees byte for
    byte on every shared oracle field:

    ``PASS_TO_PASS``
        26 rows hit Excel's 32,767-character cell limit and are truncated to
        invalid JSON. ``swebench_eval.parse_node_ids`` swallows that and
        returns ``[]``, which would grade those instances against an *empty*
        regression suite and report them resolved. This is the single most
        dangerous silent failure in the integration. Note that a truncated cell
        is not the same as an empty one: 11 instances legitimately carry ``[]``
        in both datasets, so the repair triggers on invalid JSON only.

    ``version``
        Excel coerced version strings to floats ("5.0" -> 5, "5.1" ->
        5.0999999999999996) in 215 rows. The official harness keys
        environments off ``instance_id``, so this never reaches grading, but a
        corrupted value should not be recorded either.
    """
    row = dict(raw)
    row["original_issue"] = row.pop(WORKBOOK_FULL_ISSUE_COLUMN, None)

    reference = arrow.get(row.get("instance_id", ""))
    if reference is not None:
        if not _parses_to_list(row.get("PASS_TO_PASS")):
            row["PASS_TO_PASS"] = reference["PASS_TO_PASS"]
        row["version"] = str(reference["version"])

    return _strip_answer_keys(row)


def _parses_to_list(value: Any) -> bool:
    try:
        return isinstance(json.loads(value or ""), list)
    except (json.JSONDecodeError, TypeError):
        return False


@lru_cache(maxsize=len(DATASETS))
def load_dataset_rows(dataset: str = DEFAULT_DATASET) -> dict[str, dict[str, Any]]:
    """Return ``{instance_id: row}`` in the canonical schema, answer keys removed.

    Every row is safe to hand to prompt construction, workspace preparation and
    the grader. The masking answer keys are not present on these rows at all.
    """
    if dataset == INTERACTIVE_SWE:
        return {
            instance_id: _strip_answer_keys(row)
            for instance_id, row in _load_arrow_rows().items()
        }
    if dataset == MISSING_INFO:
        arrow = _load_arrow_rows()
        rows: dict[str, dict[str, Any]] = {}
        for raw in _load_workbook():
            row = _normalize_workbook_row(raw, arrow)
            rows[row["instance_id"]] = row
        return rows
    raise ValueError(
        f"unknown dataset {dataset!r}: expected one of {', '.join(DATASETS)}"
    )


def split_categories(value: Any) -> list[str]:
    """Parse a ``hidden_categories_k`` cell into category labels.

    Several cells carry a trailing comma ("Error Information, Reproduction
    Steps, "), which yields an empty final segment; 121 such artifacts exist
    across variant 3. Empty segments are dropped.
    """
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_answer_keys(
    dataset: str = MISSING_INFO, variant: int = 3
) -> dict[str, dict[str, Any]]:
    """Return the masking answer key for scoring clarification questions.

    EVALUATOR ONLY. Never import this from the run path -- these fields state
    exactly which information was withheld from the prompt.

    ``hidden_categories`` is the compact label ("did the agent ask for the
    right *kind* of information?"); ``hidden_info`` is the detailed key ("could
    the agent's question recover a specific removed fact?").

    Note that ``hidden_categories_k`` under-reports Implementation Details; the
    matching ``hidden_info_k`` segment is authoritative when the two disagree.
    """
    if dataset != MISSING_INFO:
        raise ValueError(f"{dataset!r} carries no masking answer key")
    if variant not in (1, 2, 3):
        raise ValueError(f"unknown rewrite variant {variant!r}: expected 1, 2 or 3")

    keys: dict[str, dict[str, Any]] = {}
    for raw in _load_workbook():
        keys[raw["instance_id"]] = {
            "instance_id": raw["instance_id"],
            "hidden_categories": split_categories(raw.get(f"hidden_categories_{variant}")),
            "hidden_info": raw.get(f"hidden_info_{variant}"),
            "present_categories": split_categories(raw.get("present_categories")),
            "category_mapping": raw.get("category_mapping"),
            "baselines": {
                "grpo": raw.get("clarification_questions_grpo_3"),
                "gpt5_nano": raw.get("clarification_questions_gpt5_nano_3"),
                "gpt5": raw.get("clarification_questions_gpt5_3"),
            },
        }
    return keys
