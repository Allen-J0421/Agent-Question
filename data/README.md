# Two datasets

This directory contains two separate datasets, both derived from the same 500-task
[SWE-bench Verified](https://www.swebench.com/) source benchmark. They serve different research
purposes and have different formats.

| Dataset | Location and format | Use it for |
|---|---|---|
| **1. Interactive SWE** | `interactive-swe/` — 15-column Hugging Face/Arrow dataset | Running software-agent experiments and evaluating patches under ambiguous versus full issue text. |
| **2. Missing Information** | `missing-info/data.xlsx` — 28-column Excel dataset | Measuring whether an agent identifies and asks for deliberately hidden categories of issue information. |

The datasets share task provenance, but neither should be treated as a replacement for the other:
Interactive SWE is the runnable benchmark input; Missing Information is the richer annotation and
clarification-analysis dataset.

## Layout

```
data/
├── README.md
├── interactive-swe/                  # Runnable 15-column Hugging Face dataset
│   ├── dataset_dict.json
│   └── test/
│       ├── data-00000-of-00001.arrow # 500 records
│       ├── dataset_info.json
│       └── state.json
└── missing-info/
    └── data.xlsx                      # 28-column annotation and baseline workbook
```

---

## Dataset 1 — `interactive-swe/`

`interactive-swe` is the dataset consumed by the experiment scripts. It presents every task in two
forms:

- **Ambiguous condition:** `problem_statement`, a short rewrite that removes reproductions,
  tracebacks/error text, and expected-versus-actual behavior.
- **Full-information condition:** `original_issue`, the real GitHub issue text.

This makes it possible to compare whether a software agent asks for clarification and whether the
missing context changes its ability to produce a correct patch.

### Dataset facts

| Property | Value |
|---|---|
| Format | Hugging Face `datasets`, saved to disk as Apache Arrow IPC |
| Split | `test` only |
| Rows | 500 |
| Columns | 15, all stored as strings |
| Source | SWE-bench Verified, across 12 Python repositories |
| Prompt size | `problem_statement`: average 377 chars; `original_issue`: average 1,700 chars |

### Fields

| Group | Fields | Purpose |
|---|---|---|
| Agent-facing issue text | `problem_statement`, `original_issue` | Use one of these as the task prompt, depending on the experimental condition. |
| Issue discussion | `hints_text` | Maintainer comments. Empty in 162 rows and generally hidden to prevent leakage. |
| Solution and test oracles | `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS` | Evaluate a proposed patch. `FAIL_TO_PASS` must pass after the fix; `PASS_TO_PASS` must remain passing. |
| File-localization oracle | `files` | Gold source paths edited by the fix. Empty in 26 rows. |
| Checkout and provenance | `repo`, `instance_id`, `base_commit`, `environment_setup_commit`, `version`, `created_at`, `difficulty` | Reproduce the task environment and stratify results. |

`base_commit` is the unresolved code revision. `patch` and `test_patch` are answer keys, not prompt
context. `FAIL_TO_PASS` and `PASS_TO_PASS` are JSON-encoded test lists and must be parsed before use.

### Load it

```python
from datasets import load_from_disk

ds = load_from_disk("data/interactive-swe")["test"]
example = ds[0]

ambiguous_prompt = example["problem_statement"]
full_prompt = example["original_issue"]
```

With only PyArrow:

```python
import pyarrow as pa
import pyarrow.ipc as ipc

path = "data/interactive-swe/test/data-00000-of-00001.arrow"
with pa.memory_map(path, "r") as src:
    table = ipc.open_stream(src).read_all()
```

### Use it

1. Check out `repo` at `base_commit`.
2. Give the agent `problem_statement` for the ambiguous condition or `original_issue` for the
   full-information control.
3. Record whether it asks a question before it edits, then capture its patch.
4. Evaluate the patch with the SWE-bench test oracles.

### Important caveats

- Django accounts for about 46% of rows, and 91% of tasks are estimated at one hour or less. Report
  repository- and difficulty-stratified results when possible.
- `problem_statement` is a rewrite, not a verbatim issue title.
- `files` is missing in 26 rows; derive paths from `patch` diff headers when needed.
- `hints_text` is missing in 162 rows and can leak withheld details when present.

---

## Dataset 2 — `missing-info/data.xlsx`

This 500-row, 28-column companion workbook explains what was hidden from each issue and supports
direct evaluation of clarification questions.

The harness runs it directly via `--dataset missing-info`, which pairs the conditions
`mi_ambiguous` (`rewrite_3`) and `mi_full` (`original issue`). See
[Running it through the harness](#running-it-through-the-harness) below.

### How one row is constructed

```text
complete original issue
  → annotate evidence across six information categories
  → identify the categories present in this issue
  → independently create three natural-sounding masked rewrites
  → store what was hidden as labels and detailed answer keys
  → collect baseline model questions for rewrite_3
```

The six categories are:

- Error Information
- Reproduction Steps
- Expected Behavior
- Version/Environment Information
- External References
- Implementation Details

### The three rewrite variants

For each variant `k` (`1`, `2`, or `3`), the pipeline normally selects **one to three** present
categories, removes their annotated evidence, and rewrites the remaining issue so it has no obvious
redaction gap.

The variants are independent samples, **not a partition**: a category can be hidden in more than one
variant. The entire hidden-category set happens to match in only 3–4 of the 500 rows, but that does
not mean categories are disjoint. The project uses variant 3; variants 1 and 2 are extra experimental
capacity.

| Field family | What it contains | Who may use it |
|---|---|---|
| `rewrite_1` / `_2` / `_3` | Natural-sounding, incomplete issue prompt | Agent and evaluator |
| `hidden_categories_1` / `_2` / `_3` | Comma-separated category labels for information withheld from the matching rewrite | Evaluator only |
| `hidden_info_1` / `_2` / `_3` | Detailed removed evidence, encoded as `Category: <probe question> \| Examples: <spans>` | Evaluator only |

`problem_statement` is a verbatim copy of `rewrite_3` in all 496 populated rows. Existing
`clarification_questions_*_3` columns were generated against rewrite 3, so score them only against
the matching `_3` answer key.

### Workbook schema

| Columns | Role | Fields |
|---|---|---|
| 0–12 | SWE-bench task provenance and solution oracles | `repo`, `instance_id`, `base_commit`, `patch`, `test_patch`, `original issue`, `hints_text`, `created_at`, `version`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `environment_setup_commit`, `difficulty` |
| 13–14 | Source annotation for masking | `category_mapping`, `present_categories` |
| 15–23 | Three masked prompts and their answer keys | `rewrite_1`–`rewrite_3`, `hidden_categories_1`–`hidden_categories_3`, `hidden_info_1`–`hidden_info_3` |
| 24–26 | Existing baseline questions for variant 3 | `clarification_questions_grpo_3`, `clarification_questions_gpt5_nano_3`, `clarification_questions_gpt5_3` |
| 27 | Harness alias | `problem_statement` |

Key fields:

| Field | Meaning and use |
|---|---|
| `category_mapping` | JSON source annotation: each category has `present` and verbatim `examples` spans. It is the source for the masking process. Four rows are empty objects. |
| `present_categories` | Compact comma-separated view of categories marked present. It is the intended candidate pool for masking. |
| `hidden_categories_k` | Compact scoring label: did the agent ask for the right *kind* of information? |
| `hidden_info_k` | Detailed answer key: could the agent’s question recover a specific fact that was removed? |
| `clarification_questions_*_3` | Pre-existing GRPO, GPT-5 nano, and GPT-5 baseline outputs. Use only for comparison, never as input to a new agent. |

### Recommended evaluation protocol

For a direct gap-identification test:

1. Use `rewrite_3` as the only issue text.
2. Ask the model to list clarification questions before proposing a solution.
3. Map each question to one or more of the six categories.
4. Compare the predicted categories with `hidden_categories_3` for category recall and coverage.
5. Inspect `hidden_info_3` to judge whether a question requests a specific withheld fact.

For a natural-behavior test, instead say only “Resolve this issue” and record whether the agent
spontaneously asks before editing. Keep this separate from the forced question-generation test.

The workbook’s existing baseline outputs have approximate category-level results:

| Model | Questions/row | Hidden-category recall | Full coverage | Questions about visible information |
|---|---:|---:|---:|---:|
| GRPO | 2.94 | 27.6% | 12.6% | 37.2% |
| GPT-5 nano | 5.26 | 53.6% | 32.1% | 47.4% |
| GPT-5 | 5.07 | 56.7% | 35.2% | 46.8% |

These scores come from a keyword classifier, not a semantic judge, and should be treated as
comparative baselines rather than exact quality measurements.

### Load it

```python
import json
import pandas as pd

df = pd.read_excel("data/missing-info/data.xlsx")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
df["version"] = df["version"].astype(str)

def split_categories(value):
    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}

df["hidden_cats_3"] = df["hidden_categories_3"].apply(split_categories)
evaluation_rows = df[df["hidden_cats_3"].apply(len) > 0]

def parse_tests(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
```

### Running it through the harness

`datasets_registry.py` normalizes both datasets onto one row schema, so the rest of the harness is
dataset-agnostic:

```bash
python experiment.py list  --dataset missing-info
python experiment.py run   <instance_id> --dataset missing-info --condition mi_ambiguous --dry-run
python experiment.py batch --dataset missing-info --condition mi_ambiguous --count 50
```

The conditions are named `mi_*` rather than reusing `ambiguous`/`full` because both datasets cover
the same 500 `instance_id`s; distinct names keep resume state, checkout directories, and every
aggregate report from conflating the two. Runs record their source in `task.dataset`, which also
appears as a `dataset` column in `reports/run-summary.csv`.

Three normalizations happen at load time:

- `original issue` is renamed to `original_issue` so one condition table serves both datasets.
- The 26 truncated `PASS_TO_PASS` cells and the Excel-mangled `version` floats are repaired from
  `interactive-swe`, which covers the same instances and agrees byte for byte on every shared
  oracle column. Without the first repair those instances would be graded against an empty
  regression suite and reported as resolved.
- Every evaluator-only column is **stripped from the row entirely** — `hidden_categories_*`,
  `hidden_info_*`, `category_mapping`, `present_categories`, the `clarification_questions_*`
  baselines, the unused `rewrite_1`/`rewrite_2`, and `hints_text`. They are reachable only through
  `datasets_registry.load_answer_keys()`, which nothing in the run path imports. The agent-facing
  prompt is unchanged from the other dataset: `"Resolve the following issue in this repository:"`
  followed by exactly one issue field.

Note that four instances have no `rewrite_3` and are skipped under `mi_ambiguous`, leaving **496
runnable**; a further 13 have no `hidden_categories_3`, leaving **483 scoreable** for category
recall.

### Workbook caveats

- **Excel truncation:** `PASS_TO_PASS` is invalid/truncated in 26 rows because Excel cells reached
  their 32,767-character limit.
- **Incomplete annotations:** four rows have no downstream annotation. Another 17 have empty
  `hidden_categories_3`; exclude both groups from category-recall evaluation, leaving 479 usable
  variant-3 rows.
- **Implementation Details label bug:** `hidden_categories_k` under-reports this category. In 174
  variant-3 rows, the corresponding `hidden_info_3` segment has an empty category name; only 36 of
  those rows name it in `hidden_categories_3`. Recover this category from `hidden_info_k` before
  reporting metrics that involve it.
- **Correlated variants:** do not put variants of the same `instance_id` into different training and
  test splits.

## Provenance

Both datasets are derived from SWE-bench Verified (Jimenez et al., *SWE-bench: Can Language Models
Resolve Real-World GitHub Issues?*). `interactive-swe` adds the ambiguous `problem_statement`,
full `original_issue`, and `files` fields. The workbook adds the information-category annotations,
rewrite variants, and clarification-question baselines.
