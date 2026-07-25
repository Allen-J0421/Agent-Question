# `interactive-swe` Dataset

A benchmark for studying **ambiguity handling and clarification behavior in software-engineering agents**. It is a modified derivative of **SWE-bench Verified** (the same 500 human-validated instances across 12 Python repositories), augmented so that each task can be presented to an agent in an *under-specified* form while the full ground-truth context is retained for evaluation.

The core idea: the normal, detail-rich GitHub issue is preserved verbatim in `original_issue`, while `problem_statement` holds a **short, paraphrased, deliberately vague summary** with all reproduction code, error strings, and expected/actual output removed. This lets you measure whether an agent recognizes missing information and asks clarifying questions — the "interactive" setting — versus how it performs given the complete spec.

---

## At a glance

| | |
|---|---|
| **Format** | HuggingFace `datasets`, saved to disk (Apache Arrow IPC) |
| **Splits** | `test` only |
| **Examples** | 500 |
| **Columns** | 15 (all `string`) |
| **On-disk size** | ~8 MB (`data-00000-of-00001.arrow`) |
| **Repositories** | 12 (Python OSS projects) |
| **Issue date range** | 2013-01-25 → 2023-08-07 |
| **Base benchmark** | SWE-bench Verified |
| **`instance_id`** | Unique for all 500; format `repo__name-<number>` |

### Directory layout

```
data/
├── README.md                        ← this file
└── interactive-swe/
    ├── dataset_dict.json            ← {"splits": ["test"]}
    └── test/
        ├── data-00000-of-00001.arrow   ← the 500 records
        ├── dataset_info.json           ← HF feature schema + stats
        └── state.json                  ← HF split state / fingerprint
```

---

## Schema

All 15 fields are stored as strings. Fields marked **★** are additions/modifications relative to standard SWE-bench; the rest are inherited unchanged.

### Task specification

| Column | Description |
|---|---|
| `problem_statement` **★** | **Rewritten, ambiguous task description.** Short prose summary of the issue (avg **377** chars, max 810). Reproduction code, tracebacks, and expected/actual output are **stripped** — 0/500 contain any code block or REPL snippet. Shorter than `original_issue` in 489/500 cases. This is the intended agent-facing prompt for the ambiguous condition. |
| `original_issue` **★** | **The real GitHub issue text**, verbatim (avg **1,700** chars, max 24,770). 256/500 contain a code repro or traceback. Use as the "full-information" control condition or as the ground-truth source of the details missing from `problem_statement`. |
| `hints_text` | Maintainer/discussion comments from the issue thread. **Empty for 162/500** instances. |

### Ground truth / solution

| Column | Description |
|---|---|
| `patch` | Gold code diff that fixes the issue (unified diff, avg 1.6 KB). Touches **1 file in 429/500** instances; the rest touch 2–6 (one outlier: 21). |
| `test_patch` | Diff adding/modifying tests that validate the fix (avg 1.8 KB). |
| `files` **★** | Gold file path(s) the fix edits (comma/newline-separated string, e.g. `astropy/modeling/separable.py`). Convenient localization ground truth. **Empty for 26/500** instances — see Caveats. |
| `FAIL_TO_PASS` | JSON-encoded list of test node IDs that must flip **fail → pass** after the fix (avg 3.0 tests, up to 438). Primary success oracle. |
| `PASS_TO_PASS` | JSON-encoded list of regression tests that must **stay passing** (avg 120 tests, up to 2,476). |

### Environment / provenance

| Column | Description |
|---|---|
| `repo` | GitHub `owner/name` of the source repository. |
| `instance_id` | Unique task ID, `repo__name-<issue/PR number>` (e.g. `astropy__astropy-12907`). |
| `base_commit` | 40-char SHA the agent should start from (issue is unresolved at this commit). |
| `environment_setup_commit` | 40-char SHA pinning the dependency/build environment. |
| `version` | Project release line the instance targets (e.g. `4.3`, `3.2`). |
| `created_at` | ISO-8601 timestamp of the original issue. |
| `difficulty` | Human effort estimate: `<15 min fix`, `15 min - 1 hour`, `1-4 hours`, `>4 hours`. |

---

## Distributions

### Repositories (× difficulty)

| repo | <15 min | 15 min–1 hr | 1–4 hr | >4 hr | **total** |
|---|--:|--:|--:|--:|--:|
| django/django | 92 | 117 | 22 | 0 | **231** |
| sympy/sympy | 25 | 43 | 6 | 1 | **75** |
| sphinx-doc/sphinx | 22 | 17 | 4 | 1 | **44** |
| matplotlib/matplotlib | 15 | 19 | 0 | 0 | **34** |
| scikit-learn/scikit-learn | 13 | 18 | 1 | 0 | **32** |
| astropy/astropy | 4 | 15 | 3 | 0 | **22** |
| pydata/xarray | 5 | 15 | 1 | 1 | **22** |
| pytest-dev/pytest | 8 | 8 | 3 | 0 | **19** |
| pylint-dev/pylint | 3 | 5 | 2 | 0 | **10** |
| psf/requests | 6 | 2 | 0 | 0 | **8** |
| mwaskom/seaborn | 0 | 2 | 0 | 0 | **2** |
| pallets/flask | 1 | 0 | 0 | 0 | **1** |
| **total** | **194** | **261** | **42** | **3** | **500** |

> ⚠️ **Django dominates (~46%)** and difficulty skews easy/medium (91% ≤ 1 hour). Control for repository and difficulty when reporting aggregate results.

---

## Loading

**With `datasets` (recommended):**
```python
from datasets import load_from_disk

ds = load_from_disk("data/interactive-swe")["test"]
print(ds)                     # 500 rows, 15 columns
ex = ds[0]
print(ex["problem_statement"])   # ambiguous prompt
print(ex["original_issue"])      # full issue
```

**With `pyarrow` only (no `datasets` install):**
```python
import pyarrow as pa, pyarrow.ipc as ipc

with pa.memory_map("data/interactive-swe/test/data-00000-of-00001.arrow", "r") as src:
    table = ipc.open_stream(src).read_all()
df = table.to_pandas()
```

**Parsing the test lists** (`FAIL_TO_PASS` / `PASS_TO_PASS` are JSON strings):
```python
import json
f2p = json.loads(ex["FAIL_TO_PASS"])   # -> ["path::test_a", "path::test_b", ...]
```

---

## Example record (`astropy__astropy-12907`)

**`problem_statement`** (ambiguous, agent-facing):
> The issue involves the incorrect computation of the separability matrix for nested compound models in a modeling library. While the separability matrix is computed correctly for simple and non-nested compound models, the expected separability is not maintained when models are nested, leading to unexpected results...

**`original_issue`** (full, with repro — truncated):
> `separability_matrix` does not compute separability correctly for nested CompoundModels
> ```python
> from astropy.modeling import models as m
> cm = m.Linear1D(10) & m.Linear1D(5)
> separability_matrix(m.Pix2Sky_TAN() & cm)
> # -> off-diagonal True where it should be False
> ```

**Ground truth:** `files = astropy/modeling/separable.py` · `patch` edits `_cstack` · `FAIL_TO_PASS` = 2 tests in `test_separable.py`.

---

## Suggested usage

- **Ambiguous vs. full condition** — prompt agents with `problem_statement` (vague) and compare against `original_issue` (complete) to isolate the cost of missing information.
- **Clarification behavior** — in the interactive setting, treat `original_issue` (and `hints_text`) as the "oracle" an agent can query, and measure whether a clarifying question recovers the detail needed to solve the task.
- **Localization** — `files` provides a cheap, run-free ground truth for file-level retrieval accuracy.
- **Resolution** — apply the agent's patch at `base_commit`, run the `test_patch`, and check that all `FAIL_TO_PASS` pass and all `PASS_TO_PASS` still pass (standard SWE-bench evaluation harnesses apply directly).

---

## Caveats

- **26 instances have an empty `files` field** even though their `patch` edits files (annotation gap). Derive file paths from the `patch` diff headers as a fallback.
- **162 instances have empty `hints_text`** — don't assume hints are always available.
- **Repository & difficulty imbalance** (Django ~46%; 91% of tasks ≤ 1 hour) — stratify or reweight for fair aggregate reporting.
- `problem_statement` is a machine/human paraphrase, not the literal issue title; do not treat it as verbatim source text.
- All columns are typed `string`, including the JSON-encoded `FAIL_TO_PASS` / `PASS_TO_PASS` and the multi-valued `files` — parse before use.

---

## Provenance

Derived from **SWE-bench Verified** (Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*), which itself samples human-validated instances from 12 popular Python repositories. The `original_issue`, `problem_statement` (rewritten), and `files` fields were added/modified for the ambiguity/interactive study in this project (`ambig-SWE`). All standard SWE-bench evaluation fields are preserved.
