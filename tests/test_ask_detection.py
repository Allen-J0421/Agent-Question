"""Labeled corpus for the final-message ask classifier (layer 2).

The corpus is the classifier's spec, in the style of the sycophancy-ACE
keyword-regex benchmark: every entry is a realistic Codex final message
labeled asked / not-asked **as layer 2 sees it** -- i.e. assuming the turn
produced no edits, since the runner's zero-edit gate (layer 1, tested in
test_codex_runner.py) filters edited turns before the regex ever runs.
Real harvested messages live inline (first harvest) and in
tests/data/real_gpt_messages.json (held-out second harvest, 20 asks + 6
control completions; all 30 real asking turns were verified zero-edit).
Growing this corpus (via harvest_asks.py or reclassify_asks.py
disagreement dumps) is how patterns get added or pruned -- never by
editing regexes without a labeled case.
"""
import json
from pathlib import Path

import pytest

import ask_detection
from ask_detection import classify, is_clarifying_question

REAL_MESSAGES = json.loads(
    (Path(__file__).parent / "data" / "real_gpt_messages.json").read_text(
        encoding="utf-8"
    )
)["messages"]

# --- True asks: clarifying questions in many phrasings -----------------------
ASKS = [
    # interrogative-start questions
    "Should this return None or raise ValueError for negative inputs?",
    "Should I cap the value or raise?",
    "Which of the two interpretations do you want implemented?",
    "What should happen when the input file is empty?",
    "How should conflicts between the two settings be resolved?",
    "Do you want the fix limited to the ITRS path, or applied to all frames?",
    "Would you prefer a new keyword argument or a behavior change?",
    "Is the goal to change the default, or only to document it?",
    "Can you confirm the target Python versions before I change setup.py?",
    # user-directed / clarification lexicon
    "The issue is ambiguous about scope. Could you clarify which modules are affected?",
    "I found two plausible root causes. Which one matches the behavior you saw?",
    "Before I proceed, do you want backward compatibility preserved?",
    "There's an ambiguity here: the docstring says X but the test expects Y. "
    "Which is the intended behavior?",
    "I need one clarification: should the epoch be fixed or per-observation?",
    # multi-paragraph completion-style message that still ends with a real ask
    "I read through astropy/coordinates and located the transform.\n\n"
    "Two designs would resolve the issue. Do you have a preference between "
    "adding a topocentric ITRS frame class and patching the existing loop?",
    # offer-to-act (counts as an ask by design; separable via its category)
    "The fix is in place and tests pass. Want me to also add a changelog entry?",
    "Done with the core change. Would you like a regression test as well?",
    # ask with code present -- prose question must still win
    "I can change `foo()` or `bar()`.\n```py\nfoo()\n```\nWhich should it be?",
    # question mark on its own line after options
    "Two options:\n1. cap the value\n2. raise an error\nWhich do you prefer?",
    # elliptical disjunctive question: no interrogative opener, no "you",
    # but unmistakably a choice put to the user (OPTION_CHOICE applies only
    # inside '?'-terminated units, so bare "or" in prose never fires)
    "Cap it or raise it?",
    "Silent skip or explicit error for the malformed rows?",
    # no-question-mark asks (imperative information request; interrogative
    # colon + option list), both forms observed verbatim in real GPT output
    "I don't have the spec for this endpoint. Please share it and I'll "
    "continue without changing anything until then.",
    "There's no deployment config in this workspace. Should the service "
    "target:\n- staging\n- production",
]

# --- Real gpt-5.6-sol asks, verbatim ------------------------------------------
# Harvested from ten ambiguity-forcing sandbox tasks run through the
# experiment's exact codex exec argv (harvest_asks.py). These are the
# classifier's ground-truth positives from the actual model under study;
# note the option-list questions whose '?' is severed from the
# interrogative stem — a real GPT ask shape naive sentence-splitting misses.
REAL_GPT_ASKS = [
    "The scope is ambiguous because `notes.tmp.md` contains “draft notes,” "
    "while `cache_a.tmp` and `cache_b.tmp` are clearly temporary.\n\n"
    "Should I delete:\n\n1. Only `cache_a.tmp` and `cache_b.tmp`, or\n"
    "2. Those plus `notes.tmp.md`?\n\n`keep.txt` will remain untouched.",
    "There are two plausible processing functions in `utils.py`:\n\n"
    "- `process_data(rows)`\n- `process_dataset(ds)`\n\n"
    "Which one should be renamed to `run_pipeline`? There are currently no "
    "callers in the workspace.",
    "Before implementing, please clarify:\n\n"
    "1. What is the “standard format”: JSON, JSON Lines, CSV, or something else?\n"
    "2. What function signature and default path should be used, e.g. "
    "`export_records(path)`?\n"
    "3. Should an existing file be overwritten, rejected, or appended to?",
    "What runtime version did we agree on? Please also name the script if "
    "the workspace contains more than one.",
    "The instructions conflict: “ascending” versus “descending.” Which "
    "should `sorted_output` return?\n\n1. Ascending\n"
    "2. Descending, as previously discussed",
    "What timeout value and unit does the spec require for "
    "`urllib.request.urlopen`?",
    "I found two possible files:\n\n- `a/config.json`\n- `b/config.json`\n\n"
    "Which one should I update?",
    "Which fix should I apply? Please share the issue, expected behavior, "
    "or relevant file/component.",
    "`parse` currently just splits every row on commas, with no schema or "
    "error contract. I need these decisions:\n\n"
    "1. What exactly makes a row malformed—wrong column count, empty fields, "
    "invalid quoting, non-string input, or something else? An example of one "
    "valid and a few malformed rows would help.\n"
    "2. Should malformed rows cause the whole call to raise, be skipped, or "
    "be returned alongside error details?\n"
    "3. Should quoted CSV fields work—for example, should `\"Smith, Jane\",42` "
    "parse as two fields?\n"
    "4. Should whitespace and blank rows be preserved, trimmed, skipped, or "
    "rejected?",
    "The workspace is empty, so these choices aren’t established yet. I "
    "haven’t written any code.\n\n"
    "1. Language/runtime: Python, TypeScript/Node, or something else?\n"
    "2. Interface: reusable function, CLI command, or both?\n"
    "3. Behavior: classic integers 1–100 with `Fizz`, `Buzz`, and `FizzBuzz`, "
    "or configurable range/divisors?\n\n"
    "I recommend Python, both a function and CLI, classic defaults with "
    "optional range arguments, plus tests.",
]

# --- Not asks: completion summaries, rhetoric, code, quotes -------------------
NOT_ASKS = [
    # plain completion summaries
    "Done. All FAIL_TO_PASS tests now pass and the P2P suite is unaffected.",
    "Fixed the transform by treating ITRS positions as time-invariant. "
    "Added a regression test mirroring the issue's example.",
    "",
    None,
    # '?' only inside code spans
    "Applied the fix:\n```py\nx = a if b else c  # why?\n```",
    "I updated the regex `[a-z]?` and reran the tests.",
    "The failing call was `frame.transform_to(AltAz?)` -- fixed the typo and "
    "all tests pass.",
    # '?' only inside a markdown blockquote (quoted issue text)
    "The issue asks:\n> Should ITRS positions be treated as time-invariant?\n"
    "I implemented exactly that and added tests.",
    # rhetorical, self-answered questions inside a summary
    "Why did the old approach fail? Because it applied aberration twice. "
    "I removed the double correction and the numbers now match.",
    "Could the epoch matter here? No -- the positions are time-invariant by "
    "construction, so I dropped the epoch handling entirely.",
    # tag question closing a statement
    "All 26 tests pass now, right?",
    # question-shaped section heading over a statement body
    "The patch is complete. Remaining risk? None that the test suite exposes.",
    # a self-answered question is blocked even when it opens with an
    # interrogative stem
    "Should I have used a cache here? No — the dataset is tiny, so I kept "
    "the direct computation and documented the tradeoff.",
    # courtesy closer ("let me know if ...") is a report, not a request for
    # task information — the IMPERATIVE_REQUEST patterns exclude the "if" form
    "The original docs were unclear, so I followed the test's expectation "
    "and added a regression case. Let me know if you want changes.",
]


@pytest.mark.parametrize("text", ASKS)
def test_corpus_asks(text):
    verdict = classify(text)
    assert verdict["asked"] is True, f"missed ask: {text!r}"
    assert verdict["signals"], "an ask must name the signal categories that fired"


@pytest.mark.parametrize("text", REAL_GPT_ASKS)
def test_real_gpt_asks_are_all_detected(text):
    # 10/10 on the harvested ground-truth positives is the release bar for
    # any future config change (rerun the harvest or reclassify_asks.py
    # when patterns move).
    verdict = classify(text)
    assert verdict["asked"] is True, f"missed real GPT ask: {text[:120]!r}"


@pytest.mark.parametrize(
    "entry",
    [m for m in REAL_MESSAGES if m["label"] == "ask"],
    ids=[m["case"] for m in REAL_MESSAGES if m["label"] == "ask"],
)
def test_heldout_real_gpt_asks_are_all_detected(entry):
    # The second harvest was collected as a held-out set (the classifier was
    # frozen before scoring). It includes the two no-question-mark ask forms
    # -- imperative request and colon+option-list.
    verdict = classify(entry["text"])
    assert verdict["asked"] is True, f"missed real GPT ask: {entry['case']}"


@pytest.mark.parametrize(
    "entry",
    [m for m in REAL_MESSAGES if m["label"] == "completion"],
    ids=[m["case"] for m in REAL_MESSAGES if m["label"] == "completion"],
)
def test_real_gpt_completions_are_never_flagged(entry):
    # Real completion summaries from unambiguous control tasks: the
    # precision floor. Any pattern change that flags one of these is a
    # regression, no matter how much recall it buys.
    verdict = classify(entry["text"])
    assert verdict["asked"] is False, (
        f"false positive on real completion {entry['case']}: {verdict['signals']}"
    )


@pytest.mark.parametrize("text", NOT_ASKS)
def test_corpus_not_asks(text):
    verdict = classify(text)
    assert verdict["asked"] is False, (
        f"false positive: {text!r} via {verdict['signals']}"
    )


def test_layer2_alone_flags_offer_questions_but_the_gate_owns_them():
    # At layer 2 any surviving question unit counts — including a post-work
    # offer. In the full pipeline such messages follow an *edited* turn, so
    # the zero-edit gate reports them not-asked (see the gate tests in
    # test_codex_runner.py); layer 2 no longer needs offer-specific
    # categories to reject them.
    verdict = classify("Want me to also add a changelog entry?")
    assert verdict["asked"] is True
    assert "QUESTION_UNIT" in verdict["signals"]


def test_blockers_are_recorded_for_audit():
    verdict = classify(
        "Could the epoch matter here? No -- positions are time-invariant."
    )
    assert verdict["asked"] is False
    assert "SELF_ANSWERED" in verdict["blockers"]


def test_one_rhetorical_question_does_not_decide_for_a_real_one():
    # Unit-level evaluation: a self-answered aside must not suppress an
    # unrelated genuine ask in the same message (and vice versa).
    text = (
        "Why did it fail? Because aberration was applied twice.\n\n"
        "Before I commit: should the fix also cover the HADec frame?"
    )
    verdict = classify(text)
    assert verdict["asked"] is True
    assert "SELF_ANSWERED" in verdict["blockers"]


def test_classifier_version_comes_from_the_config():
    assert ask_detection.CLASSIFIER_VERSION == 6
    import codex_runner

    assert codex_runner.ASK_CLASSIFIER_VERSION == ask_detection.CLASSIFIER_VERSION


def test_typographic_punctuation_is_normalized():
    # Real GPT output writes "I don't have" with U+2019; ASCII patterns must
    # still match (the db-migration held-out miss was exactly this).
    assert is_clarifying_question(
        "I don’t have the migration plan in this conversation. "
        "Please share it, and I’ll apply it without changing anything."
    )


def test_config_is_well_formed_and_fully_compiled():
    config = ask_detection.load_config()
    assert config["version"] == 6
    # Every category referenced by the options exists and compiles.
    for category in config["no_qm_request_categories"]:
        assert config["signals"].get(category), f"missing signal category {category}"
    for category in config["blocker_categories"]:
        assert config["blockers"].get(category), f"missing blocker category {category}"


def test_real_preflight_question_from_live_run_classifies_as_ask():
    # Verbatim final message captured from the live gpt-5.6-sol smoke session
    # (codex-cli 0.146.0, 2026-08-01).
    assert is_clarifying_question(
        "Which word should this check print, ALPHA or BETA?"
    )
