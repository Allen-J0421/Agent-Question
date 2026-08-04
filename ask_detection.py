"""Detect user-directed clarifying questions in Codex final messages.

This is **layer 2** of the GPT arm's two-layer ask detector (v5), with the
architecture adopted from the sycophancy-ACE study's refusal/clarification
detector (``refusal_analysis/keyword_regex.py`` +
``experiment_runner/clarification.py``):

* **Layer 1 -- zero-edit turn gate (in the runner, not here).** Empirically,
  asking and editing are mutually exclusive per turn: across 36 harvested
  real gpt-5.6-sol turns, all 30 asking turns changed nothing (0
  ``file_change`` items, workspaces byte-identical) and all 6 completing
  turns edited. The runner therefore reports an edited turn as not-asked
  regardless of its text (recording ``question_with_edits`` if the regex
  would have fired, so the assumption stays auditable), and only zero-edit
  turns reach this module. This mirrors sycophancy-ACE's
  ``_layer1_zero_change`` structural gate.
* **Layer 2 -- this module.** With completion summaries gated out, the
  keyword layer only separates zero-edit *asks* from zero-edit *reports*
  (refusals, blocked-status messages, diagnoses), so it is deliberately
  small: after normalization and code/quote stripping, any ``?``-terminated
  question unit counts unless a **blocker** fires (tag questions,
  rhetorical self-answered questions); a message with no question mark
  counts only via an explicit information request (``IMPERATIVE_REQUEST``)
  or an interrogative colon-plus-option-list (``COLON_OPTION_LIST``).
* **Config-driven and versioned** (``config/ask_detection.json``): patterns
  live in a category dictionary with provenance and a changelog in
  ``meta``; every verdict returns its matched categories for per-pattern
  ablation over stored runs (``reclassify_asks.py``), and the labeled
  corpus in ``tests/test_ask_detection.py`` is the spec -- patterns change
  only against labeled cases.

The classifier stays deliberately deterministic (no LLM judge): the study's
primary outcome must be reproducible from stored artifacts alone.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "ask_detection.json"

FLAG_MAP = {"IGNORECASE": re.I, "MULTILINE": re.M, "DOTALL": re.S, "VERBOSE": re.X}

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_BLOCKQUOTE = re.compile(r"^\s*>.*$", re.MULTILINE)
_QUESTION_UNIT = re.compile(r"[^.!?]*\?")

# GPT final messages use typographic punctuation ("I don't have" with
# U+2019); patterns are written in ASCII, so the text is normalized once
# here instead of every pattern carrying Unicode variants.
_PUNCT_NORMALIZATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

# Characters of following context a SELF_ANSWERED blocker may look at: just
# enough to see "? No -- ..." / "? Because ...", not the next paragraph.
FOLLOWING_CONTEXT_CHARS = 60


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and compile the pattern dictionary; returns a ready-to-use bundle."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    flags = 0
    for flag_name in config.get("options", {}).get("flags", ["IGNORECASE"]):
        flags |= FLAG_MAP[flag_name]

    def compile_map(section: dict[str, list[str]]) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
        return {
            category: [(pattern, re.compile(pattern, flags)) for pattern in patterns]
            for category, patterns in section.items()
        }

    options = config["options"]
    return {
        "version": config["meta"]["version"],
        "meta": config["meta"],
        "options": options,
        "signals": compile_map(config["signals"]),
        "blockers": compile_map(config["blockers"]),
        "blocker_categories": list(options["blocker_categories"]),
        "no_qm_request_categories": list(options.get("no_qm_request_categories", [])),
    }


_DEFAULT_CONFIG: dict[str, Any] | None = None


def default_config() -> dict[str, Any]:
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = load_config()
    return _DEFAULT_CONFIG


CLASSIFIER_VERSION = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["meta"]["version"]


def strip_nonprose(text: str) -> str:
    """Remove the spans whose question marks are never the agent asking.

    Fenced code, inline code, and markdown blockquotes (quoted issue text),
    plus typographic-punctuation normalization so ASCII patterns match real
    GPT output.
    """
    stripped = text.translate(_PUNCT_NORMALIZATION)
    stripped = _FENCED_CODE.sub(" ", stripped)
    stripped = _INLINE_CODE.sub(" ", stripped)
    return _BLOCKQUOTE.sub(" ", stripped)


def question_units(stripped: str) -> list[tuple[str, str]]:
    """Return (unit, following_context) for every ``?``-terminated span."""
    units: list[tuple[str, str]] = []
    for match in _QUESTION_UNIT.finditer(stripped):
        unit = match.group(0)
        if not unit.strip(" \n\t?"):
            continue  # a bare '?' with no words is not a question
        following = stripped[match.end() : match.end() + FOLLOWING_CONTEXT_CHARS]
        units.append((unit, following))
    return units


def _matches(
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]],
    categories: list[str],
    text: str,
) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for category in categories:
        matched = [pattern for pattern, rx in compiled.get(category, []) if rx.search(text)]
        if matched:
            hits[category] = matched
    return hits


def _merge(into: dict[str, list[str]], hits: dict[str, list[str]]) -> None:
    for category, patterns in hits.items():
        bucket = into.setdefault(category, [])
        for pattern in patterns:
            if pattern not in bucket:
                bucket.append(pattern)


def classify(text: str | None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify one **zero-edit turn's** final message (layer 2 only).

    The caller (the runner's gate, or an offline audit) is responsible for
    layer 1: this function assumes the turn produced no edits, and its
    verdict is meaningless for a turn that did. ``signals`` names what made
    the verdict positive -- ``QUESTION_UNIT`` carries the asking units'
    text snippets; the no-question-mark categories carry their matched
    patterns -- and ``blockers`` records every suppressed unit, so
    pattern-level ablation can run over stored outcomes later.
    """
    config = config or default_config()
    if not text:
        return {"asked": False, "question_units": 0, "signals": {}, "blockers": {}}

    stripped = strip_nonprose(text)
    units = question_units(stripped)

    asked = False
    signals: dict[str, list[str]] = {}
    blockers: dict[str, list[str]] = {}

    for unit, following in units:
        unit_blockers = _matches(config["blockers"], ["TAG_QUESTION"], unit)
        _merge(
            unit_blockers,
            _matches(config["blockers"], ["SELF_ANSWERED"], following),
        )
        unit_blockers = {
            category: patterns
            for category, patterns in unit_blockers.items()
            if category in config["blocker_categories"]
        }
        if unit_blockers:
            _merge(blockers, unit_blockers)
            continue
        asked = True
        _merge(signals, {"QUESTION_UNIT": [" ".join(unit.split())[:80]]})

    # No-question-mark asks: an imperative information request ("I don't
    # have the plan... Please share it") or an interrogative stem ending in
    # ':' over an option list ("Should this be:\n- Patch..."). Single-factor
    # is safe here because only zero-edit turns reach this layer; it also
    # deliberately runs when every '?' unit was blocked -- "Could it be the
    # cache? No. Please share the trace." is still a request for input.
    if not asked and config["no_qm_request_categories"]:
        request = _matches(config["signals"], config["no_qm_request_categories"], stripped)
        if request:
            asked = True
            _merge(signals, request)

    return {
        "asked": asked,
        "question_units": len(units),
        "signals": signals,
        "blockers": blockers,
    }


def is_clarifying_question(text: str | None, config: dict[str, Any] | None = None) -> bool:
    """Layer-2 verdict alone; only meaningful for a zero-edit turn."""
    return classify(text, config)["asked"]
