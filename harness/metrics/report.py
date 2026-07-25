"""Compute and print the Q1-Q8 evidence metric families, overall and stratified by
repo and difficulty. Reads the aggregated table (or builds it on the fly).

The metric families map to the plan's evidence questions:
  Q1 recognition, Q2 when/how-much-asks, Q3 cost-of-ambiguity, Q4 localization,
  Q5 regression, Q6 variance-across-repeats, Q7 effort/cost, Q8 failure-modes.
"""
from __future__ import annotations

import pandas as pd

from harness.config import PathsConfig
from harness.constants import CONDITION_AMBIGUOUS, CONDITION_FULL
from harness.metrics.aggregate import build_table


def _rate(series) -> float:
    s = series.dropna()
    return float(s.mean()) if len(s) else float("nan")


def _fmt(x) -> str:
    return "n/a" if x != x else f"{x:.3f}"  # x!=x catches NaN


def q1_recognition(df: pd.DataFrame) -> None:
    print("\n=== Q1: Ambiguity recognition (spontaneous ask rate) ===")
    amb = df[df.condition == CONDITION_AMBIGUOUS]
    full = df[df.condition == CONDITION_FULL]
    print(f"  ask_rate(ambiguous) = {_fmt(_rate(amb.asked))}  (n={len(amb)})")
    print(f"  ask_rate(full)      = {_fmt(_rate(full.asked))}  (n={len(full)})  [false-positive baseline]")
    print(f"  sensitivity delta   = {_fmt(_rate(amb.asked) - _rate(full.asked))}")
    if len(amb):
        print("  -- ambiguous ask_rate by repo --")
        for repo, g in amb.groupby("repo"):
            print(f"     {repo:28s} {_fmt(_rate(g.asked))}  (n={len(g)})")
        print("  -- ambiguous ask_rate by difficulty --")
        for d, g in amb.groupby("difficulty"):
            print(f"     {d:20s} {_fmt(_rate(g.asked))}  (n={len(g)})")
        print("  -- exit-reason split (ambiguous) --")
        print("    ", amb.exit_reason.value_counts().to_dict())


def q2_when_asks(df: pd.DataFrame) -> None:
    print("\n=== Q2: When / how much it asks (ambiguous, asked only) ===")
    a = df[(df.condition == CONDITION_AMBIGUOUS) & (df.asked)]
    if not len(a):
        print("  (no asks recorded yet)")
        return
    print(f"  mean n_questions   = {_fmt(a.n_questions.mean())}")
    print(f"  mean first_ask_turn= {_fmt(a.first_ask_turn.mean())}")
    print(f"  median n_turns     = {_fmt(a.n_turns.median())}")


def q3_cost_of_ambiguity(df: pd.DataFrame) -> None:
    print("\n=== Q3: Cost of ambiguity (resolve rate) ===")
    amb = df[df.condition == CONDITION_AMBIGUOUS]
    full = df[df.condition == CONDITION_FULL]
    amb_proceeded = amb[amb.produced_patch]
    rr_full = _rate(full.resolved)
    rr_amb_all = _rate(amb.resolved.fillna(False)) if len(amb) else float("nan")
    rr_amb_proc = _rate(amb_proceeded.resolved)
    print(f"  resolve_rate(full)              = {_fmt(rr_full)}  (n={len(full)})")
    print(f"  resolve_rate(ambiguous, all)    = {_fmt(rr_amb_all)}  [asks count as unresolved]")
    print(f"  resolve_rate(ambiguous, proceed)= {_fmt(rr_amb_proc)}  (n={len(amb_proceeded)})")
    print(f"  cost-of-ambiguity delta (all)   = {_fmt(rr_full - rr_amb_all)}")
    print(f"  cost-of-ambiguity delta (proceed)= {_fmt(rr_full - rr_amb_proc)}")


def q4_localization(df: pd.DataFrame) -> None:
    print("\n=== Q4: Localization (produced-patch runs) ===")
    for cond in (CONDITION_AMBIGUOUS, CONDITION_FULL):
        g = df[(df.condition == cond) & (df.produced_patch)]
        print(f"  [{cond}] hit_any={_fmt(_rate(g.loc_hit_any))} "
              f"hit_all={_fmt(_rate(g.loc_hit_all))} "
              f"recall={_fmt(_rate(g.loc_recall))} precision={_fmt(_rate(g.loc_precision))} "
              f"(n={len(g)})")


def q5_regression(df: pd.DataFrame) -> None:
    print("\n=== Q5: Regression (patched & evaluated runs) ===")
    for cond in (CONDITION_AMBIGUOUS, CONDITION_FULL):
        g = df[(df.condition == cond) & (df.produced_patch) & (df.regression.notna())]
        print(f"  [{cond}] regression_rate={_fmt(_rate(g.regression))} (n={len(g)})")


def q6_variance(df: pd.DataFrame) -> None:
    print("\n=== Q6: Variance across ambiguous repeats ===")
    amb = df[df.condition == CONDITION_AMBIGUOUS]
    if not len(amb):
        print("  (no ambiguous runs yet)")
        return
    grp = amb.groupby("instance_id")
    ask_frac = grp.asked.mean()
    always = int((ask_frac == 1.0).sum())
    never = int((ask_frac == 0.0).sum())
    mixed = int(((ask_frac > 0) & (ask_frac < 1)).sum())
    print(f"  instances: always-asks={always}, never-asks={never}, mixed={mixed}")
    # resolve consistency among proceeded
    res = grp.resolved.apply(lambda s: s.dropna().nunique())
    inconsistent = int((res > 1).sum())
    print(f"  instances with inconsistent resolved outcome across repeats: {inconsistent}")


def q7_effort(df: pd.DataFrame) -> None:
    print("\n=== Q7: Effort & cost distributions (median) ===")
    for cond in (CONDITION_AMBIGUOUS, CONDITION_FULL):
        g = df[df.condition == cond]
        if not len(g):
            continue
        print(f"  [{cond}] n_turns={_fmt(g.n_turns.median())} "
              f"tool_calls={_fmt(g.n_tool_calls.median())} "
              f"tokens={_fmt(g.tokens_total.median())} "
              f"cost=${_fmt(g.cost_usd.median())} "
              f"wall_s={_fmt(g.wall_time_s.median())}")


def q8_failure_modes(df: pd.DataFrame) -> None:
    print("\n=== Q8: Failure modes (exit reasons, all runs) ===")
    print("  ", df.exit_reason.value_counts().to_dict())
    err = df[df.exit_reason == "error"]
    print(f"  infra/error runs: {len(err)} (excluded from behavioral rates)")


def print_report(paths: PathsConfig | None = None) -> None:
    df = build_table(paths or PathsConfig())
    if df.empty:
        print("no records found. run a sweep first.")
        return
    print(f"loaded {len(df)} run records "
          f"({df.instance_id.nunique()} instances, conditions={sorted(df.condition.unique())})")
    q1_recognition(df)
    q2_when_asks(df)
    q3_cost_of_ambiguity(df)
    q4_localization(df)
    q5_regression(df)
    q6_variance(df)
    q7_effort(df)
    q8_failure_modes(df)
