"""Orchestrator CLI entrypoint.

Subcommands:
  sample     freeze the M1 pilot manifest (stratified instance_ids)
  run        run agent sweep over instances (--manifest / --instances / --all),
             optionally limited by --conditions and --repeats
  eval       run the decoupled swebench evaluation pass over produced-patch runs
  aggregate  scan records -> long-format table (parquet + csv)
  report     print stratified Q1-Q8 metrics

Run agent sweeps and eval separately so CLI throughput and Docker throughput decouple.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from harness.config import HarnessConfig
from harness.data.loader import instances_by_id, load_instances
from harness.data.sampling import read_manifest, stratified_pilot, write_manifest
from harness.orchestrator.executor import execute_unit
from harness.orchestrator.plan import RunUnit, enumerate_units, pending_units
from harness.record.store import RunStore


def _select_instances(args, cfg):
    by_id = instances_by_id()
    if args.all:
        return list(load_instances())
    if args.manifest:
        ids = read_manifest(args.manifest, cfg.paths)
    elif args.instances:
        ids = args.instances
    else:
        raise SystemExit("specify --all, --manifest NAME, or --instances ID [ID ...]")
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown instance_ids: {missing}")
    return [by_id[i] for i in ids]


def cmd_sample(args):
    cfg = HarnessConfig()
    ids = stratified_pilot(n=args.n, seed=args.seed)
    path = write_manifest(ids, name=args.name, paths=cfg.paths,
                          meta={"n": args.n, "seed": args.seed})
    print(f"wrote manifest {path} with {len(ids)} instances")
    # quick composition summary
    by_id = instances_by_id()
    from collections import Counter
    repos = Counter(by_id[i].repo for i in ids)
    diffs = Counter(by_id[i].difficulty for i in ids)
    print("by repo:", dict(repos))
    print("by difficulty:", dict(diffs))


def cmd_run(args):
    cfg = HarnessConfig()
    if args.model:
        cfg = HarnessConfig(run=type(cfg.run)(**{**cfg.run.__dict__, "model": args.model}))
    store = RunStore(cfg.paths)
    instances = _select_instances(args, cfg)
    by_id = {i.instance_id: i for i in instances}

    units = enumerate_units(instances, cfg.run)
    if args.conditions:
        units = [u for u in units if u.condition in args.conditions]
    units = pending_units(units, store)
    if args.limit:
        units = units[: args.limit]

    print(f"{len(units)} pending run unit(s); cli_workers={cfg.concurrency.cli_workers}")
    if not units:
        return

    done = 0
    with ThreadPoolExecutor(max_workers=cfg.concurrency.cli_workers) as ex:
        futs = {ex.submit(execute_unit, u, by_id[u.instance_id], cfg, store): u
                for u in units}
        for fut in as_completed(futs):
            u = futs[fut]
            rec = fut.result()
            done += 1
            print(f"[{done}/{len(units)}] {rec.run_id} -> {rec.exit.reason}"
                  f" (asked={rec.ask.asked}, patch={rec.patch.produced_patch})")


def cmd_eval(args):
    cfg = HarnessConfig()
    from harness.eval.driver import run_eval_pass
    only = None
    if args.manifest:
        only = read_manifest(args.manifest, cfg.paths)
    elif args.instances:
        only = args.instances
    stats = run_eval_pass(cfg, only_instance_ids=only)
    print("eval stats:", stats)


def cmd_aggregate(args):
    from harness.metrics.aggregate import aggregate_to_table
    cfg = HarnessConfig()
    out = aggregate_to_table(cfg.paths)
    print("wrote:", out)


def cmd_report(args):
    from harness.metrics.report import print_report
    cfg = HarnessConfig()
    print_report(cfg.paths)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sample")
    ps.add_argument("--n", type=int, default=60)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--name", default="pilot_60")
    ps.set_defaults(func=cmd_sample)

    pr = sub.add_parser("run")
    g = pr.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true")
    g.add_argument("--manifest")
    g.add_argument("--instances", nargs="+")
    pr.add_argument("--conditions", nargs="+", choices=["ambiguous", "full"])
    pr.add_argument("--limit", type=int)
    pr.add_argument("--model")
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("eval")
    pe.add_argument("--manifest")
    pe.add_argument("--instances", nargs="+")
    pe.set_defaults(func=cmd_eval)

    pa = sub.add_parser("aggregate")
    pa.set_defaults(func=cmd_aggregate)

    prep = sub.add_parser("report")
    prep.set_defaults(func=cmd_report)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
