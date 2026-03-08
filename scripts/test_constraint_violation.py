#!/usr/bin/env python3
"""
Quick test: run one implementation, sample, and report constraint violation rate only.

No full metrics or sweep; just fit -> sample -> constraint_violation_summary.

Usage:
  python scripts/test_constraint_violation.py --schema schemas/lung_schema.json --data data/lung_clean.csv --relax-n
  python scripts/test_constraint_violation.py --schema schemas/lung_schema.json --data data/lung_clean.csv --impl crn --n 200 --relax-n
  python scripts/test_constraint_violation.py --schema ... --data ... --impl crn --impl dpmm --relax-n
  (Use --relax-n when schema was generated from a different-sized dataset.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from metrics.constraint.validator import constraint_violation_summary

WRAPPERS = {
    "crn": "implementations.crn_wrapper.CRNWrapper",
    "dpmm": "implementations.dpmm_wrapper.DPMMWrapper",
    "synthcity": "implementations.synthcity_wrapper.SynthCityWrapper",
}


def load_wrapper(name: str, epsilon: float = 1.0, seed: int = 0):
    mod, _, cls = WRAPPERS[name].rpartition(".")
    m = __import__(mod, fromlist=[cls])
    return getattr(m, cls)(epsilon=epsilon, seed=seed)


def main():
    p = argparse.ArgumentParser(description="Quick constraint violation rate test")
    p.add_argument("--schema", required=True, help="Schema JSON path")
    p.add_argument("--data", required=True, help="Real data CSV path")
    p.add_argument("--impl", action="append", default=None, dest="impls",
                   choices=list(WRAPPERS), help="Implementation(s) to test (default: crn); can repeat")
    p.add_argument("--n", type=int, default=None, help="Synthetic sample size (default: len(data))")
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--relax-n", action="store_true",
                   help="Set schema dataset_info.n_records to len(data) so fit accepts (quick test only)")
    args = p.parse_args()
    if not args.impls:
        args.impls = ["crn"]

    with open(args.schema) as f:
        schema = json.load(f)
    df = pd.read_csv(args.data)
    if args.relax_n:
        schema.setdefault("dataset_info", {})["n_records"] = len(df)
    n = args.n if args.n is not None else len(df)

    print(f"Schema: {args.schema}")
    print(f"Data: {args.data} (n={len(df)})")
    print(f"Sample size: {n}")
    print()

    for impl in args.impls:
        print(f"--- {impl.upper()} ---")
        try:
            wrapper = load_wrapper(impl, epsilon=args.epsilon, seed=args.seed)
            wrapper.fit(df, schema=schema)
            synth_df = wrapper.sample(n)
            summary = constraint_violation_summary(synth_df, schema)
            rate = summary["overall_violation_rate"]
            ev = summary.get("survival_pair_event_violation_rate")
            tm = summary.get("survival_pair_time_violation_rate")
            print(f"  Overall violation rate: {rate:.6f}")
            if ev is not None:
                print(f"  From event ∉ {{0,1}}:   {ev:.6f}")
            if tm is not None:
                print(f"  From time < threshold: {tm:.6f}")
            if summary.get("column_constraints"):
                nonzero = [(c, v) for c, v in summary["column_constraints"].items() if v and v > 0]
                if nonzero:
                    print(f"  Column violations: {dict(nonzero)}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
