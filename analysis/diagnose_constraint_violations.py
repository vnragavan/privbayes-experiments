"""
analysis/diagnose_constraint_violations.py

Print constraint violation breakdown (event vs time) per implementation
from sweep result JSONs. Run after a sweep so result files contain
survival_pair_event_violation_rate and survival_pair_time_violation_rate.

Usage:
  python analysis/diagnose_constraint_violations.py
  python analysis/diagnose_constraint_violations.py --results-dir results/eps_sweep/lung_clean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.load_results import load_all_results


def main():
    parser = argparse.ArgumentParser(description="Report constraint violation breakdown (event vs time) per implementation.")
    parser.add_argument("--results-dir", default="results/eps_sweep", help="Base results dir (e.g. results/eps_sweep or results/eps_sweep/lung_clean)")
    args = parser.parse_args()

    # Try results_dir and results_dir/<dataset> if it's a single dir
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    df = load_all_results(str(results_dir))
    if df.empty and results_dir.exists():
        for sub in sorted(results_dir.iterdir()):
            if sub.is_dir():
                df = load_all_results(str(sub))
                if not df.empty:
                    results_dir = sub
                    break
    if df.empty:
        print("No result files found. Run a sweep first so result JSONs exist under results/eps_sweep/<dataset>/")
        return 1

    has_event = "constraint_viol_event" in df.columns
    has_time = "constraint_viol_time" in df.columns
    if not has_event or not has_time:
        print("Result JSONs do not yet have event/time breakdown. Re-run the sweep (or refresh metrics)")
        print("so that constraint_violation_summary returns survival_pair_event_violation_rate and survival_pair_time_violation_rate.")
        print("Showing overall constraint_viol only.")
        has_event = has_time = False

    print(f"\nConstraint violation breakdown (from {results_dir})")
    print("=" * 70)
    for impl in df["implementation"].dropna().unique():
        sub = df[df["implementation"] == impl]
        n = len(sub)
        viol = sub["constraint_viol"].mean()
        viol_s = sub["constraint_viol"].std()
        print(f"\n{impl.upper()} (n={n} runs)")
        print(f"  Overall violation rate   : {viol:.4f}  (std {viol_s:.4f})")
        if has_event:
            ev = sub["constraint_viol_event"].mean()
            ev_s = sub["constraint_viol_event"].std()
            print(f"  From event ∉ {{0,1}}     : {ev:.4f}  (std {ev_s:.4f})")
        if has_time:
            tm = sub["constraint_viol_time"].mean()
            tm_s = sub["constraint_viol_time"].std()
            print(f"  From time < threshold   : {tm:.4f}  (std {tm_s:.4f})")
        if has_event and has_time:
            print("  (Event = status/event column not in allowed values; Time = duration column below schema minimum)")
    print("\nNote: CRN/PrivBayes is schema-native; SynthCity is not (it reads data directly).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
