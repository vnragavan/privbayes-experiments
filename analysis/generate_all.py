"""
analysis/generate_all.py

Generates all figures and tables from sweep results.
Run after completing run_sweep.py.
"""

import argparse
import os
import sys
import traceback


def run(label, fn):
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    try:
        fn()
        print(f"  OK")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate figures and tables from sweep results.")
    parser.add_argument("--results-dir", default="results/eps_sweep",
                        help="Directory containing results_eps*_seed*.json")
    parser.add_argument("--figures-dir", default=None,
                        help="Directory for figure PDFs (default: outputs/figures)")
    parser.add_argument("--tables-dir", default=None,
                        help="Directory for table .tex files (default: outputs/tables)")
    args = parser.parse_args()
    results_dir = args.results_dir.rstrip("/")
    figures_dir = args.figures_dir or "outputs/figures"
    tables_dir = args.tables_dir or "outputs/tables"

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    from analysis.figures.fig2_to_6 import (
        fig2_utility_curves,
        fig4_compliance_adjusted, fig5_privacy_risk, fig6_performance,
        fig_survival_all_in_one)
    from analysis.tables.tables import (
        tab1_compliance_audit, tab_compliance_ledger, tab2_utility_survival, tab3_privacy_performance)

    single_run = os.path.join(results_dir, "results_eps1.0_seed0.json")
    tasks = [
        ("Figure 2: Utility curves",            lambda: fig2_utility_curves(results_dir=results_dir, out=os.path.join(figures_dir, "fig2_utility_curves.pdf"))),
        ("Figure: Survival metrics all-in-one", lambda: fig_survival_all_in_one(results_dir=results_dir, out=os.path.join(figures_dir, "fig_survival_all_in_one.pdf"))),
        ("Figure 4: Compliance-adjusted",       lambda: fig4_compliance_adjusted(results_dir=results_dir, out=os.path.join(figures_dir, "fig4_compliance_adjusted.pdf"))),
        ("Figure 5: Privacy risk",              lambda: fig5_privacy_risk(results_dir=results_dir, out=os.path.join(figures_dir, "fig5_privacy_risk.pdf"))),
        ("Figure 6: Performance",               lambda: fig6_performance(
            results_dir=results_dir,
            single_run_json=single_run,
            out=os.path.join(figures_dir, "fig6_performance.pdf"))),
        ("Table 1: Compliance audit",           lambda: tab1_compliance_audit(out=os.path.join(tables_dir, "tab1_compliance_audit.tex"))),
        ("Table: DBC and report completeness",  lambda: tab_compliance_ledger(results_dir=results_dir, out=os.path.join(tables_dir, "tab_compliance_ledger.tex"))),
        ("Table 2: Utility + survival",          lambda: tab2_utility_survival(results_dir=results_dir, out=os.path.join(tables_dir, "tab2_utility_survival.tex"))),
        ("Table 3: Privacy + performance",      lambda: tab3_privacy_performance(
            results_dir=results_dir,
            single_run_json=single_run,
            out=os.path.join(tables_dir, "tab3_privacy_performance.tex"))),
    ]

    failed = []
    for label, fn in tasks:
        if not run(label, fn):
            failed.append(label)

    print(f"\n{'=' * 55}")
    if failed:
        print(f"FAILED ({len(failed)}/{len(tasks)}):")
        for f in failed:
            print(f"  - {f}")
    else:
        print(f"All {len(tasks)} outputs generated successfully.")

    print("\nOutputs:")
    for d in [figures_dir, tables_dir]:
        for fname in sorted(os.listdir(d)):
            print(f"  {d}/{fname}")


if __name__ == "__main__":
    main()
