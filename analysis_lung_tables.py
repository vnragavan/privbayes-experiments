#!/usr/bin/env python3
"""
analysis_lung_tables.py

Quick helper to turn the lung eps-sweep JSON outputs into paper-ready
tables (CSV + markdown).

Usage (from repo root, venv optional but recommended):

    python analysis_lung_tables.py \
        --results-dir results/eps_sweep \
        --out-csv     metrics/lung_summary.csv

This expects files like:
    results/eps_sweep/results_eps{epsilon}_seed{seed}.json
which are produced by compute_sweep_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("results_eps*_seed*.json")):
        with path.open() as f:
            blob = json.load(f)
        # One JSON per (eps, seed), with keys = implementation names
        # and values = metrics dicts.
        # Example key shape:
        #   blob["crn"]["privacy"]["epsilon"]
        #   blob["crn"]["survival"]["km_l1"]
        #   blob["crn"]["performance"]["fit_time_sec"]
        #   blob["crn"]["performance"]["sample_time_sec"]
        #   blob["crn"]["performance"]["peak_memory_max_mb"]
        for impl, payload in blob.items():
            if not isinstance(payload, dict):
                continue
            status = payload.get("status")
            if status and status != "ok":
                continue
            priv = payload.get("privacy", {})
            surv = payload.get("survival", {})
            perf = payload.get("performance", {})
            comp = payload.get("compliance", {})
            ledger = (comp or {}).get("ledger", {})

            row: dict[str, Any] = {
                "epsilon": priv.get("epsilon"),
                "seed": payload.get("seed"),
                "implementation": impl,
                "status": status or "ok",
                # Survival utility
                "km_l1": surv.get("km_l1"),
                # Optional: downstream metrics if present
                "c_index": surv.get("c_index"),
                "brier": surv.get("brier_score"),
                # Performance
                "fit_time_sec": perf.get("fit_time_sec"),
                "sample_time_sec": perf.get("sample_time_sec"),
                "peak_memory_mb": perf.get("peak_memory_max_mb"),
                # Compliance headline
                "n_source": ledger.get("n_source"),
                "epsilon_total_declared": ledger.get("epsilon_total_declared"),
                "gap_flag": (comp or {}).get("composition", {}).get("gap_flag"),
                "ledger_completeness": comp.get("ledger_completeness"),
            }
            rows.append(row)
    return rows


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds: median + IQR per (impl, epsilon)."""
    if df.empty:
        return df

    def _iqr(x: pd.Series) -> float:
        return float(x.quantile(0.75) - x.quantile(0.25))

    group_cols = ["implementation", "epsilon"]
    agg_spec: Dict[str, List[str]] = {
        "km_l1": ["median", _iqr],
        "c_index": ["median", _iqr],
        "brier": ["median", _iqr],
        "fit_time_sec": ["median"],
        "sample_time_sec": ["median"],
        "peak_memory_mb": ["median"],
        "ledger_completeness": ["median"],
    }
    g = df.groupby(group_cols).agg(agg_spec)

    # Flatten MultiIndex columns to nicer names
    g.columns = [
        f"{metric}_{func if isinstance(func, str) else 'iqr'}"
        for metric, func in g.columns
    ]
    g = g.reset_index()
    return g.sort_values(group_cols)


def _print_markdown_table(summary: pd.DataFrame) -> None:
    """Print a compact markdown table suitable for the paper."""
    if summary.empty:
        print("No rows to summarise.")
        return

    cols = [
        "implementation",
        "epsilon",
        "km_l1_median",
        "km_l1_iqr",
        "fit_time_sec_median",
        "peak_memory_mb_median",
        "ledger_completeness_median",
    ]
    df_view = summary[cols].copy()
    # Round numeric columns for nicer display
    for c in df_view.columns:
        if pd.api.types.is_numeric_dtype(df_view[c]):
            df_view[c] = df_view[c].round(4)
    print("\nMarkdown table (per-impl, per-epsilon):\n")
    print(df_view.to_markdown(index=False))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/eps_sweep"),
        help="Directory containing results_eps*_seed*.json",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("metrics/lung_summary.csv"),
        help="Path to write the aggregated CSV (will create parent dirs).",
    )
    args = ap.parse_args()

    rows = _load_results(args.results_dir)
    if not rows:
        print(f"No JSON metrics found under {args.results_dir}")
        return

    df = pd.DataFrame(rows)
    summary = _summarise(df)

    # Ensure output directory exists
    os.makedirs(args.out_csv.parent, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(f"Wrote aggregated metrics CSV → {args.out_csv}")

    _print_markdown_table(summary)


if __name__ == "__main__":
    main()

