"""
experiments/run_adapter_ablation_example.py

Example script to run the adapter ablation metrics (raw / adapted / optional wrong_schema)
and write Table 1 (schema interpretation), Table 2 (output structure), Table 3 (benchmark),
plus a flat CSV of all metrics.

Usage:
  # CSV + schema (default for this repo)
  python experiments/run_adapter_ablation_example.py --schema schemas/lung_schema.json --data data/lung_clean.csv

  # With wrong_schema condition and custom output dir
  python experiments/run_adapter_ablation_example.py --schema schemas/lung_schema.json --data data/lung_clean.csv --wrong-schema --out-dir outputs/ablation_metrics

  # Multiple runs: writes adapter_ablation_summary.csv with mean, std, se, 95%% CI per benchmark metric
  python experiments/run_adapter_ablation_example.py --schema schemas/lung_schema.json --data data/lung_clean.csv --n-runs 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_ablation_metrics import (
    run_adapter_ablation_metrics,
    build_table_schema_interpretation,
    build_table_output_structure,
    build_table_benchmark,
    build_benchmark_summary_with_ci,
)


def main():
    parser = argparse.ArgumentParser(description="Run adapter ablation metrics and write tables + CSV.")
    parser.add_argument("--schema", required=True, help="Path to schema JSON")
    parser.add_argument("--data", required=True, help="Path to dataset CSV (or Parquet)")
    parser.add_argument("--out-dir", default="outputs/ablation_metrics", help="Output directory for tables and CSV")
    parser.add_argument("--wrong-schema", action="store_true", help="Include wrong_schema condition")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=1, help="Number of runs (seeds seed..seed+n_runs-1); summary CSV gets mean, SE, 95%% CI")
    parser.add_argument("--n-synth", type=int, default=None, help="Synthetic sample size (default: len(train))")
    args = parser.parse_args()

    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=args.seed)
    test_real_df, holdout_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    print("Running adapter ablation metrics (synthcity + dpmm, raw + adapted" + (" + wrong_schema" if args.wrong_schema else "") + f", n_runs={args.n_runs})...")
    all_runs = []
    for r in range(args.n_runs):
        run_seed = args.seed + r
        df_metrics = run_adapter_ablation_metrics(
            train_df=train_df,
            test_real_df=test_real_df,
            holdout_df=holdout_df,
            schema=args.schema,
            epsilon=args.epsilon,
            seed=run_seed,
            n_synth=args.n_synth,
            include_wrong_schema=args.wrong_schema,
        )
        all_runs.append(df_metrics)
    df_metrics = pd.concat(all_runs, ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # One row per (impl, condition) for tables: use first run for schema/output; use summary mean for benchmark when n_runs > 1
    df_first = df_metrics.drop_duplicates(subset=["implementation", "condition"], keep="first")
    table1 = build_table_schema_interpretation(df_first)
    table2 = build_table_output_structure(df_first)
    summary = build_benchmark_summary_with_ci(df_metrics, confidence=0.95) if args.n_runs > 1 else None
    if args.n_runs > 1 and summary is not None:
        mean_cols = ["implementation", "condition"] + [c for c in summary.columns if c.endswith("_mean")]
        table3 = summary[mean_cols].copy()
        table3.columns = [c.replace("_mean", "") if c.endswith("_mean") else c for c in table3.columns]
    else:
        table3 = build_table_benchmark(df_metrics)

    print("\n=== Table 1: Schema interpretation ===")
    print(table1.to_string(index=False))
    print("\n=== Table 2: Output structure ===")
    print(table2.to_string(index=False))
    print("\n=== Table 3: Benchmark metrics ===")
    print(table3.to_string(index=False))

    table1.to_csv(os.path.join(args.out_dir, "table_schema_interpretation.csv"), index=False)
    table2.to_csv(os.path.join(args.out_dir, "table_output_structure.csv"), index=False)
    table3.to_csv(os.path.join(args.out_dir, "table_benchmark.csv"), index=False)
    df_metrics.to_csv(os.path.join(args.out_dir, "adapter_ablation_all_metrics.csv"), index=False)

    if args.n_runs > 1 and summary is not None:
        summary.to_csv(os.path.join(args.out_dir, "adapter_ablation_summary.csv"), index=False)
        print("\nWrote adapter_ablation_summary.csv (mean, std, se, 95% CI per metric).")

    print(f"\nWrote CSV/table files to {args.out_dir}")


if __name__ == "__main__":
    main()
