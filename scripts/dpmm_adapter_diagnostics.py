#!/usr/bin/env python3
"""
DPMM adapter diagnostics: run DPMM raw/adapted, report metrics and any metric errors.

Surfaces fit-time dtype mismatches, output structure issues, benchmark metrics,
and the actual error messages when a metric fails (e.g. TSTR C-index "single class
in synthetic", MIA "no shared columns").

Usage:
  python scripts/dpmm_adapter_diagnostics.py --schema schemas/lung_schema.json --data data/lung_clean.csv --relax-n
  python scripts/dpmm_adapter_diagnostics.py --schema ... --data ... --condition adapted --condition raw --relax-n
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
from sklearn.model_selection import train_test_split

from experiments.adapter_ablation_metrics import (
    _run_dpmm_raw,
    _run_dpmm_adapted,
    extract_benchmark_metrics,
    compute_fit_dtype_metrics,
    compute_output_structure_metrics,
)
from experiments.adapter_ablation_metrics import _load_schema
from metrics.report import compute_metrics


def _collect_errors(metrics: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Recursively find all dict entries with an 'error' key (non-empty)."""
    out = []
    for k, v in (metrics or {}).items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            if v.get("error"):
                out.append((path, str(v["error"])))
            out.extend(_collect_errors(v, path))
    return out


def main():
    p = argparse.ArgumentParser(description="DPMM adapter diagnostics: metrics + error reasons")
    p.add_argument("--schema", required=True, help="Schema JSON path")
    p.add_argument("--data", required=True, help="Real data CSV path")
    p.add_argument("--condition", action="append", default=None, dest="conditions",
                   choices=["raw", "adapted"], help="DPMM condition(s) (default: both)")
    p.add_argument("--n", type=int, default=None, help="Synthetic sample size (default: len(train))")
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--relax-n", action="store_true",
                   help="Set schema dataset_info.n_records to len(data) for fit")
    args = p.parse_args()
    if not args.conditions:
        args.conditions = ["raw", "adapted"]

    with open(args.schema) as f:
        schema = json.load(f)
    df = pd.read_csv(args.data)
    if args.relax_n:
        schema.setdefault("dataset_info", {})["n_records"] = len(df)

    schema = _load_schema(schema)
    train_df, temp = train_test_split(df, test_size=0.4, random_state=args.seed)
    test_df, holdout_df = train_test_split(temp, test_size=0.5, random_state=args.seed)
    n_synth = args.n if args.n is not None else len(train_df)

    print("DPMM adapter diagnostics")
    print("=" * 60)
    print(f"Schema: {args.schema}")
    print(f"Data: {args.data} (n={len(df)}, train={len(train_df)})")
    print(f"Sample size: {n_synth}")
    print()

    for condition in args.conditions:
        print(f"--- DPMM {condition.upper()} ---")
        try:
            if condition == "raw":
                fit_metrics, output_metrics, synth_df, privacy_report = _run_dpmm_raw(
                    train_df=train_df, schema=schema, epsilon=args.epsilon,
                    n_synth=n_synth, seed=args.seed,
                )
            else:
                fit_metrics, output_metrics, synth_df, privacy_report = _run_dpmm_adapted(
                    train_df=train_df, schema=schema, epsilon=args.epsilon,
                    n_synth=n_synth, seed=args.seed,
                )

            print("  Fit-time dtype:")
            print(f"    n_columns={fit_metrics.n_columns}, n_mismatches={fit_metrics.n_dtype_mismatches}, rate={fit_metrics.mismatch_rate:.4f}")

            print("  Output structure (pre-normalization):")
            for attr in ("binary_invalid_rate", "categorical_invalid_rate", "integer_float_column_rate", "out_of_bounds_rate"):
                v = getattr(output_metrics, attr, None)
                if v is not None:
                    print(f"    {attr}={v:.4f}")

            metrics = compute_metrics(
                implementation=f"dpmm_{condition}",
                real_df=train_df,
                synth_df=synth_df,
                schema=schema,
                privacy_report=privacy_report,
                test_real_df=test_df,
                train_df=train_df,
                holdout_df=holdout_df,
                performance={},
            )
            # Synthetic event column (for TSTR C-index): Cox requires both 0 and 1
            schema_target = (schema.get("target_spec") or {}).get("targets") or []
            event_col = None
            for c in ["status", "event"]:
                if c in synth_df.columns:
                    event_col = c
                    break
            if not event_col and schema_target:
                event_col = schema_target[0] if isinstance(schema_target[0], str) else None
            if event_col and event_col in synth_df.columns:
                evt = pd.to_numeric(synth_df[event_col], errors="coerce").dropna()
                n_evt = len(evt)
                uniq = evt.unique()
                n_uniq = len(uniq)
                print("  Synthetic event (for TSTR C-index):")
                print(f"    column={event_col}, n={n_evt}, n_unique={n_uniq}, values={sorted(uniq.tolist())[:10]}{'...' if len(uniq) > 10 else ''}")
                if n_uniq < 2:
                    print(f"    -> TSTR C-index will be NaN (Cox needs both events and non-events)")

            bench = extract_benchmark_metrics(metrics)
            surv = metrics.get("survival") or {}
            tstr_cindex = surv.get("tstr_cindex")
            tstr_cindex_val = tstr_cindex.get("value") if isinstance(tstr_cindex, dict) else tstr_cindex
            tstr_cindex_err = tstr_cindex.get("error") if isinstance(tstr_cindex, dict) else None
            print("  Benchmark:")
            print(f"    TSTR AUC={bench.tstr_auc:.4f}" if bench.tstr_auc == bench.tstr_auc else "    TSTR AUC=NaN")
            print(f"    TSTR C-index={tstr_cindex_val:.4f}" if tstr_cindex_val is not None and tstr_cindex_val == tstr_cindex_val else "    TSTR C-index=NaN")
            if tstr_cindex_err:
                print(f"    TSTR C-index error: {tstr_cindex_err}")
            print(f"    MIA AUC={bench.mia_auc:.4f}" if bench.mia_auc == bench.mia_auc else "    MIA AUC=NaN")
            print(f"    KM-L1={bench.km_l1:.4f}" if bench.km_l1 == bench.km_l1 else "    KM-L1=NaN")
            print(f"    constraint_violation_rate={bench.constraint_violation_rate:.6f}")

            errors = _collect_errors(metrics)
            if errors:
                print("  Metric errors:")
                for path, msg in errors:
                    print(f"    {path}: {msg}")
            else:
                print("  Metric errors: none")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    # Root cause summary when both conditions were run
    if set(args.conditions) >= {"raw", "adapted"}:
        print("=" * 60)
        print("TSTR C-index: root cause (adapter vs raw)")
        print("  1. Raw: DPMM output is not schema-normalized (wrong dtypes/categories).")
        print("     Cox is fitted on mis-typed covariates/event → C-index can be misleading.")
        print("     Raw event column may be floats; coercion can accidentally yield both 0 and 1")
        print("     even when DPMM effectively produced one class, so raw sometimes reports a value.")
        print("  2. Adapted: Output is normalized to schema (binary 0/1, correct dtypes).")
        print("     Cox sees the true synthetic event distribution.")
        print("  3. When adapted TSTR C-index is NaN: synthetic event has only one value (all 0 or 1).")
        print("     DPMM generated no variation in events; Cox correctly cannot fit. The adapter")
        print("     does not cause this—it exposes the true content. Raw may show a value only")
        print("     because mis-typed data coerces to two classes.")
        print("  See docs/DPMM_SINGLE_EVENT_ROOT_CAUSE.md for why adapted uses")
        print("  categorical decoding (discrete) and raw uses numeric decoding (within-bin spread).")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
