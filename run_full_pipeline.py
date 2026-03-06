#!/usr/bin/env python3
"""
run_full_pipeline.py

One-shot script that runs the full experiment workflow:
  1. Epsilon sweep (run_sweep) → synthetic CSVs + result JSONs
  2. [Optional] Refresh metrics from CSVs (compute_sweep_metrics)
  3. Generate all figures and tables (generate_all)
  4. [Optional] Compile report PDF (pdflatex) only if outputs/report.tex exists (not in repo)

Usage:
  # Full pipeline: sweep + figures/tables + report (results go to results/eps_sweep/<dataset from schema>)
  python run_full_pipeline.py --schema schemas/lung_schema.json --data data/lung_clean.csv

  # Skip sweep; only regenerate figures, tables, and report from existing results
  python run_full_pipeline.py --schema schemas/lung_schema.json --data data/lung_clean.csv --skip-sweep --results-dir results/eps_sweep/lung_clean

  # Refresh metrics from existing CSVs (e.g. to add attribute_inference), then figures + report
  python run_full_pipeline.py --schema schemas/lung_schema.json --data data/lung_clean.csv --skip-sweep --refresh-metrics --results-dir results/eps_sweep/lung_clean
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _dataset_name_from_schema(schema_path: str) -> str:
    try:
        with open(schema_path) as f:
            s = json.load(f)
        name = s.get("dataset", "")
        if isinstance(name, str) and name.strip():
            return name.strip().replace(" ", "_").replace("/", "_")
    except Exception:
        pass
    return Path(schema_path).stem


def _run(cmd: list[str], step_name: str, env: dict | None = None) -> bool:
    print(f"\n{'─' * 60}")
    print(f"  {step_name}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 60}\n")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=run_env)
    if r.returncode != 0:
        print(f"\nFAILED: {step_name} (exit code {r.returncode})", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full experiment pipeline: sweep → figures/tables → report PDF.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--schema", required=True, help="Path to schema JSON (e.g. schemas/lung_schema.json)")
    parser.add_argument("--data", required=True, help="Path to dataset CSV (e.g. data/lung_clean.csv)")
    parser.add_argument(
        "--output-dir",
        default="results/eps_sweep",
        help="Base directory for sweep results; dataset name is appended (default: results/eps_sweep)",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the epsilon sweep; use existing result JSONs and only regenerate figures, tables, and report",
    )
    parser.add_argument(
        "--refresh-metrics",
        action="store_true",
        help="Recompute all metrics from existing CSVs (e.g. to add new metrics) before generating figures; implies --skip-sweep if used without running sweep",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures",
        help="Directory for figure PDFs (default: outputs/figures)",
    )
    parser.add_argument(
        "--tables-dir",
        default="outputs/tables",
        help="Directory for table .tex files (default: outputs/tables)",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs",
        help="Directory containing report.tex and output for PDF (default: outputs)",
    )
    parser.add_argument(
        "--results-dir",
        dest="results_dir_override",
        metavar="DIR",
        default=None,
        help="Override results directory (default: <output-dir>/<dataset from schema>); e.g. results/eps_sweep/lung_clean",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    schema_path = Path(args.schema)
    data_path = Path(args.data)
    if not schema_path.is_absolute():
        schema_path = repo_root / schema_path
    if not data_path.is_absolute():
        data_path = repo_root / data_path

    if not schema_path.exists():
        print(f"ERROR: Schema not found: {schema_path}", file=sys.stderr)
        return 1
    if not data_path.exists():
        print(f"ERROR: Data not found: {data_path}", file=sys.stderr)
        return 1

    pythonpath = str(repo_root)
    run_env = {"PYTHONPATH": pythonpath}

    dataset_name = _dataset_name_from_schema(str(schema_path))
    if args.results_dir_override:
        results_dir = Path(args.results_dir_override)
    else:
        results_dir = Path(args.output_dir) / dataset_name
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    results_dir = results_dir.resolve()
    figures_dir = repo_root / args.figures_dir
    tables_dir = repo_root / args.tables_dir
    report_dir = repo_root / args.report_dir

    # ── 1. Sweep (unless skipped) ─────────────────────────────────────
    if not args.skip_sweep and not args.refresh_metrics:
        ok = _run(
            [
                sys.executable,
                str(repo_root / "run_sweep.py"),
                "--schema", str(schema_path),
                "--data", str(data_path),
                "--output-dir", str(repo_root / args.output_dir),
            ],
            "Step 1: Epsilon sweep (run_sweep.py)",
            env=run_env,
        )
        if not ok:
            return 1
    elif args.skip_sweep or args.refresh_metrics:
        if not (results_dir / "results_eps1.0_seed0.json").exists():
            print(
                f"WARNING: No results found in {results_dir}. "
                "Run without --skip-sweep first, or use --refresh-metrics with existing CSVs.",
                file=sys.stderr,
            )
            if not args.refresh_metrics:
                return 1

    # ── 2. Refresh metrics from CSVs (optional) ──────────────────────────
    if args.refresh_metrics:
        ok = _run(
            [
                sys.executable,
                str(repo_root / "compute_sweep_metrics.py"),
                "--schema", str(schema_path),
                "--real-data", str(data_path),
                "--sweep-dir", str(results_dir),
                "--out-dir", str(results_dir),
            ],
            "Step 2: Refresh metrics from CSVs (compute_sweep_metrics.py)",
            env=run_env,
        )
        if not ok:
            return 1

    # ── 3. Generate figures and tables ──────────────────────────────────
    gen_cmd = [
        sys.executable,
        str(repo_root / "analysis" / "generate_all.py"),
        "--results-dir", str(results_dir),
        "--figures-dir", str(figures_dir),
        "--tables-dir", str(tables_dir),
    ]
    ok = _run(gen_cmd, "Step 3: Generate figures and tables (analysis/generate_all.py)", env=run_env)
    if not ok:
        return 1

    # ── 4. Compile report PDF (optional; report.tex not in repo) ─────────
    report_tex = report_dir / "report.tex"
    if report_tex.exists():
        for run_num in (1, 2):
            ok = _run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory", str(report_dir),
                    str(report_tex),
                ],
                f"Step 4: Compile report PDF (pdflatex run {run_num}/2)",
            )
            if not ok:
                return 1
        report_pdf = report_dir / "report.pdf"
        print(f"\n{'=' * 60}")
        print("Pipeline complete.")
        print(f"  Results : {results_dir}")
        print(f"  Figures : {figures_dir}")
        print(f"  Tables  : {tables_dir}")
        print(f"  Report  : {report_pdf}")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n{'=' * 60}")
        print("Pipeline complete (report skipped — report.tex not found).")
        print(f"  Results : {results_dir}")
        print(f"  Figures : {figures_dir}")
        print(f"  Tables  : {tables_dir}")
        print(f"  Report  : (optional; add outputs/report.tex to build)")
        print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
