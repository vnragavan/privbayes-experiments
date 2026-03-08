"""
experiments/generate_adapter_ablation_artifacts.py

Runs the adapter ablation (raw vs adapted SynthCity and DPMM), then generates:
  - Table 1: Adapter necessity at fit time (dtype mismatches)
  - Table 2: Raw output representation before evaluation normalization
  - Table 3: Effect of adapters on benchmark metrics
  - Table 4 (optional): Column-level datatype examples
  - Figure: Grouped bar chart (TSTR AUC, MIA AUC, KM-L1) Raw vs Adapted

  When any metric is all-NaN (e.g. DPMM adapted TSTR), writes
  outputs/figures/fig_adapter_ablation_diagnostics.txt listing which (impl, condition, metric).
  For deeper DPMM diagnostics (fit dtype, output structure, and metric error messages),
  run: python scripts/dpmm_adapter_diagnostics.py --schema ... --data ... --relax-n

Usage:
  python experiments/generate_adapter_ablation_artifacts.py \\
    --schema schemas/lung_schema.json --data data/lung_clean.csv \\
    --tables-dir outputs/tables --figures-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root on path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapter_ablation import run_adapter_ablation, AblationResult


def _fmt_num(x, decimals=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{float(x):.{decimals}f}"


def _fmt_pct(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{100 * float(x):.{decimals}f}\\%"


def write_tab_adapter_fit(results: list[AblationResult], out_path: str) -> None:
    """Table 1: Fit-time datatype interpretation with and without adapters."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{",
        r"Fit-time datatype interpretation with and without minimal schema adapters.",
        r"Mismatch count denotes the number of columns whose fit-time representation",
        r"is inconsistent with the declared schema.",
        r"}",
        r"\label{tab:adapter_fit}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Implementation & Condition & \# Columns & \# Dtype Mismatches & Mismatch Rate \\",
        r"\midrule",
    ]
    for r in results:
        n_cols = r.fit_dtype_summary.get("n_columns", 0)
        n_mis = r.fit_dtype_summary.get("n_mismatches", 0)
        rate = (n_mis / n_cols) if n_cols else 0.0
        impl = "SynthCity" if r.implementation == "synthcity" else "DPMM"
        cond = "Raw" if r.condition == "raw" else "Adapted"
        lines.append(f"{impl} & {cond} & {n_cols} & {n_mis} & {_fmt_pct(rate)} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def write_tab_adapter_output(results: list[AblationResult], out_path: str) -> None:
    """Table 2: Pre-normalization output diagnostics."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        r"Pre-normalization output diagnostics for raw and adapted baselines.",
        r"These statistics are computed on synthetic outputs before the common",
        r"evaluation-time schema normalization step.",
        r"}",
        r"\label{tab:adapter_output}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Implementation & Condition & Binary Invalid Rate & Categorical Invalid Rate & Integer-as-Float Column Rate & Out-of-Bounds Rate \\",
        r"\midrule",
    ]
    for r in results:
        s = r.pre_normalization_summary
        impl = "SynthCity" if r.implementation == "synthcity" else "DPMM"
        cond = "Raw" if r.condition == "raw" else "Adapted"
        b = _fmt_pct(s.get("binary_invalid_rate"))
        c = _fmt_pct(s.get("categorical_invalid_rate"))
        i = _fmt_pct(s.get("integer_float_column_rate"))
        o = _fmt_pct(s.get("out_of_bounds_rate"))
        lines.append(f"{impl} & {cond} & {b} & {c} & {i} & {o} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def write_tab_adapter_metrics(results: list[AblationResult], out_path: str) -> None:
    """Table 3: Effect of adapters on benchmark metrics."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        r"Adapter ablation under common evaluation-time normalization.",
        r"All synthetic outputs are normalized to the declared schema before scoring,",
        r"so differences reflect fit-time interpretation rather than arbitrary output formatting.",
        r"}",
        r"\label{tab:adapter_metrics}",
        r"\small",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Implementation & Condition & Marginal & Corr. & TSTR AUC & Attr. Inf. AUC & MIA AUC & NNDR & KM-L1 & Cox Rank \\",
        r"\midrule",
    ]
    keys = [
        "utility.marginal.mean_l1",
        "utility.correlation.value",
        "utility.tstr.roc_auc",
        "privacy.attribute_inference.auc",
        "privacy.mia.auc",
        "privacy.nndr.value",
        "survival.km_l1.value",
        "survival.cox_spearman.value",
    ]
    for r in results:
        impl = "SynthCity" if r.implementation == "synthcity" else "DPMM"
        cond = "Raw" if r.condition == "raw" else "Adapted"
        cells = [impl, cond]
        for k in keys:
            v = r.metrics.get(k)
            cells.append(_fmt_num(v, 3))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def write_tab_adapter_examples(
    results: list[AblationResult],
    schema: dict,
    out_path: str,
    example_columns: list[str] | None = None,
) -> None:
    """Table 4 (optional): Column-level datatype examples."""
    col_types = schema.get("column_types", {})
    if not example_columns:
        example_columns = [c for c in list(col_types.keys())[:6] if col_types.get(c)]

    # Get raw and adapted fit_dtype_summary for one implementation (e.g. SynthCity)
    raw_cols = {}
    adp_cols = {}
    for r in results:
        if r.implementation != "synthcity":
            continue
        for row in r.fit_dtype_summary.get("columns", []):
            col = row.get("column")
            if col not in example_columns:
                continue
            if r.condition == "raw":
                raw_cols[col] = row.get("dtype_seen", "")
            else:
                adp_cols[col] = row.get("dtype_seen", "")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Illustrative column-level datatype interpretation examples.}",
        r"\label{tab:adapter_examples}",
        r"\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Column & Declared Schema Type & Raw Fit Dtype & Adapted Fit Dtype \\",
        r"\midrule",
    ]
    for col in example_columns:
        stype = col_types.get(col, "---")
        raw_d = raw_cols.get(col, "---")
        adp_d = adp_cols.get(col, "---")
        # Escape underscores for LaTeX
        col_tex = col.replace("_", r"\_")
        lines.append(f"{col_tex} & {stype} & {raw_d} & {adp_d} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def write_fig_adapter_ablation(
    results: list[AblationResult],
    out_path: str,
    error_bar: str = "se",
) -> None:
    """Figure: Grouped bar chart — TSTR AUC, MIA AUC, KM-L1 (Raw vs Adapted), with optional error bars.

    error_bar: 'se' = ±1 standard error (narrower bars), '95ci' = 95% CI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    try:
        from analysis.figures import apply_style, save_figure
    except ImportError:
        def apply_style():
            plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
        def save_figure(fig, path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fig.savefig(path, bbox_inches="tight", format="pdf")
            print(f"Saved: {path}")

    apply_style()

    # Group by (impl, condition); if multiple runs, compute mean and error-bar half-width
    key_to_metrics: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        key = (r.implementation, r.condition)
        key_to_metrics.setdefault(key, []).append({
            "TSTR AUC": r.metrics.get("utility.tstr.roc_auc"),
            "MIA AUC": r.metrics.get("privacy.mia.auc"),
            "KM-L1": r.metrics.get("survival.km_l1.value"),
        })

    use_95ci = error_bar.lower() == "95ci"
    PLACEHOLDER_WHEN_MISSING = 0.01  # small visible bar when metric is all NaN
    missing_metrics: list[tuple[str, str, str]] = []  # (impl, condition, metric)

    def _mean_and_err(vals: list, impl: str, condition: str, metric: str) -> tuple[float, float]:
        """Return (mean, half_width). Use ±1 SE or 95% CI. Use placeholder for all-NaN so bar is visible."""
        arr = np.array([float(x) if x is not None and np.isfinite(x) else np.nan for x in vals])
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            missing_metrics.append((impl, condition, metric))
            return PLACEHOLDER_WHEN_MISSING, 0.0
        mean = float(np.nanmean(arr))
        if len(valid) < 2:
            return mean, 0.0
        n = len(valid)
        sem = float(np.nanstd(arr, ddof=1) / np.sqrt(n))
        half = float(stats.t.ppf(0.975, n - 1) * sem) if use_95ci else sem
        return mean, half

    metrics = ["TSTR AUC", "MIA AUC", "KM-L1"]
    x_impls = ["SynthCity", "DPMM"]
    raw_vals = {m: [] for m in metrics}
    raw_err = {m: [] for m in metrics}
    adp_vals = {m: [] for m in metrics}
    adp_err = {m: [] for m in metrics}
    for impl in ("synthcity", "dpmm"):
        for m in metrics:
            raw_list = [d.get(m) for d in key_to_metrics.get((impl, "raw"), [{}])]
            adp_list = [d.get(m) for d in key_to_metrics.get((impl, "adapted"), [{}])]
            m_r, e_r = _mean_and_err(raw_list, impl, "raw", m)
            m_a, e_a = _mean_and_err(adp_list, impl, "adapted", m)
            raw_vals[m].append(m_r)
            raw_err[m].append(e_r)
            adp_vals[m].append(m_a)
            adp_err[m].append(e_a)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3.8))
    x = np.arange(len(x_impls))
    w = 0.35

    ylabels = {"TSTR AUC": "TSTR AUC ↑", "MIA AUC": "MIA AUC ↓", "KM-L1": "KM-L1 ↓"}
    for ax, m in zip(axes, metrics):
        raw = np.asarray(raw_vals[m], dtype=float)
        adp = np.asarray(adp_vals[m], dtype=float)
        raw_err_arr = np.asarray(raw_err[m], dtype=float)
        adp_err_arr = np.asarray(adp_err[m], dtype=float)
        bars1 = ax.bar(x - w / 2, raw, w, yerr=raw_err_arr, label="Raw", color="tab:gray", alpha=0.8,
                       capsize=2, error_kw={"linewidth": 0.8})
        bars2 = ax.bar(x + w / 2, adp, w, yerr=adp_err_arr, label="Adapted", color="tab:blue", alpha=0.8,
                       capsize=2, error_kw={"linewidth": 0.8})
        ax.set_ylabel(ylabels.get(m, m))
        ax.set_xticks(x)
        ax.set_xticklabels(x_impls)
        ax.set_ylim(0, None)

    # Legend just below the title (in the margin between title and panels)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="tab:gray", alpha=0.8),
               plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.8)]
    fig.legend(handles, ["Raw", "Adapted"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.90), fontsize=8, frameon=True)

    fig.suptitle("Raw vs adapted metric change", fontsize=10, y=1.02)
    caption_extra = ""
    if missing_metrics:
        caption_extra = " Bars at 0.01 indicate missing metric (e.g. single-class synthetic labels or survival evaluation failure)."
    n_runs = len(results) // 4 if len(results) >= 4 else 1  # 4 = (synthcity, dpmm) x (raw, adapted)
    err_note = ""
    if n_runs >= 2:
        err_note = " Error bars: ±1 SE over runs." if not use_95ci else " Error bars: 95% CI over runs."
    else:
        err_note = " Single run; no confidence interval."
    fig.text(
        0.5, -0.02,
        "Effect of minimal schema adapters on benchmark outcomes for non-schema-native implementations. "
        "Adapters alter only datatype interpretation and output normalization and do not modify internal learning algorithms."
        + err_note
        + caption_extra,
        ha="center", fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    save_figure(fig, out_path)
    plt.close(fig)

    # Write diagnostics when any metric was missing (e.g. DPMM adapted)
    if missing_metrics:
        diag_path = os.path.join(os.path.dirname(out_path), "fig_adapter_ablation_diagnostics.txt")
        impl_label = lambda s: "SynthCity" if s == "synthcity" else "DPMM"
        lines = [
            "Adapter ablation figure: metrics that were all-NaN (bar shown at 0.01).",
            "Possible causes: TSTR AUC = single-class synthetic labels; MIA/KM-L1 = evaluation failure.",
            "",
        ]
        for impl, cond, m in missing_metrics:
            lines.append(f"  {impl_label(impl)} {cond}: {m}")
        try:
            with open(diag_path, "w") as f:
                f.write("\n".join(lines))
            print(f"Wrote diagnostics: {diag_path}")
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Run adapter ablation and generate tables and figure.",
    )
    parser.add_argument("--schema", required=True, help="Schema JSON path")
    parser.add_argument("--data", required=True, help="Real data CSV path")
    parser.add_argument("--tables-dir", default="outputs/tables", help="Directory for .tex tables")
    parser.add_argument("--figures-dir", default="outputs/figures", help="Directory for figure PDF")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-synth", type=int, default=None, help="Synthetic sample size (default: len(train))")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of runs for figure error bars (default: 1; use 3+ for SE or CI)")
    parser.add_argument("--error-bar", choices=["se", "95ci"], default="se", help="Error bar: se=±1 SE (narrower), 95ci=95%% CI (default: se)")
    parser.add_argument("--no-table4", action="store_true", help="Skip optional Table 4 (column examples)")
    args = parser.parse_args()

    with open(args.schema) as f:
        schema = json.load(f)
    df = pd.read_csv(args.data)

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=args.seed)
    test_df, holdout_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    all_results: list[AblationResult] = []
    for run in range(args.n_runs):
        seed = args.seed + run
        print(f"Running adapter ablation (SynthCity and DPMM, raw vs adapted) run {run + 1}/{args.n_runs} (seed={seed})...")
        run_results = run_adapter_ablation(
            train_df=train_df,
            test_real_df=test_df,
            holdout_df=holdout_df,
            schema=schema,
            epsilon=args.epsilon,
            n_synth=args.n_synth,
            seed=seed,
        )
        all_results.extend(run_results)

    # Tables use first run only
    results_first = all_results[:4] if len(all_results) >= 4 else all_results
    tables_dir = args.tables_dir
    figures_dir = args.figures_dir

    write_tab_adapter_fit(results_first, os.path.join(tables_dir, "tab_adapter_fit.tex"))
    write_tab_adapter_output(results_first, os.path.join(tables_dir, "tab_adapter_output.tex"))
    write_tab_adapter_metrics(results_first, os.path.join(tables_dir, "tab_adapter_metrics.tex"))
    if not args.no_table4:
        write_tab_adapter_examples(results_first, schema, os.path.join(tables_dir, "tab_adapter_examples.tex"))
    # Figure uses all runs for mean ± error bar when n_runs > 1
    write_fig_adapter_ablation(
        all_results,
        os.path.join(figures_dir, "fig_adapter_ablation.pdf"),
        error_bar=args.error_bar,
    )

    print("Done.")


if __name__ == "__main__":
    main()
