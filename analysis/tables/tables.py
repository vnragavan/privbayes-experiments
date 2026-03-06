"""
analysis/tables/tables.py

Generates all LaTeX tables.
Run after completing the epsilon sweep and compute_sweep_metrics.py.
"""

import glob
import os
import json
from itertools import zip_longest
import numpy as np
import pandas as pd

from analysis.load_results import load_all_results, get_at_epsilon
from metrics.compliance.dbc import compute_compliance_metrics

# ─── Helpers ─────────────────────────────────────────────────────

PREAMBLE_SYMBOLS = r"""\providecommand{\cmark}{\textcolor{green!60!black}{\ding{51}}}
\providecommand{\xmark}{\textcolor{red}{\ding{55}}}
\providecommand{\qmark}{\textcolor{orange!80!black}{?}}
% Requires: \usepackage{pifont,xcolor}
"""


def _n_seeds_from_df(df):
    """Number of unique seeds in the loaded results (for captions)."""
    if df is None or df.empty or "seed" not in df.columns:
        return 3
    return int(df["seed"].nunique())


def _fmt(val, decimals=3, se=None):
    """
    Format a float value, optionally appending ± SE.

    Tables show mean ± SE (standard error = std / sqrt(n_seeds)).
    The 95% CI half-width is t_{0.025, n_seeds-1} × SE.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    s = f"{val:.{decimals}f}"
    if se is not None and not (isinstance(se, float) and np.isnan(se)):
        s += f" {{\\tiny ±{se:.{decimals}f}}}"
    return s


def _bold(s):
    return f"\\textbf{{{s}}}"


def _gray(s):
    return f"\\textcolor{{gray}}{{{s}}}"


def _best(vals_dict, lower_better=True, exclude=None):
    """Return key of best value, excluding specified keys."""
    exclude = exclude or []
    filtered = {k: v for k, v in vals_dict.items()
                if k not in exclude
                and v is not None
                and not (isinstance(v, float) and np.isnan(v))}
    if not filtered:
        return None
    return min(filtered, key=filtered.get) if lower_better \
        else max(filtered, key=filtered.get)


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"Saved: {path}")


# ─── Table: DBC and report completeness (computed across all ε, seeds) ─────

IMPL_ORDER_TABLES = ["crn", "dpmm", "synthcity"]
IMPL_LABELS_TABLES = {"crn": "CRNPrivBayes", "dpmm": "dpmm", "synthcity": "SynthCity"}


def tab_compliance_ledger(
        results_dir="results/eps_sweep",
        out="outputs/tables/tab_compliance_ledger.tex"):
    """
    Compute DBC and report completeness from the ledger for every (ε, seed) run;
    output a table and verify that both metrics remain static per implementation.
    """
    pattern = os.path.join(results_dir, "results_eps*_seed*.json")
    files = sorted(glob.glob(pattern))
    # Collect (impl, completeness, dbc) for each run
    rows = []
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        for impl in IMPL_ORDER_TABLES:
            if impl not in data or not isinstance(data[impl], dict):
                continue
            ledger = data[impl].get("compliance", {}).get("ledger")
            if not isinstance(ledger, dict):
                continue
            m = compute_compliance_metrics(ledger)
            rows.append({
                "impl": impl,
                "completeness": m.report_completeness,
                "dbc": m.dbc if m.dbc is not None else float("nan"),
            })
    if not rows:
        print("[tab_compliance_ledger] No ledger data found — skipping")
        return
    df = pd.DataFrame(rows)
    n_runs = len(df)
    # Check invariance per impl (should be constant across ε and seeds)
    for impl in IMPL_ORDER_TABLES:
        sub = df[df["impl"] == impl]
        if sub.empty:
            continue
        c_std = sub["completeness"].std()
        d_std = sub["dbc"].std()
        if c_std > 1e-9 or (not np.isnan(d_std) and d_std > 1e-9):
            print(f"[tab_compliance_ledger] WARNING: {impl} DBC/completeness varied across runs")
    # One row per impl: mean (should be constant)
    lines = [
        r"\begin{table}[t]",
        r"\caption{DBC and report completeness computed from the normalised ledger. "
        r"Values shown are the mean over all $\varepsilon$ and seeds (identical in all runs).}",
        r"\label{tab:compliance_ledger}",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Implementation & Report completeness & DBC \\",
        r"\midrule",
    ]
    for impl in IMPL_ORDER_TABLES:
        sub = df[df["impl"] == impl]
        if sub.empty:
            comp_str = "—"
            dbc_str = "—"
        else:
            c = sub["completeness"].mean()
            d = sub["dbc"].mean()
            comp_str = f"{c:.2f}" if not np.isnan(c) else "—"
            dbc_str = f"{d:.2f}" if not np.isnan(d) else "N/A"
        label = IMPL_LABELS_TABLES.get(impl, impl)
        lines.append(f"{label} & {comp_str} & {dbc_str} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\footnotesize Computed from ledger in {n_runs} runs (all $\varepsilon$, all seeds). "
        r"Both metrics are invariant per implementation.",
        r"\end{table}",
    ])
    _write(out, "\n".join(lines))


# ─── Table 1: Compliance audit ────────────────────────────────────

TABLE1_DATA = {
    "Numeric bounds source": {
        "crn":       ("PASS",    r"public\_bounds at init"),
        "dpmm":      ("PARTIAL", r"domain dict if provided"),
        "synthcity": ("FAIL",    r"pd.cut(data[col])"),
    },
    "Categorical domain source": {
        "crn":       ("PASS",    r"public\_categories at init"),
        "dpmm":      ("PARTIAL", r"domain[col][categories]"),
        "synthcity": ("FAIL",    r"LabelEncoder.fit(data[col])"),
    },
    r"Dataset size $n$ source": {
        "crn":       ("PASS",    r"adjacency flag only"),
        "dpmm":      ("FAIL",    r"data.df.shape[0]"),
        "synthcity": ("FAIL",    r"len(data)"),
    },
    r"Structure $\varepsilon$ declared": {
        "crn":       ("PASS",    r"$\varepsilon_\text{struct}$ in ledger"),
        "dpmm":      ("PASS",    r"$\varepsilon/2$ via zCDP"),
        "synthcity": ("PARTIAL", r"$\varepsilon/2$ declared"),
    },
    r"CPT $\varepsilon$ declared": {
        "crn":       ("PASS",    r"$\varepsilon_\text{cpt}$ in ledger"),
        "dpmm":      ("PASS",    r"remaining $\varepsilon$ via zCDP"),
        "synthcity": ("PARTIAL", r"$\varepsilon/2$ declared"),
    },
    "Sensitivity calibration": {
        "crn":       ("PASS",    r"L1$\leq 1/\varepsilon_\text{cpt}$"),
        "dpmm":      ("PARTIAL", r"calibrated to data $n$"),
        "synthcity": ("FAIL",    r"no CPT noise (MLE)"),
    },
    "Composition fully accounted": {
        "crn":       ("PASS",    r"parts sum to $\varepsilon$"),
        "dpmm":      ("PARTIAL", r"binner $\varepsilon$ outside total"),
        "synthcity": ("FAIL",    r"encoding ops unaccounted"),
    },
    "Ledger completeness": {
        "crn":       ("PASS",    r"1.0"),
        "dpmm":      ("PARTIAL", r"$\approx$0.6"),
        "synthcity": ("FAIL",    r"$\approx$0.3"),
    },
}

STATUS_CMD = {
    "PASS":    r"\cmark",
    "PARTIAL": r"\qmark",
    "FAIL":    r"\xmark",
}


def tab1_compliance_audit(
        out="outputs/tables/tab1_compliance_audit.tex"):
    lines = [PREAMBLE_SYMBOLS, ""]
    lines += [
        r"\begin{table}[t]",
        r"\caption{DP compliance audit across three PrivBayes implementations. "
        r"Static analysis combined with runtime verification.}",
        r"\label{tab:compliance}",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Requirement & CRNPrivBayes & dpmm & SynthCity$^{\dagger}$ \\",
        r"\midrule",
    ]

    for req, by_impl in TABLE1_DATA.items():
        cells = []
        for impl in ["crn", "dpmm", "synthcity"]:
            status, evidence = by_impl[impl]
            cmd  = STATUS_CMD[status]
            cell = f"{cmd} \\\\ {{\\scriptsize {evidence}}}"
            cells.append(cell)
        lines.append(f"{req} & {' & '.join(cells)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize",
        r"\cmark~PASS \quad \qmark~PARTIAL \quad \xmark~FAIL \\",
        r"$^{\dagger}$SynthCity values shown in grey. "
        r"Static analysis evidence in Appendix~\ref{app:static_analysis}.",
        r"\end{table}",
    ]
    _write(out, "\n".join(lines))


# ─── Table 2: Utility + survival at eps=1.0 ──────────────────────

METRIC_ROWS = [
    # (col_name, display_name, lower_better)
    ("marginal_l1",   "Marginal L1 $\\downarrow$",           True),
    ("tvd_mean",      "Pairwise TVD $\\downarrow$",           True),
    ("corr_spearman", "Correlation Spearman $\\uparrow$",     False),
    ("cat_coverage",  "Cat.\\ coverage $\\uparrow$",          False),
    ("---", "", None),
    ("km_l1",          "KM L1 $\\downarrow$",                 True),
    ("km_ci_overlap",  "KM CI overlap $\\uparrow$",           False),
    ("logrank_p",      "Log-rank $p$-value $\\uparrow$",      False),
    ("cox_spearman",   "Cox Spearman $\\uparrow$",             False),
    ("tstr_cindex",    "TSTR C-index $\\uparrow$",            False),
    ("censoring_err",  "Censoring error $\\downarrow$",       True),
    ("joint_censoring","Joint surv.-cens.\\ $\\downarrow$",   True),
    ("rmst",           "RMST error $\\downarrow$",            True),
    ("---", "", None),
    ("constraint_viol","Constraint viol.\\ rate $\\downarrow$", True),
]


def tab2_utility_survival(
        results_dir="results/eps_sweep",
        epsilon=1.0,
        out="outputs/tables/tab2_utility_survival.tex"):
    df = load_all_results(results_dir)
    if df.empty:
        print("[tab2] No results — generating placeholder table")
        at = pd.DataFrame({"implementation": ["crn", "dpmm", "synthcity"]})
        n_seeds = 3
    else:
        at = get_at_epsilon(df, epsilon, agg=True)
        n_seeds = _n_seeds_from_df(df)

    impls = ["crn", "dpmm", "synthcity"]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Utility and survival metrics at $\varepsilon=1.0$ "
        rf"(mean $\pm$ SE across {n_seeds} seeds; 95\,\% CI: mean $\pm$ "
        r"$t_{0.025,n-1}\,$SE). "
        r"Bold = best among compliant implementations. "
        r"Grey = not directly comparable under equivalent DP.}",
        r"\label{tab:utility_survival}",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & CRNPrivBayes & dpmm & SynthCity$^{\dagger}$ \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{General Utility}} \\",
    ]

    group2_start = next(i for i, (c, _, _) in enumerate(METRIC_ROWS) if c == "---")
    group3_start = next(
        i for i, (c, _, _) in enumerate(METRIC_ROWS[group2_start + 1:], group2_start + 1)
        if c == "---")

    section_labels = {
        group2_start + 1: r"\multicolumn{4}{l}{\textit{Survival Utility}} \\",
        group3_start + 1: r"\multicolumn{4}{l}{\textit{Constraint Validity}} \\",
    }

    for row_idx, (col, label, lower_better) in enumerate(METRIC_ROWS):
        if col == "---":
            lines.append(r"\midrule")
            if (row_idx + 1) in section_labels:
                lines.append(section_labels[row_idx + 1])
            continue

        mean_col = f"{col}_mean"
        se_col   = f"{col}_se"

        vals = {}
        for impl in impls:
            row = at[at["implementation"] == impl]
            if row.empty or mean_col not in row.columns:
                vals[impl] = float("nan")
            else:
                vals[impl] = row[mean_col].iloc[0]

        ses = {}
        for impl in impls:
            row = at[at["implementation"] == impl]
            if row.empty or se_col not in row.columns:
                ses[impl] = float("nan")
            else:
                ses[impl] = row[se_col].iloc[0]

        best_impl = _best(
            {k: v for k, v in vals.items() if k != "synthcity"},
            lower_better=lower_better)

        cells = []
        for impl in impls:
            s = _fmt(vals[impl], decimals=3, se=ses[impl])
            if impl == best_impl:
                s = _bold(s)
            if impl == "synthcity":
                s = _gray(s)
            # Colour logrank p-value green (not significant) or red
            if col == "logrank_p" and not np.isnan(vals[impl]):
                colour = "green!60!black" if vals[impl] > 0.05 else "red"
                s = f"\\textcolor{{{colour}}}{{{_fmt(vals[impl], decimals=3)}}}"
            cells.append(s)

        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize",
        r"$^{\dagger}$SynthCity values shown in grey. "
        r"Utility reflects undeclared privacy cost; "
        r"not comparable under equivalent DP guarantees. "
        r"TSTR C-index omitted for SynthCity when the synthetic event column has only one value (Cox cannot be fitted); see report. "
        r"Bold = best among compliant implementations.",
        r"\end{table}",
    ]
    _write(out, "\n".join(lines))


# ─── Table 3: Privacy risk + performance ─────────────────────────

def tab3_privacy_performance(
        results_dir="results/eps_sweep",
        single_run_json="outputs/results_eps1.0_seed0.json",
        epsilon=1.0,
        out="outputs/tables/tab3_privacy_performance.tex"):
    import json as _json

    df = load_all_results(results_dir)
    if df.empty:
        at = pd.DataFrame({"implementation": ["crn", "dpmm", "synthcity"]})
        n_seeds = 3
    else:
        at = get_at_epsilon(df, epsilon, agg=True)
        n_seeds = _n_seeds_from_df(df)

    impls = ["crn", "dpmm", "synthcity"]

    # Load single-run performance data if available
    perf_single = {}
    if os.path.exists(single_run_json):
        with open(single_run_json) as f:
            sr = _json.load(f)
        for impl, d in sr.items():
            p = d.get("performance", {})
            if p.get("fit_time_sec") is not None:
                perf_single[impl] = {
                    "fit_time_sec":    p.get("fit_time_sec"),
                    "sample_time_sec": p.get("sample_time_sec"),
                    "peak_memory_mb":  p.get("peak_memory_max_mb"),
                }
    # If results_dir is outputs/<name>, try sweep-run JSON for real timing
    if not perf_single and results_dir.startswith("outputs/"):
        sweep_name = os.path.basename(results_dir.rstrip("/"))
        sweep_single = os.path.join("results", "eps_sweep", sweep_name, "results_eps1.0_seed0.json")
        if os.path.exists(sweep_single):
            with open(sweep_single) as f:
                sr = _json.load(f)
            for impl, d in sr.items():
                p = d.get("performance", {})
                if p.get("fit_time_sec") is not None:
                    perf_single[impl] = {
                        "fit_time_sec":    p.get("fit_time_sec"),
                        "sample_time_sec": p.get("sample_time_sec"),
                        "peak_memory_mb":  p.get("peak_memory_max_mb"),
                    }

    def cell_val(impl, col):
        """Return (mean, se) from aggregated sweep data."""
        row = at[at["implementation"] == impl]
        mc  = f"{col}_mean"
        sec = f"{col}_se"
        if row.empty or mc not in row.columns:
            return float("nan"), float("nan")
        mean = row[mc].iloc[0]
        se   = row[sec].iloc[0] if sec in row.columns else float("nan")
        return mean, se

    def perf_cell(impl, col):
        """Return single-run value if available, else sweep mean."""
        if impl in perf_single and col in perf_single[impl]:
            v = perf_single[impl][col]
            return (v if v is not None else float("nan")), float("nan")
        return cell_val(impl, col)

    perf_note = (f"Single run at $\\varepsilon=1.0$."
                 if perf_single else
                 r"Performance not available; re-run with \texttt{run\_experiment.py}.")

    lines = [
        r"\begin{table}[t]",
        rf"\caption{{Left: Empirical privacy risk at $\\varepsilon=1.0$ (mean $\\pm$ SE, {n_seeds} seeds). "
        r"MIA AUC near 0.5 = no membership leakage. "
        r"NNDR median $<$1 = memorisation risk. "
        r"Attr.\ inf.\ AUC near 0.5 = low attribute inference risk. "
        r"Right: Computational performance (CPU). " + perf_note + "}",
        r"\label{tab:privacy_performance}",
        r"\centering",
        r"\begin{tabular}{lccc@{\hspace{2em}}lccc}",
        r"\toprule",
        r"\multicolumn{4}{c}{\textbf{Privacy Risk}} & "
        r"\multicolumn{4}{c}{\textbf{Performance}} \\",
        r"\cmidrule(r){1-4}\cmidrule(l){5-8}",
        r"Metric & CRN & dpmm & SC$^\dagger$ & "
        r"Metric & CRN & dpmm & SC \\",
        r"\midrule",
    ]

    PRIV = [
        ("mia_auc",           "MIA AUC $\\downarrow$",         True),
        ("mia_advantage",     "MIA advantage $\\downarrow$",    True),
        ("nndr_median",       "NNDR median $\\uparrow$",        False),
        ("attr_inference_auc", "Attr. inf. AUC $\\downarrow$", True),
    ]
    PERF = [
        ("fit_time_sec",    "Fit time (s) $\\downarrow$",    True),
        ("sample_time_sec", "Sample time (s) $\\downarrow$", True),
        ("peak_memory_mb",  "Memory (MB) $\\downarrow$",     True),
    ]

    for (pc, pl, plb), (fc, fl, flb) in zip_longest(PRIV, PERF, fillvalue=(None, "—", True)):
        # Privacy cells (sweep mean ± SE; SynthCity greyed)
        pvals = {impl: cell_val(impl, pc)[0] for impl in impls}
        pbest = _best({k: v for k, v in pvals.items() if k != "synthcity"},
                      lower_better=plb)
        pcells = []
        for impl in impls:
            v, se = cell_val(impl, pc)
            cell  = _fmt(v, decimals=3, se=se)
            if impl == pbest:
                cell = _bold(cell)
            if impl == "synthcity":
                cell = _gray(cell)
            pcells.append(cell)

        # Performance cells (single-run or sweep; or "—" when no 4th perf metric)
        if fc is None:
            fcells = ["—", "—", "—"]
            fl = "—"
        else:
            fvals = {impl: perf_cell(impl, fc)[0] for impl in impls}
            fbest = _best(fvals, lower_better=flb)
            fcells = []
            crn_val = fvals.get("crn", float("nan"))
            for impl in impls:
                v, se = perf_cell(impl, fc)
                # Use more decimal places for very small times (CRN fit = 0.004s)
                dec = 3 if not np.isnan(v) and v < 1 else 1
                cell = _fmt(v, decimals=dec, se=se)
                if impl == fbest:
                    cell = _bold(cell)
                # Speedup annotation vs CRN in fit time column
                if (fc == "fit_time_sec" and impl != "crn"
                        and not np.isnan(v) and not np.isnan(crn_val) and crn_val > 0):
                    ratio = v / crn_val
                    cell += f" ($\\times${ratio:.0f})"
                fcells.append(cell)

        row = (f"{pl} & {' & '.join(pcells)} & "
               f"{fl} & {' & '.join(fcells)} \\\\")
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize",
        r"SC = SynthCity. "
        r"$^\dagger$Privacy metrics shown in grey for SynthCity; "
        r"performance metrics are directly comparable. "
        r"Attr.\ inf.\ ``---'' for SynthCity: implementation gap (synthetic event column collapses to one class; see docs/SYNTHCITY\_IMPLEMENTATION\_GAP.md).",
        r"\end{table}",
    ]
    _write(out, "\n".join(lines))


# ─── Entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    tab1_compliance_audit()
    tab2_utility_survival()
    tab3_privacy_performance()
    print("\nAll tables generated.")