"""
analysis/figures/fig2_to_6.py

Figures 2-6 using loaded sweep results.
Run after completing the epsilon sweep and compute_sweep_metrics.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.figures import (
    apply_style, save_figure, COLOURS, LABELS,
    LINESTYLES, MARKERS, IMPL_ORDER, NONCOMPLIANT_FOOTNOTE)
from analysis.load_results import load_all_results, aggregate_over_seeds
EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # sweep values — used for xlim guard

# Compliance gap ratios for Fig 4 — derived from runtime compliance ledger
# CRN:       gap = 1.0  — epsilon_total_declared = epsilon_requested (0 gaps)
# dpmm:      gap = 1.1  — declares full epsilon but 1 undeclared binner phase
# SynthCity: gap = 2.0  — epsilon_total_declared = epsilon/2 (halves internally),
#                          PLUS 3 undeclared data-dependent phases.
#                          Confirmed from JSON: at eps=1.0, declared=0.5 → ratio=2.0
GAP_RATIOS = {"synthcity": 2.0, "dpmm": 1.1, "crn": 1.0}

GAP_EVIDENCE = {
    "synthcity": "ε_declared = ε_claimed/2 (confirmed from ledger) + 3 undeclared phases → ×2.0",
    "dpmm":      "1 undeclared binner phase, conservative estimate → ×1.1",
    "crn":       "0 gaps, fully accounted → ×1.0",
}

# SynthCity often produces synthetic survival data with only one event type (e.g. all censored),
# so the Cox model cannot be fitted and TSTR C-index is not computed (shown as missing/—).
TSTR_SYNTHCITY_FOOTNOTE = (
    "TSTR C-index: SynthCity may be missing when synthetic event column has only one value "
    "(Cox model not fitted on synthetic data)."
)


def _safe_log_x(ax):
    """Set log x-scale with explicit positive xlim to avoid matplotlib crash
    when axes contain no plotted data (all-NaN or all-zero panels)."""
    lo = min(e for e in EPSILONS) * 0.7
    hi = max(e for e in EPSILONS) * 1.5
    ax.set_xlim(lo, hi)
    ax.set_xscale("log")




def _plot_line(ax, agg_df, metric_col, impl, label=None, alpha=1.0,
               band="se"):
    """
    Plot a metric line with shaded uncertainty band.

    Parameters
    ----------
    band : "se"  — shade mean ± 1 SE  (default; appropriate for n=3 seeds)
           "std" — shade mean ± 1 std  (wider; shows raw spread)
           "ci"  — shade 95 % CI  (very wide at n=3, t_crit=4.303)
           None  — no shading
    """
    sub = agg_df[agg_df["implementation"] == impl].sort_values("epsilon")
    if sub.empty:
        return

    x     = sub["epsilon"].values
    m_col = f"{metric_col}_mean"
    y     = sub[m_col].values if m_col in sub.columns else np.full(len(x), np.nan)

    ax.plot(x, y,
            color=COLOURS[impl], ls=LINESTYLES[impl],
            marker=MARKERS[impl], markersize=4, alpha=alpha,
            label=label or LABELS[impl])

    if band == "se":
        se_col = f"{metric_col}_se"
        if se_col in sub.columns:
            ye = sub[se_col].values
            ax.fill_between(x, y - ye, y + ye,
                            color=COLOURS[impl], alpha=0.15 * alpha)
    elif band == "std":
        s_col = f"{metric_col}_std"
        if s_col in sub.columns:
            ye = sub[s_col].values
            ax.fill_between(x, y - ye, y + ye,
                            color=COLOURS[impl], alpha=0.12 * alpha)
    elif band == "ci":
        lo_col = f"{metric_col}_ci_lo"
        hi_col = f"{metric_col}_ci_hi"
        if lo_col in sub.columns and hi_col in sub.columns:
            ax.fill_between(x,
                            sub[lo_col].values,
                            sub[hi_col].values,
                            color=COLOURS[impl], alpha=0.12 * alpha)


# ─── Figure 2: Utility curves ────────────────────────────────────

def _n_seeds_from_df(df):
    """Number of unique seeds in the loaded results (for captions)."""
    if df.empty or "seed" not in df.columns:
        return 3  # default for backward compatibility
    return int(df["seed"].nunique())


def fig2_utility_curves(results_dir="results/eps_sweep",
                         out="outputs/figures/fig2_utility_curves.pdf"):
    apply_style()
    df = load_all_results(results_dir)
    if df.empty:
        print("[fig2] No results found — skipping")
        return
    agg = aggregate_over_seeds(df)
    n_seeds = _n_seeds_from_df(df)

    PANELS = [
        ("marginal_l1",   "Marginal L1 ↓",              True),
        ("tvd_mean",      "Pairwise TVD ↓",             True),
        ("wasserstein_mean", "Mean Wasserstein ↓",      True),
        ("corr_spearman", "Correlation Spearman ↑",      False),
        ("cat_coverage",  "Cat. coverage ↑",             False),
        ("constraint_viol","Constraint viol. rate ↓",    True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for idx, (metric, ylabel, _lower_better) in enumerate(PANELS):
        ax = axes[idx]
        for impl in IMPL_ORDER:
            alpha = 0.5 if impl == "synthcity" else 1.0
            _plot_line(ax, agg, metric, impl, alpha=alpha)
        _safe_log_x(ax)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ylabel, fontsize=9)

    axes[0].legend(fontsize=7)
    # Hide any unused subplot slots (last panel in 2x3 grid)
    for j in range(len(PANELS), len(axes)):
        axes[j].set_visible(False)
    fig.text(0.5, -0.02, NONCOMPLIANT_FOOTNOTE,
             ha="center", fontsize=7.5, style="italic")
    fig.text(0.5, -0.05,
             f"Shaded bands: mean ± 1 SE across {n_seeds} seeds.  {TSTR_SYNTHCITY_FOOTNOTE}",
             ha="center", fontsize=7, color="gray")
    fig.suptitle("Utility metrics across privacy budget", y=1.02)
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


# ─── Figure 3: Survival metrics vs privacy budget ─────────────────

def fig3_survival_curves(results_dir="results/eps_sweep",
                          data_dir="outputs",
                          out="outputs/figures/fig3_survival_curves.pdf"):
    """
    Survival metrics vs ε (all six metrics from sweep results).
    Uses the same data and layout as fig_survival_all_in_one so the
    pipeline's main survival figure is fully driven by sweep data.
    """
    apply_style()
    df = load_all_results(results_dir)
    if df.empty:
        print("[fig3] No results found — skipping")
        return
    agg = aggregate_over_seeds(df)
    n_seeds = _n_seeds_from_df(df)

    panels = [
        ("km_l1",              "KM L1 ↓",                    None),
        ("km_ci_overlap",      "KM CI overlap ↑",            None),
        ("tstr_cindex",        "TSTR C-index ↑",             None),
        ("cox_spearman",       "Cox coef. Spearman ↑",       None),
        ("censoring_err",      "1 − Censoring error ↑",      lambda m, s: (1.0 - m, s)),
        ("joint_censoring",    "Joint surv.-cens. ↓",         None),
    ]
    n_panels = len(panels)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for idx, (metric_col, ylabel, transform) in enumerate(panels):
        ax = axes[idx]
        for impl in IMPL_ORDER:
            sub = agg[agg["implementation"] == impl].sort_values("epsilon")
            if sub.empty:
                continue
            x = sub["epsilon"].values
            m_col = f"{metric_col}_mean"
            se_col = f"{metric_col}_se"
            if m_col not in sub.columns:
                continue
            y = sub[m_col].values.copy()
            ye = sub[se_col].values if se_col in sub.columns else None
            if transform is not None:
                y, ye = transform(y, ye)
            alpha = 0.5 if impl == "synthcity" else 1.0
            ax.plot(x, y, color=COLOURS[impl], ls=LINESTYLES[impl],
                    marker=MARKERS[impl], markersize=3, alpha=alpha, label=LABELS[impl])
            if ye is not None and not (np.isnan(ye).all() if hasattr(ye, '__iter__') else np.isnan(ye)):
                ax.fill_between(x, y - ye, y + ye, color=COLOURS[impl], alpha=0.12 * alpha)
        _safe_log_x(ax)
        ax.set_xlabel(r"$\varepsilon$", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ylabel, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(axis="both", labelsize=7)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Survival metrics vs privacy budget", fontsize=11, y=1.02)
    fig.text(0.5, -0.02,
             f"Mean ± 1 SE across {n_seeds} seeds.  {NONCOMPLIANT_FOOTNOTE}  {TSTR_SYNTHCITY_FOOTNOTE}",
             ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


# ─── Survival metrics vs ε (all in one figure) ───────────────────

def fig_survival_all_in_one(results_dir="results/eps_sweep",
                            out="outputs/figures/fig_survival_all_in_one.pdf"):
    """All survival metrics vs ε in one figure (one panel per metric)."""
    apply_style()
    df = load_all_results(results_dir)
    if df.empty:
        print("[fig_survival_all_in_one] No results found — skipping")
        return
    agg = aggregate_over_seeds(df)
    n_seeds = _n_seeds_from_df(df)

    # Panels: (metric_col, ylabel, transform). transform=None means plot mean as-is.
    panels = [
        ("km_l1",              "KM L1 ↓",                    None),
        ("km_ci_overlap",      "KM CI overlap ↑",            None),
        ("tstr_cindex",        "TSTR C-index ↑",             None),
        ("cox_spearman",       "Cox coef. Spearman ↑",       None),
        ("censoring_err",      "1 − Censoring error ↑",      lambda m, s: (1.0 - m, s)),  # plot 1 - mean, same SE
        ("joint_censoring",    "Joint surv.-cens. ↓",        None),
    ]
    n_panels = len(panels)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for idx, (metric_col, ylabel, transform) in enumerate(panels):
        ax = axes[idx]
        for impl in IMPL_ORDER:
            sub = agg[agg["implementation"] == impl].sort_values("epsilon")
            if sub.empty:
                continue
            x = sub["epsilon"].values
            m_col = f"{metric_col}_mean"
            se_col = f"{metric_col}_se"
            if m_col not in sub.columns:
                continue
            y = sub[m_col].values.copy()
            ye = sub[se_col].values if se_col in sub.columns else None
            if transform is not None:
                y, ye = transform(y, ye)
            else:
                ye = ye  # no change
            alpha = 0.5 if impl == "synthcity" else 1.0
            ax.plot(x, y, color=COLOURS[impl], ls=LINESTYLES[impl],
                    marker=MARKERS[impl], markersize=3, alpha=alpha, label=LABELS[impl])
            if ye is not None and not (np.isnan(ye).all() if hasattr(ye, '__iter__') else np.isnan(ye)):
                ax.fill_between(x, y - ye, y + ye, color=COLOURS[impl], alpha=0.12 * alpha)
        _safe_log_x(ax)
        ax.set_xlabel(r"$\varepsilon$", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ylabel, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(axis="both", labelsize=7)

    # Hide any unused subplot slots (if panels < grid size)
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Survival metrics vs privacy budget", fontsize=11, y=1.02)
    fig.text(0.5, -0.02,
             f"Mean ± 1 SE across {n_seeds} seeds.  {NONCOMPLIANT_FOOTNOTE}  {TSTR_SYNTHCITY_FOOTNOTE}",
             ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


# ─── Survival metrics vs ε (separate graphs) ─────────────────────

def fig_survival_km_ci(results_dir="results/eps_sweep",
                       out="outputs/figures/fig_survival_km_ci.pdf"):
    """KM CI overlap vs ε (one graph)."""
    apply_style()
    df = load_all_results(results_dir)
    if df.empty:
        print("[fig_survival_km_ci] No results found — skipping")
        return
    agg = aggregate_over_seeds(df)
    n_seeds = _n_seeds_from_df(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    for impl in IMPL_ORDER:
        alpha = 0.5 if impl == "synthcity" else 1.0
        _plot_line(ax, agg, "km_ci_overlap", impl, alpha=alpha)
    _safe_log_x(ax)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("KM CI overlap ↑")
    ax.set_title("KM CI overlap vs privacy budget")
    ax.legend(fontsize=7)
    fig.text(0.5, -0.05,
             f"Mean ± 1 SE across {n_seeds} seeds.  {NONCOMPLIANT_FOOTNOTE}",
             ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def fig_survival_1_minus_censoring_err(results_dir="results/eps_sweep",
                                       out="outputs/figures/fig_survival_1_minus_censoring_err.pdf"):
    """1 − Censoring error vs ε (higher is better)."""
    apply_style()
    df = load_all_results(results_dir)
    if df.empty:
        print("[fig_survival_1_minus_censoring_err] No results found — skipping")
        return
    agg = aggregate_over_seeds(df)
    n_seeds = _n_seeds_from_df(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    for impl in IMPL_ORDER:
        sub = agg[agg["implementation"] == impl].sort_values("epsilon")
        if sub.empty:
            continue
        x = sub["epsilon"].values
        m_col = "censoring_err_mean"
        se_col = "censoring_err_se"
        y = 1.0 - sub[m_col].values if m_col in sub.columns else np.full(len(x), np.nan)
        ye = sub[se_col].values if se_col in sub.columns else None
        alpha = 0.5 if impl == "synthcity" else 1.0
        ax.plot(x, y, color=COLOURS[impl], ls=LINESTYLES[impl],
                marker=MARKERS[impl], markersize=4, alpha=alpha, label=LABELS[impl])
        if ye is not None and not np.isnan(ye).all():
            ax.fill_between(x, y - ye, y + ye, color=COLOURS[impl], alpha=0.15 * alpha)
    _safe_log_x(ax)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("1 − Censoring error ↑")
    ax.set_title("1 − Censoring error vs privacy budget")
    ax.legend(fontsize=7)
    fig.text(0.5, -0.05,
             f"Mean ± 1 SE across {n_seeds} seeds.  {NONCOMPLIANT_FOOTNOTE}",
             ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def fig_survival_joint_censoring(results_dir="results/eps_sweep",
                                 out="outputs/figures/fig_survival_joint_censoring.pdf"):
    """Joint survival-censoring distance vs ε (lower is better)."""
    apply_style()
    df = load_all_results(results_dir)
    if df.empty:
        print("[fig_survival_joint_censoring] No results found — skipping")
        return
    agg = aggregate_over_seeds(df)
    n_seeds = _n_seeds_from_df(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    for impl in IMPL_ORDER:
        alpha = 0.5 if impl == "synthcity" else 1.0
        _plot_line(ax, agg, "joint_censoring", impl, alpha=alpha)
    _safe_log_x(ax)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Joint survival-censoring distance ↓")
    ax.set_title("Joint survival-censoring vs privacy budget")
    ax.legend(fontsize=7)
    fig.text(0.5, -0.05,
             f"Mean ± 1 SE across {n_seeds} seeds.  {NONCOMPLIANT_FOOTNOTE}",
             ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


# ─── Figure 4: Compliance-adjusted utility ───────────────────────

def fig4_compliance_adjusted(
        results_dir="results/eps_sweep",
        out="outputs/figures/fig4_compliance_adjusted.pdf"):
    """
    Naive vs compliance-adjusted utility comparison.
    Gap ratios: SynthCity ×1.25 (3 undeclared phases), dpmm ×1.10 (1 binner phase).
    CRN is unchanged (0 gaps).
    """
    apply_style()
    df = load_all_results(results_dir)
    if df.empty:
        print("[fig4] No results found — skipping")
        return
    agg = aggregate_over_seeds(df)
    n_seeds = _n_seeds_from_df(df)

    metric = "tstr_cindex"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for panel_idx, (ax, title) in enumerate(zip(axes, [
            "(a) Naive comparison (claimed ε)",
            "(b) Honest ε after compliance adjustment"])):
        for impl in IMPL_ORDER:
            sub = agg[agg["implementation"] == impl].sort_values("epsilon")
            if sub.empty:
                continue
            col = f"{metric}_mean"
            x   = sub["epsilon"].values
            y   = sub[col].values if col in sub.columns else np.full(len(x), np.nan)
            gap = GAP_RATIOS.get(impl, 1.0)

            if panel_idx == 1 and gap > 1.0:
                x_plot = x * gap
                # Arrow annotation at ε=1.0 for non-CRN implementations
                ix = np.argmin(np.abs(x - 1.0))
                if ix < len(x):
                    y_pt = y[ix] if not np.isnan(y[ix]) else 0.5
                    offset = 0.015 if impl == "synthcity" else -0.015
                    ax.annotate(
                        "", xy=(x_plot[ix], y_pt),
                        xytext=(x[ix], y_pt),
                        arrowprops=dict(
                            arrowstyle="->", color=COLOURS[impl], lw=1.5))
                    ax.text(
                        (x[ix] + x_plot[ix]) / 2,
                        y_pt + offset,
                        f"×{gap}",
                        fontsize=7, color=COLOURS[impl], ha="center")
            else:
                x_plot = x

            alpha = 0.5 if impl == "synthcity" else 1.0
            ax.plot(x_plot, y,
                    color=COLOURS[impl], ls=LINESTYLES[impl],
                    marker=MARKERS[impl], markersize=4, alpha=alpha,
                    label=LABELS[impl])

            # SE band
            se_col = f"{metric}_se"
            if se_col in sub.columns:
                ye = sub[se_col].values
                ax.fill_between(x_plot, y - ye, y + ye,
                                color=COLOURS[impl], alpha=0.15 * alpha)
        if panel_idx == 0:
            # Annotate SynthCity at lowest epsilon as "appears superior"
            sub_sc = agg[agg["implementation"] == "synthcity"].sort_values("epsilon")
            col = f"{metric}_mean"
            if not sub_sc.empty and col in sub_sc.columns:
                x0 = sub_sc["epsilon"].iloc[0]
                y0 = sub_sc[col].iloc[0]
                if not np.isnan(y0):
                    ax.annotate("Appears superior\nat low ε",
                                xy=(x0, y0), xytext=(x0 * 2, y0 - 0.04),
                                fontsize=7.5, color=COLOURS["synthcity"],
                                arrowprops=dict(arrowstyle="->",
                                                color=COLOURS["synthcity"]))

        _safe_log_x(ax)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_title(title, fontsize=10)

    axes[0].set_ylabel("TSTR C-index ↑")
    axes[0].legend(fontsize=7)

    fig.text(
        0.5, -0.05,
        r"Honest $\varepsilon$ = claimed $\varepsilon$ $\times$ gap ratio.  "
        r"CRN: $\times$1.0 (0 gaps, fully declared).  "
        r"dpmm: $\times$1.1 (1 undeclared binner phase).  "
        r"SynthCity: $\times$2.0 ($\varepsilon_\mathrm{declared}=\varepsilon_\mathrm{claimed}/2$"
        r" confirmed from ledger, plus 3 undeclared phases).",
        ha="center", fontsize=7.5, style="italic")
    fig.text(0.5, -0.09,
             f"Shaded bands: mean ± 1 SE across {n_seeds} seeds.  {TSTR_SYNTHCITY_FOOTNOTE}",
             ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


# ─── Figure 5: Privacy risk ───────────────────────────────────────

def fig5_privacy_risk(results_dir="results/eps_sweep",
                       out="outputs/figures/fig5_privacy_risk.pdf"):
    apply_style()
    df = load_all_results(results_dir)
    if df.empty:
        print("[fig5] No results found — skipping")
        return
    agg = aggregate_over_seeds(df)
    n_seeds = _n_seeds_from_df(df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # MIA AUC
    ax = axes[0]
    for impl in IMPL_ORDER:
        alpha = 0.5 if impl == "synthcity" else 1.0
        _plot_line(ax, agg, "mia_auc", impl, alpha=alpha)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random baseline")
    _safe_log_x(ax)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("MIA AUC")
    ax.set_title("Membership Inference Attack AUC")
    ax.legend(fontsize=7)

    # NNDR
    ax2 = axes[1]
    for impl in IMPL_ORDER:
        alpha = 0.5 if impl == "synthcity" else 1.0
        _plot_line(ax2, agg, "nndr_mean", impl, alpha=alpha)
    ax2.axhline(1.0, color="gray", ls="--", lw=1, label="Ideal (ratio=1)")
    _safe_log_x(ax2)
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel("NNDR mean ratio")
    ax2.set_title("Nearest-Neighbour Distance Ratio")
    ax2.legend(fontsize=7)

    # Attribute inference AUC (wider ylim: AUC can be < 0.5 when synthetic data
    # weakens or inverts the relationship, e.g. dpmm at some ε)
    ax3 = axes[2]
    for impl in IMPL_ORDER:
        alpha = 0.5 if impl == "synthcity" else 1.0
        _plot_line(ax3, agg, "attr_inference_auc", impl, alpha=alpha)
    ax3.axhline(0.5, color="gray", ls="--", lw=1, label="Random baseline")
    _safe_log_x(ax3)
    ax3.set_ylim(0.25, 1.0)
    ax3.set_xlabel(r"$\varepsilon$")
    ax3.set_ylabel("Attr. inf. AUC")
    ax3.set_title("Attribute Inference AUC")
    ax3.legend(fontsize=7)

    fig.text(0.5, -0.03, NONCOMPLIANT_FOOTNOTE,
             ha="center", fontsize=7.5, style="italic")
    fig.text(0.5, -0.06,
             f"Shaded bands: mean ± 1 SE across {n_seeds} seeds. "
             "AUC < 0.5 can occur when synthetic data weakens or inverts the target–feature relationship.",
             ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


# ─── Figure 6: Performance ────────────────────────────────────────

def _load_single_run_perf(path):
    """Load performance dict from a single-run JSON; returns {} if missing or all null."""
    import json as _json
    perf = {}
    if path and os.path.exists(path):
        with open(path) as f:
            single = _json.load(f)
        for impl, data in single.items():
            p = data.get("performance", {})
            if p.get("fit_time_sec") is not None:
                perf[impl] = {
                    "fit_time_sec":    p.get("fit_time_sec"),
                    "sample_time_sec": p.get("sample_time_sec"),
                    "peak_memory_mb":  p.get("peak_memory_max_mb"),
                }
    return perf


def _perf_results_dir(results_dir: str) -> str:
    """Directory that has performance data. If results_dir is outputs/<name>, use results/eps_sweep/<name>."""
    if results_dir.startswith("outputs/"):
        name = os.path.basename(results_dir.rstrip("/"))
        return os.path.join("results", "eps_sweep", name)
    return results_dir


def _single_run_json_from_results_dir(results_dir: str) -> str:
    """Preferred single-run JSON path: results_dir/results_eps1.0_seed0.json."""
    return os.path.join(results_dir.rstrip("/"), "results_eps1.0_seed0.json")


def fig6_performance(results_dir="results/eps_sweep",
                      single_run_json=None,
                      out="outputs/figures/fig6_performance.pdf"):
    """
    Performance figure: mean ± 95% CI of fit time, sample time, and peak memory
    across seeds for each epsilon (using sweep data). Covers all epsilons in the sweep.
    If sweep has no timing (e.g. results from compute_sweep_metrics only),
    falls back to single-run bar chart at ε=1.0.
    Legend is placed above the panels.
    """
    apply_style()
    if single_run_json is None:
        single_run_json = _single_run_json_from_results_dir(results_dir)
    perf_dir = _perf_results_dir(results_dir)
    df = load_all_results(perf_dir)
    has_sweep_perf = (
        not df.empty
        and "fit_time_sec" in df.columns
        and df["fit_time_sec"].notna().any()
    )

    if has_sweep_perf:
        # Use grouped bar charts: for each epsilon, one bar per implementation.
        agg = aggregate_over_seeds(df)
        n_seeds = _n_seeds_from_df(df)
        epsilons_used = sorted(agg["epsilon"].unique())

        PERF_PANELS = [
            ("fit_time_sec",    "Fit time (s) ↓"),
            ("sample_time_sec", "Sample time (s) ↓"),
            ("peak_memory_mb",  "Peak memory (MB) ↓"),
        ]

        # Only keep implementations that actually appear in the results
        present_impls = [
            impl for impl in IMPL_ORDER
            if impl in agg["implementation"].unique()
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        x_eps = np.arange(len(epsilons_used))
        n_impl = max(len(present_impls), 1)
        width = 0.8 / n_impl  # keep groups compact within each epsilon

        for ax, (metric, ylabel) in zip(axes, PERF_PANELS):
            for j, impl in enumerate(present_impls):
                sub = agg[agg["implementation"] == impl].sort_values("epsilon")
                if sub.empty:
                    continue
                # Align rows to the global epsilon grid so bars line up
                sub = sub.set_index("epsilon").reindex(epsilons_used)

                mean_col = f"{metric}_mean"
                se_col   = f"{metric}_se"
                means = sub[mean_col].values if mean_col in sub.columns else np.full(len(x_eps), np.nan)
                ses   = sub[se_col].values   if se_col in sub.columns   else np.zeros(len(x_eps))

                # Center the group of bars around each epsilon tick
                offset = (j - (n_impl - 1) / 2) * width
                x_pos = x_eps + offset

                alpha = 0.5 if impl == "synthcity" else 1.0
                bars = ax.bar(
                    x_pos, means, width,
                    color=COLOURS[impl], alpha=alpha,
                    label=LABELS[impl]
                )

                # 95% CI error bars using 1.96 * SE (n≈3 seeds)
                if np.any(ses > 0):
                    ax.errorbar(
                        x_pos, means, yerr=1.96 * ses,
                        fmt="none", ecolor="black",
                        capsize=3, linewidth=0.8
                    )

            ax.set_xticks(x_eps)
            ax.set_xticklabels([str(e) for e in epsilons_used])
            ax.set_xlabel(r"$\varepsilon$")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(ylabel, fontsize=9)
            ax.yaxis.grid(True, alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

        # Legend above the panels
        if present_impls:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=7, loc="upper center",
                       bbox_to_anchor=(0.5, 1.02), ncol=len(present_impls), frameon=True)
        fig.text(0.5, -0.02, NONCOMPLIANT_FOOTNOTE,
                 ha="center", fontsize=7.5, style="italic")
        fig.text(0.5, -0.05,
                 f"Mean ± 95% CI across {n_seeds} seeds per ε. "
                 f"Epsilons: {', '.join(str(e) for e in epsilons_used)}.",
                 ha="center", fontsize=7, color="gray")
        fig.suptitle("Performance across privacy budget", y=1.02)
        plt.tight_layout()
        save_figure(fig, out)
        plt.close(fig)
        return

    # Fallback: single-run bar chart at ε=1.0 (use results_dir first)
    perf = _load_single_run_perf(single_run_json)
    if not perf:
        alt_path = _single_run_json_from_results_dir(results_dir)
        if alt_path != single_run_json:
            perf = _load_single_run_perf(alt_path)
            if perf:
                single_run_json = alt_path
    if not perf and results_dir.startswith("outputs/"):
        sweep_name = os.path.basename(results_dir.rstrip("/"))
        sweep_single = os.path.join("results", "eps_sweep", sweep_name, "results_eps1.0_seed0.json")
        perf = _load_single_run_perf(sweep_single)
        if perf:
            single_run_json = sweep_single

    if not perf:
        df_fb = load_all_results(results_dir)
        if not df_fb.empty:
            agg_fb = aggregate_over_seeds(df_fb)
            for impl in IMPL_ORDER:
                sub = agg_fb[agg_fb["implementation"] == impl]
                if sub.empty:
                    continue
                perf[impl] = {
                    "fit_time_sec":    sub["fit_time_sec_mean"].mean(),
                    "sample_time_sec": sub["sample_time_sec_mean"].mean(),
                    "peak_memory_mb":  sub["peak_memory_mb_mean"].mean(),
                }

    if not perf:
        print("[fig6] No performance data available — skipping")
        return

    PERF_PANELS = [
        ("fit_time_sec",    "Fit time (s) ↓",       True),
        ("sample_time_sec", "Sample time (s) ↓",    True),
        ("peak_memory_mb",  "Peak memory (MB) ↓",   True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    impl_list = [i for i in IMPL_ORDER if i in perf]
    x = np.arange(len(impl_list))
    bar_labels = [LABELS[i].replace(" (proposed)", "\n(proposed)") for i in impl_list]

    for ax, (metric, ylabel, _) in zip(axes, PERF_PANELS):
        vals = [perf[i].get(metric) for i in impl_list]
        colours = [COLOURS[i] for i in impl_list]
        bars = ax.bar(x, vals, color=colours, alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{val:.3g}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        if metric == "fit_time_sec" and "crn" in perf:
            crn_t = perf["crn"].get("fit_time_sec", 0)
            if crn_t and crn_t > 0:
                for impl in ["dpmm", "synthcity"]:
                    if impl in perf:
                        other_t = perf[impl].get("fit_time_sec")
                        if other_t and other_t > 0:
                            ratio = other_t / crn_t
                            xi = impl_list.index(impl)
                            ax.text(xi, other_t * 0.5,
                                    f"×{ratio:.0f} slower",
                                    ha="center", va="center",
                                    fontsize=7.5, color="white", fontweight="bold")

    # Legend above the panels (single-run fallback)
    if impl_list:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=COLOURS[i], alpha=0.85, label=LABELS[i]) for i in impl_list]
        fig.legend(handles, [LABELS[i] for i in impl_list], fontsize=7, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=len(impl_list), frameon=True)
    fig.text(0.5, -0.03,
             f"Single run at ε=1.0 (no sweep timing in {results_dir}).",
             ha="center", fontsize=7.5, color="gray")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    fig2_utility_curves()
    fig3_survival_curves()
    fig4_compliance_adjusted()
    fig5_privacy_risk()
    fig6_performance()
