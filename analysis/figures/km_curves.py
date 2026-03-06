"""
analysis/figures/km_curves.py

Multi-panel figure: real vs synthetic Kaplan-Meier curves with confidence intervals,
one panel per epsilon. Shows how well synthetic survival curves overlap the real one.
"""

import json
import os
import re
import glob
import sys
from pathlib import Path

# Allow running as script: ensure project root is on path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from analysis.figures import (
    apply_style, save_figure, COLOURS, LABELS,
    IMPL_ORDER, NONCOMPLIANT_FOOTNOTE,
)


def _get_survival_cols(schema: dict):
    """Return (event_col, duration_col) from schema target_spec, or (None, None)."""
    ts = schema.get("target_spec", {})
    if ts.get("kind") != "survival_pair":
        return None, None
    event_col = ts.get("primary_target")
    targets = ts.get("targets", [])
    duration_col = next((t for t in targets if t != event_col), None)
    return event_col, duration_col


def _km_curve_and_ci(df, duration_col, event_col, timeline):
    """
    Fit KM on df and return survival curve and 95% CI interpolated at timeline.
    Returns (surv, ci_lo, ci_hi) arrays of length len(timeline).
    """
    kmf = KaplanMeierFitter()
    dur = pd.to_numeric(df[duration_col], errors="coerce").fillna(0)
    evt = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    kmf.fit(dur, evt)
    surv = kmf.survival_function_at_times(timeline).values
    ci = kmf.confidence_interval_survival_function_
    if ci is None or ci.empty:
        return surv, surv, surv
    t_ci = ci.index.values.astype(float)
    ci_lo = np.interp(timeline, t_ci, ci.iloc[:, 0].values)
    ci_hi = np.interp(timeline, t_ci, ci.iloc[:, 1].values)
    return surv, ci_lo, ci_hi


def _load_synthetic_curves_per_epsilon(results_dir, epsilon, event_col, duration_col, timeline):
    """
    For one epsilon, load all synthetic CSVs (crn/dpmm/synthcity × seeds),
    compute KM curve on timeline for each, return dict impl -> (surv_mean, ci_lo, ci_hi).
    """
    out = {}
    for impl in IMPL_ORDER:
        pattern = os.path.join(results_dir, f"{impl}_eps{epsilon}_seed*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        curves = []
        for path in files:
            df = pd.read_csv(path)
            if event_col not in df.columns or duration_col not in df.columns:
                continue
            surv, _, _ = _km_curve_and_ci(df, duration_col, event_col, timeline)
            curves.append(surv)
        if not curves:
            continue
        curves = np.array(curves)
        out[impl] = (
            np.nanmean(curves, axis=0),
            np.nanpercentile(curves, 2.5, axis=0),
            np.nanpercentile(curves, 97.5, axis=0),
        )
    return out


def fig_km_real_vs_synthetic(
    results_dir,
    schema_path,
    data_path,
    out="outputs/figures/fig_km_real_vs_synthetic.pdf",
    n_grid=200,
    epsilons=None,
):
    """
    Plot real vs synthetic Kaplan-Meier curves with confidence intervals,
    one panel per epsilon.

    Parameters
    ----------
    results_dir : str
        Directory containing synthetic CSVs (e.g. results/eps_sweep/ncctg_lung).
    schema_path : str
        Path to schema JSON (for event_col, duration_col).
    data_path : str
        Path to real data CSV.
    out : str
        Output PDF path.
    n_grid : int
        Number of time points for the survival curve.
    epsilons : list of float, optional
        Epsilons to plot; if None, inferred from result filenames in results_dir.
    """
    apply_style()

    with open(schema_path) as f:
        schema = json.load(f)
    event_col, duration_col = _get_survival_cols(schema)
    if not event_col or not duration_col:
        print("[fig_km_real_vs_synthetic] Schema is not survival_pair — skipping")
        return

    real_df = pd.read_csv(data_path)
    if event_col not in real_df.columns or duration_col not in real_df.columns:
        print(f"[fig_km_real_vs_synthetic] Real data missing {event_col} or {duration_col} — skipping")
        return

    # Discover epsilons from synthetic CSVs if not provided
    if epsilons is None:
        pattern = os.path.join(results_dir, "crn_eps*_seed*.csv")
        files = glob.glob(pattern)
        epsilons = []
        for path in files:
            m = re.search(r"eps([\d.]+)_seed", os.path.basename(path))
            if m:
                try:
                    epsilons.append(float(m.group(1)))
                except ValueError:
                    pass
        epsilons = sorted(set(epsilons)) if epsilons else [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Common timeline: 0 to max duration (real and synthetic)
    t_max_real = float(pd.to_numeric(real_df[duration_col], errors="coerce").max())
    t_max = t_max_real
    for eps in epsilons:
        for impl in IMPL_ORDER:
            g = os.path.join(results_dir, f"{impl}_eps{eps}_seed*.csv")
            for path in glob.glob(g):
                df = pd.read_csv(path)
                if duration_col in df.columns:
                    t_max = max(t_max, float(pd.to_numeric(df[duration_col], errors="coerce").max()))
    timeline = np.linspace(0, t_max, n_grid)

    # Real KM and CI
    surv_real, ci_lo_real, ci_hi_real = _km_curve_and_ci(
        real_df, duration_col, event_col, timeline
    )

    n_panels = len(epsilons)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, eps in enumerate(epsilons):
        ax = axes[idx]
        # Real: black line and shaded CI
        ax.fill_between(timeline, ci_lo_real, ci_hi_real, color="black", alpha=0.2)
        ax.plot(timeline, surv_real, color="black", lw=2, label="Real", zorder=10)

        # Synthetic per implementation
        syn = _load_synthetic_curves_per_epsilon(
            results_dir, eps, event_col, duration_col, timeline
        )
        for impl in IMPL_ORDER:
            if impl not in syn:
                continue
            mean_s, lo_s, hi_s = syn[impl]
            ax.fill_between(timeline, lo_s, hi_s, color=COLOURS[impl], alpha=0.2)
            ax.plot(timeline, mean_s, color=COLOURS[impl], lw=1.5,
                    label=LABELS[impl], ls="-")

        ax.set_xlim(0, t_max)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        ax.set_title(rf"$\varepsilon = {eps}$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Real vs synthetic Kaplan–Meier curves (mean ± 95% CI)", fontsize=11, y=1.02)
    fig.text(0.5, -0.02,
             "Real: black. Synthetic: mean across seeds with pointwise 95% band. " + NONCOMPLIANT_FOOTNOTE,
             ha="center", fontsize=7, color="gray")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot real vs synthetic KM curves, one panel per epsilon.")
    p.add_argument("--results-dir", required=True, help="Sweep results dir (e.g. results/eps_sweep/ncctg_lung)")
    p.add_argument("--schema", required=True, help="Schema JSON path")
    p.add_argument("--data", required=True, help="Real data CSV path")
    p.add_argument("--out", default="outputs/figures/fig_km_real_vs_synthetic.pdf")
    args = p.parse_args()
    fig_km_real_vs_synthetic(
        results_dir=args.results_dir,
        schema_path=args.schema,
        data_path=args.data,
        out=args.out,
    )
