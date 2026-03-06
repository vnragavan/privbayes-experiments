"""
Figure 1: Compliance metrics — DBC and report completeness from normalised ledger.

Compliance is an implementation property: it does not depend on ε. Report completeness
and DBC are the same for all epsilon runs for a given implementation. We use one run
(e.g. ε = 1.0) only to obtain the ledger; the metrics would be identical for any ε.
Ledgers come from compliance.ledger in result files, or REPRESENTATIVE_LEDGERS as fallback.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.figures import apply_style, save_figure, COLOURS, LABELS, IMPL_ORDER
from metrics.compliance.dbc import compute_compliance_metrics

# Fallback when no result file or no compliance.ledger (all values should come from
# privacy report in normal use; run_experiment + compute_sweep_metrics preserve ledger).
REPRESENTATIVE_LEDGERS = {
    "crn": {
        "epsilon_total_declared": 1.0,
        "epsilon_structure": 0.3,
        "epsilon_cpt": 0.7,
        "epsilon_disc": 0.0,
    },
    "dpmm": {
        "epsilon_total_declared": 1.0,
        "epsilon_structure": "undeclared",
        "epsilon_cpt": "undeclared",
        "epsilon_disc": "undeclared",
    },
    "synthcity": {},
}


def _get_ledgers_from_results(results_dir: str, single_run_path: str):
    """Load compliance.ledger for each impl from a single run JSON if present."""
    path = (
        single_run_path
        if os.path.isabs(single_run_path)
        else os.path.join(results_dir or "results/eps_sweep", os.path.basename(single_run_path))
    )
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    out = {}
    for impl in IMPL_ORDER:
        if impl not in data or not isinstance(data[impl], dict):
            continue
        comp = data[impl].get("compliance", {})
        ledger = comp.get("ledger")
        if isinstance(ledger, dict):
            out[impl] = ledger
    return out if out else None


def _verify_compliance_invariant(results_dir: str, baseline_metrics: dict, seed: str = "seed0") -> None:
    """Check that compliance metrics are the same across a few epsilon runs (same seed)."""
    base = results_dir or "results/eps_sweep"
    checked = []
    for eps in ("0.1", "10.0"):
        name = f"results_eps{eps}_{seed}.json"
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            continue
        ledgers = _get_ledgers_from_results(results_dir, path)
        if not ledgers:
            continue
        same = True
        for impl in IMPL_ORDER:
            if impl not in ledgers:
                continue
            m = compute_compliance_metrics(ledgers[impl])
            b = baseline_metrics[impl]
            if m.report_completeness != b.report_completeness or m.dbc != b.dbc:
                print(f"  Warning: compliance for {impl} differs at ε={eps} (expected invariant)")
                same = False
        if same:
            checked.append(name)
    if checked:
        print(f"Figure 1: compliance identical across ε (checked: {', '.join(checked)})")


def main(out: str = "outputs/figures/fig1_compliance_heatmap.pdf",
         results_dir: str = "results/eps_sweep",
         single_run_json: str = "results_eps1.0_seed0.json"):
    apply_style()

    ledgers = REPRESENTATIVE_LEDGERS.copy()
    from_results = _get_ledgers_from_results(results_dir, single_run_json)
    data_source: str  # for subtitle and return
    if from_results:
        for impl in IMPL_ORDER:
            if impl in from_results:
                ledgers[impl] = from_results[impl]
        data_source = f"Ledger from: {single_run_json}"
        print(f"Figure 1: using ledger from result file ({single_run_json})")
    else:
        data_source = "Representative ledgers (no result file)"
        print("Figure 1: using fallback representative ledgers (no result file or no compliance.ledger)")

    metrics_by_impl = {impl: compute_compliance_metrics(ledgers[impl]) for impl in IMPL_ORDER}

    if from_results:
        # Compliance is implementation-defined and should not vary with ε
        seed = "seed0"
        if "_seed" in single_run_json:
            seed = single_run_json.split("_seed")[-1].replace(".json", "")
            seed = f"seed{seed}"
        _verify_compliance_invariant(results_dir, metrics_by_impl, seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    x = np.arange(len(IMPL_ORDER))
    width = 0.45

    # Left: Report completeness (0–1)
    completeness = [metrics_by_impl[impl].report_completeness for impl in IMPL_ORDER]
    bars1 = ax1.bar(x - width / 2, completeness, width, color=[COLOURS[impl] for impl in IMPL_ORDER], edgecolor="white", linewidth=1)
    ax1.set_ylabel("Report completeness")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[impl] for impl in IMPL_ORDER])
    ax1.set_title("Report completeness\n(required fields specified)")
    ax1.yaxis.grid(True, alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # Right: DBC (0–1 or N/A)
    dbc_vals = []
    dbc_labels = []
    for impl in IMPL_ORDER:
        m = metrics_by_impl[impl]
        if m.dbc is not None:
            dbc_vals.append(m.dbc)
            dbc_labels.append(None)
        else:
            dbc_vals.append(0.0)
            dbc_labels.append("N/A")
    bars2 = ax2.bar(x - width / 2, dbc_vals, width, color=[COLOURS[impl] for impl in IMPL_ORDER], edgecolor="white", linewidth=1)
    for i, label in enumerate(dbc_labels):
        if label:
            bars2[i].set_hatch("//")
            bars2[i].set_edgecolor("gray")
    ax2.set_ylabel("Declared Budget Coverage (DBC)")
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[impl] for impl in IMPL_ORDER])
    ax2.set_title("DBC\n(phase ε sum / ε total)")
    ax2.yaxis.grid(True, alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)
    # Annotate N/A bars
    for i, (val, lbl) in enumerate(zip(dbc_vals, dbc_labels)):
        if lbl:
            ax2.text(x[i] - width / 2, 0.02, "N/A", ha="center", fontsize=8, color="gray")

    fig.suptitle("Compliance metrics (invariant to ε; ledger from one run)", fontsize=11, y=1.02)
    fig.text(0.5, -0.02, data_source, ha="center", fontsize=8, color="gray", style="italic")
    plt.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    main()
