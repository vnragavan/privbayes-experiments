"""
analysis/figures/__init__.py

Shared visual style for all figures.
Import apply_style() and COLOURS at the top of every figure script.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

COLOURS = {
    "crn":       "#2ca02c",
    "dpmm":      "#ff7f0e",
    "synthcity": "#1f77b4",
}

LABELS = {
    "crn":       "CRNPrivBayes (proposed)",
    "dpmm":      "dpmm",
    "synthcity": r"SynthCity$^\dagger$",
}

LINESTYLES = {"crn": "-", "dpmm": "--", "synthcity": ":"}
MARKERS    = {"crn": "o", "dpmm": "s", "synthcity": "^"}
IMPL_ORDER = ["crn", "dpmm", "synthcity"]

NONCOMPLIANT_FOOTNOTE = (
    r"$^\dagger$SynthCity $\varepsilon$ reflects claimed, "
    "not verified, DP guarantee."
)

def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "pdf.fonttype": 42,
    })

def save_figure(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    print(f"Saved: {path}")
