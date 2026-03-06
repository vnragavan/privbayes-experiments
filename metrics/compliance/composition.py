"""
metrics/compliance/composition.py

Identifies which privacy budget phases are declared and which
data-dependent operations are unaccounted for.
"""

from __future__ import annotations

# Known unaccounted data-dependent operations per implementation
_KNOWN_GAPS = {
    "crn": [],
    "dpmm": [
        "priv_tree_binner_epsilon (numeric binning epsilon outside main epsilon total)"
    ],
    "synthcity": [
        "pd.cut encoding (bounds inferred from data before DP)",
        "LabelEncoder.fit (categories inferred from data before DP)",
        "_compute_K uses len(data) — n is data-derived",
    ],
}

_DECLARED_PHASES = {
    "crn": [
        "structure_learning (eps_struct, exponential mechanism)",
        "cpt_estimation (eps_cpt, Laplace mechanism)",
        "metadata_bounds (eps_disc, when require_public=False)",
    ],
    "dpmm": [
        "structure_learning (epsilon/2, zCDP)",
        "cpt_estimation (remaining epsilon/2, Gaussian mechanism)",
    ],
    "synthcity": [
        "structure_learning (epsilon/2, exponential mechanism)",
        "cpt_estimation (epsilon/2, Laplace mechanism — but see gaps)",
    ],
}


def composition_summary(ledger: dict) -> dict:
    impl = ledger.get("_implementation", "unknown")
    gaps = _KNOWN_GAPS.get(impl, [])
    declared = _DECLARED_PHASES.get(impl, [])

    epsilon_declared = ledger.get("epsilon_total_declared")
    epsilon_struct = ledger.get("epsilon_structure")
    epsilon_cpt = ledger.get("epsilon_cpt")
    epsilon_disc = ledger.get("epsilon_disc", 0)

    # For CRN: check that parts sum to declared total
    gap_ratio = None
    if impl == "crn":
        try:
            parts = [
                float(epsilon_struct) if epsilon_struct not in (None, "undeclared") else 0.0,
                float(epsilon_cpt) if epsilon_cpt not in (None, "undeclared") else 0.0,
                float(epsilon_disc) if epsilon_disc not in (None, "undeclared") else 0.0,
            ]
            total_parts = sum(parts)
            total_declared = float(epsilon_declared) if epsilon_declared not in (None, "undeclared") else None
            if total_declared and total_declared > 0:
                gap_ratio = round(total_parts / total_declared, 4)
        except (TypeError, ValueError):
            pass

    return {
        "declared_phases": declared,
        "undeclared_data_dependent_phases": gaps,
        "gap_flag": len(gaps) > 0,
        "n_gaps": len(gaps),
        "composition_gap_ratio": gap_ratio,
        "note": (
            "gap_ratio=1.0 means declared total matches sum of declared phases. "
            "gap_ratio>1.0 means more was consumed than declared. "
            "None means not computable from available ledger data."
        ),
    }
