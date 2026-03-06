"""
metrics/compliance/ledger.py

Normalises privacy_report() from any wrapper into a standard ledger format.
"""

from __future__ import annotations

STANDARD_LEDGER_KEYS = [
    "epsilon_total_declared",
    "epsilon_structure",
    "epsilon_cpt",
    "epsilon_disc",
    "delta",
    "mechanism_structure",
    "mechanism_cpt",
    "adjacency",
    "n_source",
    "composition_method",
]

_CRN_MAP = {
    "epsilon_total_declared": "epsilon",
    "epsilon_structure":      "eps_struct",
    "epsilon_cpt":            "eps_cpt",
    "epsilon_disc":           "eps_disc",
    "delta":                  "delta",
    "mechanism_structure":    "mechanism_structure",
    "mechanism_cpt":          "mechanism_cpt",
    "adjacency":              "adjacency",
    "n_source":               "metadata_mode",
    "composition_method":     "composition",
}

_DPMM_MAP = {
    "epsilon_total_declared": "epsilon",
    "epsilon_structure":      None,
    "epsilon_cpt":            None,
    "epsilon_disc":           None,
    "delta":                  "delta",
    "mechanism_structure":    None,
    "mechanism_cpt":          None,
    "adjacency":              None,
    "n_source":               "n_source",
    "composition_method":     None,
}

_SYNTHCITY_MAP = {
    "epsilon_total_declared": "epsilon",
    "epsilon_structure":      None,
    "epsilon_cpt":            None,
    "epsilon_disc":           None,
    "delta":                  None,
    "mechanism_structure":    None,
    "mechanism_cpt":          None,
    "adjacency":              None,
    "n_source":               "n_source",
    "composition_method":     None,
}

_MAPS = {"crn": _CRN_MAP, "dpmm": _DPMM_MAP, "synthcity": _SYNTHCITY_MAP}


def build_ledger(privacy_report: dict, implementation: str) -> dict:
    """
    Normalise privacy_report() to standard ledger format.
    Missing fields → "undeclared".
    """
    mapping = _MAPS.get(implementation, {})
    ledger = {}
    for key in STANDARD_LEDGER_KEYS:
        src = mapping.get(key)
        if src and src in privacy_report:
            ledger[key] = privacy_report[src]
        else:
            ledger[key] = "undeclared"

    # Pass through schema section if present (post-patch CRN)
    if "schema" in privacy_report:
        ledger["schema"] = privacy_report["schema"]

    ledger["_implementation"] = implementation
    ledger["_raw"] = privacy_report
    return ledger


def ledger_completeness_score(ledger: dict) -> float:
    declared = sum(
        1 for k in STANDARD_LEDGER_KEYS
        if ledger.get(k) not in ("undeclared", None))
    return declared / len(STANDARD_LEDGER_KEYS)


def print_ledger_comparison(ledgers: dict) -> None:
    impls = list(ledgers.keys())
    col_w = 18
    header = "Requirement".ljust(30) + "".join(i.rjust(col_w) for i in impls)
    print(header)
    print("-" * len(header))
    for key in STANDARD_LEDGER_KEYS:
        row = key.ljust(30)
        for impl in impls:
            val = ledgers.get(impl, {}).get(key, "-")
            row += str(val)[:col_w - 2].rjust(col_w)
        print(row)
    print("-" * len(header))
    row = "Completeness score".ljust(30)
    for impl in impls:
        score = ledger_completeness_score(ledgers.get(impl, {}))
        row += f"{score:.2f}".rjust(col_w)
    print(row)
