"""
Declared Budget Coverage (DBC) and report completeness on the normalised ledger.

Used by Figure 1 (compliance metrics plot). Ledger keys: epsilon_total_declared,
epsilon_structure, epsilon_cpt, epsilon_disc (no "accountant").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Normalised-ledger fields we use (no "accountant")
REQUIRED_KEYS: Tuple[str, ...] = (
    "epsilon_total_declared",
    "epsilon_structure",
    "epsilon_cpt",
    "epsilon_disc",
)

UNDECLARED_SENTINELS = {"undeclared", "n/a", "na", "null", "none", ""}


def _is_undeclared(x: Any) -> bool:
    """Treat None or sentinel strings as unspecified."""
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in UNDECLARED_SENTINELS:
        return True
    return False


def _to_float(x: Any) -> Optional[float]:
    """Parse numeric values; return None if unspecified or unparseable."""
    if _is_undeclared(x):
        return None
    if isinstance(x, (int, float)):
        try:
            v = float(x)
            return v if v == v else None  # NaN guard
        except Exception:
            return None
    if isinstance(x, str):
        try:
            v = float(x.strip())
            return v if v == v else None
        except Exception:
            return None
    return None


def compute_dbc(ledger: Dict[str, Any]) -> Optional[float]:
    """
    Declared Budget Coverage (DBC) on the normalised ledger.

    DBC = (sum of numeric declared phase epsilons) / epsilon_total_declared

    Phase epsilons: epsilon_structure, epsilon_cpt, epsilon_disc (if numeric).
    Returns None if:
      - epsilon_total_declared missing/non-numeric, or
      - no phase epsilon is numeric.
    """
    total = _to_float(ledger.get("epsilon_total_declared"))
    if total is None or total <= 0:
        return None

    parts = [
        _to_float(ledger.get("epsilon_structure")),
        _to_float(ledger.get("epsilon_cpt")),
        _to_float(ledger.get("epsilon_disc")),
    ]
    parts = [p for p in parts if p is not None]

    if not parts:
        return None

    return sum(parts) / total


def compute_report_completeness(ledger: Dict[str, Any]) -> float:
    """
    Report completeness on the normalised ledger (value-based).

    Completeness = (# required fields with specified values) / (# required fields)

    A field counts as specified if it is present and not 'undeclared'/None/empty.
    """
    if not REQUIRED_KEYS:
        return 1.0

    specified = 0
    for k in REQUIRED_KEYS:
        if k in ledger and not _is_undeclared(ledger.get(k)):
            specified += 1

    return specified / len(REQUIRED_KEYS)


@dataclass(frozen=True)
class ComplianceMetrics:
    dbc: Optional[float]
    report_completeness: float


def compute_compliance_metrics(ledger: Dict[str, Any]) -> ComplianceMetrics:
    return ComplianceMetrics(
        dbc=compute_dbc(ledger),
        report_completeness=compute_report_completeness(ledger),
    )
