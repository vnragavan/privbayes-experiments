#!/usr/bin/env python3
"""
schema_generator.py — Generates a schema JSON from a CSV file, aligned with:

  - schema_validator.py (column_types, public_bounds, public_categories, target_spec,
    constraints including survival_pair, provenance; optional sensitive_attributes)
  - implementations: CRN (crn_privbayes), dpmm (via adapters/schema_to_dpmm),
    SynthCity (wrapper uses public_bounds/public_categories for clipping)
  - metrics/report.py and compute_sweep_metrics (target_spec.primary_target,
    get_attribute_inference_target(schema) which uses sensitive_attributes or primary_target)

Schema generation modes
-----------------------
  --schema-mode public  (default)
      The schema is treated as public knowledge — inferred bounds and categories
      consume no privacy budget.  The full synthesis epsilon is available for M_synth.

  --schema-mode private  --schema-epsilon FLOAT
      The schema is inferred from the private dataset D.  Numeric bounds and tau
      are released under epsilon_1-DP (Laplace mechanism with conservative one-sided
      noise), tracking per-column budget via domain_provenance.  The remaining
      epsilon_2 = epsilon_total - epsilon_1 should be allocated to synthesis.

      Budget allocation: sequential composition across columns (conservative).
        epsilon_per_col = schema_epsilon / n_private_numeric_cols
        min query:  epsilon_per_col / 2
        max query:  epsilon_per_col / 2
        tau:        subsumed by the time column max query (no additional cost)

      Sensitivity model: local sensitivity = empirical range (max - min).
      Noise direction: one-sided and conservative — noise widens bounds, never
      narrows them, so the calibration contract is never violated.

      DP coverage in private mode
      ---------------------------
      Covered (epsilon-DP with budget accounting):
        numeric bounds (continuous / integer / ordinal / datetime / survival time)
        n_records  — Laplace noise, sensitivity 1     (opt-in via --dp-n-records)
        missing_value_rates — Laplace per column,    (opt-in via --dp-missing-rates)
                              sensitivity 1/n, clipped to [0,1], parallel composition

      Not covered — marked as non_private_auxiliary with explicit warnings:
        categorical domains inferred via --infer-categories
        ordinal level lists inferred from small-integer columns
      These require DPSU (Differentially Private Set Union) for strict DP;
      they must be declared as public overrides for a fully DP pipeline.

      Sensitivity model: local sensitivity = empirical range (max - min).
      Noise direction: one-sided and conservative for bounds.

Theory alignment  S = (A, T, Omega, Gamma, Pi, P)
--------------------------------------------------
  A       → column_types.keys()
  T       → column_types values
  Omega   → public_bounds + public_categories  (domain per column)
  Gamma   → sensitivity_bounds  (explicit per-column sensitivity, new v1.2)
  Pi      → mechanism_hints     (advisory default mechanism; new v1.2)
  P       → domain_provenance   (public | private(epsilon_1) | non_private_auxiliary)

Gamma derivation by type
  continuous / integer : max - min  (L1 sensitivity of range-bounded queries)
  binary               : 1.0        (single record changes 0<->1)
  ordinal              : max - min  (numeric range)
  categorical          : 2.0        (L1 sensitivity of the full normalised histogram
                                     vector — NOT a single-cell count sensitivity;
                                     downstream consumers must interpret accordingly)

Pi assignment by type — ADVISORY HINTS ONLY, not normative DP requirements.
  PrivBayes-style histogram mechanisms typically use Laplace after discretization
  regardless of the per-column Pi hint.
  binary / categorical : exponential  (finite non-numeric domain)
  ordinal / integer    : laplace      (discrete numeric range)
  continuous           : gaussian     (continuous range; (eps,delta)-DP)
                         use laplace if pure eps-DP (no delta) is required
  survival event col   : exponential
  survival time col    : laplace      (sensitivity = tau)

Changes in v1.6 (remove redundant user-facing arguments)
-----------------------------------
  R. --pb-max-numeric-bins now defaults to None (auto).  When not explicitly
     set, the cap is derived from n_records as:
       cap = max(sturges(n), ceil(n^(1/3)), 5)  capped at hard ceiling 20
     This means pb-max-numeric-bins never needs to appear in a standard
     command — the data-driven Sturges bin count is already ≤ the auto cap
     for all typical clinical dataset sizes.  Override only when targeting
     a specific CPT memory budget.
  S. --pad-frac default is 0.0 (unchanged from v1.0).  Clarified in docs
     that this argument should be omitted unless non-zero padding is wanted.
     Explicitly passing --pad-frac 0.0 is always a no-op.
Changes in v1.5 (data-driven PrivBayes parameters)
-----------------------------------
  K. Public mode: n_bins now uses Sturges rule (ceil(log2(n)+1), capped at
     pb_max_numeric_bins) instead of a flat default of 8.
  L. Public mode: discretization strategy (equal_width vs quantile) now
     chosen by Bowley skewness — quantile if |skew| > 0.2, else equal_width.
  M. Public mode: dirichlet_alpha now uses Perks prior (1/K) where K is
     n_bins_total, instead of a flat Jeffreys prior of 0.5.
  N. Public mode: max_parents now chosen as the smallest k such that
     max_bins_total^k >= n (sensitivity crossover point).
  O. Private mode: all four parameters above use DP-safe equivalents:
       n_bins     — Sturges on DP-released n_records (post-processing, free)
       strategy   — Bowley skewness via Laplace-noised quartiles,
                    5% of schema_epsilon, sequential composition
       alpha      — Perks on DP-derived n_bins_total (post-processing, free)
       max_parents — crossover on DP n_records and n_bins_total (free)
  P. New --pb-strategy-epsilon-frac CLI arg (default 0.05) controls the
     fraction of schema_epsilon spent on DP strategy selection.
  Q. Per-column discretization entries now carry:
       skewness         : Bowley skewness value used for strategy decision
       skewness_source  : "data" | "dp_laplace" | "default"
-----------------------------------
  G. _infer_privbayes_extensions now accepts missing_value_rates and
     pb_max_numeric_bins; per-column discretization entries gain:
       n_bins_total  = n_bins + 1 if column has any missing values
       has_nan_bin   = true/false
       nan_bin_index = n_bins  (0-indexed; NaN bin is always the last bin)
     This makes the NaN-as-extra-bin strategy explicit and machine-readable.
  H. n_bins in the PrivBayes extension's per_column entries is capped at
     pb_max_numeric_bins (default 10) to prevent CPT explosion.
     NOTE: the core schema's public_bounds[col]["n_bins"] still stores
     min(n_unique, 100) — the raw domain metadata, not the modeling cap.
     The cap applies only inside extensions.privbayes.discretization.per_column.
     Consumers should use public_bounds n_bins for domain queries and
     extensions per_column n_bins for CPT sizing.
  I. New --pb-max-numeric-bins CLI arg (default 10) controls the cap.
  J. Top-level missing_value_handling block added to extensions.
-----------------------------------
  A. n_records privatised under Laplace(1) with --dp-n-records flag
  B. missing_value_rates privatised per column under Laplace(1/n)
     with --dp-missing-rates flag; parallel composition
  C. Categorical / ordinal domains inferred from data marked
     non_private_auxiliary in domain_provenance; explicit warning emitted
  D. Sensitivity docstring clarifies categorical 2.0 = histogram L1 vector
  E. mechanism_hints docstring clarifies advisory (not normative) status
  F. _default_pb_max_parents labelled explicitly as heuristic in provenance
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.6.0"
MAX_INTEGER_LEVELS = 20   # integer columns with ≤ this many unique values → ordinal

# Schema provenance modes
_MODE_PUBLIC  = "public"
_MODE_PRIVATE = "private"

_UUID_RE = re.compile(
    r"^(?:"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    r"|[0-9a-fA-F]{32}"
    r"|\{[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\}"
    r"|urn:uuid:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    r")$"
)

_TARGET_HINTS = [
    "target", "label", "class", "outcome", "income",
    "event", "status", "died", "death", "recurrence",
    "arrest", "failure", "relapse", "deceased", "censored",
]
_TIME_HINTS = [
    "time", "duration", "survival_time", "follow_up",
    "days", "weeks", "months", "time_to_event", "week",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _infer_target_col(cols: list[str]) -> str | None:
    cl = [c.lower() for c in cols]
    for hint in _TARGET_HINTS:
        if hint in cl:
            return cols[cl.index(hint)]
    return None


def _infer_time_col(cols: list[str], exclude: str | None = None) -> str | None:
    cl = [c.lower() for c in cols]
    for hint in _TIME_HINTS:
        if hint in cl:
            c = cols[cl.index(hint)]
            if c != exclude:
                return c
    return None


def _parse_csv_list(v: str | None) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def _infer_csv_delimiter(path: Path) -> str:
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
        if not sample.strip():
            return ","
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        d = getattr(dialect, "delimiter", ",")
        return d if d else ","
    except Exception:
        return ","


def _infer_target_dtype(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return "unknown"
    s = df[col]
    if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
        return "categorical"
    x = pd.to_numeric(s, errors="coerce")
    xn = x[np.isfinite(x)].to_numpy(dtype=float)
    if xn.size == 0:
        return "continuous"
    if np.all(np.isclose(xn, np.round(xn), atol=1e-8)):
        return "integer"
    return "continuous"


def _target_dtype_from_column_type(column_type: str | None) -> str | None:
    if not isinstance(column_type, str):
        return None
    t = column_type.strip().lower()
    if t in {"integer", "continuous", "categorical", "ordinal", "binary"}:
        return t
    return None


def _ordinal_bounds_from_categories(cat_list: list[str]) -> dict[str, Any]:
    nums: list[float] = []
    for x in cat_list:
        try:
            nums.append(float(str(x).strip().replace(",", ".")))
        except ValueError:
            continue
    if not nums:
        return {"min": 0, "max": 1, "n_bins": 2}
    lo, hi = int(min(nums)), int(max(nums))
    if lo == hi:
        hi = lo + 1
    return {"min": lo, "max": hi, "n_bins": max(len(cat_list), 2)}


def _guess_datetime_output_format(raw: pd.Series) -> str:
    samples = pd.Series(raw, copy=False).astype("string").dropna().astype(str).str.strip()
    if samples.empty:
        return "%Y-%m-%dT%H:%M:%S"
    patterns = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
        "%d/%m/%Y %H:%M", "%d/%m/%Y",
        "%m-%d-%Y %H:%M", "%m-%d-%Y",
        "%Y/%m/%dT%H:%M:%SZ", "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d", "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d",
    ]
    for fmt in patterns:
        try:
            dt = pd.to_datetime(samples, format=fmt, errors="coerce")
            if float(dt.notna().mean()) >= 0.8:
                return fmt
        except Exception:
            continue
    return "%Y-%m-%dT%H:%M:%S"


def _maybe_parse_datetime_like(
    s: pd.Series, *, min_parse_frac: float
) -> tuple[pd.Series, bool, str | None]:
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(pd.Series(s, copy=False), errors="coerce")
        v = dt.astype("int64").astype("float64")
        v[dt.isna()] = np.nan
        return v, True, "%Y-%m-%dT%H:%M:%S"
    if pd.api.types.is_timedelta64_dtype(s):
        td = pd.to_timedelta(pd.Series(s, copy=False), errors="coerce")
        v = td.astype("int64").astype("float64")
        v[td.isna()] = np.nan
        return v, True, "%Y-%m-%dT%H:%M:%S"
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        raw = pd.Series(s, copy=False).astype("string").replace(
            {"": pd.NA, " ": pd.NA, "null": pd.NA, "NULL": pd.NA,
             "none": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA,
             "nat": pd.NA, "NaT": pd.NA}
        )
        parse_attempts = [
            {"format": "mixed", "utc": True}, {"utc": True},
            {"dayfirst": True, "utc": True}, {"yearfirst": True, "utc": True},
            {"dayfirst": True, "yearfirst": True, "utc": True},
        ]
        best_dt, best_frac = None, -1.0
        for kw in parse_attempts:
            try:
                dt = pd.to_datetime(raw, errors="coerce", **kw)
                frac = float(dt.notna().mean())
                if frac > best_frac:
                    best_frac, best_dt = frac, dt
            except Exception:
                continue
        if best_dt is not None and best_frac >= float(min_parse_frac):
            v = best_dt.astype("int64").astype("float64")
            v[best_dt.isna()] = np.nan
            return v, True, _guess_datetime_output_format(raw)
    return s, False, None


def _is_guid_like_series(s: pd.Series, *, min_match_frac: float = 0.95) -> bool:
    if not (s.dtype == "object" or pd.api.types.is_string_dtype(s)):
        return False
    x = pd.Series(s, copy=False).astype("string").dropna()
    if x.empty:
        return False
    # FIX #3: use _UUID_RE.pattern (string), not the compiled object
    return float(x.str.strip().str.fullmatch(_UUID_RE.pattern, na=False).mean()) >= float(min_match_frac)


def _is_number_like_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
        return True
    if s.dtype == "object":
        return bool(pd.to_numeric(s, errors="coerce").notna().mean() >= 0.95)
    return False


def _bounds_for_number_like(
    s: pd.Series, pad_frac: float, *, integer_like: bool = False
) -> list[float | int]:
    """Public-mode bound estimation — no DP noise applied."""
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0, 1] if integer_like else [0.0, 1.0]
    vmin, vmax = float(np.min(x)), float(np.max(x))
    span = vmax - vmin
    pad = (pad_frac * span) if span > 0 else (max(abs(vmin) * pad_frac, 1.0) if pad_frac > 0 else 0.0)
    lo, hi = vmin - pad, vmax + pad
    if integer_like:
        return [int(np.floor(lo)), int(np.ceil(hi))]
    return [lo, hi]


# ---------------------------------------------------------------------------
# NEW v1.2: Private-mode DP bound estimation
# ---------------------------------------------------------------------------

def _dp_private_bounds(
    s: pd.Series,
    epsilon: float,
    pad_frac: float,
    *,
    integer_like: bool = False,
) -> tuple[list[float | int], float]:
    """
    Estimate numeric bounds under epsilon-DP using the Laplace mechanism.

    Budget split: epsilon/2 per bound (min query + max query, sequential composition).
    Sensitivity model: local sensitivity proxy = empirical range (max - min).
    Noise direction: one-sided and conservative — noise is subtracted from the
    lower bound and added to the upper bound, so released bounds always contain
    the true domain.  This upholds the calibration contract: sensitivity derived
    from S will never underestimate the true data range.

    Returns (bounds, epsilon_spent).  epsilon_spent = epsilon if data was present,
    0.0 for degenerate (empty/constant) columns.

    Limitations
    -----------
    Local sensitivity (empirical range) is used as a proxy for global sensitivity.
    This is standard practice in schema estimation but is not strictly epsilon-DP
    in the worst case (a record at the boundary can shift the range by up to the
    full domain width).  For rigorous epsilon-DP use smooth sensitivity or
    propose-test-release with a declared outer bound.
    """
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return ([0, 1] if integer_like else [0.0, 1.0]), 0.0

    emp_min, emp_max = float(np.min(x)), float(np.max(x))
    emp_range = emp_max - emp_min

    if emp_range <= 0.0:
        # Degenerate / constant column — widen minimally, no meaningful DP noise
        lo, hi = emp_min - 1.0, emp_max + 1.0
        eps_spent = 0.0
    else:
        # Each bound query gets epsilon/2 (sequential composition)
        eps_half = max(epsilon / 2.0, 1e-12)
        scale = emp_range / eps_half
        rng = np.random.default_rng()  # intentionally non-seeded for DP randomness

        # Absolute noise values — always widen, never narrow
        noise_lo = abs(float(rng.laplace(loc=0.0, scale=scale)))
        noise_hi = abs(float(rng.laplace(loc=0.0, scale=scale)))
        lo = emp_min - noise_lo
        hi = emp_max + noise_hi
        eps_spent = float(epsilon)

    # Apply pad_frac on top of DP-widened bounds
    span = hi - lo
    pad = (pad_frac * span) if span > 0 else (max(abs(lo) * pad_frac, 1.0) if pad_frac > 0 else 0.0)
    lo -= pad
    hi += pad

    if integer_like:
        return [int(np.floor(lo)), int(np.ceil(hi))], eps_spent
    return [lo, hi], eps_spent


def _dp_private_tau(noised_upper_bound: float) -> int:
    """
    Derive the DP tau (RMST horizon) from the already-noised upper bound of the
    time column.  No additional privacy budget is consumed — tau is a deterministic
    post-processing function of the released bound.

    Using ceil() ensures tau >= true max, which is the conservative direction
    required by the calibration contract.
    """
    return int(np.ceil(noised_upper_bound))


def _dp_laplace_scalar(
    true_value: float,
    sensitivity: float,
    epsilon: float,
    *,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
) -> tuple[float, float]:
    """
    Release a single scalar under epsilon-DP via the Laplace mechanism.

    Returns (noised_value, epsilon_spent).  Result is clipped to [clip_lo, clip_hi]
    if provided (post-processing; does not affect the privacy guarantee).
    """
    scale = sensitivity / max(epsilon, 1e-12)
    noise = float(np.random.default_rng().laplace(loc=0.0, scale=scale))
    result = true_value + noise
    if clip_lo is not None:
        result = max(result, clip_lo)
    if clip_hi is not None:
        result = min(result, clip_hi)
    return result, epsilon


# ---------------------------------------------------------------------------
# Data-driven PrivBayes parameter helpers
# ---------------------------------------------------------------------------

def _sturges_bins(n: int, cap: int) -> int:
    """
    Sturges' rule: k = ceil(log2(n) + 1), capped at `cap`.
    Valid for n >= 30; for smaller n the formula underestimates variance but is
    still better than a flat default.  Lower-bounded at 5 to avoid degenerate CPTs.

    Public mode:  call with true n.
    Private mode: call with DP-released n_records (post-processing — free).
    """
    return max(5, min(int(np.ceil(np.log2(max(n, 2)) + 1)), cap))


def _bowley_skewness(x: np.ndarray) -> float:
    """
    Bowley (quartile) skewness = (Q3 + Q1 - 2·Q2) / (Q3 - Q1).
    Range [-1, 1].  Threshold |skew| > 0.2 → quantile binning.

    Preferred over Pearson skewness for DP because quartiles have
    a simple sensitivity: sensitivity(Qi) = (u - l) / n.
    """
    if len(x) < 4:
        return 0.0
    q1, q2, q3 = float(np.percentile(x, 25)), float(np.percentile(x, 50)), float(np.percentile(x, 75))
    denom = q3 - q1
    if denom < 1e-10:
        return 0.0
    return (q3 + q1 - 2.0 * q2) / denom


def _dp_bowley_skewness(
    s: pd.Series,
    bounds: list,
    epsilon: float,
) -> tuple[float, float]:
    """
    Release Bowley skewness under epsilon-DP via Laplace-noised quartiles.

    Three quartile queries (Q1, Q2, Q3) under sequential composition,
    epsilon/3 each.  Sensitivity of each quantile = (u - l) / n
    (a record at an extreme can shift a quantile by at most one rank,
    which in the worst case moves the quantile by (u-l)/n).

    Returns (noised_bowley, epsilon_spent).
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    x = np.clip(x, lo, hi)
    n = len(x)
    if n < 4:
        return 0.0, 0.0

    sens_q = (hi - lo) / n          # sensitivity of each quantile query
    eps_each = max(epsilon / 3.0, 1e-12)
    scale = sens_q / eps_each
    rng = np.random.default_rng()

    q1 = float(np.percentile(x, 25)) + rng.laplace(0.0, scale)
    q2 = float(np.percentile(x, 50)) + rng.laplace(0.0, scale)
    q3 = float(np.percentile(x, 75)) + rng.laplace(0.0, scale)

    denom = q3 - q1
    if abs(denom) < 1e-10:
        return 0.0, epsilon
    bowley = float(np.clip((q3 + q1 - 2.0 * q2) / denom, -1.0, 1.0))
    return bowley, epsilon


def _optimal_max_parents(n: int, max_bins_total: int) -> int:
    """
    Find the smallest k such that max_bins_total^k >= n.
    At this k, CPT sensitivity transitions from n-limited (2/n)
    to domain-limited (2/domain).  Going higher buys no noise reduction
    but multiplies CPT size exponentially — so k at the crossover is optimal.

    Public/private: takes DP-released n_records (post-processing — free).
    """
    for k in range(1, 6):
        if max_bins_total ** k >= n:
            return k
    return 3   # fallback for very large n


def _auto_max_numeric_bins(n: int, k_default: int = 3, hard_ceiling: int = 20) -> int:
    """
    Derive the PrivBayes discretisation cap automatically from n_records.

    Logic: at the sensitivity crossover point, CPT domain = n, so
    max_bins_total = ceil(n ^ (1/k)).  Using k=k_default (the most common
    max_parents value) gives a cap that is exactly tight — bins above this
    add CPT size with no noise benefit.

    The cap is also lower-bounded by Sturges(n) so it never prevents the
    data-driven bin count from being used in full.

    hard_ceiling prevents absurd caps for very large n (n=1M → cap=100+).

    Called in public mode with true n, or in private mode with DP-released
    n_records (post-processing — free).
    """
    sturges = int(np.ceil(np.log2(max(n, 2)) + 1))
    crossover_cap = int(np.ceil(n ** (1.0 / max(k_default, 1))))
    # Take the larger of the two so Sturges is never the binding constraint
    return min(max(sturges, crossover_cap, 5), hard_ceiling)


def _perks_alpha(n_bins_total: int) -> float:
    """
    Perks prior: alpha = 1/K where K = n_bins_total.
    Scales Dirichlet smoothing to the actual domain size.
    Smaller domain → larger smoothing (less data per cell).
    Larger domain → smaller smoothing (many cells, sparse CPT).

    This is a post-processing function of n_bins_total — free in both modes.
    """
    return round(1.0 / max(n_bins_total, 1), 6)


# ---------------------------------------------------------------------------
# NEW v1.2: Gamma — explicit sensitivity bounds
# ---------------------------------------------------------------------------

def _infer_sensitivity_bounds(
    column_types: dict[str, str],
    public_bounds: dict[str, Any],
    public_categories: dict[str, list[str]],
) -> dict[str, float]:
    """
    Derive Gamma: A -> R+  — explicit per-column sensitivity bounds.

    Sensitivity is derived deterministically from the declared domain Omega
    (public_bounds and public_categories) and type T, without re-accessing D.
    This implements the Schema Sufficiency property: noise calibration for any
    downstream mechanism M is fully determined by S alone.

    Sensitivity model
    -----------------
    binary      : 1.0   — single record changes 0<->1
    categorical : 2.0   — L1 sensitivity of the *full normalised histogram vector*.
                          Adding/removing one record changes two bins by 1 each,
                          giving L1 norm = 2.  This is NOT the sensitivity of a
                          single-cell count query (which is 1).  Downstream consumers
                          that interpret sensitivity_bounds[col] as a scalar query
                          sensitivity (e.g. for a single category frequency) should
                          use 1.0, not 2.0.  The 2.0 value is correct when the
                          entire distribution vector is released jointly.
    ordinal     : max - min  (numeric range, treated like integer)
    integer     : max - min
    continuous  : max - min
    """
    sensitivity: dict[str, float] = {}
    for col, ctype in column_types.items():
        if ctype == "binary":
            sensitivity[col] = 1.0
        elif ctype == "categorical":
            sensitivity[col] = 2.0
        elif ctype in {"continuous", "integer", "ordinal"}:
            bounds = public_bounds.get(col)
            lo: float | None = None
            hi: float | None = None
            if isinstance(bounds, dict):
                lo = bounds.get("min")
                hi = bounds.get("max")
            elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lo, hi = float(bounds[0]), float(bounds[1])
            if lo is not None and hi is not None:
                try:
                    sensitivity[col] = max(float(hi) - float(lo), 1.0)
                except (TypeError, ValueError):
                    sensitivity[col] = 1.0
            else:
                # Ordinal without explicit bounds: derive from categories
                cats = public_categories.get(col)
                if cats:
                    try:
                        nums = [float(c) for c in cats]
                        sensitivity[col] = max(max(nums) - min(nums), 1.0)
                    except (ValueError, TypeError):
                        sensitivity[col] = 1.0
                else:
                    sensitivity[col] = 1.0
        else:
            sensitivity[col] = 1.0
    return sensitivity


# ---------------------------------------------------------------------------
# NEW v1.2: Pi — default mechanism hints
# ---------------------------------------------------------------------------

def _infer_mechanism_hints(
    column_types: dict[str, str],
    target_spec: dict[str, Any] | None,
) -> dict[str, str]:
    """
    Derive Pi: A -> MechanismClass — ADVISORY default mechanism hints per column.

    These hints are informational metadata for the synthesis algorithm and downstream
    consumers.  They are NOT normative DP requirements.  In particular:

    - PrivBayes-style implementations typically Laplace-privatise discretised
      histogram CPTs regardless of per-column Pi assignments.
    - "gaussian" for continuous columns implies (epsilon,delta)-DP; use "laplace"
      for pure epsilon-DP.  The hint does not enforce either choice.
    - Synthesis algorithms may override any hint without violating the schema
      calibration contract, provided they honour sensitivity_bounds[col].

    Default assignments by type:
      binary / categorical : exponential  (finite non-numeric domain)
      ordinal / integer    : laplace      (discrete numeric; pure epsilon-DP)
      continuous           : gaussian     (continuous; (epsilon,delta)-DP)
      survival event col   : exponential  (binary outcome; finite domain)
      survival time col    : laplace      (positive numeric; sensitivity = tau)
    """
    survival_event_col: str | None = None
    survival_time_col: str | None = None
    if isinstance(target_spec, dict) and str(target_spec.get("kind")) == "survival_pair":
        targets = target_spec.get("targets")
        if isinstance(targets, list) and len(targets) >= 2:
            survival_event_col = str(targets[0])
            survival_time_col = str(targets[1])

    hints: dict[str, str] = {}
    for col, ctype in column_types.items():
        if col == survival_event_col:
            hints[col] = "exponential"
        elif col == survival_time_col:
            hints[col] = "laplace"
        elif ctype == "binary":
            hints[col] = "exponential"
        elif ctype == "categorical":
            hints[col] = "exponential"
        elif ctype == "ordinal":
            hints[col] = "laplace"
        elif ctype == "integer":
            hints[col] = "laplace"
        elif ctype == "continuous":
            hints[col] = "gaussian"
        else:
            hints[col] = "laplace"
    return hints


# ---------------------------------------------------------------------------
# NEW v1.2: P — domain provenance map
# ---------------------------------------------------------------------------

def _build_domain_provenance(
    column_types: dict[str, str],
    bound_sources: dict[str, str],
    schema_mode: str,
    schema_epsilon: float | None,
    epsilon_spent_per_col: dict[str, float] | None,
    non_private_aux_cols: set[str] | None = None,
) -> dict[str, Any]:
    """
    Build P: A -> {public | private(epsilon_1) | non_private_auxiliary}

    public mode  : all components marked "public"; no privacy budget consumed.
    private mode : components inferred from data under a DP mechanism are marked
                   "private" with the epsilon_1 sub-budget recorded.
                   Components inferred from data WITHOUT a DP mechanism (e.g.
                   categorical domain discovery via unique()) are marked
                   "non_private_auxiliary" with an explicit warning.  Downstream
                   consumers MUST treat these as public data or apply DPSU
                   (Differentially Private Set Union) before claiming end-to-end DP.

    Downstream synthesis algorithms MUST subtract sum(epsilon_spent) from their
    declared epsilon_total to obtain epsilon_2 available for synthesis, satisfying:
        epsilon_total >= epsilon_1 + epsilon_2   (sequential composition).
    """
    _NPA = "non_private_auxiliary"
    provenance: dict[str, Any] = {}
    for col in column_types:
        source = bound_sources.get(col, "public")
        if schema_mode == _MODE_PRIVATE and (non_private_aux_cols or set()) and col in (non_private_aux_cols or set()):
            provenance[col] = {
                "source": _NPA,
                "warning": (
                    "Domain inferred from raw data without a DP mechanism. "
                    "This violates end-to-end DP if the category set is sensitive. "
                    "Declare domain as a public override or apply DPSU for strict DP."
                ),
            }
        elif schema_mode == _MODE_PRIVATE and source == "inferred_from_data":
            entry: dict[str, Any] = {"source": "private"}
            if epsilon_spent_per_col and col in epsilon_spent_per_col:
                entry["epsilon_spent"] = round(epsilon_spent_per_col[col], 10)
        else:
            entry = {"source": "public"}
        if col not in provenance:
            provenance[col] = entry
    return provenance


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def _build_constraints(
    *,
    column_types: dict[str, str],
    public_categories: dict[str, list[str]],
    public_bounds: dict[str, Any],
    guid_like_columns: list[str],
    constant_columns: list[str],
    target_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    column_constraints: dict[str, Any] = {}
    cross_column_constraints: list[dict[str, Any]] = []

    for col, ctype in column_types.items():
        c: dict[str, Any] = {"type": ctype}
        if col in public_categories:
            c["allowed_values"] = public_categories[col]
        if col in public_bounds:
            bv = public_bounds[col]
            if isinstance(bv, dict):
                if "min" in bv:
                    c["min"] = bv["min"]
                if "max" in bv:
                    c["max"] = bv["max"]
                if "n_bins" in bv:
                    c["n_bins"] = bv["n_bins"]
            elif isinstance(bv, (list, tuple)) and len(bv) == 2:
                c["min"] = bv[0]
                c["max"] = bv[1]
        if col in constant_columns:
            c["note"] = "constant_column_excluded_from_synthesis"
        column_constraints[col] = c

    if isinstance(target_spec, dict) and str(target_spec.get("kind")) == "survival_pair":
        tcols = target_spec.get("targets")
        if isinstance(tcols, list) and len(tcols) >= 2:
            # FIX #4: unified naming — "event_col"/"time_col" throughout
            event_col = str(tcols[0])
            time_col = str(tcols[1])
            cross_column_constraints.append({
                "name": "survival_pair_definition",
                "type": "survival_pair",
                "event_col": event_col,       # FIX #4 (was event_col already — keep consistent)
                "time_col": time_col,          # FIX #4
                "event_allowed_values": [0, 1],
                "time_min_exclusive": 0,
            })
            ec = column_constraints.get(event_col, {"type": "binary"})
            ec["allowed_values"] = ["0", "1"]
            column_constraints[event_col] = ec
            tc = column_constraints.get(time_col, {"type": column_types.get(time_col, "continuous")})
            tc["min_exclusive"] = 0
            column_constraints[time_col] = tc

    return {
        "column_constraints": column_constraints,
        "cross_column_constraints": cross_column_constraints,
        "row_group_constraints": [],
    }


def _merge_constraints(base: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    import copy
    out = copy.deepcopy(base)
    out.setdefault("column_constraints", {})
    out.setdefault("cross_column_constraints", [])
    out.setdefault("row_group_constraints", [])
    if isinstance(user.get("column_constraints"), dict):
        for col, rules in user["column_constraints"].items():
            if isinstance(rules, dict):
                out["column_constraints"].setdefault(col, {})
                out["column_constraints"][col].update(rules)
    if isinstance(user.get("cross_column_constraints"), list):
        out["cross_column_constraints"].extend(copy.deepcopy(user["cross_column_constraints"]))
    if isinstance(user.get("row_group_constraints"), list):
        out["row_group_constraints"].extend(copy.deepcopy(user["row_group_constraints"]))
    return out


# ---------------------------------------------------------------------------
# PrivBayes extensions
# ---------------------------------------------------------------------------

def _default_pb_max_parents(n_cols: int, is_survival: bool) -> int:
    # HEURISTIC — not derived from theory.  Chosen empirically to balance
    # PrivBayes computational cost (exponential in k) against structure expressivity.
    # Survival mode allows one extra parent because the time→event edge is required.
    # FIX #2: non-survival logic was inverted — more columns → smaller k
    if is_survival:
        return 3 if n_cols <= 20 else 2
    return 3 if n_cols <= 12 else 2   # FIX #2: was "2 if <= 12 else 3"


def _infer_privbayes_extensions(
    *,
    column_types: dict[str, str],
    target_spec: dict[str, Any] | None,
    public_bounds: dict[str, Any],
    missing_value_rates: dict[str, float],
    df: "pd.DataFrame",
    n_records: int,
    schema_mode: str,
    eps_for_pb_strategy: float,
    pb_max_parents: int | None,
    pb_default_numeric_bins: int,
    pb_max_numeric_bins: int,
    pb_time_bins: int,
    pb_default_strategy: str,
    pb_time_strategy: str,
    pb_dirichlet_alpha: float | None,
    pb_emit_parent_constraints: bool,
    pb_emit_partial_order: bool,
) -> dict[str, Any]:
    """
    Build optional mechanism-specific hints for PrivBayes.
    These are public algorithm hints, not core schema semantics.

    v1.5: all key parameters are now data-driven in public mode and
    DP-privatised in private mode.

    Parameter derivation
    --------------------
    n_bins       Public:  Sturges rule on true n
                 Private: Sturges rule on DP-released n_records (free post-processing)

    strategy     Public:  Bowley skewness on raw data; quantile if |skew| > 0.2
                 Private: Bowley skewness via Laplace-noised quartiles,
                          eps_for_pb_strategy / K_numeric_cols per column

    alpha        Both:    Perks prior 1/K where K = n_bins_total (free post-processing)

    max_parents  Both:    smallest k s.t. max_bins_total^k >= n (free post-processing)
    """
    all_cols = list(column_types.keys())
    is_survival = isinstance(target_spec, dict) and str(target_spec.get("kind")) == "survival_pair"

    survival_event_col: str | None = None
    survival_time_col: str | None = None
    if is_survival:
        targets = target_spec.get("targets")
        if isinstance(targets, list) and len(targets) >= 2:
            survival_event_col = str(targets[0])
            survival_time_col = str(targets[1])
            assert survival_event_col is not None and survival_time_col is not None, (
                "survival_pair target_spec must have two non-None column names"
            )

    covariates = [
        c for c in all_cols
        if c not in {survival_event_col, survival_time_col}
    ]

    # ------------------------------------------------------------------
    # Count numeric columns that need strategy estimation (for DP budget split)
    # ------------------------------------------------------------------
    numeric_cols_for_strategy = [
        c for c, t in column_types.items()
        if t in {"continuous", "integer"}
        and not (survival_time_col and c == survival_time_col)
    ]
    n_strat_cols = max(len(numeric_cols_for_strategy), 1)
    eps_per_strat_col = eps_for_pb_strategy / n_strat_cols  # sequential composition

    # tracking
    strategy_epsilon_spent: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Discretization hints — per column
    # ------------------------------------------------------------------
    discretization_per_column: dict[str, dict[str, Any]] = {}
    max_n_bins_total_seen = 1   # track for max_parents computation

    for col, ctype in column_types.items():
        if ctype not in {"continuous", "integer"}:
            continue

        # ---- n_bins -------------------------------------------------------
        if survival_time_col and col == survival_time_col:
            bins     = pb_time_bins
            strategy = pb_time_strategy
            skewness_val    = None
            skewness_source = "fixed_survival_time"
        else:
            # Sturges on (DP-released) n_records — post-processing, free
            bins = _sturges_bins(n_records, pb_max_numeric_bins)

            # ---- strategy via Bowley skewness ----------------------------
            bounds = public_bounds.get(col)
            bounds_list: list | None = None
            if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
                bounds_list = [bounds["min"], bounds["max"]]
            elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                bounds_list = list(bounds)

            if bounds_list is None:
                # no bounds — cannot compute skewness, use default
                strategy        = pb_default_strategy
                skewness_val    = None
                skewness_source = "default_no_bounds"
            elif schema_mode == _MODE_PRIVATE and eps_per_strat_col > 0:
                # DP Bowley skewness
                skew, eps_spent = _dp_bowley_skewness(
                    df[col], bounds_list, eps_per_strat_col
                )
                strategy_epsilon_spent[col] = eps_spent
                strategy        = "quantile" if abs(skew) > 0.2 else "equal_width"
                skewness_val    = round(skew, 4)
                skewness_source = "dp_laplace"
                print(f"  [DP-STRAT] {col!r}  bowley={skew:+.4f}  "
                      f"strategy={strategy}  eps_spent={eps_spent:.6f}")
            else:
                # Public mode — direct computation
                x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
                if bounds_list:
                    x = np.clip(x, bounds_list[0], bounds_list[1])
                skew            = _bowley_skewness(x)
                strategy        = "quantile" if abs(skew) > 0.2 else "equal_width"
                skewness_val    = round(skew, 4)
                skewness_source = "data"

        # ---- NaN bin accounting -------------------------------------------
        col_missing_rate = missing_value_rates.get(col, 0.0)
        has_nan_bin      = col_missing_rate > 0.0
        n_bins_total     = bins + (1 if has_nan_bin else 0)
        max_n_bins_total_seen = max(max_n_bins_total_seen, n_bins_total)

        # ---- Perks dirichlet alpha (post-processing of n_bins_total) ------
        if pb_dirichlet_alpha is None:
            col_alpha = _perks_alpha(n_bins_total)
        else:
            col_alpha = pb_dirichlet_alpha   # user override

        entry: dict[str, Any] = {
            "strategy":    strategy,
            "n_bins":      int(bins),
            "n_bins_total": int(n_bins_total),
            "has_nan_bin": has_nan_bin,
            "dirichlet_alpha": col_alpha,
        }
        if skewness_val is not None:
            entry["skewness"]        = skewness_val
            entry["skewness_source"] = skewness_source
        else:
            entry["skewness_source"] = skewness_source
        if has_nan_bin:
            entry["nan_bin_index"] = int(bins)
            entry["missing_rate"]  = round(col_missing_rate, 6)

        discretization_per_column[col] = entry

    # ------------------------------------------------------------------
    # Data-driven max_parents (post-processing — free in both modes)
    # ------------------------------------------------------------------
    if pb_max_parents is not None:
        k = int(pb_max_parents)
        k_source = "user_override"
    else:
        k = _optimal_max_parents(n_records, max_n_bins_total_seen)
        k_source = "sensitivity_crossover"

    # ------------------------------------------------------------------
    # Global dirichlet_alpha: Perks on the median n_bins_total
    # (used as the top-level smoothing hint; per-column values take priority)
    # ------------------------------------------------------------------
    if pb_dirichlet_alpha is not None:
        global_alpha  = pb_dirichlet_alpha
        alpha_source  = "user_override"
    elif discretization_per_column:
        median_k = int(np.median([e["n_bins_total"] for e in discretization_per_column.values()]))
        global_alpha  = _perks_alpha(median_k)
        alpha_source  = f"perks_prior_median_K={median_k}"
    else:
        global_alpha  = 0.5
        alpha_source  = "jeffreys_fallback"

    # ------------------------------------------------------------------
    # Partial order, preferred_root, parent constraints (unchanged)
    # ------------------------------------------------------------------
    partial_order: list[str] = []
    if pb_emit_partial_order and is_survival and survival_time_col and survival_event_col:
        partial_order = covariates + [survival_time_col, survival_event_col]

    preferred_root = (covariates[0] if covariates else None) if pb_emit_partial_order else None

    allowed_parents: dict[str, list[str]] = {}
    forbidden_parents: dict[str, list[str]] = {}
    if pb_emit_parent_constraints and is_survival and survival_time_col and survival_event_col:
        allowed_parents[survival_time_col]  = covariates.copy()
        allowed_parents[survival_event_col] = covariates + [survival_time_col]
        forbidden_parents[survival_time_col] = [survival_event_col]

    structure: dict[str, Any] = {
        "emitted":          pb_emit_partial_order or pb_emit_parent_constraints,
        "preferred_root":   preferred_root,
        "partial_order":    partial_order,
        "allowed_parents":  allowed_parents,
        "forbidden_parents": forbidden_parents,
    }

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------
    cols_with_missing = {
        c: round(r, 6)
        for c, r in missing_value_rates.items()
        if r > 0.0 and c in column_types
    }
    missing_value_handling: dict[str, Any] = {
        "strategy": "nan_as_extra_bin",
        "description": (
            "Missing values are encoded as one extra discretisation bin "
            "(bin index = n_bins, 0-based) appended after the regular bins. "
            "The Bayesian network learns the joint distribution over all bins "
            "including the NaN bin, preserving missingness correlations. "
            "At generation time, missing_value_rates from the core schema "
            "should be used to validate that synthetic NaN proportions match."
        ),
        "columns_affected": cols_with_missing,
        "n_columns_affected": len(cols_with_missing),
    }

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    total_strat_eps = sum(strategy_epsilon_spent.values())

    ext: dict[str, Any] = {
        "version": "1.2",
        "enabled": True,
        "max_parents": k,
        "max_parents_source": k_source,
        "structure": structure,
        "discretization": {
            "default_strategy":    pb_default_strategy,
            "default_numeric_bins": int(pb_default_numeric_bins),
            "max_numeric_bins":    int(pb_max_numeric_bins),
            "per_column":          discretization_per_column,
        },
        "missing_value_handling": missing_value_handling,
        "smoothing": {
            "dirichlet_alpha":        global_alpha,
            "dirichlet_alpha_source": alpha_source,
            "note": (
                "Per-column dirichlet_alpha values in discretization.per_column "
                "take priority over this global value."
            ),
        },
        "provenance": {
            "source": "public_heuristic",
            "parameter_mode": schema_mode,
            "strategy_epsilon_spent": round(total_strat_eps, 10) if schema_mode == _MODE_PRIVATE else 0.0,
            "note": (
                "Mechanism-specific hints for PrivBayes. These fields are optional, public, "
                "and separate from the core schema used for DP calibration. "
                "max_parents is determined by sensitivity crossover (see max_parents_source). "
                "n_bins uses Sturges rule. strategy uses Bowley skewness threshold |skew|>0.2. "
                "dirichlet_alpha uses Perks prior 1/K."
            ),
        },
    }

    if is_survival and survival_time_col and survival_event_col:
        # FIX #4: use "event_col"/"time_col" to match constraints block naming
        ext["survival"] = {
            "time_col": survival_time_col,
            "event_col": survival_event_col,
            "time_binning_strategy": pb_time_strategy,
            "time_bins": int(pb_time_bins),
            "event_depends_on_time": True,
        }

    return ext


# ---------------------------------------------------------------------------
# Binary / small-integer domain helpers (unchanged)
# ---------------------------------------------------------------------------

def _binary_integer_domain_values(s: pd.Series) -> list[str] | None:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    ux = np.unique(x)
    if ux.size != 2 or not np.all(np.isclose(ux, np.round(ux), atol=1e-8)):
        return None
    ints = sorted([int(round(v)) for v in ux.tolist()])
    if pd.api.types.is_float_dtype(s):
        return [f"{v}.0" for v in ints]
    return [str(v) for v in ints]


def _small_integer_domain_values(s: pd.Series, max_levels: int) -> list[str] | None:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    if not np.all(np.isclose(x, np.round(x), atol=1e-8)):
        return None
    ux = np.unique(x)
    if ux.size > max_levels:
        return None
    if pd.api.types.is_float_dtype(s):
        return [f"{int(round(v))}.0" for v in sorted(ux.tolist())]
    return [str(int(round(v))) for v in sorted(ux.tolist())]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, default=None)
    ap.add_argument("--target-col", type=str, default=None)
    ap.add_argument("--target-cols", type=str, default=None)
    ap.add_argument("--target-kind", type=str, default=None)
    ap.add_argument("--survival-event-col", type=str, default=None)
    ap.add_argument("--survival-time-col", type=str, default=None)
    ap.add_argument("--column-types", type=Path, default=None)
    ap.add_argument("--target-spec-file", type=Path, default=None)
    ap.add_argument("--constraints-file", type=Path, default=None)
    ap.add_argument("--sensitive-attributes", type=str, default=None,
                    help="Comma-separated column names for attribute-inference metric")
    ap.add_argument("--delimiter", type=str, default="auto")
    ap.add_argument("--pad-frac", type=float, default=0.0)
    ap.add_argument("--pad-frac-integer", type=float, default=None)
    ap.add_argument("--pad-frac-continuous", type=float, default=None)
    ap.add_argument("--infer-categories", action="store_true")
    ap.add_argument("--max-categories", type=int, default=200)
    ap.add_argument("--max-integer-levels", type=int, default=MAX_INTEGER_LEVELS,
                    help="Integer columns with ≤ this many unique values are promoted to ordinal")
    ap.add_argument("--infer-binary-domain", action="store_true")
    ap.add_argument("--infer-datetimes", action="store_true")
    ap.add_argument("--datetime-min-parse-frac", type=float, default=0.95)
    ap.add_argument("--datetime-output-format", type=str, default="preserve")
    ap.add_argument("--guid-min-match-frac", type=float, default=0.95)
    ap.add_argument("--target-is-classifier", action="store_true")
    ap.add_argument("--no-publish-label-domain", action="store_true")
    ap.add_argument("--redact-source-path", action="store_true")
    ap.add_argument("--n-records", type=int, default=None,
                    help="Declared row count (default: len(data)); use train size if using a fixed split")

    # -----------------------------------------------------------------------
    # NEW v1.2: Schema provenance mode
    # -----------------------------------------------------------------------
    ap.add_argument(
        "--schema-mode",
        type=str,
        default=_MODE_PUBLIC,
        choices=[_MODE_PUBLIC, _MODE_PRIVATE],
        help=(
            "public (default): schema bounds are treated as public knowledge; "
            "no privacy budget consumed.  "
            "private: bounds are inferred from D under epsilon_1-DP; "
            "requires --schema-epsilon."
        ),
    )
    ap.add_argument(
        "--schema-epsilon",
        type=float,
        default=None,
        help=(
            "Total epsilon budget for schema generation (epsilon_1). "
            "Required when --schema-mode private.  "
            "Split equally across private numeric columns (sequential composition). "
            "Remaining epsilon_2 = epsilon_total - epsilon_1 is reported in provenance."
        ),
    )
    ap.add_argument(
        "--epsilon-total",
        type=float,
        default=None,
        help=(
            "Total pipeline epsilon (epsilon_1 + epsilon_2).  Optional; when provided "
            "the remaining synthesis budget epsilon_2 is computed and written to provenance."
        ),
    )
    ap.add_argument(
        "--dp-n-records",
        action="store_true",
        default=False,
        help=(
            "In private mode: release n_records under Laplace(sensitivity=1, epsilon=schema_epsilon). "
            "Consumes epsilon from schema_epsilon budget (sequential composition). "
            "If omitted in private mode, n_records is marked non_private_auxiliary."
        ),
    )
    ap.add_argument(
        "--dp-missing-rates",
        action="store_true",
        default=False,
        help=(
            "In private mode: release each missing_value_rate under Laplace(sensitivity=1/n, "
            "epsilon=schema_epsilon/K_missing) via parallel composition (disjoint columns). "
            "If omitted in private mode, missing_value_rates is marked non_private_auxiliary."
        ),
    )

    # PrivBayes-specific extensions
    ap.add_argument("--emit-privbayes-extensions", action="store_true",
                    help="Add schema['extensions']['privbayes'] with public algorithm hints")
    ap.add_argument("--pb-max-parents", type=int, default=None,
                    help="Optional override for PrivBayes max_parents")
    ap.add_argument("--pb-default-numeric-bins", type=int, default=8,
                    help="Default numeric bin count hint for PrivBayes")
    ap.add_argument("--pb-time-bins", type=int, default=10,
                    help="Suggested bin count for the survival time column")
    ap.add_argument("--pb-default-strategy", type=str, default="equal_width",
                    choices=["equal_width", "quantile"],
                    help="Default discretization strategy hint for numeric columns")
    ap.add_argument("--pb-time-strategy", type=str, default="quantile",
                    choices=["equal_width", "quantile"],
                    help="Preferred discretization strategy hint for the survival time column")
    ap.add_argument("--pb-dirichlet-alpha", type=float, default=None,
                    help=(
                        "Override CPT smoothing alpha for PrivBayes. "
                        "Default: Perks prior (1/K per column, where K=n_bins_total). "
                        "Pass a fixed float (e.g. 0.5) to use Jeffreys prior globally."
                    ))
    ap.add_argument("--pb-max-numeric-bins", type=int, default=None,
                    help=(
                        "Hard cap on n_bins for integer/continuous columns in PrivBayes extensions. "
                        "Default: auto — derived from n_records as ceil(n^(1/3)), lower-bounded by "
                        "Sturges(n) and upper-bounded at 20. Override only if you have a specific "
                        "CPT memory budget. Does not affect public_bounds[col].n_bins."
                    ))
    ap.add_argument("--pb-strategy-epsilon-frac", type=float, default=0.05,
                    help=(
                        "Fraction of schema_epsilon reserved for DP strategy selection "
                        "(Bowley skewness queries) in private mode. Default: 0.05 (5%%). "
                        "Ignored in public mode."
                    ))
    ap.add_argument("--pb-no-parent-constraints", action="store_true",
                    help="Do not emit survival-specific allowed/forbidden parent hints")
    ap.add_argument("--pb-no-partial-order", action="store_true",
                    help="Do not emit partial ordering hints for survival variables")

    args = ap.parse_args()

    # Validate schema-mode / schema-epsilon consistency
    if args.schema_mode == _MODE_PRIVATE and args.schema_epsilon is None:
        raise SystemExit("--schema-mode private requires --schema-epsilon FLOAT")
    if args.schema_mode == _MODE_PUBLIC and args.schema_epsilon is not None:
        print("  [WARN] --schema-epsilon ignored in public mode")
    if (args.epsilon_total is not None
            and args.schema_epsilon is not None
            and args.schema_epsilon > args.epsilon_total):
        raise SystemExit(
            f"--schema-epsilon ({args.schema_epsilon}) must be <= --epsilon-total ({args.epsilon_total})"
        )
    if not (0.0 <= args.pb_strategy_epsilon_frac <= 1.0):
        raise SystemExit(
            f"--pb-strategy-epsilon-frac ({args.pb_strategy_epsilon_frac}) must be in [0, 1]"
        )
    # Sanity check: sum of auxiliary fractions must not exceed 1.0, otherwise
    # eps_remaining for numeric bounds would go negative.
    _frac_sum = (
        (0.05 if args.dp_n_records else 0.0)
        + (0.10 if args.dp_missing_rates else 0.0)
        + (args.pb_strategy_epsilon_frac if args.emit_privbayes_extensions else 0.0)
    )
    if _frac_sum > 1.0:
        raise SystemExit(
            f"Combined auxiliary epsilon fractions ({_frac_sum:.3f}) exceed 1.0. "
            "Reduce --pb-strategy-epsilon-frac or disable optional queries."
        )

    schema_mode: str = args.schema_mode
    schema_epsilon: float | None = args.schema_epsilon

    delimiter = (
        _infer_csv_delimiter(args.data)
        if str(args.delimiter).strip().lower() == "auto"
        else str(args.delimiter)
    )
    df = pd.read_csv(args.data, sep=delimiter, engine="python")
    cols = [str(c) for c in df.columns]
    n_records = int(args.n_records) if args.n_records is not None else len(df)

    survival_event_col = args.survival_event_col
    survival_time_col = args.survival_time_col

    if bool(survival_event_col) != bool(survival_time_col):
        raise SystemExit("Both --survival-event-col and --survival-time-col must be provided together")

    if args.target_kind == "survival_pair" and not survival_event_col:
        survival_event_col = _infer_target_col(cols)
        survival_time_col = _infer_time_col(cols, exclude=survival_event_col)
        if not survival_event_col or not survival_time_col:
            raise SystemExit(
                "Could not auto-infer survival columns. "
                "Please supply --survival-event-col and --survival-time-col explicitly."
            )
        print(f"Auto-inferred survival columns: event={survival_event_col}, time={survival_time_col}")

    target_col = args.target_col or (survival_event_col if survival_event_col else _infer_target_col(cols))

    public_bounds: dict[str, Any] = {}
    public_categories: dict[str, list[str]] = {}
    missing_value_rates: dict[str, float] = {}
    column_types: dict[str, str] = {}
    datetime_spec: dict[str, dict] = {}
    guid_like_columns: list[str] = []
    constant_columns: list[str] = []
    bound_sources: dict[str, str] = {}
    epsilon_spent_per_col: dict[str, float] = {}

    pad_frac_global = float(args.pad_frac)
    pad_frac_integer = float(args.pad_frac_integer) if args.pad_frac_integer is not None else pad_frac_global
    pad_frac_continuous = float(args.pad_frac_continuous) if args.pad_frac_continuous is not None else pad_frac_global
    max_int_levels = int(args.max_integer_levels)

    type_overrides: dict[str, Any] = {}
    if args.column_types is not None:
        type_overrides = json.loads(args.column_types.read_text())
        if not isinstance(type_overrides, dict):
            raise SystemExit("--column-types must be a JSON object")

    def _override_for(col: str) -> dict[str, Any] | None:
        if col not in type_overrides:
            return None
        v = type_overrides[col]
        if isinstance(v, str):
            return {"type": v}
        if isinstance(v, dict):
            return dict(v)
        raise SystemExit(f"--column-types[{col}] must be a string or object")

    # -------------------------------------------------------------------
    # Private mode: budget pre-computation
    # Sequential composition over all DP queries on D.
    # Queries: numeric bounds (2 per col), optionally n_records and
    # per-column missing rates (parallel composition within that group).
    # -------------------------------------------------------------------
    non_private_aux_cols: set[str] = set()   # domains inferred non-privately in private mode

    def _count_private_numeric_cols() -> int:
        """Count continuous/integer columns that will need DP bounds."""
        count = 0
        for c in cols:
            s0 = df[c]
            if _is_guid_like_series(s0, min_match_frac=float(args.guid_min_match_frac)):
                continue
            if s0.dropna().nunique() <= 1:
                continue
            ov = _override_for(c)
            if ov is not None:
                t = str(ov.get("type") or "").strip().lower()
                if t in {"continuous", "integer"}:
                    count += 1
            elif _is_number_like_series(s0):
                # bool columns become binary (no bounds needed)
                if not pd.api.types.is_bool_dtype(s0):
                    count += 1
        return max(count, 1)

    # Budget fractions for optional auxiliary queries (sequential composition).
    # We reserve a fixed fraction of schema_epsilon for each opt-in query type
    # and allocate the remainder to numeric bounds.
    _NRECORDS_FRAC   = 0.05   # 5 % of schema_epsilon for n_records
    _MISSING_FRAC    = 0.10   # 10 % of schema_epsilon for all missing rates (parallel)
    # PB strategy fraction: user-configurable (default 5%)
    _PB_STRAT_FRAC   = float(args.pb_strategy_epsilon_frac) if args.emit_privbayes_extensions else 0.0

    eps_for_n_records:    float = 0.0
    eps_for_missing:      float = 0.0
    eps_for_pb_strategy:  float = 0.0

    if schema_mode == _MODE_PRIVATE and schema_epsilon is not None:
        eps_remaining = schema_epsilon
        if args.dp_n_records:
            eps_for_n_records = schema_epsilon * _NRECORDS_FRAC
            eps_remaining -= eps_for_n_records
        if args.dp_missing_rates:
            eps_for_missing = schema_epsilon * _MISSING_FRAC
            eps_remaining -= eps_for_missing
        if args.emit_privbayes_extensions:
            eps_for_pb_strategy = schema_epsilon * _PB_STRAT_FRAC
            eps_remaining -= eps_for_pb_strategy

        n_private_numeric = _count_private_numeric_cols()
        epsilon_per_col = max(eps_remaining, 0.0) / n_private_numeric
        print(
            f"  [DP-SCHEMA] mode=private  epsilon_1={schema_epsilon}  "
            f"n_numeric_cols={n_private_numeric}  epsilon_per_col={epsilon_per_col:.6f}"
            + (f"  eps_n_records={eps_for_n_records:.4f}" if args.dp_n_records else "")
            + (f"  eps_missing={eps_for_missing:.4f}"     if args.dp_missing_rates else "")
            + (f"  eps_pb_strategy={eps_for_pb_strategy:.4f}" if args.emit_privbayes_extensions else "")
        )
    else:
        epsilon_per_col = 0.0

    # -------------------------------------------------------------------
    # Per-column processing
    # -------------------------------------------------------------------
    raw_missing_value_rates: dict[str, float] = {}   # pre-noise rates, computed once

    for c in cols:
        s0 = df[c]
        raw_missing_value_rates[c] = float(s0.isna().mean())
        missing_value_rates[c] = raw_missing_value_rates[c]  # updated below if DP

        if _is_guid_like_series(s0, min_match_frac=float(args.guid_min_match_frac)):
            guid_like_columns.append(c)
            print(f"  [GUID]     {c!r} — excluded from synthesis")
            continue

        n_unique = s0.dropna().nunique()
        if n_unique <= 1:
            constant_columns.append(c)
            print(f"  [CONST]    {c!r} — {n_unique} unique value(s), excluded from synthesis")
            continue

        s, dt_converted, dt_fmt_hint = (
            _maybe_parse_datetime_like(s0, min_parse_frac=float(args.datetime_min_parse_frac))
            if bool(args.infer_datetimes) else (s0, False, None)
        )

        ov = _override_for(c)
        if ov is not None:
            t = str(ov.get("type") or "").strip().lower()
            if t not in {"continuous", "integer", "categorical", "ordinal", "binary"}:
                raise SystemExit(f"--column-types[{c}].type invalid: {t!r}")
            column_types[c] = t
            dom = ov.get("domain")
            if t in {"categorical", "ordinal", "binary"} and dom is not None:
                if not isinstance(dom, list):
                    raise SystemExit(f"--column-types[{c}].domain must be list")
                public_categories[c] = [str(x) for x in dom]
                if t == "ordinal":
                    public_bounds[c] = _ordinal_bounds_from_categories(public_categories[c])
                bound_sources[c] = "override"
            elif t in {"continuous", "integer"} and _is_number_like_series(s):
                this_pad = pad_frac_integer if t == "integer" else pad_frac_continuous
                if schema_mode == _MODE_PRIVATE and epsilon_per_col > 0:
                    bnds, eps_spent = _dp_private_bounds(
                        s, epsilon_per_col, this_pad, integer_like=(t == "integer")
                    )
                    epsilon_spent_per_col[c] = eps_spent
                    bound_sources[c] = "inferred_from_data"
                    print(f"  [DP-BOUND] {c!r}  eps_spent={eps_spent:.6f}")
                else:
                    bnds = _bounds_for_number_like(s, this_pad, integer_like=(t == "integer"))
                    bound_sources[c] = "inferred_from_data"
                public_bounds[c] = bnds
            if dt_converted:
                out_fmt = (dt_fmt_hint if str(args.datetime_output_format).strip().lower() == "preserve"
                           else str(args.datetime_output_format))
                datetime_spec[c] = {
                    "storage": "epoch_ns",
                    "output_format": out_fmt or "%Y-%m-%dT%H:%M:%S",
                    "timezone": "UTC",
                }
            continue

        if pd.api.types.is_bool_dtype(s0):
            column_types[c] = "binary"
            public_categories[c] = ["0", "1"]
            bound_sources[c] = "type_inference"

        elif _is_number_like_series(s):
            if dt_converted:
                column_types[c] = "integer"
                out_fmt = (dt_fmt_hint if str(args.datetime_output_format).strip().lower() == "preserve"
                           else str(args.datetime_output_format))
                datetime_spec[c] = {
                    "storage": "epoch_ns",
                    "output_format": out_fmt or "%Y-%m-%dT%H:%M:%S",
                    "timezone": "UTC",
                }
                if schema_mode == _MODE_PRIVATE and epsilon_per_col > 0:
                    bnds, eps_spent = _dp_private_bounds(s, epsilon_per_col, pad_frac_integer, integer_like=True)
                    epsilon_spent_per_col[c] = eps_spent
                    print(f"  [DP-BOUND] {c!r} (datetime)  eps_spent={eps_spent:.6f}")
                else:
                    _lo, _hi = _bounds_for_number_like(s, pad_frac_integer, integer_like=True)
                    bnds = {"min": _lo, "max": _hi, "n_bins": 100}
                public_bounds[c] = bnds if isinstance(bnds, dict) else {"min": bnds[0], "max": bnds[1], "n_bins": 100}
                bound_sources[c] = "inferred_from_data"
            else:
                if pd.api.types.is_integer_dtype(s0):
                    raw_type = "integer"
                else:
                    xn = pd.to_numeric(s, errors="coerce")
                    xn = xn[np.isfinite(xn)]
                    raw_type = (
                        "integer"
                        if xn.size > 0 and np.all(np.isclose(xn, np.round(xn), atol=1e-8))
                        else "continuous"
                    )

                bin_dom = _binary_integer_domain_values(s)

                if bin_dom is not None:
                    normalized = sorted([str(x).strip() for x in bin_dom])
                    if normalized in (["0", "1"], ["0.0", "1.0"]):
                        column_types[c] = "binary"
                        public_categories[c] = ["0", "1"]
                        bound_sources[c] = "type_inference"
                        print(f"  [BINARY]   {c!r} — values [0, 1]")
                    else:
                        column_types[c] = "categorical"
                        public_categories[c] = bin_dom
                        bound_sources[c] = "inferred_from_data"
                        print(f"  [CATEG]    {c!r} — two values {bin_dom} (not 0/1, categorical for CRN)")
                        if schema_mode == _MODE_PRIVATE:
                            non_private_aux_cols.add(c)
                            print(f"  [NPA-WARN] {c!r} two-value domain inferred non-privately; "
                                  "declare domain as public override for strict DP.")

                elif raw_type == "integer":
                    small_dom = _small_integer_domain_values(s, max_int_levels)
                    if small_dom is not None:
                        column_types[c] = "ordinal"
                        public_categories[c] = small_dom
                        public_bounds[c] = _ordinal_bounds_from_categories(small_dom)
                        bound_sources[c] = "inferred_from_data"
                        print(f"  [ORDINAL]  {c!r} — {len(small_dom)} integer levels promoted to ordinal")
                        # Ordinal level list inferred non-privately — mark in private mode
                        if schema_mode == _MODE_PRIVATE:
                            non_private_aux_cols.add(c)
                            print(f"  [NPA-WARN] {c!r} ordinal levels inferred non-privately; "
                                  "declare as public override for strict DP.")
                    else:
                        column_types[c] = "integer"
                        n_unique_num = int(pd.to_numeric(s, errors="coerce").nunique())
                        if schema_mode == _MODE_PRIVATE and epsilon_per_col > 0:
                            bnds_raw, eps_spent = _dp_private_bounds(
                                s, epsilon_per_col, pad_frac_integer, integer_like=True
                            )
                            lo, hi = bnds_raw[0], bnds_raw[1]
                            epsilon_spent_per_col[c] = eps_spent
                            print(f"  [DP-BOUND] {c!r}  eps_spent={eps_spent:.6f}")
                        else:
                            lo, hi = _bounds_for_number_like(s, pad_frac_integer, integer_like=True)
                        public_bounds[c] = {"min": lo, "max": hi, "n_bins": min(n_unique_num, 100)}
                        bound_sources[c] = "inferred_from_data"

                else:  # continuous
                    column_types[c] = "continuous"
                    if schema_mode == _MODE_PRIVATE and epsilon_per_col > 0:
                        bnds_raw, eps_spent = _dp_private_bounds(
                            s, epsilon_per_col, pad_frac_continuous, integer_like=False
                        )
                        lo, hi = bnds_raw[0], bnds_raw[1]
                        epsilon_spent_per_col[c] = eps_spent
                        print(f"  [DP-BOUND] {c!r}  eps_spent={eps_spent:.6f}")
                    else:
                        lo, hi = _bounds_for_number_like(s, pad_frac_continuous, integer_like=False)
                    public_bounds[c] = {"min": lo, "max": hi}
                    bound_sources[c] = "inferred_from_data"

        else:
            column_types[c] = "categorical"
            bound_sources[c] = "type_inference"
            if args.infer_categories:
                u = pd.Series(s, copy=False).astype("string").dropna().unique().tolist()
                u = sorted([str(x) for x in u])
                if 0 < len(u) <= int(args.max_categories):
                    public_categories[c] = u
                    # Categories inferred from raw data without DP — mark in private mode
                    if schema_mode == _MODE_PRIVATE:
                        non_private_aux_cols.add(c)
                        print(f"  [NPA-WARN] {c!r} categories inferred non-privately via unique(); "
                              "declare domain as public override or apply DPSU for strict DP.")
                else:
                    print(f"  [WARN]     {c!r} has {len(u)} unique string values — "
                          f"too many for public_categories (max={args.max_categories}).")

    if survival_event_col and survival_event_col in column_types:
        column_types[survival_event_col] = "binary"
        public_categories[survival_event_col] = ["0", "1"]
        public_bounds.pop(survival_event_col, None)
        print(f"  [SURVIVAL] {survival_event_col!r} forced to binary {{0,1}}")

    # -------------------------------------------------------------------
    # DP release of n_records and missing_value_rates (private mode, opt-in)
    # Sequential composition for n_records (1 query, sensitivity 1).
    # Parallel composition for missing rates (disjoint per-column queries,
    # sensitivity 1/n, all using the same eps_for_missing budget).
    # -------------------------------------------------------------------
    n_records_aux_note: str | None = None
    missing_rates_aux_note: str | None = None

    if schema_mode == _MODE_PRIVATE:
        if args.dp_n_records and eps_for_n_records > 0:
            noised_n, eps_spent_n = _dp_laplace_scalar(
                float(n_records), sensitivity=1.0, epsilon=eps_for_n_records,
                clip_lo=1.0,
            )
            n_records = max(1, int(round(noised_n)))
            epsilon_spent_per_col["__n_records__"] = eps_spent_n
            print(f"  [DP-NREC]  n_records noised → {n_records}  eps_spent={eps_spent_n:.6f}")
        else:
            n_records_aux_note = (
                "non_private_auxiliary: n_records released without DP. "
                "Use --dp-n-records for a privatised release."
            )
            print("  [NPA-WARN] n_records released without DP budget (use --dp-n-records).")

        if args.dp_missing_rates and eps_for_missing > 0:
            # Parallel composition: each column uses the same eps_for_missing budget
            # because queries operate on disjoint column domains.
            sens_missing = 1.0 / max(n_records, 1)
            for c in list(missing_value_rates.keys()):
                noised_rate, _ = _dp_laplace_scalar(
                    raw_missing_value_rates.get(c, 0.0),
                    sensitivity=sens_missing,
                    epsilon=eps_for_missing,
                    clip_lo=0.0,
                    clip_hi=1.0,
                )
                missing_value_rates[c] = round(noised_rate, 6)
            epsilon_spent_per_col["__missing_rates__"] = eps_for_missing
            print(f"  [DP-MISS]  missing_value_rates noised (parallel)  "
                  f"sensitivity={sens_missing:.2e}  eps_spent={eps_for_missing:.6f}")
        else:
            missing_rates_aux_note = (
                "non_private_auxiliary: missing_value_rates released without DP. "
                "Use --dp-missing-rates for a privatised release."
            )
            print("  [NPA-WARN] missing_value_rates released without DP budget (use --dp-missing-rates).")

    # -------------------------------------------------------------------
    # Tau: inferred from time column bounds (DP-noised in private mode)
    # No additional epsilon spent — tau is post-processing of the released max.
    # -------------------------------------------------------------------
    tau: int | None = None
    if survival_time_col and survival_time_col in df.columns:
        if schema_mode == _MODE_PRIVATE and survival_time_col in public_bounds:
            # Use the already-noised upper bound — no extra budget
            noised_upper = public_bounds[survival_time_col]
            if isinstance(noised_upper, dict):
                noised_upper = noised_upper.get("max")
            elif isinstance(noised_upper, (list, tuple)) and len(noised_upper) == 2:
                noised_upper = noised_upper[1]
            if noised_upper is not None and np.isfinite(float(noised_upper)):
                tau = _dp_private_tau(float(noised_upper))
                print(f"  [DP-TAU]   tau={tau} (post-processing of noised max({survival_time_col!r}))")
        else:
            max_t = pd.to_numeric(df[survival_time_col], errors="coerce").max()
            if np.isfinite(max_t):
                tau = int(np.ceil(max_t))

    # -------------------------------------------------------------------
    # target_spec
    # -------------------------------------------------------------------
    target_spec: dict[str, Any] | None = None

    if args.target_spec_file is not None:
        target_spec = json.loads(args.target_spec_file.read_text())
        if not isinstance(target_spec, dict):
            raise SystemExit("--target-spec-file must contain a JSON object")

    elif survival_event_col:
        target_spec = {
            "targets": [survival_event_col, survival_time_col],
            "kind": "survival_pair",
            "primary_target": survival_event_col,
            "dtypes": {
                # FIX #1: event col is binary, not ordinal
                survival_event_col: "binary",
                survival_time_col: _infer_target_dtype(df, survival_time_col),
            },
        }
        if tau is not None:
            target_spec["tau"] = tau
            print(f"  [TAU]      RMST horizon = {tau} from {'noised ' if schema_mode == _MODE_PRIVATE else ''}max({survival_time_col!r})")

    else:
        targets = _parse_csv_list(args.target_cols)
        if not targets and target_col:
            targets = [target_col]
        if targets:
            target_kind = args.target_kind or ("single" if len(targets) == 1 else "multi_target")
            target_spec = {
                "targets": targets,
                "kind": target_kind,
                "primary_target": target_col,
                "dtypes": {t: _infer_target_dtype(df, t) for t in targets},
            }

    if target_spec is not None:
        tcols = target_spec.get("targets")
        if isinstance(tcols, list) and tcols:
            existing_dtypes = target_spec.get("dtypes") if isinstance(target_spec.get("dtypes"), dict) else {}
            allowed = {"integer", "continuous", "categorical", "ordinal", "binary"}
            normalised: dict[str, str] = {}
            for t in [str(x) for x in tcols]:
                mapped = _target_dtype_from_column_type(column_types.get(t))
                if mapped is not None:
                    normalised[t] = mapped
                elif isinstance(existing_dtypes.get(t), str) and existing_dtypes[t].strip().lower() in allowed:
                    normalised[t] = existing_dtypes[t].strip().lower()
                else:
                    normalised[t] = _infer_target_dtype(df, t)
            target_spec["dtypes"] = normalised

    label_domain: list[str] = []
    if target_col and target_col in df.columns and not args.no_publish_label_domain:
        is_survival = target_spec is not None and str(target_spec.get("kind")) == "survival_pair"
        is_categorical = column_types.get(target_col) in {"categorical", "ordinal", "binary"}
        if (is_categorical or args.target_is_classifier) and not is_survival:
            u = pd.Series(df[target_col], copy=False).astype("string").dropna().unique().tolist()
            u_sorted = sorted([str(x) for x in u])
            if 0 < len(u_sorted) <= int(args.max_categories):
                label_domain = u_sorted
                public_categories[target_col] = label_domain

    if args.no_publish_label_domain:
        scrub = []
        if target_spec is not None and isinstance(target_spec.get("targets"), list):
            scrub.extend([str(x) for x in target_spec["targets"]])
        elif target_col:
            scrub.append(target_col)
        for t in scrub:
            public_categories.pop(t, None)

    # -------------------------------------------------------------------
    # NEW v1.2: Gamma, Pi, P
    # -------------------------------------------------------------------
    sensitivity_bounds = _infer_sensitivity_bounds(column_types, public_bounds, public_categories)
    mechanism_hints = _infer_mechanism_hints(column_types, target_spec)
    domain_provenance = _build_domain_provenance(
        column_types=column_types,
        bound_sources=bound_sources,
        schema_mode=schema_mode,
        schema_epsilon=schema_epsilon,
        epsilon_spent_per_col=epsilon_spent_per_col,
        non_private_aux_cols=non_private_aux_cols if schema_mode == _MODE_PRIVATE else None,
    )

    # NOTE: total_epsilon_spent and epsilon_remaining are computed AFTER the
    # PrivBayes extensions block so that __pb_strategy__ is included.
    # Do not move this computation above the emit_privbayes_extensions block.

    # -------------------------------------------------------------------
    # Assemble schema
    # -------------------------------------------------------------------
    dataset_info: dict[str, Any] = {"n_records": n_records}
    if n_records_aux_note:
        dataset_info["n_records_note"] = n_records_aux_note

    missing_val_block: dict[str, Any] = dict(missing_value_rates)
    if missing_rates_aux_note:
        missing_val_block["__note__"] = missing_rates_aux_note

    schema: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset_name or args.data.stem,
        "dataset_info": dataset_info,
        "target_col": target_col,
        "label_domain": label_domain,
        "missing_value_rates": missing_val_block,
        "public_bounds": public_bounds,
        "public_categories": public_categories,
        "column_types": column_types,
        "datetime_spec": datetime_spec,
        # NEW v1.2: Gamma — explicit sensitivity per column
        "sensitivity_bounds": sensitivity_bounds,
        # NEW v1.2: Pi — default mechanism per column
        "mechanism_hints": mechanism_hints,
        # NEW v1.2: P — per-component provenance
        "domain_provenance": domain_provenance,
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_csv": (
                "example_data_path_to_csv_file" if bool(args.redact_source_path)
                else str(args.data)
            ),
            "source_delimiter": delimiter,
            "schema_mode": schema_mode,
            # DP schema budget accounting
            # NOTE: epsilon_spent_schema and epsilon_remaining_synthesis are set to
            # placeholder None here and patched to their correct values after the
            # PrivBayes extensions block, where __pb_strategy__ epsilon is registered.
            "schema_epsilon": schema_epsilon,
            "epsilon_total": args.epsilon_total,
            "epsilon_spent_schema": None,           # patched below
            "epsilon_remaining_synthesis": None,    # patched below
            # Composition note for downstream consumers
            "composition_note": (
                "Sequential composition: epsilon_total >= epsilon_spent_schema + epsilon_synthesis. "
                "epsilon_remaining_synthesis is the maximum budget available for M_synth. "
                + (
                    f"WARNING: {len(non_private_aux_cols)} column(s) have non_private_auxiliary "
                    "domain provenance — categorical/ordinal domains inferred non-privately. "
                    "End-to-end DP requires these domains to be declared as public overrides."
                    if non_private_aux_cols else ""
                )
            ) if schema_mode == _MODE_PRIVATE else "Schema treated as public; full epsilon available for synthesis.",
            "pad_frac": pad_frac_global,
            "pad_frac_integer": pad_frac_integer,
            "pad_frac_continuous": pad_frac_continuous,
            "inferred_categories": bool(args.infer_categories),
            "max_categories": int(args.max_categories),
            "max_integer_levels": max_int_levels,
            "inferred_datetimes": bool(args.infer_datetimes),
            "datetime_min_parse_frac": float(args.datetime_min_parse_frac),
            "inferred_binary_domain": bool(args.infer_binary_domain),
            "guid_min_match_frac": float(args.guid_min_match_frac),
            "guid_like_columns": guid_like_columns,
            "constant_columns": constant_columns,
            "datetime_output_format": str(args.datetime_output_format),
            "no_publish_label_domain": bool(args.no_publish_label_domain),
            "column_types_overrides": str(args.column_types) if args.column_types is not None else None,
            "bound_sources": bound_sources,
        },
    }

    if target_spec is not None:
        schema["target_spec"] = target_spec

    if args.sensitive_attributes is not None:
        sens = _parse_csv_list(args.sensitive_attributes)
        if sens:
            schema["sensitive_attributes"] = sens

    constraints = _build_constraints(
        column_types=column_types,
        public_categories=public_categories,
        public_bounds=public_bounds,
        guid_like_columns=guid_like_columns,
        constant_columns=constant_columns,
        target_spec=target_spec,
    )
    if args.constraints_file is not None:
        user_constraints = json.loads(args.constraints_file.read_text())
        if not isinstance(user_constraints, dict):
            raise SystemExit("--constraints-file must contain a JSON object")
        constraints = _merge_constraints(constraints, user_constraints)
    schema["constraints"] = constraints

    if args.emit_privbayes_extensions:
        schema.setdefault("extensions", {})
        # Resolve auto cap: derive from n_records when user did not override.
        # k_default=3 is used because _optimal_max_parents almost always returns 3
        # for clinical datasets (n typically 100–10,000). The cap is lower-bounded
        # by Sturges(n) so it never suppresses the data-driven bin count.
        pb_max_numeric_bins_resolved: int = (
            args.pb_max_numeric_bins
            if args.pb_max_numeric_bins is not None
            else _auto_max_numeric_bins(n_records, k_default=3)
        )
        pb_ext = _infer_privbayes_extensions(
            column_types=column_types,
            target_spec=target_spec,
            public_bounds=public_bounds,
            missing_value_rates=missing_value_rates,
            df=df,
            n_records=n_records,
            schema_mode=schema_mode,
            eps_for_pb_strategy=eps_for_pb_strategy,
            pb_max_parents=args.pb_max_parents,
            pb_default_numeric_bins=int(args.pb_default_numeric_bins),
            pb_max_numeric_bins=pb_max_numeric_bins_resolved,
            pb_time_bins=int(args.pb_time_bins),
            pb_default_strategy=str(args.pb_default_strategy),
            pb_time_strategy=str(args.pb_time_strategy),
            pb_dirichlet_alpha=args.pb_dirichlet_alpha,   # None → Perks
            pb_emit_parent_constraints=not bool(args.pb_no_parent_constraints),
            pb_emit_partial_order=not bool(args.pb_no_partial_order),
        )
        schema["extensions"]["privbayes"] = pb_ext
        # Record strategy epsilon in budget accounting — must happen before
        # total_epsilon_spent is computed below.
        strat_eps = pb_ext.get("provenance", {}).get("strategy_epsilon_spent", 0.0)
        if strat_eps > 0:
            epsilon_spent_per_col["__pb_strategy__"] = strat_eps

    # ------------------------------------------------------------------
    # Budget totals — computed HERE, after all queries (including PrivBayes
    # strategy) have registered their epsilon in epsilon_spent_per_col.
    # __pb_strategy__ is a legitimate budget spend and must be included.
    # ------------------------------------------------------------------
    _SPECIAL_BUDGET_KEYS = {"__n_records__", "__missing_rates__", "__pb_strategy__"}
    total_epsilon_spent = sum(
        v for k, v in epsilon_spent_per_col.items()
        if not k.startswith("__") or k in _SPECIAL_BUDGET_KEYS
    )
    epsilon_remaining: float | None = None
    if args.epsilon_total is not None:
        epsilon_remaining = round(args.epsilon_total - total_epsilon_spent, 10)

    # Patch the two provenance fields that depend on total_epsilon_spent
    # (schema dict was already assembled above with placeholder 0.0 values).
    schema["provenance"]["epsilon_spent_schema"] = (
        round(total_epsilon_spent, 10) if schema_mode == _MODE_PRIVATE else 0.0
    )
    schema["provenance"]["epsilon_remaining_synthesis"] = epsilon_remaining

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, indent=2) + "\n")

    print("\nSchema summary:")
    print(f"  Schema version    : {SCHEMA_VERSION}")
    print(f"  Schema mode       : {schema_mode.upper()}")
    if schema_mode == _MODE_PRIVATE:
        print(f"  Epsilon (schema)  : {schema_epsilon}  →  spent={total_epsilon_spent:.6f}")
        if epsilon_remaining is not None:
            print(f"  Epsilon remaining : {epsilon_remaining:.6f}  (available for synthesis)")
        print(f"  DP n_records      : {'yes' if args.dp_n_records else 'no (NPA)'}")
        print(f"  DP missing rates  : {'yes' if args.dp_missing_rates else 'no (NPA)'}")
        if non_private_aux_cols:
            print(f"  NPA columns       : {sorted(non_private_aux_cols)}"
                  "  ← categorical/ordinal domains not DP-covered")
    print(f"  Columns in schema : {len(column_types)}")
    print(f"  GUID excluded     : {len(guid_like_columns)}  {guid_like_columns}")
    print(f"  Constant excluded : {len(constant_columns)}  {constant_columns}")
    print(f"  Binary columns    : {[c for c, t in column_types.items() if t == 'binary']}")
    print(f"  Ordinal columns   : {[c for c, t in column_types.items() if t == 'ordinal']}")
    print(f"  Survival kind     : {target_spec.get('kind') if target_spec else 'n/a'}")
    if target_spec and target_spec.get("kind") == "survival_pair":
        print(f"  Event col         : {target_spec.get('primary_target')}")
        print(f"  Time col          : {target_spec['targets'][1] if len(target_spec.get('targets', [])) > 1 else 'n/a'}")
        print(f"  Tau (RMST)        : {target_spec.get('tau')}")
    print(f"  Sensitivity bounds: {list(sensitivity_bounds.keys())[:6]}{'...' if len(sensitivity_bounds) > 6 else ''}")
    print(f"  Mechanism hints   : { {k: v for k, v in list(mechanism_hints.items())[:4]} }{'...' if len(mechanism_hints) > 4 else ''}")
    print(f"  Bounds n_bins     : {[c for c, b in public_bounds.items() if isinstance(b, dict) and 'n_bins' in b]}")
    if args.emit_privbayes_extensions:
        print("  PrivBayes ext     : emitted under schema['extensions']['privbayes']")
    print(f"\nWrote schema → {args.out}")


if __name__ == "__main__":
    main()
