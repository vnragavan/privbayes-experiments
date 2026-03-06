"""
schema_validator.py

Validates a schema-generator JSON before it is used to guide
DP synthetic data generation.

TWO MODES:
  1. Schema-only validation — checks internal consistency and completeness
  2. Schema + data cross-validation — checks that schema declarations
     match the actual data before any model sees the data

USAGE:
  # Validate schema only (no data needed)
  python schema_validator.py schemas/my_schema.json

  # Validate schema AND cross-check against data
  python schema_validator.py schemas/my_schema.json data/my_data.csv

  # Validate multiple schemas at once
  python schema_validator.py schemas/lung.json schemas/adult.json

  # As a library (import into run_experiment.py)
  from schema_validator import validate, validate_against_data

WHAT IT CHECKS:
  Schema-only:
    1. column_types present and all types are recognised
    2. Every numeric column has public_bounds (min < max)
    3. Every categorical/ordinal/binary column has public_categories
    4. Binary columns have exactly 2 categories
    5. target_spec is complete (kind, targets, primary_target)
    6. Survival pair has cross_column_constraint defined
    7. Survival pair has tau defined
    8. Provenance block present and documents bound sources
    9. Suspicious tight bounds flagged (possible data-derived)

  Cross-validation against data:
    10. Every schema column exists in the dataframe
    11. Every dataframe column exists in the schema
    12. Categorical columns — schema categories match actual values
    13. Numeric columns — observed range fits within declared bounds
    14. Binary columns — exactly two unique values in data
    15. Column types consistent with pandas dtype

EXIT CODES:
  0 — passed (possibly with warnings)
  1 — failed (hard errors that would break synthesis)
"""

from __future__ import annotations

import json
import sys
from typing import List, Optional, Tuple


# ─── Type definitions ─────────────────────────────────────────────

NUMERIC_TYPES  = {"continuous", "integer"}
CAT_TYPES      = {"categorical", "ordinal", "binary"}
ALL_TYPES      = NUMERIC_TYPES | CAT_TYPES

PANDAS_NUMERIC_DTYPES = {
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
}


# ─── Exceptions ────────────────────────────────────────────────────

class SchemaValidationError(Exception):
    """Raised when the schema has hard errors that would break synthesis."""
    pass


# ─── Helpers ──────────────────────────────────────────────────────

def _get_bounds(bv) -> Tuple[Optional[float], Optional[float]]:
    """Extract (min, max) from either dict or list format."""
    if isinstance(bv, dict):
        return bv.get("min"), bv.get("max")
    elif isinstance(bv, (list, tuple)) and len(bv) == 2:
        return bv[0], bv[1]
    return None, None


def _is_round(v: float) -> bool:
    """True if v is a round number (divisible by 5 or a power of 10)."""
    if v == 0:
        return True
    return (v % 5 == 0) or (v % 10 == 0) or (v % 100 == 0)


# ─── Schema-only validation ────────────────────────────────────────

def validate(schema: dict, strict: bool = True) -> List[str]:
    """
    Validate schema internal consistency and completeness.

    Parameters
    ----------
    schema : dict
        Loaded schema-generator JSON.
    strict : bool
        If True, raise SchemaValidationError on hard failures.
        If False, return errors as strings in the list (prefixed FAIL).

    Returns
    -------
    List[str]
        Warning and info messages. Hard errors raise if strict=True,
        otherwise returned as FAIL-prefixed strings.
    """
    errors   = []
    warnings = []

    col_types     = schema.get("column_types", {})
    public_bounds = schema.get("public_bounds", {})
    public_cats   = schema.get("public_categories", {})
    target_spec   = schema.get("target_spec", {})
    constraints   = schema.get("constraints", {})
    provenance    = schema.get("provenance", {})

    # ── 1. column_types must be non-empty ─────────────────────────
    if not col_types:
        errors.append(
            "FAIL column_types is empty or missing. "
            "Every column must have a declared type.")

    # ── 2. All types must be recognised ───────────────────────────
    for col, t in col_types.items():
        if t not in ALL_TYPES:
            errors.append(
                f"FAIL column '{col}' has unknown type '{t}'. "
                f"Valid types: {sorted(ALL_TYPES)}")

    # ── 3. Every numeric column must have public_bounds ───────────
    for col, t in col_types.items():
        if t not in NUMERIC_TYPES:
            continue
        if col not in public_bounds:
            errors.append(
                f"FAIL numeric column '{col}' (type: {t}) "
                f"has no entry in public_bounds. "
                f"Add: \"public_bounds\": {{\"{col}\": "
                f"{{\"min\": <value>, \"max\": <value>}}}}")
        else:
            lo, hi = _get_bounds(public_bounds[col])
            if lo is None or hi is None:
                errors.append(
                    f"FAIL public_bounds['{col}'] is missing "
                    f"min or max. Got: {public_bounds[col]}")
            else:
                try:
                    lo_f, hi_f = float(lo), float(hi)
                    if lo_f >= hi_f:
                        errors.append(
                            f"FAIL public_bounds['{col}'] is invalid: "
                            f"min={lo_f} >= max={hi_f}. "
                            f"min must be strictly less than max.")
                except (TypeError, ValueError):
                    errors.append(
                        f"FAIL public_bounds['{col}'] min or max "
                        f"is not numeric: min={lo}, max={hi}")

    # ── 4. Every categorical/ordinal/binary column must have
    #       public_categories ────────────────────────────────────
    for col, t in col_types.items():
        if t not in CAT_TYPES:
            continue
        if col not in public_cats:
            errors.append(
                f"FAIL {t} column '{col}' has no entry in "
                f"public_categories. "
                f"Declare the full set of valid values from "
                f"domain knowledge, e.g.: "
                f"\"public_categories\": {{\"{col}\": [val1, val2, ...]}}")
        else:
            cats = public_cats[col]
            if not isinstance(cats, list):
                errors.append(
                    f"FAIL public_categories['{col}'] must be a list, "
                    f"got {type(cats).__name__}")
            elif len(cats) < 2:
                errors.append(
                    f"FAIL public_categories['{col}'] has fewer than "
                    f"2 values: {cats}. "
                    f"A categorical column must have at least 2 categories.")
            if t == "binary":
                if isinstance(cats, list) and len(cats) != 2:
                    errors.append(
                        f"FAIL binary column '{col}' must have exactly "
                        f"2 categories, got {len(cats)}: {cats}. "
                        f"If there are more valid values, use type "
                        f"'categorical' instead.")

    # ── 5. Columns should not appear in both bounds and categories ─
    both = set(public_bounds.keys()) & set(public_cats.keys())
    for col in both:
        ctype = col_types.get(col, "unknown")
        if ctype in CAT_TYPES:
            warnings.append(
                f"WARN '{col}' (type: {ctype}) appears in both "
                f"public_bounds and public_categories. "
                f"For categorical columns only public_categories "
                f"is needed. public_bounds entry will be ignored "
                f"by CRNPrivBayes.")
        elif ctype in NUMERIC_TYPES:
            warnings.append(
                f"WARN '{col}' (type: {ctype}) appears in both "
                f"public_bounds and public_categories. "
                f"For numeric columns only public_bounds is needed.")

    # ── 6. target_spec must be present and complete ───────────────
    if not target_spec:
        errors.append(
            "FAIL target_spec is missing. "
            "Must declare at minimum: kind, targets, primary_target. "
            "For survival data use kind='survival_pair'.")
    else:
        kind    = target_spec.get("kind")
        targets = target_spec.get("targets", [])
        primary = target_spec.get("primary_target")

        if not kind:
            errors.append(
                "FAIL target_spec.kind is missing. "
                "Use 'single' for classification or "
                "'survival_pair' for survival analysis.")

        if not targets:
            errors.append(
                "FAIL target_spec.targets is empty. "
                "Must list the target column(s).")

        if not primary:
            errors.append(
                "FAIL target_spec.primary_target is missing. "
                "Must name the primary target column.")

        if primary and col_types and primary not in col_types:
            errors.append(
                f"FAIL target_spec.primary_target '{primary}' "
                f"is not in column_types. "
                f"Check spelling — column names are case-sensitive.")

        # ── 7. Survival pair specific checks ──────────────────────
        if kind == "survival_pair":
            if len(targets) != 2:
                errors.append(
                    f"FAIL survival_pair requires exactly 2 targets "
                    f"(event column and duration column), "
                    f"got {len(targets)}: {targets}")
            else:
                duration_col = next(
                    (t for t in targets if t != primary), None)

                if duration_col and col_types and \
                        duration_col not in col_types:
                    errors.append(
                        f"FAIL duration column '{duration_col}' "
                        f"is not in column_types.")

                if duration_col and duration_col not in public_bounds:
                    errors.append(
                        f"FAIL duration column '{duration_col}' "
                        f"has no entry in public_bounds. "
                        f"The maximum follow-up time must be declared "
                        f"from the study design.")

                # Must have survival_pair cross_column_constraint
                cross = constraints.get("cross_column_constraints", [])
                has_survival = any(
                    c.get("type") == "survival_pair" for c in cross)
                if not has_survival:
                    errors.append(
                        "FAIL survival_pair target_spec requires a "
                        "cross_column_constraint of type 'survival_pair'. "
                        "None found in constraints.cross_column_constraints. "
                        "Add: {\"type\": \"survival_pair\", "
                        "\"event_col\": \"<event>\", "
                        "\"time_col\": \"<duration>\", "
                        "\"event_allowed_values\": [0, 1], "
                        "\"time_min_exclusive\": 0}")

                # tau should be defined for RMST metrics
                if not target_spec.get("tau"):
                    warnings.append(
                        "WARN target_spec.tau (maximum follow-up horizon) "
                        "is not set. RMST metrics will be skipped. "
                        "Add tau equal to the study follow-up window.")

    # ── 8. cross_column_constraints integrity ─────────────────────
    for i, c in enumerate(constraints.get("cross_column_constraints", [])):
        ctype = c.get("type")
        if ctype == "survival_pair":
            event_col = c.get("event_col")
            time_col  = c.get("time_col")
            if not event_col:
                errors.append(
                    f"FAIL cross_column_constraints[{i}] of type "
                    f"'survival_pair' is missing event_col.")
            if not time_col:
                errors.append(
                    f"FAIL cross_column_constraints[{i}] of type "
                    f"'survival_pair' is missing time_col.")
            if event_col and col_types and event_col not in col_types:
                errors.append(
                    f"FAIL cross_column_constraints[{i}].event_col "
                    f"'{event_col}' not in column_types.")
            if time_col and col_types and time_col not in col_types:
                errors.append(
                    f"FAIL cross_column_constraints[{i}].time_col "
                    f"'{time_col}' not in column_types.")
            if not c.get("event_allowed_values"):
                errors.append(
                    f"FAIL cross_column_constraints[{i}] missing "
                    f"event_allowed_values. "
                    f"Declare the valid event codes e.g. [0, 1] or [1, 2].")

    # ── 8.5. sensitive_attributes (optional, for attribute-inference metric) ─
    sensitive = schema.get("sensitive_attributes")
    if sensitive is not None:
        if not isinstance(sensitive, list):
            errors.append(
                "FAIL sensitive_attributes must be a list of column names "
                "(e.g. [\"status\"] or [\"race\", \"income\"]).")
        elif len(sensitive) == 0:
            errors.append(
                "FAIL sensitive_attributes is empty. "
                "Omit the key or list at least one column to infer.")
        elif col_types:
            for col in sensitive:
                if col not in col_types:
                    errors.append(
                        f"FAIL sensitive_attributes contains '{col}' "
                        f"which is not in column_types.")

    # ── 9. Provenance ──────────────────────────────────────────────
    if not provenance:
        warnings.append(
            "WARN provenance block is missing. "
            "Cannot verify that bounds are schema-authoritative "
            "and not derived from data inspection. "
            "Add a provenance block documenting the source of "
            "each bound (study protocol, clinical standard, etc.)")
    else:
        bound_sources = provenance.get("bound_sources", {})
        for col, t in col_types.items():
            if t in NUMERIC_TYPES and col not in bound_sources:
                warnings.append(
                    f"WARN no provenance.bound_sources entry for "
                    f"numeric column '{col}'. "
                    f"Document where this bound comes from so "
                    f"reviewers can verify it is not data-derived.")

    # ── 10. Suspicious tight bounds ───────────────────────────────
    bound_sources = provenance.get("bound_sources", {})
    tight_cols = []
    for col, bv in public_bounds.items():
        lo, hi = _get_bounds(bv)
        if lo is None or hi is None:
            continue
        try:
            lo_f, hi_f = float(lo), float(hi)
        except (TypeError, ValueError):
            continue
        # Non-round bounds with no documented source (or auto-inferred) are suspicious
        src = bound_sources.get(col, "")
        undocumented = (col not in bound_sources) or (src == "inferred_from_data")
        if undocumented and not _is_round(lo_f) and not _is_round(hi_f):
            tight_cols.append(
                f"'{col}': min={lo_f}, max={hi_f}  [source: {src or 'missing'}]")

    if tight_cols:
        warnings.append(
            "WARN these columns have non-round bounds that appear data-derived. "
            "Replace provenance.bound_sources[col] value from "
            "'inferred_from_data' with a domain-knowledge citation "
            "(e.g. 'study_protocol_age_18_to_90'). "
            "Data-derived bounds weaken the DP guarantee:\n"
            + "\n".join(f"    {c}" for c in tight_cols))

    # ── Hard failure ───────────────────────────────────────────────
    if errors and strict:
        raise SchemaValidationError(
            "\n\nSchema validation FAILED — "
            "fix these errors before running synthesis:\n\n"
            + "\n".join(f"  {e}" for e in errors))

    return warnings + (errors if not strict else [])


# ─── Cross-validation against data ────────────────────────────────

def validate_against_data(schema: dict, df) -> List[str]:
    """
    Cross-check schema declarations against the actual dataframe.

    Catches representation mismatches before they cause silent errors
    inside the model — e.g. schema says ["Male","Female"] but CSV
    has ["male","female"] lowercase.

    Parameters
    ----------
    schema : dict
        Loaded schema-generator JSON.
    df : pd.DataFrame
        The dataset that will be passed to the synthesiser.

    Returns
    -------
    List[str]
        Warnings and failures. Caller decides whether to abort.
        Lines prefixed FAIL are hard errors.
        Lines prefixed WARN are worth investigating.
        Lines prefixed INFO are informational only.
    """
    import pandas as pd
    import numpy as np

    messages = []

    col_types     = schema.get("column_types", {})
    public_bounds = schema.get("public_bounds", {})
    public_cats   = schema.get("public_categories", {})

    schema_cols = set(col_types.keys())
    data_cols   = set(df.columns)

    # ── 11. Column presence ───────────────────────────────────────
    in_schema_not_data = schema_cols - data_cols
    in_data_not_schema = data_cols - schema_cols

    for col in sorted(in_schema_not_data):
        messages.append(
            f"FAIL column '{col}' is declared in schema "
            f"but not found in the dataframe. "
            f"Check column name spelling — names are case-sensitive.")

    for col in sorted(in_data_not_schema):
        messages.append(
            f"WARN column '{col}' is in the dataframe "
            f"but not declared in schema column_types. "
            f"It will be ignored by the synthesiser.")

    # ── 12. Binary columns — exactly 2 unique values ──────────────
    for col, t in col_types.items():
        if t != "binary" or col not in df.columns:
            continue
        unique_vals = set(df[col].dropna().unique())
        if len(unique_vals) > 2:
            messages.append(
                f"FAIL binary column '{col}' has {len(unique_vals)} "
                f"unique values in data: {unique_vals}. "
                f"Binary columns must have exactly 2. "
                f"Change type to 'categorical' if more values are valid.")
        elif len(unique_vals) < 2:
            messages.append(
                f"WARN binary column '{col}' has only "
                f"{len(unique_vals)} unique value(s) in data: "
                f"{unique_vals}. "
                f"This may cause issues with CPT estimation.")

    # ── 13. Categorical columns — schema categories match data ─────
    for col, cats in public_cats.items():
        if col not in df.columns:
            continue
        actual   = set(str(v) for v in df[col].dropna().unique())
        declared = set(str(v) for v in cats)

        in_data_not_declared = actual - declared
        in_declared_not_data = declared - actual

        if in_data_not_declared:
            messages.append(
                f"FAIL '{col}' has values in data that are NOT in "
                f"public_categories: {sorted(in_data_not_declared)}. "
                f"The synthesiser will not know how to handle these. "
                f"Either add them to public_categories or recode the data. "
                f"Schema has: {sorted(declared)}")

        if in_declared_not_data:
            messages.append(
                f"INFO '{col}' has schema values not seen in this "
                f"data sample: {sorted(in_declared_not_data)}. "
                f"This is correct if they are valid but rare values "
                f"(e.g. a cancer stage not present in this cohort). "
                f"The synthesiser will still know about them.")

    # ── 14. Numeric columns — observed range fits declared bounds ──
    for col, bv in public_bounds.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        lo, hi = _get_bounds(bv)
        if lo is None or hi is None:
            continue
        lo_f, hi_f     = float(lo), float(hi)
        obs_min        = float(vals.min())
        obs_max        = float(vals.max())

        if obs_min < lo_f:
            messages.append(
                f"FAIL '{col}' observed minimum {obs_min} is below "
                f"declared schema bound min={lo_f}. "
                f"Records with out-of-bounds values will cause "
                f"discretization errors. "
                f"Lower the schema min or clip the data.")

        if obs_max > hi_f:
            messages.append(
                f"FAIL '{col}' observed maximum {obs_max} exceeds "
                f"declared schema bound max={hi_f}. "
                f"Records with out-of-bounds values will cause "
                f"discretization errors. "
                f"Raise the schema max or clip the data.")

        # Warn if bounds are extremely wide relative to observed range
        observed_range = obs_max - obs_min
        declared_range = hi_f - lo_f
        if declared_range > 0 and observed_range / declared_range < 0.05:
            messages.append(
                f"WARN '{col}' uses only {observed_range / declared_range:.1%} "
                f"of its declared range "
                f"(observed [{obs_min}, {obs_max}] vs "
                f"declared [{lo_f}, {hi_f}]). "
                f"This is fine if bounds come from domain knowledge. "
                f"Most discretization bins will be empty.")

    # ── 15. Type consistency with pandas dtype ─────────────────────
    for col, t in col_types.items():
        if col not in df.columns:
            continue
        dtype_str = str(df[col].dtype)

        if t in NUMERIC_TYPES:
            dtype_base = dtype_str.rstrip("0123456789")
            if dtype_base not in ("int", "uint", "float"):
                messages.append(
                    f"WARN '{col}' is declared as numeric type '{t}' "
                    f"but has pandas dtype '{dtype_str}'. "
                    f"The model will attempt pd.to_numeric() conversion. "
                    f"Verify the column contains numeric values.")

        elif t in CAT_TYPES:
            dtype_base = dtype_str.rstrip("0123456789")
            if dtype_base in ("float",):
                # Float categorical is unusual but allowed (e.g. 0.0, 1.0)
                messages.append(
                    f"INFO '{col}' is declared as categorical type '{t}' "
                    f"but has float dtype '{dtype_str}'. "
                    f"Values will be matched as strings. "
                    f"Ensure public_categories uses the same "
                    f"representation (e.g. 0 not 0.0).")

    return messages


# ─── Synthesis readiness summary ──────────────────────────────────

def synthesis_readiness(schema: dict) -> dict:
    """
    Returns a structured summary of what the schema provides
    to guide synthesis. Useful for reporting in privacy_report().

    Returns
    -------
    dict with keys:
        n_columns_typed         : int
        n_numeric_with_bounds   : int
        n_categorical_with_cats : int
        n_binary_correct        : int
        survival_pair_defined   : bool
        cross_constraints       : int
        provenance_complete     : bool
        ready_for_synthesis     : bool
        missing_bounds          : list[str]
        missing_categories      : list[str]
    """
    col_types     = schema.get("column_types", {})
    public_bounds = schema.get("public_bounds", {})
    public_cats   = schema.get("public_categories", {})
    target_spec   = schema.get("target_spec", {})
    constraints   = schema.get("constraints", {})
    provenance    = schema.get("provenance", {})

    numeric_cols  = [c for c, t in col_types.items()
                     if t in NUMERIC_TYPES]
    cat_cols      = [c for c, t in col_types.items()
                     if t in CAT_TYPES]
    binary_cols   = [c for c, t in col_types.items()
                     if t == "binary"]

    missing_bounds = [c for c in numeric_cols
                      if c not in public_bounds]
    missing_cats   = [c for c in cat_cols
                      if c not in public_cats]

    cross = constraints.get("cross_column_constraints", [])
    survival_pair = any(c.get("type") == "survival_pair" for c in cross)

    bound_sources = provenance.get("bound_sources", {})
    prov_complete = all(c in bound_sources for c in numeric_cols)

    ready = (
        len(missing_bounds) == 0 and
        len(missing_cats) == 0 and
        bool(target_spec)
    )

    return {
        "n_columns_typed":         len(col_types),
        "n_numeric_cols":          len(numeric_cols),
        "n_numeric_with_bounds":   len(numeric_cols) - len(missing_bounds),
        "n_categorical_cols":      len(cat_cols),
        "n_categorical_with_cats": len(cat_cols) - len(missing_cats),
        "n_binary_cols":           len(binary_cols),
        "survival_pair_defined":   survival_pair,
        "tau":                     target_spec.get("tau"),
        "cross_constraints":       len(cross),
        "provenance_complete":     prov_complete,
        "ready_for_synthesis":     ready,
        "missing_bounds":          missing_bounds,
        "missing_categories":      missing_cats,
    }


# ─── Full report ──────────────────────────────────────────────────

def validate_and_report(
        schema_path: str,
        data_path: Optional[str] = None,
        strict: bool = True) -> bool:
    """
    Load schema (and optionally data), run all checks, print report.

    Returns True if schema passes all hard checks.
    """
    print(f"\n{'=' * 65}")
    print(f"  Schema: {schema_path}")
    if data_path:
        print(f"  Data:   {data_path}")
    print(f"{'=' * 65}")

    # Load schema
    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except FileNotFoundError:
        print(f"FAIL schema file not found: {schema_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"FAIL schema is not valid JSON: {e}")
        return False

    # Print summary
    ds = schema.get("dataset", {})
    if isinstance(ds, str):
        ds = {"name": ds, "description": "—"}
    ts = schema.get("target_spec", {})
    print(f"\n  Dataset       : {ds.get('name', '(unnamed)')}")
    print(f"  Description   : {ds.get('description', '—')}")

    readiness = synthesis_readiness(schema)
    print(f"\n  Columns typed          : {readiness['n_columns_typed']}")
    print(f"  Numeric with bounds    : "
          f"{readiness['n_numeric_with_bounds']} / "
          f"{readiness['n_numeric_cols']}")
    print(f"  Categorical with cats  : "
          f"{readiness['n_categorical_with_cats']} / "
          f"{readiness['n_categorical_cols']}")
    print(f"  Binary columns         : {readiness['n_binary_cols']}")
    print(f"  Target kind            : {ts.get('kind', '—')}")
    print(f"  Primary target         : {ts.get('primary_target', '—')}")
    sens = schema.get("sensitive_attributes")
    print(f"  Sensitive (attr. inf.) : {sens if sens else '(primary_target)'}")
    print(f"  Survival pair defined  : {readiness['survival_pair_defined']}")
    print(f"  Tau (follow-up horizon): {readiness['tau']}")
    print(f"  Provenance complete    : {readiness['provenance_complete']}")

    passed = True

    # Schema-only validation
    print(f"\n{'─' * 65}")
    print("  Schema validation")
    print(f"{'─' * 65}")
    try:
        schema_warnings = validate(schema, strict=True)
        print(f"  PASS — schema is internally consistent")
        if schema_warnings:
            print(f"\n  {len(schema_warnings)} warning(s):")
            for w in schema_warnings:
                # Wrap long warnings
                lines = w.split(". ")
                print(f"\n    {lines[0]}.")
                for line in lines[1:]:
                    if line:
                        print(f"    {line}.")
    except SchemaValidationError as e:
        print(str(e))
        passed = False

    # Cross-validation against data
    if data_path:
        print(f"\n{'─' * 65}")
        print(f"  Cross-validation against data")
        print(f"{'─' * 65}")
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

            cross_messages = validate_against_data(schema, df)

            hard = [m for m in cross_messages if m.startswith("FAIL")]
            warn = [m for m in cross_messages if m.startswith("WARN")]
            info = [m for m in cross_messages if m.startswith("INFO")]

            if hard:
                passed = False
                print(f"\n  {len(hard)} hard error(s):")
                for m in hard:
                    print(f"\n    {m}")

            if warn:
                print(f"\n  {len(warn)} warning(s):")
                for m in warn:
                    lines = m.split(". ")
                    print(f"\n    {lines[0]}.")
                    for line in lines[1:]:
                        if line:
                            print(f"    {line}.")

            if info:
                print(f"\n  {len(info)} info message(s):")
                for m in info:
                    print(f"    {m}")

            if not hard and not warn:
                print("  PASS — schema declarations match data exactly")
            elif not hard:
                print("  PASS with warnings — check warnings above")
            else:
                print("  FAIL — fix errors before running synthesis")

        except ImportError:
            print("  SKIP — pandas not installed, "
                  "cannot cross-validate against data")
        except FileNotFoundError:
            print(f"  FAIL data file not found: {data_path}")
            passed = False

    # Final verdict
    print(f"\n{'─' * 65}")
    if passed:
        ready = readiness["ready_for_synthesis"]
        if ready:
            print("  RESULT: PASS — schema is ready to guide synthesis")
        else:
            missing = (readiness["missing_bounds"] +
                       readiness["missing_categories"])
            print(f"  RESULT: PASS with gaps — schema is not fully complete.")
            print(f"  Missing metadata for: {missing}")
            print(f"  Columns without public metadata will fall back "
                  f"to data-derived inference (compliance gap).")
    else:
        print("  RESULT: FAIL — fix errors above before running synthesis")
    print(f"{'─' * 65}\n")

    return passed


# ─── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    schema_paths = [a for a in args if a.endswith(".json")]
    data_paths   = [a for a in args if a.endswith(".csv")]

    if not schema_paths:
        print("Error: no .json schema file provided.")
        print("Usage: python schema_validator.py schema.json [data.csv]")
        sys.exit(1)

    data_path = data_paths[0] if data_paths else None

    all_passed = True
    for schema_path in schema_paths:
        ok = validate_and_report(
            schema_path=schema_path,
            data_path=data_path,
            strict=True)
        if not ok:
            all_passed = False

    sys.exit(0 if all_passed else 1)
