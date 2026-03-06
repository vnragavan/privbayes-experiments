#!/usr/bin/env python3
"""
schema_generator.py — Generates a schema JSON from a CSV file, aligned with:

  - schema_validator.py (column_types, public_bounds, public_categories, target_spec,
    constraints including survival_pair, provenance; optional sensitive_attributes)
  - implementations: CRN (crn_privbayes), dpmm (via adapters/schema_to_dpmm),
    SynthCity (wrapper uses public_bounds/public_categories for clipping)
  - metrics/report.py and compute_sweep_metrics (target_spec.primary_target,
    get_attribute_inference_target(schema) which uses sensitive_attributes or primary_target)

Key behaviour:
  - GUID columns are EXCLUDED from column_types (not modelled by PrivBayes)
  - Small-domain integer columns (≤ MAX_INTEGER_LEVELS unique values) promoted to ordinal
  - Constant columns detected and flagged; excluded from synthesis
  - primary_target correctly set for survival schemas (always event_col)
  - Survival target_spec targets order enforced: [event_col, time_col]
  - tau (RMST horizon) inferred from duration column max and written to target_spec
  - Optional sensitive_attributes (--sensitive-attributes) for attribute-inference metric
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

SCHEMA_VERSION = "1.0.0"
MAX_INTEGER_LEVELS = 20   # integer columns with ≤ this many unique values → ordinal

_UUID_RE = re.compile(
    r"^(?:"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    r"|[0-9a-fA-F]{32}"
    r"|\{[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\}"
    r"|urn:uuid:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    r")$"
)

# Common target/event column names — extended for clinical/survival datasets
_TARGET_HINTS = [
    "target", "label", "class", "outcome", "income",       # classification
    "event", "status", "died", "death", "recurrence",       # survival events
    "arrest", "failure", "relapse", "deceased", "censored", # more survival
]
_TIME_HINTS = [
    "time", "duration", "survival_time", "follow_up",
    "days", "weeks", "months", "time_to_event", "week",
]


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
    """CRN requires public_bounds for ordinal columns. Derive min/max/n_bins from category list."""
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
        "%d/%m/%Y %H:%M",    "%d/%m/%Y",
        "%m-%d-%Y %H:%M",    "%m-%d-%Y",
        "%Y/%m/%dT%H:%M:%SZ","%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",          "%Y.%m.%d %H:%M:%S",
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
            {"dayfirst": True, "utc": True},  {"yearfirst": True, "utc": True},
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
    return float(x.str.strip().str.fullmatch(_UUID_RE, na=False).mean()) >= float(min_match_frac)


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
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0.0, 1.0]
    vmin, vmax = float(np.min(x)), float(np.max(x))
    span = vmax - vmin
    pad = (pad_frac * span) if span > 0 else (max(abs(vmin) * pad_frac, 1.0) if pad_frac > 0 else 0.0)
    lo, hi = vmin - pad, vmax + pad
    if integer_like:
        return [int(np.floor(lo)), int(np.ceil(hi))]
    return [lo, hi]


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
    """Return sorted string category list if integer column has ≤ max_levels unique values.

    Category strings are written to match how pandas re-reads the CSV:
    - If the column is integer dtype → "0", "1", "2"
    - If the column is float dtype (integers stored as float due to NaN) → "0.0", "1.0"
    This prevents cross-validation mismatches.
    """
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    if not np.all(np.isclose(x, np.round(x), atol=1e-8)):
        return None  # not integer-like
    ux = np.unique(x)
    if ux.size > max_levels:
        return None
    # Use float representation if original series is float dtype (NaN present)
    if pd.api.types.is_float_dtype(s):
        return [f"{int(round(v))}.0" for v in sorted(ux.tolist())]
    return [str(int(round(v))) for v in sorted(ux.tolist())]


def _build_constraints(
    *,
    column_types: dict[str, str],
    public_categories: dict[str, list[str]],
    public_bounds: dict[str, list[float | int]],
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
                if "min" in bv: c["min"] = bv["min"]
                if "max" in bv: c["max"] = bv["max"]
                if "n_bins" in bv: c["n_bins"] = bv["n_bins"]
            elif isinstance(bv, (list, tuple)) and len(bv) == 2:
                c["min"] = bv[0]
                c["max"] = bv[1]
        if col in constant_columns:
            c["note"] = "constant_column_excluded_from_synthesis"
        column_constraints[col] = c

    if isinstance(target_spec, dict) and str(target_spec.get("kind")) == "survival_pair":
        tcols = target_spec.get("targets")
        if isinstance(tcols, list) and len(tcols) >= 2:
            event_col = str(tcols[0])
            time_col  = str(tcols[1])
            cross_column_constraints.append({
                "name":                 "survival_pair_definition",
                "type":                 "survival_pair",
                "event_col":            event_col,
                "time_col":             time_col,
                "event_allowed_values": [0, 1],
                "time_min_exclusive":   0,
            })
            ec = column_constraints.get(event_col, {"type": "binary"})
            ec["allowed_values"] = ["0", "1"]
            column_constraints[event_col] = ec
            tc = column_constraints.get(time_col, {"type": column_types.get(time_col, "continuous")})
            tc["min_exclusive"] = 0
            column_constraints[time_col] = tc

    return {
        "column_constraints":       column_constraints,
        "cross_column_constraints": cross_column_constraints,
        "row_group_constraints":    [],
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",                  type=Path,  required=True)
    ap.add_argument("--out",                   type=Path,  required=True)
    ap.add_argument("--dataset-name",          type=str,   default=None)
    ap.add_argument("--target-col",            type=str,   default=None)
    ap.add_argument("--target-cols",           type=str,   default=None)
    ap.add_argument("--target-kind",           type=str,   default=None)
    ap.add_argument("--survival-event-col",    type=str,   default=None)
    ap.add_argument("--survival-time-col",     type=str,   default=None)
    ap.add_argument("--column-types",          type=Path,  default=None)
    ap.add_argument("--target-spec-file",      type=Path,  default=None)
    ap.add_argument("--constraints-file",      type=Path,  default=None)
    ap.add_argument("--sensitive-attributes", type=str,   default=None,
                    help="Comma-separated column names for attribute-inference metric (e.g. status); optional, defaults to primary_target")
    ap.add_argument("--delimiter",             type=str,   default="auto")
    ap.add_argument("--pad-frac",              type=float, default=0.0)
    ap.add_argument("--pad-frac-integer",      type=float, default=None)
    ap.add_argument("--pad-frac-continuous",   type=float, default=None)
    ap.add_argument("--infer-categories",      action="store_true")
    ap.add_argument("--max-categories",        type=int,   default=200)
    ap.add_argument("--max-integer-levels",    type=int,   default=MAX_INTEGER_LEVELS,
                    help="Integer columns with ≤ this many unique values are promoted to ordinal")
    ap.add_argument("--infer-binary-domain",   action="store_true")
    ap.add_argument("--infer-datetimes",       action="store_true")
    ap.add_argument("--datetime-min-parse-frac", type=float, default=0.95)
    ap.add_argument("--datetime-output-format",  type=str,   default="preserve")
    ap.add_argument("--guid-min-match-frac",   type=float, default=0.95)
    ap.add_argument("--target-is-classifier",  action="store_true")
    ap.add_argument("--no-publish-label-domain", action="store_true")
    ap.add_argument("--redact-source-path",    action="store_true")
    ap.add_argument("--n-records",            type=int, default=None,
                    help="Declared row count for CRN (default: len(data)); use train size if using a fixed split")
    args = ap.parse_args()

    delimiter = (
        _infer_csv_delimiter(args.data)
        if str(args.delimiter).strip().lower() == "auto"
        else str(args.delimiter)
    )
    df = pd.read_csv(args.data, sep=delimiter, engine="python")
    cols = [str(c) for c in df.columns]
    n_records = int(args.n_records) if args.n_records is not None else len(df)

    # ── Resolve survival columns first so hints work correctly ─────
    survival_event_col = args.survival_event_col
    survival_time_col  = args.survival_time_col

    if bool(survival_event_col) != bool(survival_time_col):
        raise SystemExit("Both --survival-event-col and --survival-time-col must be provided together")

    # Auto-infer survival columns if neither was given but target-kind is survival_pair
    if args.target_kind == "survival_pair" and not survival_event_col:
        survival_event_col = _infer_target_col(cols)
        survival_time_col  = _infer_time_col(cols, exclude=survival_event_col)
        if not survival_event_col or not survival_time_col:
            raise SystemExit(
                "Could not auto-infer survival columns. "
                "Please supply --survival-event-col and --survival-time-col explicitly."
            )
        print(f"Auto-inferred survival columns: event={survival_event_col}, time={survival_time_col}")

    target_col = args.target_col or (survival_event_col if survival_event_col else _infer_target_col(cols))

    # ── Storage ───────────────────────────────────────────────────
    public_bounds:       dict[str, list] = {}
    public_categories:   dict[str, list] = {}
    missing_value_rates: dict[str, float] = {}
    column_types:        dict[str, str]   = {}
    datetime_spec:       dict[str, dict]  = {}
    guid_like_columns:   list[str]        = []
    constant_columns:    list[str]        = []

    pad_frac_global    = float(args.pad_frac)
    pad_frac_integer   = float(args.pad_frac_integer)   if args.pad_frac_integer   is not None else pad_frac_global
    pad_frac_continuous= float(args.pad_frac_continuous) if args.pad_frac_continuous is not None else pad_frac_global
    max_int_levels     = int(args.max_integer_levels)

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

    # ── Per-column inference ───────────────────────────────────────
    for c in cols:
        s0 = df[c]
        missing_value_rates[c] = float(s0.isna().mean())

        # 1. GUID → excluded from column_types entirely
        if _is_guid_like_series(s0, min_match_frac=float(args.guid_min_match_frac)):
            guid_like_columns.append(c)
            print(f"  [GUID]     {c!r} — excluded from synthesis")
            continue

        # 2. Constant column → flag and exclude
        n_unique = s0.dropna().nunique()
        if n_unique <= 1:
            constant_columns.append(c)
            print(f"  [CONST]    {c!r} — {n_unique} unique value(s), excluded from synthesis")
            continue

        # 3. Optional datetime parsing
        s, dt_converted, dt_fmt_hint = (
            _maybe_parse_datetime_like(s0, min_parse_frac=float(args.datetime_min_parse_frac))
            if bool(args.infer_datetimes) else (s0, False, None)
        )

        # 4. User type override
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
            elif t in {"continuous", "integer"} and _is_number_like_series(s):
                this_pad = pad_frac_integer if t == "integer" else pad_frac_continuous
                public_bounds[c] = _bounds_for_number_like(s, this_pad, integer_like=(t == "integer"))
            if dt_converted:
                out_fmt = (dt_fmt_hint if str(args.datetime_output_format).strip().lower() == "preserve"
                           else str(args.datetime_output_format))
                datetime_spec[c] = {
                    "storage": "epoch_ns",
                    "output_format": out_fmt or "%Y-%m-%dT%H:%M:%S",
                    "timezone": "UTC",
                }
            continue

        # 5. Bool → binary {0,1}  (False/True → "0"/"1")
        if pd.api.types.is_bool_dtype(s0):
            column_types[c] = "binary"
            public_categories[c] = ["0", "1"]

        # 6. Numeric / datetime-like
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
                _lo, _hi = _bounds_for_number_like(s, pad_frac_integer, integer_like=True)
                public_bounds[c] = {"min": _lo, "max": _hi, "n_bins": 100}
            else:
                # Determine integer vs continuous
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

                # Binary check: CRN accepts only 0/1 for binary. Two integer values that are
                # not 0/1 (e.g. sex 1/2) become categorical.
                bin_dom = _binary_integer_domain_values(s)

                if bin_dom is not None:
                    normalized = sorted([str(x).strip() for x in bin_dom])
                    if normalized in (["0", "1"], ["0.0", "1.0"]):
                        column_types[c] = "binary"
                        public_categories[c] = ["0", "1"]
                        print(f"  [BINARY]   {c!r} — values [0, 1]")
                    else:
                        column_types[c] = "categorical"
                        public_categories[c] = bin_dom
                        print(f"  [CATEG]    {c!r} — two values {bin_dom} (not 0/1, so categorical for CRN)")

                elif raw_type == "integer":
                    # Small-domain integer → ordinal (e.g. Likert, coded variables)
                    small_dom = _small_integer_domain_values(s, max_int_levels)
                    if small_dom is not None:
                        column_types[c] = "ordinal"
                        public_categories[c] = small_dom
                        public_bounds[c] = _ordinal_bounds_from_categories(small_dom)
                        print(f"  [ORDINAL]  {c!r} — {len(small_dom)} integer levels promoted to ordinal")
                    else:
                        column_types[c] = "integer"
                        # Fix 3: write dict format with n_bins for dpmm adapter
                        lo, hi = _bounds_for_number_like(s, pad_frac_integer, integer_like=True)
                        n_unique = int(pd.to_numeric(s, errors="coerce").nunique())
                        public_bounds[c] = {"min": lo, "max": hi, "n_bins": min(n_unique, 100)}

                else:  # continuous
                    column_types[c] = "continuous"
                    lo, hi = _bounds_for_number_like(s, pad_frac_continuous, integer_like=False)
                    public_bounds[c] = {"min": lo, "max": hi}

        # 7. Categorical / string
        else:
            column_types[c] = "categorical"
            if args.infer_categories:
                u = pd.Series(s, copy=False).astype("string").dropna().unique().tolist()
                u = sorted([str(x) for x in u])
                if 0 < len(u) <= int(args.max_categories):
                    public_categories[c] = u
                else:
                    print(f"  [WARN]     {c!r} has {len(u)} unique string values — "
                          f"too many for public_categories (max={args.max_categories}). "
                          f"Provide bounds manually or exclude this column.")

    # ── Survival: force event column to binary {0,1} ─────────────
    if survival_event_col and survival_event_col in column_types:
        column_types[survival_event_col] = "binary"
        public_categories[survival_event_col] = ["0", "1"]
        public_bounds.pop(survival_event_col, None)
        print(f"  [SURVIVAL] {survival_event_col!r} forced to binary {{0,1}}")

    # Infer tau (RMST horizon) from observed max duration
    tau: int | None = None
    if survival_time_col and survival_time_col in df.columns:
        max_t = pd.to_numeric(df[survival_time_col], errors="coerce").max()
        if np.isfinite(max_t):
            tau = int(np.ceil(max_t))

    # ── Build target_spec ─────────────────────────────────────────
    target_spec: dict[str, Any] | None = None

    if args.target_spec_file is not None:
        target_spec = json.loads(args.target_spec_file.read_text())
        if not isinstance(target_spec, dict):
            raise SystemExit("--target-spec-file must contain a JSON object")

    elif survival_event_col:
        # Survival schema — targets order: [event_col, time_col]
        target_spec = {
            "targets":        [survival_event_col, survival_time_col],
            "kind":           "survival_pair",
            "primary_target": survival_event_col,   # always event, never None
            "dtypes": {
                survival_event_col: "ordinal",
                survival_time_col:  _infer_target_dtype(df, survival_time_col),
            },
        }
        if tau is not None:
            target_spec["tau"] = tau
            print(f"  [TAU]      RMST horizon inferred as {tau} from max({survival_time_col!r})")

    else:
        targets = _parse_csv_list(args.target_cols)
        if not targets and target_col:
            targets = [target_col]
        if targets:
            target_kind = args.target_kind or ("single" if len(targets) == 1 else "multi_target")
            target_spec = {
                "targets":        targets,
                "kind":           target_kind,
                "primary_target": target_col,
                "dtypes":         {t: _infer_target_dtype(df, t) for t in targets},
            }

    # Normalise dtypes in target_spec to match column_types
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

    # ── Label domain ──────────────────────────────────────────────
    label_domain: list[str] = []
    if target_col and target_col in df.columns and not args.no_publish_label_domain:
        is_survival   = target_spec is not None and str(target_spec.get("kind")) == "survival_pair"
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

    # ── Build provenance.bound_sources ───────────────────────────
    # Every numeric column gets an explicit source tag so the validator
    # does not flag them all as undocumented. User should replace
    # "inferred_from_data" with a domain-knowledge citation where applicable.
    bound_sources: dict[str, str] = {}
    for col, t in column_types.items():
        if t in {"continuous", "integer"}:
            bound_sources[col] = "inferred_from_data"

    # ── Assemble schema ───────────────────────────────────────────
    # CRN PrivBayes (require_public=True) needs n_records: in dataset dict or dataset_info.
    # We keep "dataset" as a string for compatibility and add dataset_info.n_records.
    schema: dict[str, Any] = {
        "schema_version":     SCHEMA_VERSION,
        "dataset":            args.dataset_name or args.data.stem,
        "dataset_info":       {"n_records": n_records},
        "target_col":         target_col,
        "label_domain":       label_domain,
        "missing_value_rates":missing_value_rates,
        "public_bounds":      public_bounds,
        "public_categories":  public_categories,
        "column_types":       column_types,
        "datetime_spec":      datetime_spec,
        "provenance": {
            "generated_at_utc":      datetime.now(timezone.utc).isoformat(),
            "source_csv":            ("example_data_path_to_csv_file" if bool(args.redact_source_path)
                                      else str(args.data)),
            "source_delimiter":      delimiter,
            "pad_frac":              pad_frac_global,
            "pad_frac_integer":      pad_frac_integer,
            "pad_frac_continuous":   pad_frac_continuous,
            "inferred_categories":   bool(args.infer_categories),
            "max_categories":        int(args.max_categories),
            "max_integer_levels":    max_int_levels,
            "inferred_datetimes":    bool(args.infer_datetimes),
            "datetime_min_parse_frac": float(args.datetime_min_parse_frac),
            "inferred_binary_domain": bool(args.infer_binary_domain),
            "guid_min_match_frac":   float(args.guid_min_match_frac),
            "guid_like_columns":     guid_like_columns,
            "constant_columns":      constant_columns,
            "datetime_output_format":str(args.datetime_output_format),
            "no_publish_label_domain": bool(args.no_publish_label_domain),
            "column_types_overrides":str(args.column_types) if args.column_types is not None else None,
            "bound_sources":         bound_sources,
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

    # ── Write ─────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, indent=2) + "\n")

    print(f"\nSchema summary:")
    print(f"  Columns in schema : {len(column_types)}")
    print(f"  GUID excluded     : {len(guid_like_columns)}  {guid_like_columns}")
    print(f"  Constant excluded : {len(constant_columns)}  {constant_columns}")
    print(f"  Binary columns    : {[c for c,t in column_types.items() if t=='binary']}")
    print(f"  Ordinal columns   : {[c for c,t in column_types.items() if t=='ordinal']}")
    print(f"  Survival kind     : {target_spec.get('kind') if target_spec else 'n/a'}")
    if target_spec and target_spec.get("kind") == "survival_pair":
        print(f"  Event col         : {target_spec.get('primary_target')}")
        print(f"  Time col          : {target_spec['targets'][1] if len(target_spec.get('targets',[])) > 1 else 'n/a'}")
        print(f"  Tau (RMST)        : {target_spec.get('tau')}")
    print(f"  Bounds with n_bins: {[c for c,b in public_bounds.items() if isinstance(b,dict) and 'n_bins' in b]}")
    print(f"\nWrote schema → {args.out}")


if __name__ == "__main__":
    main()
