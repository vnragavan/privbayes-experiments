"""
generate_dummy_data.py

Generate dummy (random) data that conforms to a CRN schema.

Usage:
    python generate_dummy_data.py --schema schemas/lung_schema.json --n 200
    python generate_dummy_data.py --schema schemas/lung_schema.json --n 500 --out data/lung_dummy.csv --seed 42

What it generates per column type:
    integer    → uniform random integers in [min, max]
    continuous → uniform random floats in [min, max]
    binary     → random choice from public_categories (["0","1"] etc.)
    ordinal    → random choice from public_categories
    categorical→ random choice from public_categories
    datetime   → not yet supported, skipped with a warning

Survival pair constraint:
    If the schema has a survival_pair cross-column constraint, the generator
    respects it: censored rows (event=0) get time drawn from the full range,
    event rows (event=1) get time drawn from [min, tau] where tau is taken
    from target_spec.tau if set.

Missing values:
    Columns with missing_value_rates > 0 get that fraction of their values
    replaced with NaN after generation.

Output:
    CSV file with columns in schema column_types order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_bounds(bv) -> tuple[float | None, float | None]:
    """Extract (min, max) from a bounds value (list or dict)."""
    if isinstance(bv, dict):
        return float(bv["min"]), float(bv["max"])
    elif isinstance(bv, (list, tuple)) and len(bv) >= 2:
        return float(bv[0]), float(bv[1])
    return None, None


def _is_float_category(cats: list) -> bool:
    """Return True if categories look like float strings e.g. '0.0', '1.0'."""
    return any("." in str(c) for c in cats)


# ── Per-column generators ─────────────────────────────────────────────────────

def _gen_integer(n: int, lo: float, hi: float, rng: np.random.Generator) -> pd.Series:
    lo_i, hi_i = int(round(lo)), int(round(hi))
    return pd.Series(rng.integers(lo_i, hi_i + 1, size=n), dtype="int64")


def _gen_continuous(n: int, lo: float, hi: float, rng: np.random.Generator) -> pd.Series:
    return pd.Series(rng.uniform(lo, hi, size=n), dtype="float64")


def _gen_categorical(n: int, cats: list, rng: np.random.Generator) -> pd.Series:
    idx = rng.integers(0, len(cats), size=n)
    return pd.Series([cats[i] for i in idx], dtype=object)


# ── Main generator ────────────────────────────────────────────────────────────

def generate(schema: dict, n: int, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    col_types   = schema.get("column_types", {})
    raw_bounds  = schema.get("public_bounds", {})
    raw_cats    = schema.get("public_categories", {})
    mvr         = schema.get("missing_value_rates", {})
    target_spec = schema.get("target_spec", {})

    # Identify survival pair if present
    survival_event_col = None
    survival_time_col  = None
    tau                = target_spec.get("tau")
    for cc in schema.get("constraints", {}).get("cross_column_constraints", []):
        if cc.get("type") == "survival_pair":
            survival_event_col = cc.get("event_col")
            survival_time_col  = cc.get("time_col")

    columns = list(col_types.keys())
    data    = {}

    # ── Generate event column first so time can be conditioned on it ──────────
    if survival_event_col and survival_event_col in col_types:
        cats = raw_cats.get(survival_event_col, ["0", "1"])
        data[survival_event_col] = _gen_categorical(n, cats, rng)

    for col in columns:
        if col == survival_event_col:
            continue  # already generated above

        ctype = col_types[col]

        # ── Survival time column ────────────────────────────────────────────
        if col == survival_time_col and survival_event_col in data:
            bv    = raw_bounds.get(col)
            lo, hi = _get_bounds(bv) if bv else (1, tau or 100)
            lo    = lo or 1

            # Event rows: time in [lo, tau]; censored rows: time in [lo, hi]
            event_vals  = data[survival_event_col]
            times       = np.empty(n, dtype="int64")
            event_hi    = float(tau) if tau else hi

            for i, ev in enumerate(event_vals):
                ev_int = int(float(ev))  # handles "0"/"1" and 0/1
                t_hi   = event_hi if ev_int == 1 else hi
                times[i] = rng.integers(int(lo), int(t_hi) + 1)

            # Ensure time > 0 (min_exclusive=0 constraint)
            times = np.clip(times, max(1, int(lo)), int(hi))
            data[col] = pd.Series(times, dtype="int64")
            continue

        # ── Other columns by type ───────────────────────────────────────────
        if ctype in ("binary", "ordinal", "categorical"):
            cats = raw_cats.get(col)
            if not cats:
                print(f"  WARN: no public_categories for {col!r}, using ['0','1']",
                      file=sys.stderr)
                cats = ["0", "1"]
            data[col] = _gen_categorical(n, cats, rng)

        elif ctype == "integer":
            bv = raw_bounds.get(col)
            if not bv:
                print(f"  WARN: no public_bounds for {col!r}, using [0, 100]",
                      file=sys.stderr)
                lo, hi = 0.0, 100.0
            else:
                lo, hi = _get_bounds(bv)
            data[col] = _gen_integer(n, lo, hi, rng)

        elif ctype == "continuous":
            bv = raw_bounds.get(col)
            if not bv:
                print(f"  WARN: no public_bounds for {col!r}, using [0.0, 1.0]",
                      file=sys.stderr)
                lo, hi = 0.0, 1.0
            else:
                lo, hi = _get_bounds(bv)
            data[col] = _gen_continuous(n, lo, hi, rng)

        elif ctype == "datetime":
            print(f"  WARN: datetime column {col!r} skipped (not supported).",
                  file=sys.stderr)
            data[col] = pd.Series([pd.NaT] * n)

        else:
            print(f"  WARN: unknown type {ctype!r} for {col!r}, filling with NaN.",
                  file=sys.stderr)
            data[col] = pd.Series([np.nan] * n)

    df = pd.DataFrame(data, columns=columns)

    # ── Apply missing value rates ─────────────────────────────────────────────
    for col, rate in mvr.items():
        if rate > 0 and col in df.columns:
            mask = rng.random(n) < rate
            df.loc[mask, col] = np.nan
            print(f"  INFO: introduced {mask.sum()} NaN in {col!r} "
                  f"(rate={rate:.1%})", file=sys.stderr)

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate dummy data conforming to a CRN schema.")
    ap.add_argument("--schema", required=True,
                    help="Path to schema JSON file")
    ap.add_argument("--n", type=int, default=200,
                    help="Number of rows to generate (default: 200)")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: <schema_stem>_dummy.csv)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
    args = ap.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"ERROR: schema file not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    with open(schema_path) as f:
        schema = json.load(f)

    out_path = Path(args.out) if args.out else \
        schema_path.parent / f"{schema_path.stem}_dummy.csv"

    print(f"Schema  : {schema_path}")
    print(f"Rows    : {args.n}")
    print(f"Seed    : {args.seed}")
    print(f"Output  : {out_path}")

    df = generate(schema, n=args.n, seed=args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nGenerated {len(df)} rows × {len(df.columns)} columns")
    print(df.dtypes.to_string())
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
