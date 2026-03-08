"""
metrics/constraint/validator.py

Evaluates raw synthetic output against schema constraints.
Reports violations — does not fix them.
"""

import numpy as np
import pandas as pd


def evaluate_column_constraints(synth_df, column_constraints: dict) -> dict:
    per = {}
    for col, spec in column_constraints.items():
        if col not in synth_df.columns:
            continue
        vals = pd.to_numeric(synth_df[col], errors="coerce")
        mask = pd.Series(False, index=synth_df.index)
        if "min_exclusive" in spec and spec["min_exclusive"] is not None:
            mask |= vals <= spec["min_exclusive"]
        if "min" in spec and spec["min"] is not None:
            mask |= vals < spec["min"]
        if "max_exclusive" in spec and spec["max_exclusive"] is not None:
            mask |= vals >= spec["max_exclusive"]
        if "max" in spec and spec["max"] is not None:
            mask |= vals > spec["max"]
        per[col] = float(mask.mean())

    overall = float(np.mean(list(per.values()))) if per else float("nan")
    return {"per_constraint": per, "overall": overall}


def evaluate_cross_column_constraints(synth_df,
                                       cross_column_constraints: list) -> dict:
    per = {}
    for constraint in cross_column_constraints:
        ctype = constraint.get("type")
        name = constraint.get("name", ctype or "unnamed")
        if ctype == "survival_pair":
            event_col = constraint.get("event_col")
            time_col = constraint.get("time_col")
            allowed = constraint.get("event_allowed_values", [0, 1])
            if event_col in synth_df.columns and time_col in synth_df.columns:
                evt = pd.to_numeric(synth_df[event_col], errors="coerce")
                time_vals = pd.to_numeric(synth_df[time_col], errors="coerce")
                if "time_min_inclusive" in constraint and constraint["time_min_inclusive"] is not None:
                    time_ok = time_vals >= constraint["time_min_inclusive"]
                else:
                    time_min_exc = constraint.get("time_min_exclusive", 0)
                    time_ok = time_vals > time_min_exc
                mask = ~evt.isin(allowed) | ~time_ok
                per[name] = float(mask.mean())

    overall = float(np.mean(list(per.values()))) if per else float("nan")
    return {"per_constraint": per, "overall": overall}


def constraint_violation_summary(synth_df, schema: dict) -> dict:
    constraints = schema.get("constraints", {})
    col_c = constraints.get("column_constraints", {})
    cross_c = constraints.get("cross_column_constraints", [])

    col_result = evaluate_column_constraints(synth_df, col_c)
    cross_result = evaluate_cross_column_constraints(synth_df, cross_c)

    # Any record with any violation
    n = len(synth_df)
    any_viol = pd.Series(False, index=synth_df.index)

    for col, spec in col_c.items():
        if col not in synth_df.columns:
            continue
        vals = pd.to_numeric(synth_df[col], errors="coerce")
        if "min_exclusive" in spec and spec["min_exclusive"] is not None:
            any_viol |= vals <= spec["min_exclusive"]
        if "max" in spec and spec["max"] is not None:
            any_viol |= vals > spec["max"]

    # Per-cause breakdown for survival_pair: event vs time
    survival_event_viol = pd.Series(False, index=synth_df.index)
    survival_time_viol = pd.Series(False, index=synth_df.index)

    for constraint in cross_c:
        ctype = constraint.get("type")
        if ctype == "survival_pair":
            event_col = constraint.get("event_col")
            time_col = constraint.get("time_col")
            allowed = constraint.get("event_allowed_values", [0, 1])
            if event_col in synth_df.columns and time_col in synth_df.columns:
                evt = pd.to_numeric(synth_df[event_col], errors="coerce")
                time_vals = pd.to_numeric(synth_df[time_col], errors="coerce")
                event_bad = ~evt.isin(allowed)
                any_viol |= event_bad
                survival_event_viol |= event_bad
                if "time_min_inclusive" in constraint and constraint["time_min_inclusive"] is not None:
                    time_bad = time_vals < constraint["time_min_inclusive"]
                else:
                    time_min_exc = constraint.get("time_min_exclusive", 0)
                    time_bad = time_vals <= time_min_exc
                any_viol |= time_bad
                survival_time_viol |= time_bad

    return {
        "overall_violation_rate": float(any_viol.mean()),
        "n_records_total": n,
        "n_records_with_any_violation": int(any_viol.sum()),
        "column_constraints": col_result["per_constraint"],
        "cross_column_constraints": cross_result["per_constraint"],
        "survival_pair_event_violation_rate": float(survival_event_viol.mean()),
        "survival_pair_time_violation_rate": float(survival_time_viol.mean()),
        "n_survival_pair_event_violations": int(survival_event_viol.sum()),
        "n_survival_pair_time_violations": int(survival_time_viol.sum()),
    }
