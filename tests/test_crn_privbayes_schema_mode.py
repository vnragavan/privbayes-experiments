import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Ensure project root is on sys.path so we can import implementations.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from implementations.crn_privbayes import PrivBayesSynthesizerEnhanced  # noqa: E402


def _make_toy_schema(n_records: int | None = 10, with_public_cats: bool = True) -> dict:
    """Minimal schema dict for tests."""
    schema = {
        "dataset": {"n_records": n_records} if n_records is not None else {},
        "public_bounds": {
            "x": {"min": 0, "max": 10},
        },
        "public_categories": {},
        "column_types": {
            "x": "integer",
            "y": "categorical",
        },
        "constraints": {
            "column_constraints": {},
            "cross_column_constraints": [],
            "row_group_constraints": [],
        },
    }
    if with_public_cats:
        schema["public_categories"]["y"] = ["a", "b"]
    return schema


def _make_toy_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(n, dtype=int),
            "y": (["a", "b"] * ((n + 1) // 2))[:n],
        }
    )


def test_schema_mode_enforces_n_records_match():
    schema = _make_toy_schema(n_records=5)
    df_ok = _make_toy_df(5)
    df_bad = _make_toy_df(6)

    m = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)

    # Matching length should work
    m.fit(df_ok)
    rep = m.privacy_report()
    assert rep["schema"]["n_declared"] == 5
    assert rep["schema"]["n_fit"] == 5
    assert rep["schema"]["n_source"] == "schema"
    assert rep["schema"]["n_match"] is True

    # Mismatched length should NOT raise now that we allow at least 1-row slack.
    # The reported n_fit remains the schema-declared n_records; equality check
    # is performed against that declared value, not the observed len(df).
    m.fit(df_bad)
    rep2 = m.privacy_report()
    assert rep2["schema"]["n_declared"] == 5
    assert rep2["schema"]["n_fit"] == 5  # still reports declared n_records
    assert rep2["schema"]["n_match"] is True


def test_schema_mode_requires_n_records():
    # No n_records in schema when require_public=True should hard-fail
    schema = _make_toy_schema(n_records=None)
    df = _make_toy_df(5)
    m = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    with pytest.raises(ValueError, match="Schema must declare n_records"):
        m.fit(df)


def test_original_data_bounds_must_match_schema_in_schema_mode():
    schema = _make_toy_schema(n_records=5)
    df = _make_toy_df(5)

    # Exact match to public_bounds should be allowed
    odb_ok = {"x": [0, 10]}
    m_ok = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        schema=schema,
        original_data_bounds=odb_ok,
    )
    m_ok.fit(df)

    # Mismatch in bounds should raise
    odb_bad = {"x": [0, 11]}
    m_bad = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        schema=schema,
        original_data_bounds=odb_bad,
    )
    with pytest.raises(ValueError, match="original_data_bounds may encode private min/max"):
        m_bad.fit(df)

    # Column not present in public_bounds should also raise
    odb_extra = {"z": [0, 1]}
    m_extra = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        schema=schema,
        original_data_bounds=odb_extra,
    )
    with pytest.raises(ValueError, match="not present in schema public_bounds"):
        m_extra.fit(df)


def test_schema_public_categories_prevent_adding_unk():
    schema = _make_toy_schema(n_records=4, with_public_cats=True)
    df = _make_toy_df(4)

    m = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    m.fit(df)

    meta_y = m._meta["y"]
    # Because schema provided public_categories for y, we should not prepend __UNK__
    assert "__UNK__" not in meta_y.cats
    assert set(meta_y.cats) == {"a", "b"}


def test_dp_discovered_categories_do_add_unk_non_schema_mode():
    # When not in schema-authoritative mode (no schema / require_public=False),
    # DP discovery is allowed and should introduce an __UNK__ bucket.
    df = _make_toy_df(4)

    m = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        # No schema => require_public remains False
        eps_disc=0.1,
        strict_dp=False,
    )
    m.fit(df)

    meta_y = m._meta["y"]
    assert "__UNK__" in meta_y.cats


def test_privacy_report_schema_authoritative_and_budget():
    schema = _make_toy_schema(n_records=5)
    df = _make_toy_df(5)

    m = PrivBayesSynthesizerEnhanced(epsilon=2.0, schema=schema, eps_disc=0.5)
    m.fit(df)
    rep = m.privacy_report()

    assert rep["schema_authoritative"] is True
    # In schema mode, metadata budget is disabled
    assert rep["eps_disc_configured"] == 0.0  # forced in schema mode
    assert rep["eps_disc_effective"] == 0.0
    assert rep["epsilon_total_configured"] == pytest.approx(2.0)
    assert rep["epsilon_total_actual"] == pytest.approx(2.0)


def test_fit_does_not_mutate_caller_dataframe():
    """Datetime coercions inside _build_meta should not modify the caller's df."""
    import datetime as dt

    n = 3
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "time": [dt.datetime(2023, 1, 1) + dt.timedelta(days=i) for i in range(n)],
        }
    )
    df_orig = df.copy(deep=True)

    # Run in non-schema mode so only the defensive copy behavior is exercised.
    m = PrivBayesSynthesizerEnhanced(epsilon=1.0)
    m.fit(df)

    # Caller dataframe should remain unchanged (time column still datetime64[ns])
    assert df.equals(df_orig)
    assert pd.api.types.is_datetime64_any_dtype(df["time"])


def test_n_records_tolerance_boundary():
    schema = _make_toy_schema(n_records=1000)

    # Within tolerance: exactly 0.1% relative error (allowed since rel_err <= tol)
    df_ok = _make_toy_df(999)  # rel_err = 1/1000
    m_ok = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    m_ok.fit(df_ok)

    # Just above tolerance: 0.2% relative error
    df_bad = _make_toy_df(998)  # rel_err = 2/1000
    m_bad = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    with pytest.raises(ValueError, match="n_records does not match"):
        m_bad.fit(df_bad)


def test_schema_column_type_validation():
    schema_good = _make_toy_schema(n_records=5)
    # Should not raise
    PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema_good)

    schema_bad = _make_toy_schema(n_records=5)
    schema_bad["column_types"]["x"] = "continous"  # typo
    with pytest.raises(ValueError, match="Unknown schema column type 'continous'"):
        PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema_bad)


# ---- Schema numeric validation (vectorized _validate_schema_numeric_column) ----


def _schema_numeric(n_records: int, col: str, stype: str) -> dict:
    """Minimal schema with a single numeric column and optional categorical for variety."""
    schema = {
        "dataset": {"n_records": n_records},
        "public_bounds": {col: {"min": 0, "max": 10}},
        "public_categories": {},
        "column_types": {col: stype},
        "constraints": {"column_constraints": {}, "cross_column_constraints": [], "row_group_constraints": []},
    }
    return schema


def test_validate_schema_numeric_sentinel_strings_treated_as_missing():
    """Sentinel strings (na, n/a, null, etc.) in an integer column should be treated as missing and not raise."""
    schema = _schema_numeric(5, "a", "integer")
    df = pd.DataFrame({"a": [1, "na", 3, "n/a", 5]})

    m = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    m.fit(df)  # should not raise


def test_validate_schema_numeric_garbage_token_raises():
    """Non-numeric string in an integer column should raise."""
    schema = _schema_numeric(3, "a", "integer")
    df = pd.DataFrame({"a": [1, "abc", 3]})

    m = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    # By default we do not reveal row indices or raw tokens; only the fact that
    # non-numeric values exist is reported.
    with pytest.raises(ValueError, match="contains 1 non-numeric value"):
        m.fit(df)


def test_validate_schema_numeric_binary_violation_raises():
    """Binary column with values other than 0 and 1 should raise."""
    schema = _schema_numeric(3, "b", "binary")
    schema["public_bounds"]["b"] = {"min": 0, "max": 1}
    df = pd.DataFrame({"b": [0, 1, 2]})

    m = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    with pytest.raises(ValueError, match="declared binary but contains values other than 0 and 1"):
        m.fit(df)


def test_validate_schema_numeric_float_in_integer_column_raises():
    """Integer/ordinal column with non-integer numeric (e.g. 1.5) should raise."""
    schema = _schema_numeric(3, "a", "integer")
    df = pd.DataFrame({"a": [1, 2, 1.5]})

    m = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    with pytest.raises(ValueError, match="contains non-integer numeric values"):
        m.fit(df)


def test_validate_schema_numeric_continuous_inf_raises():
    """Continuous column with inf should raise."""
    schema = _schema_numeric(3, "a", "continuous")
    df = pd.DataFrame({"a": [1.0, 2.0, float("inf")]})

    m = PrivBayesSynthesizerEnhanced(epsilon=1.0, schema=schema)
    with pytest.raises(ValueError, match="non-finite values \\(inf\\)"):
        m.fit(df)


def test_validate_schema_numeric_reveal_row_index_off():
    """When reveal_row_index_in_errors=False, error message should not contain row index."""
    schema = _schema_numeric(3, "a", "integer")
    df = pd.DataFrame({"a": [1, "garbage", 3]})

    m = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        schema=schema,
        reveal_row_index_in_errors=False,
    )
    with pytest.raises(ValueError) as exc_info:
        m.fit(df)
    msg = str(exc_info.value)
    assert "at row " not in msg  # should not reveal row index
    assert "non-numeric" in msg.lower()


