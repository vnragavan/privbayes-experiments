# Review: implementations/crn_privbayes.py

## Summary

The implementation is **correct, consistent, and coherent** with a few minor notes and one bug fix applied.

---

## 1. Correctness

### Fixed during review
- **Empty `public_categories` in inferred path**: When `pub = []` we had `cats = [unk]` then `cats = list(pub) = []`, yielding empty categories and broken discretization. Fixed by only doing `cats = list(pub)` when `len(pub) > 0` (so we only normalize the "pub was only __UNK__" case).

### Verified
- **Schema path**: `_build_meta_from_schema()` uses only schema (no dtype/value inspection). Every df column must be in `_schema_col_types`. Numeric types use `_NUMERIC_STYPES` / `_INT_STYPES`; binary gets fixed [0,1]; categorical requires `public_categories`; labels require `public_categories` and get closed vocab without __UNK__.
- **Inferred path**: DP bounds/categorical discovery, optional partial schema types, and **always** injecting __UNK__ for non-label categoricals when `public_categories` is provided, so unseen values map to __UNK__ index (no silent index-0 assumption).
- **Discretize**: Schema-mode validation (non-numeric → error, binary 0/1, int types integer-only, continuous no inf); binary strict check; categorical unseen → __UNK__ index or raise. Non-schema: unseen → __UNK__ if in cats, else strict_dp → raise, else legacy index 0.
- **Fit**: Schema required when `require_public`; `df.copy()` before meta so caller is safe; n_records tolerance `max(0.001*n, 1)`; `_n_observed` stored; `original_data_bounds` must match schema bounds when provided.
- **Privacy report**: `n_observed`, `n_declared`, `n_fit`, `n_match` (using same tolerance), `n_source`; `schema_authoritative`, `eps_disc_effective`; no leakage of observed n in schema mode.

---

## 2. Consistency

- **Schema type table** (class comment): Matches implementation. `continuous`/`integer`/`binary`/`ordinal`/`categorical`/`datetime`/`timedelta`; numeric types require bounds (binary fixed); categorical requires `public_categories`; output types (float, int64, 0/1, string) match decode behaviour.
- **Constants**: `_NUMERIC_STYPES`, `_INT_STYPES`, `_KNOWN_STYPES` used in `load_schema()` validation, `_build_meta_from_schema()`, and `_discretize()` schema validation. Single source of truth.
- **Two paths**: `require_public` → `_build_meta_from_schema()` (no private-data scanning in meta); else `_build_meta_inferred()` (legacy inference + DP). No mixing of schema vs inferred rules inside one path.
- **__UNK__**: Schema path never adds __UNK__ to schema-declared categoricals (closed vocab). Inferred path always injects __UNK__ for non-label categoricals when using `public_categories`, so unseen handling is consistent.

---

## 3. Coherence

- **Docstrings**: `_build_meta()` documents both paths; `_build_meta_from_schema()` says "no dtype/value inspection"; `_build_meta_inferred()` says "legacy path". `_discretize()` and fit/report behaviour align with these.
- **Naming**: `_n_declared` (schema), `_n_observed` (actual rows), `_n_fit` (used for sampling), `_n_source` ("schema" | "observed"); `_schema_public_category_cols` (columns with schema-provided categories). Clear and used consistently.
- **Privacy**: Schema mode: eps_disc forced to 0, budget recomputed, no observed n in report; `original_data_bounds` must match schema or be empty; dataset size from schema with tolerance. Coherent "schema-authoritative = no private metadata" story.

---

## 4. Minor notes (no code change)

- **`target_spec.primary_target`**: Code sets `_event_col = primary`, `_duration_col = other`. In `lung_schema_real.json`, `primary_target` is `"os_42"` (time) and the other target is `"os_42_status"` (event). So `_event_col`/`_duration_col` names may be reversed relative to that schema's intent. Behaviour is consistent; naming convention is schema-dependent.
- **`public_binary_numeric`**: Only populated for binary columns (schema path) or when inferred as bool/binary (inferred path). Callers must not assume a key for every numeric column.
- **Datetime in-place**: `_build_meta_from_schema()` and `_build_meta_inferred()` still convert datetime/timedelta columns in-place on the passed df; `fit()` passes `df.copy()`, so the caller's dataframe is not mutated. Comment in code documents this.

---

## 5. Test status

- `pytest tests/test_crn_privbayes_schema_mode.py` — all 9 tests pass after the empty-`pub` fix.

---

**Conclusion**: The module is correct, consistent, and coherent. One correctness fix (empty `pub` in inferred path) was applied; the rest of the design and behaviour align with the intended schema vs inferred semantics and privacy guarantees.
