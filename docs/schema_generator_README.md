# schema_generator.py

**Version 1.6** — DP-aware schema generator for PrivBayes synthesis experiments.

Generates a `schema.json` from a CSV file that drives all downstream components: `CRNPrivBayes`, the DPMM adapter, the SynthCity wrapper, the metrics pipeline, and `run_experiment.py`. It is the single authoritative source of truth about your dataset's domain, types, sensitivity bounds, and PrivBayes algorithm parameters.

---

## Table of Contents

1. [Design philosophy](#1-design-philosophy)
2. [Two-phase pipeline and ε budget](#2-two-phase-pipeline-and-ε-budget)
3. [Schema modes: public vs private](#3-schema-modes-public-vs-private)
4. [Running the schema generator](#4-running-the-schema-generator)
5. [CLI reference](#5-cli-reference)
6. [Schema JSON field reference](#6-schema-json-field-reference)
7. [PrivBayes extensions block](#7-privbayes-extensions-block)
8. [DP coverage and NPA warnings](#8-dp-coverage-and-npa-warnings)
9. [Using the schema in an experiment](#9-using-the-schema-in-an-experiment)
10. [Version history](#10-version-history)

---

## 1. Design philosophy

The schema generator treats the schema itself as a DP interface between your raw private data and the synthesis pipeline. Formally, the schema is:

```
S = (A, T, Ω, Γ, Π, P)
```

| Symbol | Meaning | JSON field |
|---|---|---|
| A | Attribute set | `column_types.keys()` |
| T | Type map | `column_types` |
| Ω | Domain map (bounds + categories) | `public_bounds`, `public_categories` |
| Γ | Sensitivity bounds per column | `sensitivity_bounds` |
| Π | Default mechanism hints per column | `mechanism_hints` |
| P | Domain provenance (how each field was derived) | `domain_provenance` |

The schema is **generated once on the full dataset** before any train/test split. `run_experiment.py` then patches only `n_records` in memory to match the training split size — the on-disk schema file is never modified during experiments.

---

## 2. Two-phase pipeline and ε budget

```
Phase 1:  D ──[M_schema, ε₁]──→ S̃        (schema_generator.py)
Phase 2:  D ──[M_synth,  ε₂]──→ D̃(S̃)    (run_experiment.py)

Total:    ε_total = ε₁ + ε₂
```

In **public mode**, ε₁ = 0 and the full `--epsilon` passed to `run_experiment.py` is available for synthesis.

In **private mode**, ε₁ = `--schema-epsilon` is spent once when you run the schema generator. Every subsequent experiment run at different ε₂ values reuses the same schema — ε₁ is a sunk cost, not re-spent. When reporting results, always state:

> ε denotes the synthesis budget (ε₂). Total DP cost is ε_total = ε_schema + ε₂, where ε_schema = 0.5 was spent once during schema generation.

---

## 3. Schema modes: public vs private

### Public mode (default)

Schema bounds and categories are inferred from the full dataset without any DP noise. Use this when the domain metadata (age range, ECOG levels, sex coding) is already public knowledge from the study protocol or dataset documentation.

- ε₁ = 0; full ε budget available for synthesis
- `provenance.schema_mode = "public"`
- `provenance.epsilon_spent_schema = 0.0`
- All domain fields have `domain_provenance[col] = "public"`

### Private mode

Numeric bounds are released under Laplace noise with `--schema-epsilon` as the total ε₁ budget. Use this when the column ranges themselves are sensitive (e.g., rare disease datasets where the mere existence of a value range reveals patient cohort characteristics).

- ε₁ = `--schema-epsilon`; synthesis budget = ε₂ = ε_total − ε₁
- `provenance.schema_mode = "private"`
- `provenance.epsilon_spent_schema` records the actual ε₁ spent
- Numeric columns get `domain_provenance[col] = "private(ε)"`
- Categorical/ordinal domains inferred via `--infer-categories` are **not** DP-covered — they are marked `non_private_auxiliary` with an explicit `[NPA-WARN]` (see [section 8](#8-dp-coverage-and-npa-warnings))

**Budget allocation in private mode:**

```
epsilon_per_col = schema_epsilon / n_private_numeric_cols
  min query:  epsilon_per_col / 2
  max query:  epsilon_per_col / 2
  tau (survival time cap): subsumed by max query — no additional cost

n_records:        Laplace(sensitivity=1, ε=schema_epsilon)   [--dp-n-records]
missing_rates:    Laplace(sensitivity=1/n) per column,       [--dp-missing-rates]
                  parallel composition across disjoint columns

PrivBayes strategy (Bowley skewness):
                  Laplace-noised quartiles, pb-strategy-epsilon-frac × schema_epsilon
                  sequential composition across numeric columns
```

Sensitivity model uses **local (empirical) sensitivity** = `max - min`. Noise is one-sided and widens bounds, never narrows them, so the calibration contract is never violated.

---

## 4. Running the schema generator

### Minimal public mode

```bash
python schema_generator.py \
  --data data/your_dataset.csv \
  --out schemas/your_schema.json \
  --schema-mode public \
  --survival-event-col status \
  --survival-time-col days_to_event \
  --infer-categories \
  --emit-privbayes-extensions
```

### Minimal private mode

```bash
python schema_generator.py \
  --data data/your_dataset.csv \
  --out schemas/your_schema_private.json \
  --schema-mode private \
  --schema-epsilon 0.5 \
  --dp-n-records \
  --dp-missing-rates \
  --survival-event-col status \
  --survival-time-col days_to_event \
  --infer-categories \
  --emit-privbayes-extensions
```

`--epsilon-total` is optional. If omitted, the schema is still fully valid and `provenance.epsilon_remaining_synthesis` will be `null`. Omit it when running a sweep across multiple ε₂ values (since no single value of ε_total applies to the whole sweep).

### With explicit column type overrides

Use `--column-types` when you have a JSON file mapping column names to types, or when you want to declare publicly-known domains to avoid NPA warnings:

```bash
python schema_generator.py \
  --data data/your_dataset.csv \
  --out schemas/your_schema.json \
  --schema-mode private \
  --schema-epsilon 0.5 \
  --column-types config/column_types.json \
  --survival-event-col status \
  --survival-time-col days_to_event \
  --emit-privbayes-extensions
```

Where `config/column_types.json` contains:
```json
{
  "sex":      "categorical",
  "ph_ecog":  "ordinal",
  "age":      "integer",
  "status":   "binary",
  "time":     "integer"
}
```

### Auto-detect survival columns

```bash
python schema_generator.py \
  --data data/your_dataset.csv \
  --out schemas/your_schema.json \
  --schema-mode public \
  --target-kind survival_pair \
  --infer-categories \
  --emit-privbayes-extensions
```

---

## 5. CLI reference

### Required

| Argument | Description |
|---|---|
| `--data PATH` | Path to input CSV file |
| `--out PATH` | Output path for schema JSON |

### Dataset and target

| Argument | Default | Description |
|---|---|---|
| `--dataset-name STR` | CSV stem | Name stored in `schema["dataset"]`; used as subdirectory name in `run_full_pipeline.py` |
| `--survival-event-col STR` | None | Column name of the binary survival event indicator (0/1) |
| `--survival-time-col STR` | None | Column name of the survival time (positive integer) |
| `--target-col STR` | None | Single target column for classification tasks |
| `--target-kind STR` | None | `survival_pair` to auto-detect event+time columns |
| `--sensitive-attributes STR` | None | Comma-separated list of sensitive columns for attribute inference metrics |
| `--delimiter STR` | auto | CSV delimiter; auto-detected if omitted |

### Column type inference

| Argument | Default | Description |
|---|---|---|
| `--column-types PATH` | None | JSON file mapping column names to types (`binary`, `categorical`, `ordinal`, `integer`, `continuous`) |
| `--infer-categories` | False | Infer categorical/ordinal levels from data. **Levels are marked `non_private_auxiliary` in private mode** |
| `--max-categories INT` | 200 | Maximum unique values for a column to be treated as categorical |
| `--max-integer-levels INT` | 20 | Columns with ≤ this many unique integer values are treated as ordinal |
| `--infer-binary-domain` | False | Infer binary columns (exactly 2 unique values) |
| `--infer-datetimes` | False | Detect and parse datetime columns |
| `--datetime-min-parse-frac FLOAT` | 0.95 | Fraction of rows that must parse as datetime for detection |
| `--datetime-output-format STR` | preserve | Output format for datetime columns |

### Bounds padding

| Argument | Default | Description |
|---|---|---|
| `--pad-frac FLOAT` | 0.0 | Fractional padding applied to both ends of numeric bounds. Omit unless needed; 0.0 is a no-op |
| `--pad-frac-integer FLOAT` | None | Override `--pad-frac` for integer columns only |
| `--pad-frac-continuous FLOAT` | None | Override `--pad-frac` for continuous columns only |

### Privacy mode

| Argument | Default | Description |
|---|---|---|
| `--schema-mode STR` | `public` | `public` or `private` |
| `--schema-epsilon FLOAT` | None | ε₁ budget for schema generation. Required when `--schema-mode private` |
| `--epsilon-total FLOAT` | None | Optional total pipeline ε. When provided, `epsilon_remaining_synthesis` is computed and stored in provenance. Omit for sweeps |
| `--dp-n-records` | False | In private mode: release `n_records` under `Laplace(1, ε)`. If omitted, `n_records` is marked `non_private_auxiliary` |
| `--dp-missing-rates` | False | In private mode: release per-column missing rates under Laplace via parallel composition. If omitted, missing rates are marked `non_private_auxiliary` |

### PrivBayes extensions

These flags are only relevant when `--emit-privbayes-extensions` is set.

| Argument | Default | Description |
|---|---|---|
| `--emit-privbayes-extensions` | False | Emit `schema["extensions"]["privbayes"]` block with all CRNPrivBayes algorithm parameters |
| `--pb-max-parents INT` | None (auto) | Override `max_parents`. Default: sensitivity crossover — smallest k such that `max_bins_total^k ≥ n` |
| `--pb-max-numeric-bins INT` | None (auto) | Hard cap on `n_bins` for integer/continuous columns. Default: `max(Sturges(n), ceil(n^(1/3)), 5)` capped at 20. Override only for CPT memory budgets |
| `--pb-default-numeric-bins INT` | 8 | Fallback bin count used only when Sturges cannot be computed |
| `--pb-time-bins INT` | 10 | Fixed bin count for the survival time column |
| `--pb-default-strategy STR` | `equal_width` | Default discretization strategy when skewness cannot be computed |
| `--pb-time-strategy STR` | `quantile` | Discretization strategy for the survival time column |
| `--pb-dirichlet-alpha FLOAT` | None | Override CPT Dirichlet smoothing globally. Default: Perks prior `1/K` per column |
| `--pb-strategy-epsilon-frac FLOAT` | 0.05 | Fraction of `schema_epsilon` spent on DP strategy selection (Bowley skewness) in private mode |
| `--pb-no-parent-constraints` | False | Do not emit `allowed_parents` / `forbidden_parents` for survival columns |
| `--pb-no-partial-order` | False | Do not emit `partial_order` column ordering hint |

### Miscellaneous

| Argument | Default | Description |
|---|---|---|
| `--n-records INT` | None | Override n_records (use only when you have a pre-split dataset) |
| `--redact-source-path` | False | Replace the source CSV path in provenance with a placeholder |
| `--constraints-file PATH` | None | JSON file with additional user-defined constraints to merge |
| `--target-spec-file PATH` | None | JSON file with a full `target_spec` override |
| `--no-publish-label-domain` | False | Do not include label domain in the schema |
| `--guid-min-match-frac FLOAT` | 0.95 | Fraction of values that must look like GUIDs for a column to be auto-excluded |

---

## 6. Schema JSON field reference

Top-level structure of the output JSON:

```json
{
  "schema_version": "1.6",
  "dataset": "ncctg_lung",
  "dataset_info": {
    "n_records": 228,
    "n_records_note": "..."
  },
  "column_types": { ... },
  "public_bounds": { ... },
  "public_categories": { ... },
  "missing_value_rates": { ... },
  "sensitivity_bounds": { ... },
  "mechanism_hints": { ... },
  "domain_provenance": { ... },
  "target_col": "status",
  "label_domain": null,
  "target_spec": { ... },
  "sensitive_attributes": [...],
  "constraints": { ... },
  "extensions": {
    "privbayes": { ... }
  },
  "provenance": { ... }
}
```

### `column_types`

Maps each column name to one of: `binary`, `categorical`, `ordinal`, `integer`, `continuous`, `datetime`, `guid`.

```json
"column_types": {
  "status":    "binary",
  "sex":       "categorical",
  "ph.ecog":   "ordinal",
  "age":       "integer",
  "meal.cal":  "integer",
  "time":      "integer"
}
```

### `public_bounds`

Per-column domain metadata for numeric columns. The `n_bins` field here is `min(n_unique, 100)` — raw domain cardinality, **not** the PrivBayes CPT bin count (which lives under `extensions.privbayes.discretization.per_column`).

```json
"public_bounds": {
  "age": {
    "min": 39, "max": 82,
    "n_unique": 42,
    "n_bins": 42
  },
  "meal.cal": {
    "min": 96.0, "max": 2600.0,
    "n_unique": 60,
    "n_bins": 60
  }
}
```

### `public_categories`

Per-column allowed value lists for categorical and ordinal columns.

```json
"public_categories": {
  "sex":     [1, 2],
  "ph.ecog": [0, 1, 2, 3]
}
```

### `missing_value_rates`

Observed (or DP-released) fraction of missing values per column. Only columns with actual missingness appear here.

```json
"missing_value_rates": {
  "meal.cal": 0.2061,
  "wt.loss":  0.0614,
  "__note__": "non_private_auxiliary [NPA-WARN]: ..."
}
```

### `sensitivity_bounds`  (Γ)

L1 sensitivity per column used for downstream mechanism calibration. Advisory — synthesisers may compute their own sensitivity, but this provides an auditable reference.

```json
"sensitivity_bounds": {
  "age":      43.0,
  "status":   1.0,
  "sex":      2.0,
  "ph.ecog":  3.0,
  "time":     1020.0
}
```

Derivation by type:

- `continuous` / `integer`: `max - min`
- `binary`: `1.0`
- `ordinal`: `max - min` (numeric range)
- `categorical`: `2.0` (L1 sensitivity of the full normalised histogram vector)

### `mechanism_hints`  (Π)

Advisory default mechanism per column. Not normative — CRNPrivBayes uses Laplace after discretization regardless of this hint.

```json
"mechanism_hints": {
  "age":      "laplace",
  "status":   "exponential",
  "sex":      "exponential",
  "time":     "laplace"
}
```

Assignment by type: `binary`/`categorical` → `exponential`; `ordinal`/`integer` → `laplace`; `continuous` → `gaussian` (use `laplace` if pure ε-DP is required); survival event → `exponential`; survival time → `laplace`.

### `domain_provenance`  (P)

Per-column record of how the domain was derived.

```json
"domain_provenance": {
  "age":     "private(ε=0.0417)",
  "sex":     "non_private_auxiliary [NPA-WARN]: categorical domain inferred non-privately",
  "ph.ecog": "non_private_auxiliary [NPA-WARN]: ordinal levels inferred non-privately",
  "time":    "private(ε=0.0417)"
}
```

In public mode all entries read `"public"`.

### `target_spec`

Describes the prediction or survival target. For survival analysis:

```json
"target_spec": {
  "kind":           "survival_pair",
  "targets":        ["status", "time"],
  "primary_target": "status",
  "time_col":       "time",
  "event_col":      "status",
  "tau":            1022
}
```

`tau` is the administrative censoring time (maximum observed survival time), used by the metrics pipeline to truncate KM curves at a consistent horizon.

### `constraints`

Auto-generated structural constraints used by the metrics pipeline to validate synthetic data.

```json
"constraints": {
  "survival_pair": {
    "event_col": "status",
    "time_col":  "time",
    "event_values": [0, 1],
    "time_min": 5,
    "time_max": 1022
  }
}
```

### `provenance`

Full audit trail of how the schema was generated.

```json
"provenance": {
  "generated_at_utc":           "2025-03-10T12:00:00+00:00",
  "source_csv":                 "data/lung_clean.csv",
  "schema_mode":                "private",
  "schema_epsilon":             0.5,
  "epsilon_total":              null,
  "epsilon_spent_schema":       0.4821,
  "epsilon_remaining_synthesis": null,
  "composition_note":           "Sequential composition...",
  "inferred_categories":        true,
  "pad_frac":                   0.0,
  "bound_sources":              { "age": "data_range", ... }
}
```

Key fields:

- `epsilon_spent_schema` — actual ε₁ spent (0.0 in public mode)
- `epsilon_remaining_synthesis` — `ε_total - ε_spent` if `--epsilon-total` was passed, else `null`
- `bound_sources` — per-column record of whether bounds came from data or a user override

---

## 7. PrivBayes extensions block

Emitted under `schema["extensions"]["privbayes"]` when `--emit-privbayes-extensions` is set. This block is consumed entirely by `CRNPrivBayes.load_schema()`. The DPMM and SynthCity wrappers read `public_bounds` and `column_types` instead.

```json
"extensions": {
  "privbayes": {
    "max_parents": 3,
    "dirichlet_alpha": 0.1,
    "dirichlet_alpha_source": "perks_prior_median_K=10",
    "preferred_root": "age",
    "partial_order": ["age", "sex", "ph.ecog", "meal.cal", "wt.loss", "time", "status"],
    "forbidden_parents": {
      "time": ["status"]
    },
    "allowed_parents": {
      "time":   ["age", "sex", "ph.ecog", "meal.cal", "wt.loss"],
      "status": ["age", "sex", "ph.ecog", "meal.cal", "wt.loss", "time"]
    },
    "missing_value_handling": {
      "strategy": "nan_bin",
      "columns_affected": {
        "meal.cal": { "nan_bin_index": 9, "missing_rate": 0.2061 },
        "wt.loss":  { "nan_bin_index": 9, "missing_rate": 0.0614 }
      }
    },
    "discretization": {
      "max_numeric_bins": 9,
      "per_column": {
        "age": {
          "strategy":       "equal_width",
          "n_bins":         9,
          "n_bins_total":   9,
          "has_nan_bin":    false,
          "dirichlet_alpha": 0.1111,
          "skewness":       0.123,
          "skewness_source": "data"
        },
        "meal.cal": {
          "strategy":       "quantile",
          "n_bins":         9,
          "n_bins_total":   10,
          "has_nan_bin":    true,
          "nan_bin_index":  9,
          "missing_rate":   0.2061,
          "dirichlet_alpha": 0.1,
          "skewness":       0.451,
          "skewness_source": "data"
        }
      }
    },
    "provenance": {
      "max_parents_source":    "sensitivity_crossover",
      "strategy_epsilon_spent": 0.0237
    }
  }
}
```

### Field-by-field

**Global parameters**

| Field | How derived | Description |
|---|---|---|
| `max_parents` | Smallest k such that `max_bins_total^k ≥ n` (sensitivity crossover) | Maximum parent set size in the Bayesian network. Passed to `CRNPrivBayes` constructor |
| `dirichlet_alpha` | Perks prior `1/K` where K = median `n_bins_total` | Global CPT smoothing fallback; per-column values override this |
| `preferred_root` | First covariate column | Greedy search root hint |
| `partial_order` | `covariates + [time_col, event_col]` | Column ordering for greedy parent search; causes survival columns to be assigned parents only from upstream covariates |
| `forbidden_parents` | Causal constraint: event cannot cause time | CRN blocks these edges in the parent search |
| `allowed_parents` | Explicit sets derived from partial order | CRN restricts parent candidates for survival columns |

**Per-column discretization (`discretization.per_column.<col>`)**

| Field | Description |
|---|---|
| `n_bins` | Sturges-rule bin count: `ceil(log2(n) + 1)`, capped at `max_numeric_bins`. Used for CPT sizing in CRN |
| `n_bins_total` | `n_bins + 1` if `has_nan_bin`, else `n_bins`. Used for sensitivity calculations |
| `strategy` | `equal_width` or `quantile`. Chosen by Bowley skewness threshold `|skew| > 0.2` |
| `has_nan_bin` | `true` if column has any missing values |
| `nan_bin_index` | Index of the NaN bin slot (always = `n_bins`). CRN routes missing values to this bin at encode time and restores `NaN` at decode time |
| `missing_rate` | Observed (or DP-released) fraction of missing values |
| `dirichlet_alpha` | Per-column Perks prior `1/n_bins_total` for CPT smoothing. Overrides the global `dirichlet_alpha` |
| `skewness` | Bowley skewness value used to select the strategy |
| `skewness_source` | `"data"` (public mode), `"dp_laplace"` (private mode), or `"default_no_bounds"` |

**Important distinction:**
`public_bounds[col]["n_bins"]` = `min(n_unique, 100)` is **domain cardinality metadata**.
`extensions.privbayes.discretization.per_column[col]["n_bins"]` is the **CPT bin count** for PrivBayes.
These are different numbers. Always use the extensions field for CPT sizing.

---

## 8. DP coverage and NPA warnings

In private mode, not all schema fields are DP-covered. The following are always marked `non_private_auxiliary` (`[NPA-WARN]`) because they require DPSU (Differentially Private Set Union) for strict DP:

- Categorical domain levels inferred via `--infer-categories`
- Ordinal level lists inferred from low-cardinality integer columns

**What this means:** the category lists (e.g. `sex ∈ {1, 2}`, `ph.ecog ∈ {0, 1, 2, 3}`) are derived by directly reading the raw data without noise. For a fully DP pipeline these must be declared as public overrides via `--column-types`.

**To suppress the warning** for columns whose domain is publicly known, provide them in a `--column-types` JSON file. The generator will then mark them `"public"` in `domain_provenance` instead of `"non_private_auxiliary"`.

---

## 9. Using the schema in an experiment

### Single run

```bash
python run_experiment.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --epsilon 1.0 \
  --seed 0 \
  --implementations crn \
  --output-dir results/test_run
```

`run_experiment.py` performs a 60/20/20 train/test/holdout split internally. It patches `n_records` in memory (the on-disk schema file is never modified). Because `schema["dataset"]` is a string in schema-generator output, the patch replaces it with `{"n_records": n_train}` — this is the key CRN's `load_schema()` reads. `schema["dataset_info"]["n_records"]` is also set as a fallback for other backends that read the legacy key.

When `--schema-mode private` was used to generate the schema, `--epsilon` here is **ε₂ only** (synthesis budget). Total DP cost = `epsilon_spent_schema + epsilon`.

### Full sweep

```bash
python run_full_pipeline.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --output-dir results/eps_sweep \
  --skip-ablation
```

The sweep varies ε₂ across values defined in `run_sweep.py`. Each sweep point reuses the same schema file. ε₁ is paid once and does not vary between sweep points.

### Verifying the schema after generation

```bash
python - <<'EOF'
import json
with open("schemas/your_schema.json") as f:
    s = json.load(f)

print("dataset     :", s.get("dataset"))
d = s.get("dataset_info", {})
print("n_records   :", d.get("n_records"))
print("columns     :", list(s.get("column_types", {}).keys()))
print("target_spec :", s.get("target_spec"))
print("ε spent     :", s["provenance"]["epsilon_spent_schema"])
print("ε remaining :", s["provenance"]["epsilon_remaining_synthesis"])

pb = s.get("extensions", {}).get("privbayes", {})
print("max_parents :", pb.get("max_parents"))
print("missing cols:", list(pb.get("missing_value_handling", {}).get("columns_affected", {}).keys()))
EOF
```

