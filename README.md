# PrivBayes experiments

Run DP synthetic data experiments (CRN, dpmm, SynthCity) over an ε-sweep and produce figures and tables.

---

## Quick start: your schema + your dataset

If you **clone the repo** and only want to **provide a schema and a dataset**, use this workflow.

### 1. Prerequisites

- **Python 3.10+**
- The three implementations available (see [SETUP.md](SETUP.md) if `dpmm` / `synthcity_standalone` are not in the repo)

### 2. One-time setup

From the repository root:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If the repo does not include the dpmm or SynthCity packages, follow [SETUP.md](SETUP.md) to add them.

### 3. Put your inputs in place

- **Schema:** a JSON file in the [schema-generator format](schema_validator.py) (e.g. `schemas/my_schema.json`).
  - Must include: `column_types`, `public_bounds` (numeric), `public_categories` (categorical), `target_spec`, and for survival: `target_spec.kind = "survival_pair"` with `targets`, `primary_target`, and a `cross_column_constraint` of type `survival_pair`.
  - Optional: `sensitive_attributes` (see below).
  - The field `dataset` in the schema is used as the results subfolder name; if missing, the schema filename stem is used.
- **Data:** a CSV with the same columns as in the schema (e.g. `data/my_data.csv`).

**What is `sensitive_attributes`?**  
If you open a schema (e.g. `schemas/lung_schema.json`) you’ll see an optional field `"sensitive_attributes": ["status"]`. This tells the pipeline **which column to use for the attribute-inference privacy metric**: we train a classifier on synthetic data to predict that column from the rest, then measure AUC on real data. High AUC means the synthetic data preserves the relationship and inference risk is higher. You don’t have to set it: if `sensitive_attributes` is omitted, the pipeline uses `target_spec.primary_target` (e.g. the event column in survival schemas). For the example lung schema, `"status"` is the binary event (e.g. 0 = censored, 1 = event), so it’s the natural choice for this metric.

**If you don’t have a schema yet**, you can generate one from your CSV using the schema-generator (see [schema-generator/README.md](schema-generator/README.md)) or the project’s `schema_generator.py`, then validate:

```bash
python schema_validator.py schemas/my_schema.json data/my_data.csv
```

### 4. Run the full pipeline

Single command from repo root (activate the venv first if you use one):

```bash
python run_full_pipeline.py --schema schemas/my_schema.json --data data/my_data.csv
```

Alternatively, using the shell script (same inputs):

```bash
./run_full_pipeline.sh schemas/my_schema.json data/my_data.csv
```

This will:

1. Run an ε-sweep (default: ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}, 5 seeds, 3 implementations).
2. Generate all figures and tables from the sweep results.

**Output locations:**

| Output | Path |
|--------|------|
| Synthetic CSVs + result JSONs | `results/eps_sweep/<dataset>/` |
| Figures (PDF) | `outputs/figures/` |
| Tables (LaTeX) | `outputs/tables/` |

`<dataset>` is taken from the schema’s `"dataset"` field, or from the schema filename if `dataset` is missing.

### 5. Optional flags

- **Regenerate only figures and tables** (no new sweep; use existing result JSONs):
  ```bash
  python run_full_pipeline.py --schema schemas/my_schema.json --data data/my_data.csv \
    --skip-sweep --results-dir results/eps_sweep/<dataset>
  ```
- **Recompute metrics** from existing CSVs (e.g. after adding a new metric), then figures and tables:
  ```bash
  python run_full_pipeline.py --schema schemas/my_schema.json --data data/my_data.csv \
    --skip-sweep --refresh-metrics --results-dir results/eps_sweep/<dataset>
  ```
- **Custom results directory** when it doesn’t match the schema’s `dataset` name:
  ```bash
  python run_full_pipeline.py --schema schemas/my_schema.json --data data/my_data.csv \
    --results-dir results/eps_sweep/my_custom_name
  ```

### 6. Adapter ablation metrics

To compare **raw** vs **schema-adapted** runs for SynthCity and DPMM (schema interpretation, output structure, and benchmark metrics), run:

```bash
python experiments/run_adapter_ablation_example.py --schema schemas/my_schema.json --data data/my_data.csv
```

- **Multiple runs with confidence intervals:** use `--n-runs` (e.g. `--n-runs 5`). The script then writes `adapter_ablation_summary.csv` with mean, std, SE, and 95% CI per benchmark metric.
- **Optional:** `--wrong-schema` to include a wrong-schema condition; `--out-dir` to change the output directory (default: `outputs/ablation_metrics`).

**Outputs** (in `outputs/ablation_metrics/` by default):

| File | Description |
|------|-------------|
| `table_schema_interpretation.csv` | Fit-time dtype mismatches (columns, mismatch counts/rates) |
| `table_output_structure.csv` | Output diagnostics (invalid rates, out-of-bounds) |
| `table_benchmark.csv` | Benchmark metrics (utility, privacy, survival, constraints); means when `--n-runs` > 1 |
| `adapter_ablation_all_metrics.csv` | Flat CSV of all metrics (one row per run × implementation × condition) |
| `adapter_ablation_summary.csv` | Only when `--n-runs` > 1: mean, std, SE, 95% CI per benchmark metric |

---

## Summary of inputs and outputs

| You provide | Where | Used for |
|-------------|--------|----------|
| Schema JSON | `--schema` (e.g. `schemas/my_schema.json`) | Column types, bounds, targets, sensitive attribute, dataset name |
| Dataset CSV | `--data` (e.g. `data/my_data.csv`) | Training data for synthesis and metric evaluation |

The pipeline produces synthetic data, metrics (utility, survival, privacy, compliance), figures, and tables without further input.

---

## More detail

- **Full setup** (including dpmm, SynthCity, schema-generator): [SETUP.md](SETUP.md)
- **Figures and tables**: [outputs/README_report.md](outputs/README_report.md)
- **Schema validation**: `python schema_validator.py --help`
