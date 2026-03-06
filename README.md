# PrivBayes experiments

Run DP synthetic data experiments (CRN, dpmm, SynthCity) over an ε-sweep and produce a compliance/utility report.

---

## Quick start: your schema + your dataset

If you **clone the repo** and only want to **provide a schema and a dataset**, use this workflow.

### 1. Prerequisites

- **Python 3.10+**
- **LaTeX** (e.g. `pdflatex`) to build the report PDF
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
  - Optional: `sensitive_attributes` (list of column names) for the attribute-inference metric; if omitted, `target_spec.primary_target` is used.
  - The field `dataset` in the schema is used as the results subfolder name; if missing, the schema filename stem is used.
- **Data:** a CSV with the same columns as in the schema (e.g. `data/my_data.csv`).

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
3. Compile `outputs/report.pdf`.

**Output locations:**

| Output | Path |
|--------|------|
| Synthetic CSVs + result JSONs | `results/eps_sweep/<dataset>/` |
| Figures (PDF) | `outputs/figures/` |
| Tables (LaTeX) | `outputs/tables/` |
| Report PDF | `outputs/report.pdf` |

`<dataset>` is taken from the schema’s `"dataset"` field, or from the schema filename if `dataset` is missing.

### 5. Optional flags

- **Regenerate only figures/tables/report** (no new sweep; use existing result JSONs):
  ```bash
  python run_full_pipeline.py --schema schemas/my_schema.json --data data/my_data.csv \
    --skip-sweep --results-dir results/eps_sweep/<dataset>
  ```
- **Recompute metrics** from existing CSVs (e.g. after adding a new metric), then figures + report:
  ```bash
  python run_full_pipeline.py --schema schemas/my_schema.json --data data/my_data.csv \
    --skip-sweep --refresh-metrics --results-dir results/eps_sweep/<dataset>
  ```
- **Custom results directory** when it doesn’t match the schema’s `dataset` name:
  ```bash
  python run_full_pipeline.py --schema schemas/my_schema.json --data data/my_data.csv \
    --results-dir results/eps_sweep/my_custom_name
  ```

---

## Summary of inputs and outputs

| You provide | Where | Used for |
|-------------|--------|----------|
| Schema JSON | `--schema` (e.g. `schemas/my_schema.json`) | Column types, bounds, targets, sensitive attribute, dataset name |
| Dataset CSV | `--data` (e.g. `data/my_data.csv`) | Training data for synthesis and metric evaluation |

The pipeline produces synthetic data, metrics (utility, survival, privacy, compliance), figures, tables, and the report PDF without further input.

---

## More detail

- **Full setup** (including dpmm, SynthCity, schema-generator): [SETUP.md](SETUP.md)
- **Report structure and RQs**: [outputs/README_report.md](outputs/README_report.md)
- **Schema validation**: `python schema_validator.py --help`
- **SynthCity implementation gap** (event column collapse / attribute inference): [docs/SYNTHCITY_IMPLEMENTATION_GAP.md](docs/SYNTHCITY_IMPLEMENTATION_GAP.md)
