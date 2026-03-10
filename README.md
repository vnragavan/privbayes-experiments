# PrivBayes Experiments

Run differentially private synthetic data experiments (CRN, DPMM, SynthCity) over an ε-sweep and produce figures and tables for benchmarking.

---

## Table of Contents

1. [Prerequisites and setup](#1-prerequisites-and-setup)
2. [Prepare your dataset](#2-prepare-your-dataset)
3. [Generate a schema](#3-generate-a-schema)
4. [Test run](#4-test-run)
5. [Full experiment](#5-full-experiment)
6. [Adapter ablation](#6-adapter-ablation)
7. [Optional pipeline flags](#7-optional-pipeline-flags)
8. [Outputs reference](#8-outputs-reference)
9. [ε budget accounting](#9-ε-budget-accounting)

---

## 1. Prerequisites and setup

**Requirements:** Python 3.10+, `pdflatex` (for report compilation, optional).

```bash
git clone <repo>
cd privbayes-experiments
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If the repo does not include the `dpmm` or `synthcity_standalone` packages, follow [SETUP.md](SETUP.md) to add them.

---

## 2. Prepare your dataset

Place your CSV under `data/`. Before running anything, drop columns that cannot be synthesised:

- Patient identifiers (MRN, name, date of birth, national ID)
- Free-text columns
- Columns with more than 80% missing values

Identify upfront:
- Your **survival time column** — positive integer, e.g. `time`
- Your **event indicator column** — binary 0/1, e.g. `status`

The full CSV (all rows) is always passed to both the schema generator and the experiment runner. The train/test/holdout split (60/20/20) is performed internally by `run_experiment.py` and is never your responsibility.

---

## 3. Generate a schema

 The schema is generated **once on the full dataset** before any experiment. It is the single authoritative source of column types, domain bounds, PrivBayes algorithm parameters, and DP provenance. See [schema_generator_README.md](docs/schema_generator_README.md) for the full field reference.

### Public mode

Use when column ranges and category levels are already public knowledge (e.g. from a published study protocol). The full ε budget is available for synthesis.

```bash
python schema_generator.py \
  --data data/your_dataset.csv \
  --out schemas/your_schema.json \
  --schema-mode public \
  --survival-event-col status \
  --survival-time-col time \
  --infer-categories \
  --emit-privbayes-extensions
```

### Private mode

Use when the column ranges themselves are sensitive. Spends ε₁ on schema generation; the remaining ε₂ = ε_total − ε₁ is available for synthesis. The schema is generated once — ε₁ is a sunk cost reused across all sweep points.

```bash
python schema_generator.py \
  --data data/your_dataset.csv \
  --out schemas/your_schema.json \
  --schema-mode private \
  --schema-epsilon 0.5 \
  --dp-n-records \
  --dp-missing-rates \
  --survival-event-col status \
  --survival-time-col time \
  --infer-categories \
  --emit-privbayes-extensions
```

`--epsilon-total` is optional and can be omitted when running a sweep (no single ε_total applies across all sweep points).

### Validate the schema

```bash
python schema_validator.py schemas/your_schema.json data/your_dataset.csv
```

---

## 4. Test run

Before committing to a full sweep, run a single implementation at one ε to confirm the pipeline works end-to-end.

```bash
python run_experiment.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --epsilon 1.5 \
  --seed 0 \
  --implementations crn \
  --output-dir results/test_run
```

When using **private mode**, `--epsilon` is ε₂ only (synthesis budget). Total DP cost = ε_schema + ε₂ = 0.5 + 1.5 = **2.0**.

When using **public mode**, `--epsilon` is the full synthesis budget.

Check the printed output for:
- No tracebacks
- `km_l1` is a small positive number (not `?` or `nan`)
- `fit` time is reasonable (CRN ~0.05s, DPMM ~20s, SynthCity ~1.5s on n≈200)

---

## 5. Full experiment

Once the test run is successful, run the full ε-sweep across all implementations:

```bash
python run_full_pipeline.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --output-dir results/eps_sweep \
  --skip-ablation
```

This runs an ε-sweep (default: ε₂ ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}, 5 seeds, 3 implementations), then generates all figures and tables. `--skip-ablation` is recommended until the main sweep results look correct.

Alternatively, using the shell script:

```bash
./run_full_pipeline.sh schemas/your_schema.json data/your_dataset.csv
```

**Output locations:**

| Output | Path |
|--------|------|
| Synthetic CSVs + result JSONs | `results/eps_sweep/<dataset>/` |
| Figures (PDF) | `outputs/figures/` |
| Tables (LaTeX) | `outputs/tables/` |

`<dataset>` is taken from the `"dataset"` field in the schema JSON, or from the schema filename stem if that field is missing.

---

## 6. Adapter ablation

Once the main sweep results look correct, re-run the full pipeline without `--skip-ablation`:

```bash
python run_full_pipeline.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --output-dir results/eps_sweep
```

This runs two additional steps after the main sweep:

**Step 4a — LaTeX tables and ablation figure**
Calls `experiments/generate_adapter_ablation_artifacts.py`. Produces `tab_adapter_*.tex` and `fig_adapter_ablation.pdf` under `outputs/tables/` and `outputs/figures/`.

**Step 4b — Ablation metrics CSV tables**
Calls `experiments/run_adapter_ablation_example.py`. Produces per-condition CSVs under `outputs/ablation_metrics/`.

To add error bars to the ablation figure, use `--ablation-n-runs` with 3 or more runs:

```bash
python run_full_pipeline.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --output-dir results/eps_sweep \
  --ablation-n-runs 3 \
  --ablation-error-bar 95ci
```

`--ablation-error-bar` accepts `se` (±1 standard error, default) or `95ci` (95% confidence interval). Error bars are only visible when `--ablation-n-runs` is 2 or more.

To run the ablation script standalone:

```bash
python experiments/run_adapter_ablation_example.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --n-runs 5 \
  --out-dir outputs/ablation_metrics
```

**Ablation outputs (in `outputs/ablation_metrics/`):**

| File | Description |
|------|-------------|
| `table_schema_interpretation.csv` | Fit-time dtype mismatches (columns, mismatch counts/rates) |
| `table_output_structure.csv` | Output diagnostics (invalid rates, out-of-bounds) |
| `table_benchmark.csv` | Benchmark metrics (utility, privacy, survival, constraints) |
| `adapter_ablation_all_metrics.csv` | Flat CSV of all metrics, one row per run × implementation × condition |
| `adapter_ablation_summary.csv` | Mean, std, SE, 95% CI per benchmark metric (only when `--n-runs` > 1) |

---

## 7. Optional pipeline flags

**Skip the sweep; regenerate only figures and tables from existing results:**
```bash
python run_full_pipeline.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --skip-sweep \
  --results-dir results/eps_sweep/<dataset>
```

**Recompute metrics from existing CSVs** (e.g. after adding a new metric), then regenerate figures and tables:
```bash
python run_full_pipeline.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --skip-sweep \
  --refresh-metrics \
  --results-dir results/eps_sweep/<dataset>
```

**Custom results directory** when it does not match the schema's `dataset` name:
```bash
python run_full_pipeline.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --results-dir results/eps_sweep/my_custom_name
```

---

## 8. Outputs reference

| You provide | Where | Used for |
|---|---|---|
| Schema JSON | `--schema` | Column types, bounds, targets, sensitive attribute, dataset name |
| Dataset CSV | `--data` | Training data for synthesis and metric evaluation |

| The pipeline produces | Path |
|---|---|
| Synthetic CSVs | `results/eps_sweep/<dataset>/` |
| Result JSONs (metrics) | `results/eps_sweep/<dataset>/results_eps<ε>_seed<s>.json` |
| Figures (PDF) | `outputs/figures/` |
| Tables (LaTeX) | `outputs/tables/` |
| Ablation metrics (CSV) | `outputs/ablation_metrics/` |
| Report PDF (if `report.tex` exists) | `outputs/report.pdf` |

**What is `sensitive_attributes`?**
An optional field in the schema JSON: `"sensitive_attributes": ["status"]`. It tells the pipeline which column to use for the attribute-inference privacy metric — a classifier is trained on synthetic data to predict that column from the rest, then evaluated on real data. High AUC means higher inference risk. If omitted, the pipeline falls back to `target_spec.primary_target`.

---

## 9. ε budget accounting

The pipeline uses a two-phase composition:

```
Phase 1:  schema_generator.py   spends ε₁ (schema mode: private only)
Phase 2:  run_experiment.py     spends ε₂ (--epsilon argument)

Total:    ε_total = ε₁ + ε₂
```

In **public mode**, ε₁ = 0. The `--epsilon` passed to `run_experiment.py` is the full DP cost.

In **private mode**, ε₁ = `--schema-epsilon` (e.g. 0.5) is spent once when the schema is generated. Every subsequent experiment run reuses the same schema. When reporting results, state:

> ε denotes the synthesis budget (ε₂). Total DP cost is ε_total = ε_schema + ε₂, where ε_schema = 0.5 was spent once during schema generation.

The sweep in `run_sweep.py` varies ε₂. The ε₁ cost does not change between sweep points and should be reported as a fixed additive offset in all figures and tables.

For the full schema generator field reference, DP coverage details, and NPA warnings see [docs/schema_generator_README.md](docs/schema_generator_README.md).
