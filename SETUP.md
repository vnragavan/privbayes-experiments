# Setup Instructions

**If you only want to run with your own schema and dataset**, see [README.md](README.md) for the quick start. The steps below are for getting the full environment (all three implementations and the integrated schema generator) in place.

---

## Step 1 — Create and activate a virtualenv

From the repo root:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This ensures `schema_generator.py`, `schema_validator.py`, and all metrics/plotting scripts have their Python dependencies available (including `pandas`, `lifelines`, and plotting libraries).

---

## Step 2 — Add the three implementation artifacts

These files/packages cannot be included in the repo and must be provided separately.

### CRNPrivBayes

Copy the enhanced CRN implementation file you received (for example `privbayes_enhanced.py`) to:

```text
implementations/crn_privbayes.py
```

The class `PrivBayesSynthesizerEnhanced` should be importable from `implementations.crn_privbayes`.

### DPMM

Extract `dpmm.zip` so that the following import works:

```python
from dpmm.models.priv_bayes import PrivBayesGM
```

The zip should contain a `dpmm/` package folder. Place it at the **project root**:

```text
privbayes-experiments/dpmm/
```

### SynthCity

Extract `synthcity_standalone.zip` so that the following import works:

```python
from synthcity_standalone.privbayes import PrivBayes
```

Place the folder at the **project root**:

```text
privbayes-experiments/synthcity_standalone/
```

---

## Step 3 — Verify imports

With the virtualenv active:

```bash
python -c "
from implementations.crn_privbayes import PrivBayesSynthesizerEnhanced
from dpmm.models.priv_bayes import PrivBayesGM
from synthcity_standalone.privbayes import PrivBayes
print('All three implementations OK')
"
```

If this prints `All three implementations OK`, the implementation layer is wired up correctly.

---

## Step 4 — Quick smoke test with your own data

You do **not** need an external schema-generator repo; this project ships its own `schema_generator.py` and detailed documentation under `docs/schema_generator_README.md`.

1. **Prepare a small CSV** under `data/`, or use an existing one such as `data/lung_clean.csv`.
2. **Generate a public schema**:

```bash
python schema_generator.py \
  --data data/your_dataset.csv \
  --out schemas/your_schema.json \
  --schema-mode public \
  --target-kind survival_pair \
  --survival-event-col status \
  --survival-time-col time \
  --infer-categories \
  --emit-privbayes-extensions
```

3. **Validate the schema against the data**:

```bash
python schema_validator.py schemas/your_schema.json data/your_dataset.csv
```

You should see a final line:

```text
RESULT: PASS — schema is ready to guide synthesis
```

Warnings (e.g. about discrete columns appearing in both `public_bounds` and `public_categories`, or bounds inferred from data) are non-fatal but recommended to clean up before publication. See `docs/schema_generator_README.md` for details on DP coverage and NPA warnings.

4. **Run a single-ε smoke test** with CRN:

```bash
python run_experiment.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --epsilon 1.0 \
  --seed 0 \
  --implementations crn \
  --output-dir results/test_run
```

This will:

- Perform a 60/20/20 train/test/holdout split internally.
- Patch `n_records` in memory (schema file on disk is unchanged).
- Fit CRN, sample synthetic data, and compute all metrics.

Check that:

- The script finishes without errors or tracebacks.
- `km_l1` is finite and reasonable.
- A synthetic CSV and a `results_eps1.0_seed0.json` file appear under `results/test_run/`.

---

## Step 5 — Run the full epsilon sweep

Once the smoke test passes, you can sweep ε across a grid of values and seeds:

```bash
python run_sweep.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --epsilons 0.1 0.5 1.0 2.0 5.0 10.0 \
  --seeds 0 1 2 3 4
```

This writes synthetic CSVs and result JSONs under:

```text
results/eps_sweep/<dataset>/
```

Where `<dataset>` comes from the `"dataset"` field in the schema (or the schema filename stem if that field is missing).

---

## Step 6 — Generate all figures and tables (and run ablation)

The recommended entry point for reproducing all paper figures/tables is the full pipeline script:

```bash
python run_full_pipeline.py \
  --schema schemas/your_schema.json \
  --data data/your_dataset.csv \
  --output-dir results/eps_sweep \
  --skip-ablation
```

This will:

1. Run the ε-sweep (`run_sweep.py`) unless `--skip-sweep` or `--refresh-metrics` is provided.
2. Recompute metrics from existing CSVs if `--refresh-metrics` is set.
3. Generate all figures and tables via `analysis/generate_all.py`.
4. Optionally run the adapter ablation (LaTeX tables, ablation figure, and CSV metrics) unless `--skip-ablation` is set.
5. Optionally compile a PDF report if `outputs/report.tex` exists.

Figures and tables land in:

- `outputs/figures/` (PDF)
- `outputs/tables/` (.tex)
- `outputs/ablation_metrics/` (CSV ablation tables)

See [README.md](README.md) for a detailed description of the ablation flags (`--ablation-n-runs`, `--ablation-error-bar`) and the outputs produced.

---

## Performance and peak memory

- **Peak memory** is the maximum process RSS (resident set size) observed during the fit and sample phases. RSS is sampled at ~20 Hz during each phase so short-lived allocations are captured.
- **One process per implementation**: each method (CRN, DPMM, SynthCity) is run in its own process so that reported peak memory is the footprint of that method in isolation, not affected by other implementations or their garbage collection.
- See `metrics/performance/tracker.py` for the implementation and `run_experiment.py` for process isolation.

