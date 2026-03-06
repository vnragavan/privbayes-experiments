# Setup Instructions

**If you only want to run with your own schema and dataset**, see [README.md](README.md) for the quick start. The steps below are for getting the full environment (all three implementations and schema-generator) in place.

---

## Step 1 — Copy the three implementation files

These files cannot be included in the zip and must be copied manually.

### CRNPrivBayes
Copy `privbayes_enhanced.py` to:
```
implementations/crn_privbayes.py
```

### dpmm
Extract `dpmm.zip` so that the following import works:
```python
from dpmm.models.priv_bayes import PrivBayesGM
```
The zip contains a `dpmm/` package folder.
Place it at the **project root**: `privbayes-experiments/dpmm/`

### SynthCity
Extract `synthcity_standalone.zip` so that the following import works:
```python
from synthcity_standalone.privbayes import PrivBayes
```
Place the folder at the **project root**: `privbayes-experiments/synthcity_standalone/`

---

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Clone schema-generator

```bash
git init
git submodule add https://github.com/vnragavan/schema-generator schema-generator
git submodule update --init --recursive
pip install -r schema-generator/requirements.txt
```

Or if you already have the repo, copy it into `schema-generator/`.

---

## Step 4 — Verify imports

```bash
python -c "
from implementations.crn_privbayes import PrivBayesSynthesizerEnhanced
from dpmm.models.priv_bayes import PrivBayesGM
from synthcity_standalone.privbayes import PrivBayes
print('All three implementations OK')
"
```

---

## Step 5 — Smoke test with Rossi dataset

```bash
python -c "
from lifelines.datasets import load_rossi
df = load_rossi()
df.to_csv('data/rossi.csv', index=False)
print(df.shape, df.dtypes)
"

python schema-generator/schema_toolkit/prepare_schema.py \
  --data data/rossi.csv \
  --out schemas/rossi_schema.json \
  --dataset-name rossi \
  --target-col arrest

python run_experiment.py \
  --schema schemas/rossi_schema.json \
  --data data/rossi.csv \
  --epsilon 1.0 \
  --seed 0 \
  --implementations crn
```

---

## Step 6 — Apply the CRNPrivBayes schema-native patch

After the smoke test passes, follow the prompts in `ANTIGRAVITY_PROMPTS.md`
to patch CRNPrivBayes to read schema-generator JSON natively.

---

## Step 7 — Run the full epsilon sweep

```bash
python run_sweep.py \
  --schema schemas/rossi_schema.json \
  --data data/rossi.csv \
  --epsilons 0.1 0.5 1.0 2.0 5.0 10.0 \
  --seeds 0 1 2 3 4
```

---

## Step 8 — Generate all figures and tables

```bash
python analysis/generate_all.py
```

Outputs land in `outputs/figures/` (PDF) and `outputs/tables/` (.tex).

---

## Performance and peak memory

- **Peak memory** is the maximum process RSS (resident set size) observed during the fit and sample phases. RSS is sampled at ~20 Hz during each phase so short-lived allocations are captured.
- **One process per implementation**: each method (CRN, dpmm, SynthCity) is run in its own process so that reported peak memory is the footprint of that method in isolation, not affected by other implementations or their garbage collection.
- See `metrics/performance/tracker.py` for the implementation and `run_experiment.py` for process isolation.
