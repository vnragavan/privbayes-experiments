#!/usr/bin/env bash
# Full pipeline: sweep → generate figures/tables → compile report.
# Matches run_full_pipeline.py (results under results/eps_sweep/<name>, report in outputs/).
#
# Usage: ./run_full_pipeline.sh [schema] [data]
# Default: schemas/lung_schema.json, data/lung_clean.csv
#
# Prerequisites: venv with deps, pdflatex. See README.md.

set -e
cd "$(dirname "$0")"
SCHEMA="${1:-schemas/lung_schema.json}"
DATA="${2:-data/lung_clean.csv}"
SWEEP_DIR="results/eps_sweep"
FIGURES_DIR="outputs/figures"
TABLES_DIR="outputs/tables"
REPORT_DIR="outputs"

# Dataset name from schema (used as subdir under results/eps_sweep)
NAME=$(python -c "
import json
with open('$SCHEMA') as f:
    d = json.load(f)
n = d.get('dataset', '').strip().replace(' ', '_').replace('/', '_') or 'dataset'
print(n)
" 2>/dev/null || echo "dataset")

RESULTS_DIR="$SWEEP_DIR/$NAME"
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR" "$TABLES_DIR"

export PYTHONPATH=.

echo "=== 1. Epsilon sweep ==="
python run_sweep.py \
  --schema "$SCHEMA" \
  --data "$DATA" \
  --output-dir "$SWEEP_DIR"

echo ""
echo "=== 2. Generate figures and tables ==="
python analysis/generate_all.py \
  --results-dir "$RESULTS_DIR" \
  --figures-dir "$FIGURES_DIR" \
  --tables-dir "$TABLES_DIR" \
  --schema "$SCHEMA" \
  --data "$DATA"

echo ""
if [[ -f "$REPORT_DIR/report.tex" ]]; then
  echo "=== 3. Compile report PDF ==="
  for _ in 1 2; do
    pdflatex -interaction=nonstopmode -output-directory "$REPORT_DIR" "$REPORT_DIR/report.tex"
  done
  echo "Done. Results: $RESULTS_DIR | Figures: $FIGURES_DIR | Tables: $TABLES_DIR | Report: $REPORT_DIR/report.pdf"
else
  echo "=== 3. Report (skipped; report.tex not in repo) ==="
  echo "Done. Results: $RESULTS_DIR | Figures: $FIGURES_DIR | Tables: $TABLES_DIR"
fi
