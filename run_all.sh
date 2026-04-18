#!/usr/bin/env bash
# Regenerate every artefact in the project from the raw data file.
# Usage:  ./run_all.sh
#
# Expects data/census-bureau.data and data/census-bureau.columns to exist.
# Produces outputs/ and figures/ contents, and refreshes notebooks/.

set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f data/census-bureau.data ]]; then
    echo "ERROR: data/census-bureau.data not found."
    echo "Place the raw CPS file at data/census-bureau.data (see README)."
    exit 1
fi

echo "=== 1/4  EDA ===================================================="
python src/eda.py

echo "=== 2/4  Classifier ============================================="
python src/train_classifier.py

echo "=== 3/4  Segmentation ==========================================="
python src/segment.py

echo "=== 4/4  Rebuild notebooks ======================================"
python build_notebooks.py

echo
echo "All done. See outputs/, figures/, notebooks/."
