#!/usr/bin/env bash
# ==============================================================================
# One-command dataset setup and download for BDD100K 10K subset
# ==============================================================================
set -euo pipefail

DATA_ROOT="data/raw/bdd100k"
echo "=== BDD100K Dataset Setup ==="

# Check if data already exists
if python scripts/verify_data.py --data-root "${DATA_ROOT}" 2>/dev/null; then
    echo "[INFO] Full BDD100K dataset already verified at ${DATA_ROOT}."
    exit 0
fi

echo "[INFO] Setting up sample dataset for offline verification..."
python scripts/setup_sample_data.py --dest "${DATA_ROOT}"

echo "[INFO] Running verification..."
python scripts/verify_data.py --data-root "${DATA_ROOT}"

echo "=============================================================="
echo "To replace with the official full 10K dataset:"
echo "1. Visit https://bdd-data.berkeley.edu and download 10K Images + Detection 2020 Labels"
echo "2. Unzip into ${DATA_ROOT}/images/100k/{train,val} and ${DATA_ROOT}/labels/det_20/"
echo "3. Re-run: python scripts/verify_data.py"
echo "=============================================================="

