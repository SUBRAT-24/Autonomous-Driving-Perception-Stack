#!/usr/bin/env bash
# ==============================================================================
# One-command fine-tuning entrypoint for YOLOv8s on BDD100K
# ==============================================================================
set -euo pipefail

CONFIG="${1:-configs/train_config.yaml}"
DATA="${2:-data/processed/bdd100k/dataset.yaml}"

echo "=== Training YOLOv8s on BDD100K ==="
echo "Config : ${CONFIG}"
echo "Data   : ${DATA}"

python -m src.detection.train --config "${CONFIG}" --data "${DATA}"

