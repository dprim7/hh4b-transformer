#!/bin/bash
set -euo pipefail

# Minimal launcher for Nautilus cluster
# Usage: ./scripts/nautilus_submit.sh [--train-only]

python -m pip install -e .
python -m pip install -r requirements.txt

if [[ "${1:-}" != "--train-only" ]]; then
  pre-commit install || true
fi

python -m hh4b_transformer.train --data configs/data.yaml --features configs/features.yaml --model configs/model.yaml --train configs/train.yaml --outdir artifacts/run-$(date +%Y%m%d-%H%M%S)
