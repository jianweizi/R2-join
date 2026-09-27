#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/demo_processed"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" "$ROOT_DIR/r2join/prepare_data.py" \
  --sample \
  --output-dir "$OUT_DIR" \
  --dataset sample

"$PYTHON_BIN" "$ROOT_DIR/r2join/train_retriever.py" \
  --data-dir "$OUT_DIR" \
  --output-dir "$ROOT_DIR/demo_models/retriever" \
  --dry-run

"$PYTHON_BIN" "$ROOT_DIR/r2join/train_reranker.py" \
  --data-dir "$OUT_DIR" \
  --output-dir "$ROOT_DIR/demo_models/reranker" \
  --dry-run

"$PYTHON_BIN" "$ROOT_DIR/r2join/evaluate.py" \
  --data-dir "$OUT_DIR" \
  --split test \
  --dry-run
