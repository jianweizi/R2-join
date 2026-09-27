#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m py_compile r2join/*.py
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

data_dir = Path("sample_data")
with (data_dir / "corpus.json").open(encoding="utf-8") as f:
    corpus = {row["id"]: row["text"] for row in json.load(f)}
with (data_dir / "train_pairs_hard_negatives.json").open(encoding="utf-8") as f:
    rows = json.load(f)

usable = []
for row in rows:
    query_id = row.get("query_col") or row.get("query_id")
    positive_id = row.get("positive_col") or row.get("positive_id")
    negatives = row.get("negatives_list") or row.get("hard_negatives") or []
    if query_id in corpus and positive_id in corpus and negatives:
        usable.append(row)

assert usable, "sample_data/train_pairs_hard_negatives.json has no usable rows"
print(f"Smoke check passed: {len(corpus)} columns, {len(usable)} training rows.")
PY
