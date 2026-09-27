#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/odoo.json}

python -m r2join.mine_hard_negatives --config "$CONFIG"
python -m r2join.train_stage1 --config "$CONFIG"
python -m r2join.train_stage2 --config "$CONFIG"
python -m r2join.evaluate_full_pipeline --config "$CONFIG"
