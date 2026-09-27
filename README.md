# R2-Join

R2-Join is a two-stage retrieval-reranking method for joinable table discovery.
Given a query column, it first retrieves candidate columns with a bi-encoder and
then reranks the retrieved pairs with a cross-encoder.

This repository contains the experiment code used for the paper. Large public
datasets, trained checkpoints, and embedding caches are not stored in Git. See
`DATA.md` for dataset instructions and expected file formats.

## Install

```bash
conda create -n r2join python=3.10 -y
conda activate r2join
pip install -r requirements.txt
```

## Smoke Check

Run a lightweight repository check without downloading model checkpoints:

```bash
bash run_demo.sh
```

This verifies Python syntax and the sample-data schema. Full experiments require
the datasets and pretrained models described below.

## Full Workflow

Edit one config file in `configs/` so that its paths point to your local dataset
copy, then run the four-step pipeline.

Prepare data and mine rank-tiered hard negatives:

```bash
python -m r2join.mine_hard_negatives --config configs/opendata.json
```

Train the Stage I retriever:

```bash
python -m r2join.train_stage1 --config configs/opendata.json
```

Train the Stage II reranker:

```bash
python -m r2join.train_stage2 --config configs/opendata.json
```

Evaluate:

```bash
python -m r2join.evaluate_full_pipeline --config configs/opendata.json
```

## Reproducing Main Results

After preparing each dataset in the format described in `DATA.md`, run the same
four-step pipeline with the corresponding config:

```bash
# OpenData
python -m r2join.mine_hard_negatives --config configs/opendata.json
python -m r2join.train_stage1 --config configs/opendata.json
python -m r2join.train_stage2 --config configs/opendata.json
python -m r2join.evaluate_full_pipeline --config configs/opendata.json

# WebTable
python -m r2join.mine_hard_negatives --config configs/webtable.json
python -m r2join.train_stage1 --config configs/webtable.json
python -m r2join.train_stage2 --config configs/webtable.json
python -m r2join.evaluate_full_pipeline --config configs/webtable.json

# Odoo
python -m r2join.mine_hard_negatives --config configs/odoo.json
python -m r2join.train_stage1 --config configs/odoo.json
python -m r2join.train_stage2 --config configs/odoo.json
python -m r2join.evaluate_full_pipeline --config configs/odoo.json
```

The expected `HIT@10`, `RECALL@10`, `MRR@10`, and `NDCG@10` values are listed
in `RESULTS.md`.

## Repository Layout

```text
r2join/mine_hard_negatives.py     Dataset preparation and rank-tiered negative mining
r2join/train_stage1.py            Stage I bi-encoder training
r2join/train_stage2.py            Stage II reranker training with hybrid loss
r2join/evaluate_full_pipeline.py  Stage I + Stage II evaluation
r2join/common.py                  Shared model, feature, metric, and IO utilities
configs/                         Dataset config templates
sample_data/                     Tiny data for schema checks
DATA.md                          Dataset notes and expected formats
RESULTS.md                       Main paper results
```

## Notes

- The scripts accept local paths through JSON configs and avoid server-specific
  absolute paths.
- OpenData and WebTable are public datasets; this repository documents how to
  prepare them instead of redistributing large files.
- Odoo is not redistributed in this public repository; `configs/odoo.json`
  expects an authorized processed copy in the documented format.
