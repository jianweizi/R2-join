# R2-Join

R2-Join is a two-stage retrieval-reranking method for joinable table discovery.
Given a query column, it first retrieves candidate columns with a bi-encoder and
then reranks the retrieved pairs with a cross-encoder.

This repository is intentionally lightweight. Large public datasets, trained
models, and embedding caches are not stored in Git. See `DATA.md` for dataset
instructions and expected file formats.

## Install

```bash
conda create -n r2join python=3.10 -y
conda activate r2join
pip install -r requirements.txt
```

## Quick Demo

Run the toy example without downloading any large model:

```bash
bash run_demo.sh
```

The demo creates processed sample data and evaluates a lexical fallback ranker.
It is only a smoke test for the repository workflow.

## Full Workflow

Prepare data:

```bash
python r2join/prepare_data.py \
  --input-dir /path/to/processed_dataset \
  --output-dir ./processed/opendata \
  --dataset opendata
```

Train the Stage I retriever:

```bash
python r2join/train_retriever.py \
  --data-dir ./processed/opendata \
  --output-dir ./models/opendata/retriever
```

Train the Stage II reranker:

```bash
python r2join/train_reranker.py \
  --data-dir ./processed/opendata \
  --output-dir ./models/opendata/reranker
```

Evaluate:

```bash
python r2join/evaluate.py \
  --data-dir ./processed/opendata \
  --retriever-model ./models/opendata/retriever \
  --reranker-model ./models/opendata/reranker \
  --split test
```

## Reproducing Main Results

After preparing each dataset in the format described in `DATA.md`, use the
following commands to train and evaluate R2-Join on each dataset.

```bash
# OpenData
python r2join/train_retriever.py \
  --data-dir ./processed/opendata \
  --output-dir ./models/opendata/retriever
python r2join/train_reranker.py \
  --data-dir ./processed/opendata \
  --output-dir ./models/opendata/reranker
python r2join/evaluate.py \
  --data-dir ./processed/opendata \
  --retriever-model ./models/opendata/retriever \
  --reranker-model ./models/opendata/reranker \
  --split test

# WebTable
python r2join/train_retriever.py \
  --data-dir ./processed/webtable \
  --output-dir ./models/webtable/retriever
python r2join/train_reranker.py \
  --data-dir ./processed/webtable \
  --output-dir ./models/webtable/reranker
python r2join/evaluate.py \
  --data-dir ./processed/webtable \
  --retriever-model ./models/webtable/retriever \
  --reranker-model ./models/webtable/reranker \
  --split test

# Odoo
python r2join/train_retriever.py \
  --data-dir ./processed/odoo \
  --output-dir ./models/odoo/retriever
python r2join/train_reranker.py \
  --data-dir ./processed/odoo \
  --output-dir ./models/odoo/reranker
python r2join/evaluate.py \
  --data-dir ./processed/odoo \
  --retriever-model ./models/odoo/retriever \
  --reranker-model ./models/odoo/reranker \
  --split test
```

The expected `HIT@10`, `RECALL@10`, `MRR@10`, and `NDCG@10` values are listed
in `RESULTS.md`.

## Repository Layout

```text
r2join/prepare_data.py       Data validation, copying, and toy data creation
r2join/train_retriever.py    Stage I bi-encoder training
r2join/train_reranker.py     Stage II cross-encoder training
r2join/evaluate.py           Hit, Recall, MRR, and NDCG evaluation
sample_data/                 Tiny data for smoke tests
DATA.md                      Dataset notes and expected formats
RESULTS.md                   Main paper results
```

## Notes

- The scripts accept local paths and avoid server-specific absolute paths.
- OpenData and WebTable are public datasets; this repository documents how to
  prepare them instead of redistributing large files.
- Put large trained checkpoints in an external release location such as Zenodo,
  Hugging Face, or GitHub Releases, then link them from `DATA.md`.
