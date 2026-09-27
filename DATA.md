# Data

Large datasets are not stored in this Git repository.

## Public Datasets

OpenData and WebTable are public datasets used for joinable table discovery.
The experiments use the LakeBench release:

https://github.com/BIT-DataLab/LakeBench

Download the OpenData and WebTable table files and the join ground-truth files,
then edit `configs/opendata.json` and `configs/webtable.json` so that
`ground_truth_file` and `raw_data_dirs` point to your local copy.

## Odoo

Odoo is an ERP-derived dataset used to evaluate R2-Join on enterprise schemas.
We do not redistribute the Odoo dataset in this public repository because it is
derived from ERP application data and may contain schema-specific information.
The Odoo results in `RESULTS.md` were produced from an internal processed
version following the same file format below.

## Expected Processed Format

Each dataset directory should contain:

```text
corpus.json
train_ground_truth.json
val_ground_truth.json
test_ground_truth.json
train_pairs_hard_negatives.json
```

Optional but recommended:

```text
train_positive_pairs.json
val_positive_pairs.json
test_positive_pairs.json
split_manifest.json
```

`corpus.json` is a list of column records:

```json
[
  {
    "id": "table_a.csv::city",
    "text": "city contains 3 values (5, 4, 4.67): Paris, Lyon, Rome"
  }
]
```

`*_ground_truth.json` maps a query column id to joinable column ids:

```json
{
  "table_a.csv::city": ["table_b.csv::city_name"]
}
```

`train_pairs_hard_negatives.json` contains training rows:

```json
[
  {
    "query_col": "table_a.csv::city",
    "positive_col": "table_b.csv::city_name",
    "query": "city contains 3 values (5, 4, 4.67): Paris, Lyon, Rome",
    "positive": "city_name contains 3 values (6, 4, 5.00): Paris, Rome, Berlin",
    "label": 1,
    "negatives_list": [
      "country contains 3 values (7, 5, 6.33): France, Germany, Italy"
    ]
  }
]
```

The training scripts also accept several common field aliases such as
`query_id`, `positive_id`, `hard_negatives`, and `negatives`.

## Reproducing the Paper Results

For OpenData and WebTable, use the raw LakeBench files and run
`r2join.mine_hard_negatives` with the corresponding config. For Odoo, place an
authorized processed copy under the path configured by `configs/odoo.json`.

Do not commit large raw tables, trained model files, embedding caches, or logs
to Git.
