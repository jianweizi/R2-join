# Data

Large datasets are not stored in this Git repository.

## Public Datasets

OpenData and WebTable are public datasets used for joinable table discovery.
The raw datasets, query files, and ground-truth files are available from
LakeBench:

https://github.com/BIT-DataLab/LakeBench

For OpenData and WebTable, use the join-search resources listed in the
Prepare Datasets table of the LakeBench README. In particular, R2-Join uses
the following raw resources before conversion to the processed format below:

- WebTable
- OpenData_SG
- OpenData_CAN
- OpenData_UK
- OpenData_USA
- WebTable_Join_Query
- WebTable_Join_Ground_Truth
- OpenData_Join_Query
- OpenData_Join_Ground_Truth

The raw files are not redistributed in this repository because of their size.
After downloading them, convert them into the R2-Join processed format described
below.

## Odoo

Odoo is an ERP-derived dataset used to evaluate R2-Join on enterprise schemas.
We do not redistribute the Odoo dataset in this public repository because it is
derived from ERP application data and may contain schema-specific information.
For transparency, this repository documents the expected processed file format
below and includes `sample_data/` as a runnable example. The Odoo results in
`RESULTS.md` were produced from an internal processed version following the same
format.

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
    "id": "table_a.city",
    "text": "[Table] table_a; [Column] city; [Values] Paris, Lyon"
  }
]
```

`*_ground_truth.json` maps a query column id to joinable column ids:

```json
{
  "table_a.city": ["table_b.city_name"]
}
```

`train_pairs_hard_negatives.json` contains training rows:

```json
[
  {
    "query_id": "table_a.city",
    "positive_id": "table_b.city_name",
    "hard_negatives": ["table_c.country"]
  }
]
```

The scripts also accept several common field aliases such as `query`,
`positive`, `pos_id`, `negative_ids`, and `negatives`.


