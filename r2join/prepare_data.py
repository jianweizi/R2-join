#!/usr/bin/env python3
"""Prepare or validate R2-Join processed data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REQUIRED_FILES = [
    "corpus.json",
    "train_ground_truth.json",
    "val_ground_truth.json",
    "test_ground_truth.json",
]

OPTIONAL_FILES = [
    "train_pairs_hard_negatives.json",
    "train_positive_pairs.json",
    "val_positive_pairs.json",
    "test_positive_pairs.json",
    "split_manifest.json",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def make_sample(output_dir: Path) -> None:
    corpus = [
        {
            "id": "cities_a.city",
            "text": "[Table] cities_a; [Column] city; [Type] text; "
            "[Values] Paris, Lyon, Berlin",
        },
        {
            "id": "cities_b.city_name",
            "text": "[Table] cities_b; [Column] city_name; [Type] text; "
            "[Values] Paris, Berlin, Rome",
        },
        {
            "id": "countries.country",
            "text": "[Table] countries; [Column] country; [Type] text; "
            "[Values] France, Germany, Italy",
        },
        {
            "id": "airports.city",
            "text": "[Table] airports; [Column] city; [Type] text; "
            "[Values] Paris, Lyon, Rome",
        },
    ]
    train_gt = {"cities_a.city": ["cities_b.city_name", "airports.city"]}
    val_gt = {"cities_b.city_name": ["cities_a.city"]}
    test_gt = {"airports.city": ["cities_a.city", "cities_b.city_name"]}
    train_pairs = [
        {
            "query_id": "cities_a.city",
            "positive_id": "cities_b.city_name",
            "hard_negatives": ["countries.country"],
        },
        {
            "query_id": "cities_a.city",
            "positive_id": "airports.city",
            "hard_negatives": ["countries.country"],
        },
    ]

    dump_json(output_dir / "corpus.json", corpus)
    dump_json(output_dir / "train_ground_truth.json", train_gt)
    dump_json(output_dir / "val_ground_truth.json", val_gt)
    dump_json(output_dir / "test_ground_truth.json", test_gt)
    dump_json(output_dir / "train_pairs_hard_negatives.json", train_pairs)
    dump_json(
        output_dir / "split_manifest.json",
        {
            "dataset": "sample",
            "note": "Toy dataset for repository smoke tests only.",
            "files": [p.name for p in output_dir.glob("*.json")],
        },
    )


def validate_processed_dir(data_dir: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {data_dir}: {missing}")

    corpus = load_json(data_dir / "corpus.json")
    if not isinstance(corpus, list):
        raise ValueError("corpus.json must be a list of records.")

    corpus_ids = set()
    for idx, item in enumerate(corpus):
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise ValueError(f"Invalid corpus record at index {idx}: {item!r}")
        corpus_ids.add(str(item["id"]))

    summary = {"num_columns": len(corpus), "splits": {}}
    for split in ["train", "val", "test"]:
        gt = load_json(data_dir / f"{split}_ground_truth.json")
        if not isinstance(gt, dict):
            raise ValueError(f"{split}_ground_truth.json must be an object.")
        num_pairs = sum(len(v) for v in gt.values())
        unknown_queries = [q for q in gt if q not in corpus_ids]
        summary["splits"][split] = {
            "num_queries": len(gt),
            "num_positive_pairs": num_pairs,
            "unknown_queries": len(unknown_queries),
        }
    return summary


def copy_processed(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--sample", action="store_true", help="Create toy data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample:
        make_sample(args.output_dir)
    else:
        if args.input_dir is None:
            raise SystemExit("--input-dir is required unless --sample is set.")
        validate_processed_dir(args.input_dir)
        copy_processed(args.input_dir, args.output_dir)

    summary = validate_processed_dir(args.output_dir)
    summary["dataset"] = args.dataset
    dump_json(args.output_dir / "data_manifest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
