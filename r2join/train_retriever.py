#!/usr/bin/env python3
"""Train the Stage I bi-encoder retriever."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def first_present(row: dict, names: list[str]):
    for name in names:
        if name in row:
            return row[name]
    return None


def load_training_triplets(data_dir: Path, max_samples: int | None = None):
    corpus = {str(x["id"]): x["text"] for x in load_json(data_dir / "corpus.json")}
    rows = load_json(data_dir / "train_pairs_hard_negatives.json")
    triplets = []

    for row in rows:
        qid = first_present(row, ["query_id", "query", "qid", "source_id"])
        pid = first_present(row, ["positive_id", "positive", "pos_id", "target_id"])
        negatives = first_present(
            row,
            ["hard_negatives", "negative_ids", "negatives", "random_negatives"],
        )
        if isinstance(negatives, str):
            negatives = [negatives]
        if not qid or not pid or not negatives:
            continue
        if qid not in corpus or pid not in corpus:
            continue
        for nid in negatives:
            if nid in corpus and nid != qid and nid != pid:
                triplets.append((corpus[qid], corpus[pid], corpus[nid]))
                break
        if max_samples and len(triplets) >= max_samples:
            break
    return triplets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer base model.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = args.max_samples or None
    triplets = load_training_triplets(args.data_dir, max_samples=max_samples)
    if not triplets:
        raise SystemExit("No valid training triplets found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "stage": "retriever",
        "data_dir": str(args.data_dir),
        "base_model": args.base_model,
        "num_triplets": len(triplets),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
    }
    dump_json(args.output_dir / "training_config.json", config)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(args.base_model)
    examples = [InputExample(texts=list(item)) for item in triplets]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.TripletLoss(model=model)
    warmup_steps = max(1, int(len(loader) * args.epochs * 0.1))

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(args.output_dir),
        show_progress_bar=True,
    )


if __name__ == "__main__":
    main()
