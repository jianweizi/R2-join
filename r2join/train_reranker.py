#!/usr/bin/env python3
"""Train the Stage II cross-encoder reranker."""

from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path


FEATURE_DIM = 32


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


def text_features(text: str, dim: int = FEATURE_DIM) -> np.ndarray:
    import numpy as np

    values = np.zeros(dim, dtype=np.float32)
    if not text:
        return values
    values[0] = len(text) / 500.0
    values[1] = text.count(" ") / 50.0
    values[2] = text.count(",") / 20.0
    values[3] = text.count(";") / 20.0
    grams = [text[i : i + 3] for i in range(max(0, len(text) - 2))][:200]
    for gram in grams:
        idx = (zlib.crc32(gram.encode("utf-8")) % (dim - 4)) + 4
        values[idx] += 1.0
    return np.log1p(values)


def load_pair_examples(data_dir: Path, max_samples: int | None = None):
    corpus = {str(x["id"]): x["text"] for x in load_json(data_dir / "corpus.json")}
    rows = load_json(data_dir / "train_pairs_hard_negatives.json")
    examples = []

    for row in rows:
        qid = first_present(row, ["query_id", "query", "qid", "source_id"])
        pid = first_present(row, ["positive_id", "positive", "pos_id", "target_id"])
        negatives = first_present(
            row,
            ["hard_negatives", "negative_ids", "negatives", "random_negatives"],
        )
        if isinstance(negatives, str):
            negatives = [negatives]
        if not qid or not pid or qid not in corpus or pid not in corpus:
            continue
        qtext = corpus[qid]
        examples.append((qtext, corpus[pid], 1.0))
        for nid in negatives or []:
            if nid in corpus and nid != qid and nid != pid:
                examples.append((qtext, corpus[nid], 0.0))
                break
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="Transformer base model for pair encoding.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    max_samples = args.max_samples or None
    examples = load_pair_examples(args.data_dir, max_samples=max_samples)
    if not examples:
        raise SystemExit("No valid reranker training examples found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "stage": "reranker",
        "data_dir": str(args.data_dir),
        "base_model": args.base_model,
        "num_pair_examples": len(examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dry_run": args.dry_run,
    }
    dump_json(args.output_dir / "training_config.json", config)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    class Reranker(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size
            self.feature_mlp = nn.Sequential(
                nn.Linear(FEATURE_DIM * 2, 32), nn.ReLU(), nn.Linear(32, 16)
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden + 16, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        def forward(self, input_ids, attention_mask, features, token_type_ids=None):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            cls = outputs.last_hidden_state[:, 0]
            feat = self.feature_mlp(features)
            return self.classifier(torch.cat([cls, feat], dim=1)).squeeze(-1)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Reranker(args.base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    def collate(batch):
        qtexts = [x[0] for x in batch]
        ctexts = [x[1] for x in batch]
        labels = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        encoded = tokenizer(
            list(zip(qtexts, ctexts)),
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        features = np.stack(
            [np.concatenate([text_features(q), text_features(c)]) for q, c in zip(qtexts, ctexts)]
        )
        return encoded, torch.tensor(features, dtype=torch.float32), labels

    loader = DataLoader(
        PairDataset(examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for encoded, features, labels in loader:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features=features, **encoded)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(loader)):.4f}")

    tokenizer.save_pretrained(args.output_dir)
    torch.save(model.state_dict(), args.output_dir / "pytorch_model.bin")
    dump_json(args.output_dir / "model_config.json", config)


if __name__ == "__main__":
    main()
