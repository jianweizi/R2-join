"""Train the Stage II R2-Join reranker with hybrid listwise/pointwise loss."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from .common import (
        DEFAULT_STAGE2_MODEL,
        FeatureExtractor,
        MultimodalCrossEncoder,
        corpus_to_maps,
        load_config,
        load_json,
        normalize_pair_row,
        set_seed,
    )
except ImportError:
    from common import (
        DEFAULT_STAGE2_MODEL,
        FeatureExtractor,
        MultimodalCrossEncoder,
        corpus_to_maps,
        load_config,
        load_json,
        normalize_pair_row,
        set_seed,
    )


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class ListwiseDataset(Dataset):
    def __init__(self, data_dir: Path, feature_extractor: FeatureExtractor, num_negatives: int):
        corpus_rows = load_json(data_dir / "corpus.json")
        _, _, id_to_text = corpus_to_maps(corpus_rows)
        raw_rows = load_json(data_dir / "train_pairs_hard_negatives.json")

        self.samples = []
        self.feature_extractor = feature_extractor
        for row in raw_rows:
            item = normalize_pair_row(row, id_to_text)
            if not item or item.get("label") != 1:
                continue
            negatives = list(item["negatives_list"])
            if not negatives:
                continue
            if len(negatives) < num_negatives:
                repeats = int(num_negatives / len(negatives)) + 1
                negatives = (negatives * repeats)[:num_negatives]
            else:
                negatives = negatives[:num_negatives]
            self.samples.append(
                {
                    "q_text": item["query"],
                    "pos_text": item["positive"],
                    "negatives": negatives,
                }
            )

        logger.info("Stage II listwise samples: %d", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        candidates = [sample["pos_text"]] + sample["negatives"]
        q_feat = self.feature_extractor.extract(sample["q_text"])
        c_feats = [self.feature_extractor.extract(candidate) for candidate in candidates]
        return {
            "q_text": sample["q_text"],
            "q_feat": q_feat,
            "candidates": candidates,
            "c_feats": c_feats,
        }


def make_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch):
        all_pairs = []
        all_features = []
        batch_size = len(batch)
        list_size = len(batch[0]["candidates"])

        for item in batch:
            for cand_text, cand_feat in zip(item["candidates"], item["c_feats"]):
                all_pairs.append([item["q_text"], cand_text])
                all_features.append(np.concatenate([item["q_feat"], cand_feat]))

        encodings = tokenizer(
            all_pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "token_type_ids": encodings.get("token_type_ids"),
            "features": torch.tensor(np.array(all_features), dtype=torch.float32),
            "batch_size": batch_size,
            "list_size": list_size,
        }

    return collate_fn


def run_epoch(model, dataloader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="train" if training else "val"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"]
        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
        features = batch["features"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(input_ids, attention_mask, token_type_ids, features)
            scores = logits.view(batch["batch_size"], batch["list_size"])
            targets_ce = torch.zeros(batch["batch_size"], dtype=torch.long, device=device)
            targets_bce = torch.zeros_like(scores, dtype=torch.float32, device=device)
            targets_bce[:, 0] = 1.0
            loss_list = criterion_ce(scores, targets_ce)
            loss_point = criterion_bce(scores, targets_bce)
            loss = beta * loss_list + (1 - beta) * loss_point

            if training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the R2-Join Stage II reranker.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--data-dir", help="Processed dataset directory.")
    parser.add_argument("--output-dir", help="Directory for the Stage II run.")
    parser.add_argument("--base-model", default=None, help="Cross-encoder backbone checkpoint or local path.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--num-negatives", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None, help="Weight for listwise CE; 1-beta weights BCE.")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow replacing an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_dir = Path(args.data_dir or config.get("data_dir", "processed/opendata"))
    beta = args.beta if args.beta is not None else float(config.get("stage2_beta", 0.5))
    output_dir = Path(args.output_dir or config.get("stage2_output_dir", data_dir / f"models/stage2_reranker_beta{beta}"))
    base_model = args.base_model or config.get("stage2_base_model", DEFAULT_STAGE2_MODEL)
    batch_size = args.batch_size or int(config.get("stage2_batch_size", 8))
    epochs = args.epochs or int(config.get("stage2_epochs", 5))
    learning_rate = args.learning_rate or float(config.get("stage2_learning_rate", 2e-5))
    max_seq_length = args.max_seq_length or int(config.get("stage2_max_seq_length", 512))
    feature_dim = args.feature_dim or int(config.get("stage2_feature_dim", 32))
    num_negatives = args.num_negatives or int(config.get("num_negatives", 7))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(config.get("stage2_val_ratio", 0.1))
    seed = args.seed or int(config.get("seed", 42))
    num_workers = args.num_workers if args.num_workers is not None else int(config.get("num_workers", 4))
    allow_overwrite = args.allow_overwrite or bool(config.get("allow_overwrite", False))

    set_seed(seed)
    if output_dir.exists():
        if not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing Stage II output directory: {output_dir}. "
                "Use --allow-overwrite or choose another output directory."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Data dir: %s", data_dir)
    logger.info("Stage II base model: %s", base_model)
    logger.info("Stage II output dir: %s", output_dir)

    model = MultimodalCrossEncoder(base_model, feature_dim * 2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = ListwiseDataset(data_dir, FeatureExtractor(feature_dim), num_negatives)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    if val_size:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset, val_dataset = dataset, dataset

    collate_fn = make_collate_fn(tokenizer, max_seq_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_path = output_dir / "best_model"
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, device, beta, optimizer)
        val_loss = run_epoch(model, val_loader, device, beta)
        logger.info("Epoch %d/%d - train loss %.4f - val loss %.4f", epoch + 1, epochs, train_loss, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(best_model_path)
            logger.info("Saved best Stage II model to %s", best_model_path)

    logger.info("Stage II training finished.")


if __name__ == "__main__":
    main()
