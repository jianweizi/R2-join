"""Train the Stage I R2-Join bi-encoder retriever."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

try:
    from .common import DEFAULT_STAGE1_MODEL, corpus_to_maps, load_config, load_json, normalize_pair_row
except ImportError:
    from common import DEFAULT_STAGE1_MODEL, corpus_to_maps, load_config, load_json, normalize_pair_row


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def build_examples(data_dir: Path, hard_negatives_per_positive: int) -> list[InputExample]:
    corpus_rows = load_json(data_dir / "corpus.json")
    _, _, id_to_text = corpus_to_maps(corpus_rows)
    rows = load_json(data_dir / "train_pairs_hard_negatives.json")

    examples = []
    skipped = 0
    for row in rows:
        item = normalize_pair_row(row, id_to_text)
        if not item or item.get("label") != 1:
            skipped += 1
            continue
        for negative in item["negatives_list"][:hard_negatives_per_positive]:
            examples.append(InputExample(texts=[item["query"], item["positive"], negative]))

    logger.info("Stage I examples: %d (skipped rows: %d)", len(examples), skipped)
    return examples


def build_evaluator(data_dir: Path, batch_size: int, name: str):
    corpus_rows = load_json(data_dir / "corpus.json")
    _, _, id_to_text = corpus_to_maps(corpus_rows)
    val_gt = load_json(data_dir / "val_ground_truth.json")

    queries = {}
    relevant_docs = {}
    for qid, target_ids in val_gt.items():
        if qid not in id_to_text:
            continue
        valid_targets = {tid for tid in target_ids if tid in id_to_text}
        if valid_targets:
            queries[qid] = id_to_text[qid]
            relevant_docs[qid] = valid_targets

    logger.info("Stage I validation queries: %d", len(queries))
    return InformationRetrievalEvaluator(
        queries,
        id_to_text,
        relevant_docs,
        name=name,
        show_progress_bar=False,
        corpus_chunk_size=50000,
        batch_size=batch_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the R2-Join Stage I bi-encoder retriever.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--data-dir", help="Processed dataset directory.")
    parser.add_argument("--output-dir", help="Directory for the trained Stage I model.")
    parser.add_argument("--base-model", default=None, help="SentenceTransformer checkpoint or local path.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--hard-negatives-per-positive", type=int, default=None)
    parser.add_argument("--evaluation-steps", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow replacing an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_dir = Path(args.data_dir or config.get("data_dir", "processed/opendata"))
    output_dir = Path(args.output_dir or config.get("stage1_output_dir", data_dir / "models/stage1_biencoder"))
    base_model = args.base_model or config.get("stage1_base_model", DEFAULT_STAGE1_MODEL)
    batch_size = args.batch_size or int(config.get("stage1_batch_size", 32))
    epochs = args.epochs or int(config.get("stage1_epochs", 10))
    learning_rate = args.learning_rate or float(config.get("stage1_learning_rate", 2e-5))
    max_seq_length = args.max_seq_length or int(config.get("stage1_max_seq_length", 256))
    negatives_per_positive = args.hard_negatives_per_positive or int(
        config.get("stage1_hard_negatives_per_positive", 2)
    )
    evaluation_steps = args.evaluation_steps or int(config.get("stage1_evaluation_steps", 500))
    allow_overwrite = args.allow_overwrite or bool(config.get("allow_overwrite", False))

    logger.info("Data dir: %s", data_dir)
    logger.info("Stage I base model: %s", base_model)
    logger.info("Stage I output dir: %s", output_dir)
    if output_dir.exists():
        if not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing Stage I output directory: {output_dir}. "
                "Use --allow-overwrite or choose another output directory."
            )
        shutil.rmtree(output_dir)

    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length

    examples = build_examples(data_dir, negatives_per_positive)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    evaluator = build_evaluator(data_dir, batch_size, data_dir.name)

    warmup_steps = int(len(train_dataloader) * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=not args.no_amp,
    )
    logger.info("Stage I training finished.")


if __name__ == "__main__":
    main()
