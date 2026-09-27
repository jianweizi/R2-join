"""Evaluate the complete R2-Join retrieval-reranking pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from .common import (
        DEFAULT_STAGE2_MODEL,
        FeatureExtractor,
        MultimodalCrossEncoder,
        calculate_metrics,
        corpus_to_maps,
        dump_json,
        load_config,
        load_json,
    )
except ImportError:
    from common import (
        DEFAULT_STAGE2_MODEL,
        FeatureExtractor,
        MultimodalCrossEncoder,
        calculate_metrics,
        corpus_to_maps,
        dump_json,
        load_config,
        load_json,
    )


def add_metrics(total, update):
    for k, values in update.items():
        for metric_name, value in values.items():
            total[k][metric_name] += value


def average_metrics(total, valid_queries: int):
    return {
        str(k): {
            metric_name: value / valid_queries if valid_queries else 0.0
            for metric_name, value in values.items()
        }
        for k, values in total.items()
    }


def print_stage_table(stage_name: str, metrics: dict):
    metric_k = [int(k) for k in metrics]
    print(f"\n{stage_name.upper()} performance:")
    print("Metric     | " + " | ".join(f"@{k:>2}" for k in metric_k) + " |")
    print("-" * (15 + 9 * len(metric_k)))
    for metric_name in ["hit", "precision", "recall", "mrr", "map", "ndcg"]:
        row = f"{metric_name.upper():<10} |"
        for k in metric_k:
            row += f" {metrics[str(k)][metric_name]:>6.4f} |"
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the R2-Join full pipeline.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--data-dir", help="Processed dataset directory.")
    parser.add_argument("--stage1-model", help="Trained Stage I SentenceTransformer directory.")
    parser.add_argument("--stage2-model", help="Trained Stage II best_model directory.")
    parser.add_argument("--stage2-base-model", default=None, help="Backbone used to initialize the Stage II reranker.")
    parser.add_argument("--output-file", help="JSON file for metrics.")
    parser.add_argument("--top-k-retrieval", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--metric-k", type=int, nargs="+", default=None)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_dir = Path(args.data_dir or config.get("data_dir", "processed/opendata"))
    model_dir = Path(config.get("model_dir", data_dir / "models"))
    stage1_model = Path(args.stage1_model or config.get("stage1_output_dir", model_dir / "stage1_biencoder"))
    stage2_model = Path(
        args.stage2_model or config.get("stage2_best_model_dir", model_dir / "stage2_reranker_beta0.5/best_model")
    )
    stage2_base_model = args.stage2_base_model or config.get("stage2_base_model", DEFAULT_STAGE2_MODEL)
    output_file = Path(args.output_file or config.get("eval_output_file", data_dir / "eval/full_pipeline_results.json"))
    top_k_retrieval = args.top_k_retrieval or int(config.get("top_k_retrieval", 100))
    rerank_top_k = args.rerank_top_k or int(config.get("rerank_top_k", 50))
    metric_k = args.metric_k or list(config.get("metric_k", [1, 5, 10, 20, 50]))
    feature_dim = args.feature_dim or int(config.get("stage2_feature_dim", 32))
    batch_size = args.batch_size or int(config.get("eval_batch_size", 32))
    max_seq_length = args.max_seq_length or int(config.get("stage2_max_seq_length", 512))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Data dir: {data_dir}")
    print(f"Stage I model: {stage1_model}")
    print(f"Stage II model: {stage2_model}")
    print(f"Device: {device}")

    corpus_rows = load_json(data_dir / "corpus.json")
    corpus_ids, corpus_texts, id_to_text = corpus_to_maps(corpus_rows)
    test_gt = load_json(data_dir / "test_ground_truth.json")

    retriever = SentenceTransformer(str(stage1_model))
    print("Encoding corpus with the Stage I retriever...")
    corpus_embeddings = retriever.encode(
        corpus_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    reranker = MultimodalCrossEncoder.from_pretrained(stage2_model, stage2_base_model, feature_dim * 2)
    reranker.to(device)
    reranker.eval()
    tokenizer = AutoTokenizer.from_pretrained(stage2_base_model)
    feature_extractor = FeatureExtractor(feature_dim)

    zero_metrics = {k: {"hit": 0.0, "precision": 0.0, "recall": 0.0, "mrr": 0.0, "map": 0.0, "ndcg": 0.0} for k in metric_k}
    totals = {"stage1": {k: dict(v) for k, v in zero_metrics.items()}, "stage2": {k: dict(v) for k, v in zero_metrics.items()}}

    valid_queries = 0
    for qid, target_ids in tqdm(list(test_gt.items()), desc="Evaluate"):
        if qid not in id_to_text:
            continue
        q_text = id_to_text[qid]
        true_set = {tid for tid in target_ids if tid in id_to_text and tid != qid}
        if not true_set:
            continue
        valid_queries += 1

        q_emb = retriever.encode(q_text, convert_to_tensor=True, normalize_embeddings=True)
        hits = util.semantic_search(q_emb, corpus_embeddings, top_k=top_k_retrieval + 5)[0]
        candidates = []
        for hit in hits:
            cid = corpus_ids[hit["corpus_id"]]
            if cid == qid:
                continue
            candidates.append({"id": cid, "text": id_to_text[cid], "score": float(hit["score"])})
            if len(candidates) >= top_k_retrieval:
                break

        add_metrics(totals["stage1"], calculate_metrics(candidates, true_set, metric_k))
        if not candidates:
            continue

        q_feat = feature_extractor.extract(q_text)
        pairs = []
        features = []
        for candidate in candidates:
            pairs.append([q_text, candidate["text"]])
            features.append(np.concatenate([q_feat, feature_extractor.extract(candidate["text"])]))

        batch_scores = []
        for start in range(0, len(pairs), batch_size):
            enc = tokenizer(
                pairs[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            feature_tensor = torch.tensor(np.array(features[start : start + batch_size]), dtype=torch.float32).to(device)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            token_type_ids = enc.get("token_type_ids")
            token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
            with torch.no_grad():
                logits = reranker(input_ids, attention_mask, token_type_ids, feature_tensor)
                scores = logits.squeeze().detach().cpu().numpy()
                if np.ndim(scores) == 0:
                    scores = [float(scores)]
                batch_scores.extend(scores)

        for candidate, score in zip(candidates, batch_scores):
            candidate["rerank_score"] = float(score)

        ranked_candidates = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:rerank_top_k]
        add_metrics(totals["stage2"], calculate_metrics(ranked_candidates, true_set, metric_k))

    averaged = {
        "stage1": average_metrics(totals["stage1"], valid_queries),
        "stage2": average_metrics(totals["stage2"], valid_queries),
    }
    print(f"\nValid queries: {valid_queries}")
    print_stage_table("stage1", averaged["stage1"])
    print_stage_table("stage2", averaged["stage2"])

    dump_json(
        output_file,
        {
            "data_dir": str(data_dir),
            "stage1_model": str(stage1_model),
            "stage2_model": str(stage2_model),
            "stage2_base_model": stage2_base_model,
            "valid_queries": valid_queries,
            "top_k_retrieval": top_k_retrieval,
            "rerank_top_k": rerank_top_k,
            "metric_k": metric_k,
            "metrics": averaged,
        },
    )
    print(f"\nSaved metrics to {output_file}")


if __name__ == "__main__":
    main()
