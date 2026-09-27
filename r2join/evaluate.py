#!/usr/bin/env python3
"""Evaluate R2-Join rankings."""

from __future__ import annotations

import argparse
import json
import math
import re
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


def tokenize(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if tok}


def lexical_score(query_text: str, candidate_text: str) -> float:
    q = tokenize(query_text)
    c = tokenize(candidate_text)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


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


def metrics_at_k(ranked_ids: list[str], true_ids: list[str], k: int) -> dict:
    true_set = set(true_ids)
    top = ranked_ids[:k]
    relevance = [1 if item in true_set else 0 for item in top]
    hits = sum(relevance)

    mrr = 0.0
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            mrr = 1.0 / idx
            break

    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(k, len(true_set))))

    return {
        "hit": 1.0 if hits else 0.0,
        "recall": hits / len(true_set) if true_set else 0.0,
        "mrr": mrr,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def average(rows: list[dict]) -> dict:
    if not rows:
        return {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def lexical_rank(query_id: str, id_to_text: dict[str, str], top_n: int) -> list[tuple[str, float]]:
    qtext = id_to_text[query_id]
    scored = []
    for cid, ctext in id_to_text.items():
        if cid == query_id:
            continue
        scored.append((cid, lexical_score(qtext, ctext)))
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return scored[:top_n]


def retriever_rank(query_id: str, id_to_text: dict[str, str], model_path: Path, top_n: int):
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(str(model_path))
    ids = list(id_to_text)
    texts = [id_to_text[cid] for cid in ids]
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    q_index = ids.index(query_id)
    hits = util.semantic_search(
        embeddings[q_index], embeddings, top_k=min(top_n + 1, len(ids))
    )[0]
    ranked = []
    for hit in hits:
        cid = ids[hit["corpus_id"]]
        if cid != query_id:
            ranked.append((cid, float(hit["score"])))
        if len(ranked) >= top_n:
            break
    return ranked


def rerank_pairs(query_id: str, candidates: list[tuple[str, float]], id_to_text, model_dir: Path):
    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    config = load_json(model_dir / "model_config.json")
    base_model = config["base_model"]

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

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Reranker(base_model).to(device)
    state = torch.load(model_dir / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state)
    model.eval()

    qtext = id_to_text[query_id]
    scored = []
    with torch.no_grad():
        for cid, stage1_score in candidates:
            ctext = id_to_text[cid]
            encoded = tokenizer(
                [(qtext, ctext)],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            features = np.concatenate([text_features(qtext), text_features(ctext)])
            features = torch.tensor(features[None, :], dtype=torch.float32).to(device)
            score = float(model(features=features, **encoded).item())
            scored.append((cid, score, stage1_score))
    scored.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    return [(cid, score) for cid, score, _ in scored]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--retriever-model", type=Path, default=None)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_json(args.data_dir / "corpus.json")
    ground_truth = load_json(args.data_dir / f"{args.split}_ground_truth.json")
    id_to_text = {str(item["id"]): item["text"] for item in corpus}

    per_query = []
    rankings = {}
    for query_id, true_ids in ground_truth.items():
        if query_id not in id_to_text:
            continue
        if args.retriever_model and not args.dry_run:
            candidates = retriever_rank(query_id, id_to_text, args.retriever_model, args.top_n)
        else:
            candidates = lexical_rank(query_id, id_to_text, args.top_n)
        if args.reranker_model and not args.dry_run:
            candidates = rerank_pairs(query_id, candidates, id_to_text, args.reranker_model)
        ranked_ids = [cid for cid, _ in candidates]
        rankings[query_id] = ranked_ids[: args.k]
        per_query.append(metrics_at_k(ranked_ids, true_ids, args.k))

    result = {
        "split": args.split,
        "k": args.k,
        "num_queries": len(per_query),
        "metrics": average(per_query),
        "rankings": rankings,
        "mode": "lexical_dry_run" if args.dry_run else "model",
    }
    print(json.dumps(result["metrics"], indent=2))
    if args.output:
        dump_json(args.output, result)


if __name__ == "__main__":
    main()
