"""Shared utilities for the R2-Join artifact code."""

from __future__ import annotations

import json
import os
import random
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


DEFAULT_STAGE1_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_STAGE2_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def config_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def corpus_to_maps(corpus_rows: Iterable[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, str]]:
    corpus_ids = []
    corpus_texts = []
    id_to_text = {}
    for row in corpus_rows:
        col_id = str(row["id"])
        text = str(row["text"])
        corpus_ids.append(col_id)
        corpus_texts.append(text)
        id_to_text[col_id] = text
    return corpus_ids, corpus_texts, id_to_text


def normalize_pair_row(row: dict[str, Any], id_to_text: dict[str, str]) -> dict[str, Any] | None:
    query_id = row.get("query_col") or row.get("query_id") or row.get("query")
    positive_id = row.get("positive_col") or row.get("positive_id") or row.get("pos_id")

    query_text = row.get("query")
    positive_text = row.get("positive")
    if query_id in id_to_text:
        query_text = id_to_text[query_id]
    if positive_id in id_to_text:
        positive_text = id_to_text[positive_id]

    negative_texts = row.get("negatives_list") or row.get("hard_negatives") or row.get("negatives") or []
    resolved_negatives = []
    for neg in negative_texts:
        resolved_negatives.append(id_to_text.get(neg, neg))

    if not query_text or not positive_text or not resolved_negatives:
        return None

    return {
        "query": query_text,
        "query_col": query_id,
        "positive": positive_text,
        "positive_col": positive_id,
        "label": row.get("label", 1),
        "negatives_list": resolved_negatives,
    }


class FeatureExtractor:
    """Deterministic text feature extractor used by the Stage II reranker."""

    def __init__(self, feature_dim: int = 32):
        if feature_dim <= 4:
            raise ValueError("feature_dim must be greater than 4")
        self.feature_dim = feature_dim

    def extract(self, text: str) -> np.ndarray:
        features = np.zeros(self.feature_dim, dtype=np.float32)
        if not text:
            return features

        features[0] = len(text) / 500.0
        features[1] = text.count(" ") / 50.0
        features[2] = text.count(",") / 20.0
        features[3] = text.count("0") / 10.0

        if len(text) > 3:
            grams = [text[i : i + 3] for i in range(len(text) - 2)]
            for gram in grams[:200]:
                hashed = zlib.crc32(gram.encode("utf-8")) & 0xFFFFFFFF
                idx = (hashed % (self.feature_dim - 4)) + 4
                features[idx] += 1.0

        return np.log1p(features)


class MultimodalCrossEncoder(nn.Module):
    """Cross-encoder reranker with auxiliary deterministic text features."""

    def __init__(self, model_name_or_path: str, feature_dim: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        fusion_dim = self.config.hidden_size + 16
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, features):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]
        feat_emb = self.feature_mlp(features)
        return self.classifier(torch.cat([cls_emb, feat_emb], dim=1))

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str | Path, base_model_name: str, feature_dim: int):
        model = cls(base_model_name, feature_dim)
        state_dict = torch.load(Path(path) / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model


def calculate_metrics(ranked_list: list[dict[str, Any]], true_set: set[str], k_list: list[int]):
    metrics = {
        k: {"acc": 0.0, "precision": 0.0, "recall": 0.0, "mrr": 0.0, "map": 0.0, "ndcg": 0.0}
        for k in k_list
    }
    hit_indices = [idx for idx, item in enumerate(ranked_list) if item["id"] in true_set]

    for k in k_list:
        current_hits = [idx for idx in hit_indices if idx < k]
        if not current_hits:
            continue
        metrics[k]["acc"] = 1.0
        metrics[k]["precision"] = len(current_hits) / k
        metrics[k]["recall"] = len(current_hits) / len(true_set)
        metrics[k]["mrr"] = 1.0 / (current_hits[0] + 1)

        ap = 0.0
        for i, rank_idx in enumerate(current_hits):
            ap += (i + 1) / (rank_idx + 1)
        metrics[k]["map"] = ap / min(k, len(true_set))

        dcg = sum(1.0 / np.log2(rank_idx + 2) for rank_idx in current_hits)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true_set))))
        metrics[k]["ndcg"] = dcg / idcg if idcg else 0.0
    return metrics
