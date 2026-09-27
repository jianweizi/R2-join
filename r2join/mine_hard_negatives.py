"""Build processed R2-Join files and mine rank-tiered hard negatives."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

try:
    from .common import DEFAULT_STAGE1_MODEL, corpus_to_maps, dump_json, load_config, load_json, set_seed
except ImportError:
    from common import DEFAULT_STAGE1_MODEL, corpus_to_maps, dump_json, load_config, load_json, set_seed


def edge_key(left: str, right: str):
    if left == right:
        return None
    return tuple(sorted((left, right)))


def build_ground_truth(edges):
    gt = defaultdict(set)
    for left, right in edges:
        gt[left].add(right)
        gt[right].add(left)
    return {qid: sorted(targets) for qid, targets in sorted(gt.items())}


def build_positive_pairs(edges, corpus):
    rows = []
    for left, right in sorted(edges):
        if left in corpus and right in corpus:
            rows.append({"query_col": left, "target_col": right, "query": corpus[left], "positive": corpus[right]})
            rows.append({"query_col": right, "target_col": left, "query": corpus[right], "positive": corpus[left]})
    return rows


def split_edges(edges, val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(edges)
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_test + n_val :], shuffled[n_test : n_test + n_val], shuffled[:n_test]


def read_column_text(file_path: Path, column_name: str, max_values: int) -> str:
    try:
        try:
            df = pd.read_csv(file_path, dtype=str, encoding="utf-8")
        except Exception:
            df = pd.read_csv(file_path, dtype=str, encoding="latin1")
    except Exception:
        return ""

    if column_name not in df.columns:
        stripped = {col.strip(): col for col in df.columns}
        if column_name.strip() not in stripped:
            return ""
        column_name = stripped[column_name.strip()]

    values = df[column_name].dropna().astype(str)
    if values.empty:
        return ""
    counts = values.value_counts()
    shown_values = ", ".join(counts.index.tolist()[:max_values])
    lengths = [len(str(value)) for value in counts.index.tolist()]
    avg_len = sum(lengths) / len(lengths)
    return (
        f"[Table] {file_path.name}; [Column] {column_name}; [Stats] "
        f"distinct={len(counts)}, max_len={max(lengths)}, min_len={min(lengths)}, avg_len={avg_len:.2f}; "
        f"[Values] {shown_values}"
    )


def resolve_table_path(table_name: str, raw_data_dirs: dict[str, str]) -> Path | None:
    table_name = table_name.strip()
    prefix = table_name.split("_")[0]
    candidate_dirs = []
    if prefix in raw_data_dirs:
        candidate_dirs.append(Path(raw_data_dirs[prefix]))
    candidate_dirs.extend(Path(path) for key, path in raw_data_dirs.items() if key != prefix)
    for directory in candidate_dirs:
        candidate = directory / table_name
        if candidate.exists():
            return candidate
    return None


def build_public_dataset(config: dict, output_dir: Path):
    gt_file = Path(config["ground_truth_file"])
    raw_data_dirs = config["raw_data_dirs"]
    max_values = int(config.get("max_values_per_column", 100))
    gt_df = pd.read_csv(gt_file, header=None)
    corpus = {}
    global_gt = defaultdict(set)
    edges = set()
    skipped = Counter()

    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Parse ground truth"):
        if len(row) < 4:
            skipped["short_ground_truth_row"] += 1
            continue
        table1, table2 = str(row[0]).strip(), str(row[1]).strip()
        col1, col2 = str(row[2]).strip(), str(row[3]).strip()
        id1 = f"{table1}::{col1}"
        id2 = f"{table2}::{col2}"
        key = edge_key(id1, id2)
        if key is None:
            skipped["self_edges"] += 1
            continue
        edges.add(key)
        global_gt[id1].add(id2)
        global_gt[id2].add(id1)

        for table_name, column_name, col_id in [(table1, col1, id1), (table2, col2, id2)]:
            if col_id in corpus:
                continue
            path = resolve_table_path(table_name, raw_data_dirs)
            if not path:
                skipped["missing_table_file"] += 1
                continue
            text = read_column_text(path, column_name, max_values)
            if not text:
                skipped["empty_or_unreadable_column"] += 1
                continue
            corpus[col_id] = text

    valid_edges = [(left, right) for left, right in edges if left in corpus and right in corpus]
    return corpus, valid_edges, global_gt, skipped


def build_existing_split_dataset(config: dict, output_dir: Path):
    source_dir = Path(config["source_data_dir"])
    corpus_rows = load_json(source_dir / "corpus.json")
    _, _, corpus = corpus_to_maps(corpus_rows)
    train_gt = load_json(source_dir / "train_ground_truth.json")
    val_gt = load_json(source_dir / "val_ground_truth.json")
    test_gt = load_json(source_dir / "test_ground_truth.json")
    global_gt = defaultdict(set)
    for gt in [train_gt, val_gt, test_gt]:
        for qid, targets in gt.items():
            for target in targets:
                global_gt[qid].add(target)

    dump_json(output_dir / "corpus.json", [{"id": col_id, "text": corpus[col_id]} for col_id in sorted(corpus)])
    dump_json(output_dir / "train_ground_truth.json", train_gt)
    dump_json(output_dir / "val_ground_truth.json", val_gt)
    dump_json(output_dir / "test_ground_truth.json", test_gt)
    for optional in ["train_positive_pairs.json", "val_positive_pairs.json", "test_positive_pairs.json"]:
        src = source_dir / optional
        if src.exists():
            dump_json(output_dir / optional, load_json(src))
    return corpus, train_gt, val_gt, test_gt, global_gt, Counter()


def sample_rank_tiered(candidate_ids: list[str], num_negatives: int, rng: random.Random):
    if len(candidate_ids) <= num_negatives:
        return list(candidate_ids)
    selected = []
    selected.extend(candidate_ids[:3])
    if len(candidate_ids) > 3:
        selected.extend(rng.sample(candidate_ids[3:20], min(3, len(candidate_ids[3:20]))))
    if len(candidate_ids) > 20 and len(selected) < num_negatives:
        selected.extend(rng.sample(candidate_ids[20:], 1))
    while len(selected) < num_negatives:
        remaining = list(set(candidate_ids) - set(selected))
        if not remaining:
            break
        selected.append(rng.choice(remaining))
    return selected[:num_negatives]


def mine_hard_negatives(corpus, train_gt, global_gt, args, config):
    model_name = args.mining_model or config.get("mining_model", DEFAULT_STAGE1_MODEL)
    top_k = args.top_k_retrieval or int(config.get("top_k_retrieval", 100))
    num_negatives = args.num_negatives or int(config.get("num_negatives", 7))
    seed = args.seed or int(config.get("seed", 42))
    rng = random.Random(seed)

    model = SentenceTransformer(model_name)
    corpus_ids = sorted(corpus)
    corpus_texts = [corpus[col_id] for col_id in corpus_ids]
    print(f"Encoding {len(corpus_ids)} columns with mining model: {model_name}")
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=int(config.get("mining_batch_size", 64)),
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    rows = []
    skipped = Counter()
    for qid in tqdm(sorted(train_gt), desc="Mine hard negatives"):
        if qid not in corpus:
            skipped["query_missing_from_corpus"] += 1
            continue
        positives = [pid for pid in train_gt[qid] if pid in corpus]
        excluded = {qid} | set(global_gt.get(qid, set()))
        q_emb = model.encode(corpus[qid], convert_to_tensor=True, normalize_embeddings=True)
        hits = util.semantic_search(q_emb, corpus_embeddings, top_k=min(len(corpus_ids), top_k + 10))[0]
        candidate_ids = []
        for hit in hits:
            cid = corpus_ids[hit["corpus_id"]]
            if cid not in excluded:
                candidate_ids.append(cid)
        sampled_ids = sample_rank_tiered(candidate_ids, num_negatives, rng)
        if not sampled_ids:
            skipped["query_without_negatives"] += 1
            continue
        negative_texts = [corpus[nid] for nid in sampled_ids]
        for pos_id in positives:
            rows.append(
                {
                    "query": corpus[qid],
                    "query_col": qid,
                    "positive": corpus[pos_id],
                    "positive_col": pos_id,
                    "label": 1,
                    "negatives_list": negative_texts,
                }
            )
    return rows, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare R2-Join data and mine hard negatives.")
    parser.add_argument("--config", required=True, help="JSON config for the dataset.")
    parser.add_argument("--output-dir", help="Processed output directory.")
    parser.add_argument("--mining-model", help="SentenceTransformer used only for hard-negative mining.")
    parser.add_argument("--top-k-retrieval", type=int)
    parser.add_argument("--num-negatives", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-mining", action="store_true", help="Only write split files; do not mine negatives.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed or int(config.get("seed", 42))
    set_seed(seed)
    output_dir = Path(args.output_dir or config.get("data_dir", "processed/dataset"))
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = config.get("prepare_mode", "lakebench_csv")

    if mode == "existing_split":
        corpus, train_gt, val_gt, test_gt, global_gt, skipped = build_existing_split_dataset(config, output_dir)
    elif mode == "lakebench_csv":
        corpus, valid_edges, global_gt, skipped = build_public_dataset(config, output_dir)
        val_ratio = float(config.get("val_ratio", 0.1))
        test_ratio = float(config.get("test_ratio", 0.1))
        train_edges, val_edges, test_edges = split_edges(valid_edges, val_ratio, test_ratio, seed)
        train_gt = build_ground_truth(train_edges)
        val_gt = build_ground_truth(val_edges)
        test_gt = build_ground_truth(test_edges)
        dump_json(output_dir / "corpus.json", [{"id": col_id, "text": corpus[col_id]} for col_id in sorted(corpus)])
        dump_json(output_dir / "train_ground_truth.json", train_gt)
        dump_json(output_dir / "val_ground_truth.json", val_gt)
        dump_json(output_dir / "test_ground_truth.json", test_gt)
        dump_json(output_dir / "train_positive_pairs.json", build_positive_pairs(train_edges, corpus))
        dump_json(output_dir / "val_positive_pairs.json", build_positive_pairs(val_edges, corpus))
        dump_json(output_dir / "test_positive_pairs.json", build_positive_pairs(test_edges, corpus))
    else:
        raise ValueError(f"Unknown prepare_mode: {mode}")

    if args.skip_mining:
        hard_negative_rows = []
        mining_skipped = Counter({"skipped_by_user": 1})
    else:
        hard_negative_rows, mining_skipped = mine_hard_negatives(corpus, train_gt, global_gt, args, config)
        dump_json(output_dir / "train_pairs_hard_negatives.json", hard_negative_rows)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepare_mode": mode,
        "output_dir": str(output_dir),
        "seed": seed,
        "corpus_rows": len(corpus),
        "train_gt_queries": len(train_gt),
        "val_gt_queries": len(val_gt),
        "test_gt_queries": len(test_gt),
        "train_hard_negative_rows": len(hard_negative_rows),
        "skipped": {"prepare": dict(skipped), "mining": dict(mining_skipped)},
    }
    dump_json(output_dir / "split_manifest.json", manifest)
    print("Done.")
    print(manifest)


if __name__ == "__main__":
    main()
