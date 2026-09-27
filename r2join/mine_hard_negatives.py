"""Build processed R2-Join files and mine rank-tiered hard negatives."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nltk
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


def ensure_fresh_dir(path: Path, allow_overwrite: bool) -> None:
    if path.exists():
        existing = [item.name for item in path.iterdir() if item.name != "logs"]
        if not existing:
            return
        if not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing output directory: {path}. "
                "Use a different data_dir or pass --allow-overwrite."
            )
        for item in path.iterdir():
            if item.name == "logs":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        path.mkdir(parents=True, exist_ok=False)


def ensure_nltk_tokenizer() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except (LookupError, OSError):
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


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

    value_counts = df[column_name].astype(str).value_counts()
    sorted_values = value_counts.index.tolist()
    if not sorted_values:
        return ""

    shown_values = ", ".join(sorted_values[:max_values])
    lengths = [len(str(value)) for value in sorted_values]
    avg_len = sum(lengths) / len(lengths)
    text = (
        f"{column_name} contains {len(sorted_values)} values "
        f"({max(lengths)}, {min(lengths)}, {avg_len}): {shown_values}"
    )
    try:
        tokens = nltk.word_tokenize(text)
        return " ".join(tokens[:512])
    except LookupError:
        return " ".join(text.split()[:512])


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
    gt_format = config.get("ground_truth_format", "opendata")
    if gt_format == "webtable":
        gt_df = pd.read_csv(gt_file)
    else:
        gt_df = pd.read_csv(gt_file, header=None)
    corpus = {}
    global_gt = defaultdict(set)
    edges = set()
    skipped = Counter()

    for table1, table2, col1, col2 in tqdm(list(iter_ground_truth_rows(gt_df, gt_format)), desc="Parse ground truth"):
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


def iter_ground_truth_rows(gt_df, gt_format: str):
    webtable_columns = {"query_table", "candidate_table", "query_column", "candidate_column"}
    if gt_format == "webtable" and webtable_columns.issubset(set(gt_df.columns)):
        for _, row in gt_df.iterrows():
            yield (
                str(row["query_table"]).strip(),
                str(row["candidate_table"]).strip(),
                str(row["query_column"]).strip(),
                str(row["candidate_column"]).strip(),
            )
    else:
        for _, row in gt_df.iterrows():
            if len(row) >= 4:
                yield (
                    str(row.iloc[0]).strip(),
                    str(row.iloc[1]).strip(),
                    str(row.iloc[2]).strip(),
                    str(row.iloc[3]).strip(),
                )


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
    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())
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
                    "negative": negative_texts[0],
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
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow replacing an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed or int(config.get("seed", 42))
    set_seed(seed)
    output_dir = Path(args.output_dir or config.get("data_dir", "processed/dataset"))
    allow_overwrite = args.allow_overwrite or bool(config.get("allow_overwrite", False))
    ensure_fresh_dir(output_dir, allow_overwrite)
    ensure_nltk_tokenizer()
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
        dump_json(output_dir / "train_edges_undirected.json", [list(edge) for edge in sorted(train_edges)])
        dump_json(output_dir / "val_edges_undirected.json", [list(edge) for edge in sorted(val_edges)])
        dump_json(output_dir / "test_edges_undirected.json", [list(edge) for edge in sorted(test_edges)])
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
        "split_method": "deduplicate ground truth as undirected edges, then split edges 80/10/10; labels are stored bidirectionally inside each split",
        "val_ratio": float(config.get("val_ratio", 0.1)),
        "test_ratio": float(config.get("test_ratio", 0.1)),
        "top_k_retrieval": args.top_k_retrieval or int(config.get("top_k_retrieval", 100)),
        "num_negs_per_query": args.num_negatives or int(config.get("num_negatives", 7)),
        "mining_model": args.mining_model or config.get("mining_model", DEFAULT_STAGE1_MODEL),
        "corpus_rows": len(corpus),
        "train_gt_queries": len(train_gt),
        "val_gt_queries": len(val_gt),
        "test_gt_queries": len(test_gt),
        "train_hard_negative_rows": len(hard_negative_rows),
        "skipped": {"prepare": dict(skipped), "mining": dict(mining_skipped)},
    }
    dump_json(output_dir / "split_manifest.json", manifest)
    print("Done.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
