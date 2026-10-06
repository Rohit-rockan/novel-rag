#!/usr/bin/env python3
"""
evaluate_retrieval.py — Stage 1b (evaluation): measures retrieval quality
(Hit@K, Recall@K, MRR) across dense-only, sparse-only (BM25), and hybrid
(RRF-fused) modes, against a synthetic ("silver") query set and/or a
hand-labeled ("gold") one.

Separates retrieval quality from generation quality on purpose: if the
right chapter never enters the candidate pool, nothing downstream (writer,
beat-classifier) can use it, no matter how good the prose generation is.

Zero additional API cost beyond whatever generate_synthetic_queries.py
already spent to build its query set — this script only runs retrieval
(dense embeddings reuse semantic_search.py's embed_query call per query,
BM25 is local/free).

Usage:
  python evaluate_retrieval.py --library-path ./library \\
      --queryset library/eval/synthetic-queries.json \\
      --queryset library/eval/gold-queries.json \\
      --k 5,10,20 --modes dense,sparse,hybrid --pool-size 50
"""
import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_search import load_db, embed_query, search as dense_search  # noqa: E402
from hybrid_search import lexical_search, rrf_fuse  # noqa: E402

CHAPTER_NUM_RE = re.compile(r"(\d+)")


def resolve_rowids(db: sqlite3.Connection, book: str, chapter) -> list[int]:
    """Match a target's (book, chapter) against the chapters table's rows.
    `chapter` may be an int (as stored in library/tags/*.json) or a heading
    string (as stored in chapters.chapter) — compare by parsed chapter number."""
    chapter_num = int(chapter) if isinstance(chapter, int) else None
    if chapter_num is None:
        m = CHAPTER_NUM_RE.search(str(chapter))
        chapter_num = int(m.group(1)) if m else None

    rows = db.execute("SELECT id, chapter FROM chapters WHERE book=?", (book,)).fetchall()
    matches = []
    for row_id, heading in rows:
        m = CHAPTER_NUM_RE.search(heading)
        if m and chapter_num is not None and int(m.group(1)) == chapter_num:
            matches.append(row_id)
    return matches


def dense_rowids_for(db, query_vec, pool_size: int) -> list[int]:
    rows = dense_search(db, query_vec, pool_size)
    rowids = []
    for book, chapter, chunk_index, _wc, _dist in rows:
        r = db.execute(
            "SELECT id FROM chapters WHERE book=? AND chapter=? AND chunk_index=?",
            (book, chapter, chunk_index),
        ).fetchone()
        if r:
            rowids.append(r[0])
    return rowids


def ranked_rowids_for_mode(db, query_text: str, model: str, pool_size: int, rrf_k: int, mode: str) -> list[int]:
    if mode == "dense":
        query_vec = embed_query(query_text, model)
        return dense_rowids_for(db, query_vec, pool_size)
    if mode == "sparse":
        return lexical_search(db, query_text, pool_size)
    if mode == "hybrid":
        query_vec = embed_query(query_text, model)
        dense_ids = dense_rowids_for(db, query_vec, pool_size)
        lexical_ids = lexical_search(db, query_text, pool_size)
        fused = rrf_fuse({"dense": dense_ids, "lexical": lexical_ids}, rrf_k)
        return [rowid for rowid, _info in sorted(fused.items(), key=lambda kv: kv[1]["rrf_score"], reverse=True)]
    raise ValueError(f"unknown mode: {mode}")


def hit_at_k(ranked: list[int], relevant: set[int], k: int) -> int:
    return 1 if set(ranked[:k]) & relevant else 0


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: list[int], relevant: set[int]) -> float:
    for i, rowid in enumerate(ranked, start=1):
        if rowid in relevant:
            return 1.0 / i
    return 0.0


def load_queryset(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (skipping {path} — not found)")
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_relevant(entry: dict) -> list[dict]:
    """Synthetic entries have a single `target`; gold entries may have a `relevant` list."""
    if "relevant" in entry:
        return entry["relevant"]
    if entry.get("target"):
        return [entry["target"]]
    return []


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (Hit@K/Recall@K/MRR) across dense/sparse/hybrid.")
    parser.add_argument("--library-path", type=Path, default=Path("library"))
    parser.add_argument("--queryset", action="append", default=[], help="Repeatable. Path to a query-set JSON file.")
    parser.add_argument("--k", default="5,10,20")
    parser.add_argument("--modes", default="dense,sparse,hybrid")
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model", default="voyage-4-lite")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ks = [int(x) for x in args.k.split(",")]
    modes = args.modes.split(",")
    db = load_db(args.library_path / "vector-index.db")

    all_rows = []
    for qs_path_str in args.queryset:
        qs_path = Path(qs_path_str)
        label = "gold" if "gold" in qs_path.stem else "silver"
        entries = load_queryset(qs_path)
        for entry in entries:
            relevant_targets = normalize_relevant(entry)
            relevant_rowids = set()
            for t in relevant_targets:
                relevant_rowids.update(resolve_rowids(db, t["book"], t["chapter"]))
            if not relevant_rowids:
                continue
            for text_field, variant in (("query_text", "plain"), ("masked_query_text", "masked")):
                query_text = entry.get(text_field)
                if not query_text:
                    continue
                all_rows.append({
                    "query_id": entry.get("query_id"), "label": label, "variant": variant,
                    "query_text": query_text, "relevant_rowids": relevant_rowids,
                })

    if not all_rows:
        print("No evaluable query rows found (check --queryset paths and that targets resolve against chapters table).")
        return

    results_per_mode = {mode: {k: [] for k in ks} for mode in modes}
    mrr_per_mode = {mode: [] for mode in modes}
    per_query_records = []

    for row in all_rows:
        record = {"query_id": row["query_id"], "label": row["label"], "variant": row["variant"]}
        for mode in modes:
            ranked = ranked_rowids_for_mode(db, row["query_text"], args.model, args.pool_size, args.rrf_k, mode)
            for k in ks:
                results_per_mode[mode][k].append(hit_at_k(ranked, row["relevant_rowids"], k))
            mrr_per_mode[mode].append(mrr(ranked, row["relevant_rowids"]))
            record[f"{mode}_hit@{ks[-1]}"] = hit_at_k(ranked, row["relevant_rowids"], ks[-1])
            record[f"{mode}_mrr"] = round(mrr(ranked, row["relevant_rowids"]), 4)
        per_query_records.append(record)

    summary_lines = [f"Evaluated {len(all_rows)} query rows across modes: {modes}, k={ks}\n"]
    summary_table = ["| Mode | " + " | ".join(f"Hit@{k}" for k in ks) + " | MRR |",
                      "|---|" + "---|" * (len(ks) + 1)]
    for mode in modes:
        hit_avgs = [sum(results_per_mode[mode][k]) / len(results_per_mode[mode][k]) for k in ks]
        mrr_avg = sum(mrr_per_mode[mode]) / len(mrr_per_mode[mode])
        row_str = f"| {mode} | " + " | ".join(f"{h:.3f}" for h in hit_avgs) + f" | {mrr_avg:.3f} |"
        summary_table.append(row_str)
        print(row_str)

    out_path = args.out or Path(f"library/eval/results/{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(per_query_records, indent=2), encoding="utf-8")

    summary_path = out_path.parent / "summary.md"
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("\n".join(summary_lines))
        f.write("\n".join(summary_table) + "\n")

    print(f"\nWrote per-query results to {out_path}")
    print(f"Appended summary to {summary_path}")


if __name__ == "__main__":
    main()
