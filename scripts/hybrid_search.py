#!/usr/bin/env python3
"""
hybrid_search.py — Stage 1b (hybrid path): fuses BM25 lexical retrieval
(build_lexical_index.py's chapter_fts) with dense semantic retrieval
(embed_library.py's chapter_vectors) via Reciprocal Rank Fusion.

Why: dense embeddings can blur past exact proper nouns, invented terms, or
character names ("Sunforge") that a reader would expect an exact match for;
pure keyword/category scoring (keyword_prefilter.py) can miss a conceptual
match with no shared vocabulary ("cold mountainous forest" vs "coniferous
woodland at high altitude"). Neither alone is reliable — this fuses both
rankings so a candidate strong in either shows up, and one strong in both
rises further.

This does NOT call any LLM for reranking. Every Claude-based judgment in
this pipeline runs through a subagent (tagger, writer, beat-classifier,
...), never a standalone script hitting the Messages API directly — keep
that boundary here too. The agent calling this script (beat-classifier) is
the one that makes any further judgment call on the shortlist.

Requires:
  VOYAGE_API_KEY (only the query gets embedded live; same as semantic_search.py)

Usage:
  python hybrid_search.py --library-path ./library \\
      --query "a tense standoff between former allies, mask slipping" \\
      --top-n 15 --boost-category emotional_beat --boost-category fight_scene
"""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_search import load_db, embed_query, search as dense_search, load_keyword_categories  # noqa: E402
from retrieve_candidates import already_tagged  # noqa: E402

WORD_RE = re.compile(r"[a-zA-Z']{3,}")


def lexical_search(db: sqlite3.Connection, query: str, pool_size: int):
    tokens = WORD_RE.findall(query.lower())
    if not tokens:
        return []
    match_expr = " OR ".join(f'"{t}"' for t in tokens)
    try:
        rows = db.execute(
            """
            SELECT rowid, bm25(chapter_fts) AS score
            FROM chapter_fts
            WHERE chapter_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match_expr, pool_size),
        ).fetchall()
    except sqlite3.OperationalError:
        # chapter_fts doesn't exist yet (build_lexical_index.py not run) — degrade to dense-only.
        return []
    return [rowid for rowid, _score in rows]


def rrf_fuse(ranked_lists: dict[str, list[int]], k: int) -> dict[int, dict]:
    """ranked_lists: {method_name: [rowid, rowid, ...]} in rank order (best first).
    Returns {rowid: {"rrf_score": float, "ranks": {method: 1-based-rank}}}."""
    fused: dict[int, dict] = {}
    for method, rowids in ranked_lists.items():
        for rank, rowid in enumerate(rowids, start=1):
            entry = fused.setdefault(rowid, {"rrf_score": 0.0, "ranks": {}})
            entry["ranks"][method] = rank
            entry["rrf_score"] += 1.0 / (k + rank)
    return fused


def main():
    parser = argparse.ArgumentParser(description="Hybrid (BM25 + dense, RRF-fused) chapter search.")
    parser.add_argument("--library-path", type=Path, default=Path("library"))
    parser.add_argument("--query", required=True, help="Plain-language description of what the beat needs.")
    parser.add_argument("--model", default="voyage-4-lite")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--pool-size", type=int, default=100, help="Candidates pulled per method before fusion.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant (standard default: 60).")
    parser.add_argument("--book", type=str, default=None, help="Restrict to books matching this substring.")
    parser.add_argument("--boost-category", action="append", default=[],
                         help="Repeatable. Chapters tagged with this keyword-index.json category get a soft RRF bonus.")
    parser.add_argument("--exclude-category", action="append", default=[],
                         help="Repeatable. Drop chapters tagged with this category from the results.")
    parser.add_argument("--exclude-book", action="append", default=[],
                         help="Repeatable substring. Drop chapters from matching books.")
    args = parser.parse_args()

    db_path = args.library_path / "vector-index.db"
    db = load_db(db_path)

    query_vec = embed_query(args.query, args.model)
    dense_rows = dense_search(db, query_vec, args.pool_size)
    dense_rowids = [
        db.execute(
            "SELECT id FROM chapters WHERE book=? AND chapter=? AND chunk_index=?",
            (book, chapter, chunk_index),
        ).fetchone()[0]
        for book, chapter, chunk_index, _word_count, _distance in dense_rows
    ]

    lexical_rowids = lexical_search(db, args.query, args.pool_size)

    fused = rrf_fuse({"dense": dense_rowids, "lexical": lexical_rowids}, args.rrf_k)

    results = []
    for rowid, info in fused.items():
        row = db.execute("SELECT book, chapter, chunk_index, word_count FROM chapters WHERE id=?", (rowid,)).fetchone()
        if row is None:
            continue
        book, chapter, chunk_index, word_count = row

        if args.book and args.book.lower() not in book.lower():
            continue
        if any(b.lower() in book.lower() for b in args.exclude_book):
            continue

        categories = load_keyword_categories(args.library_path, book, chapter)

        if any(cat in categories for cat in args.exclude_category):
            continue

        rrf_score = info["rrf_score"]
        boost_hits = [cat for cat in args.boost_category if cat in categories]
        if boost_hits:
            rrf_score += 1.0 / (args.rrf_k + 1)

        results.append({
            "book": book,
            "chapter": chapter,
            "chunk_index": chunk_index,
            "file": str(args.library_path / "extracted" / book / f"{chapter}.md"),
            "word_count": word_count,
            "already_tagged": already_tagged(args.library_path, book, chapter),
            "source": "hybrid",
            "rrf_score": round(rrf_score, 5),
            "dense_rank": info["ranks"].get("dense"),
            "bm25_rank": info["ranks"].get("lexical"),
            "boost_categories_matched": boost_hits,
            "keyword_categories": categories,
        })

    results.sort(key=lambda r: r["rrf_score"], reverse=True)
    print(json.dumps(results[: args.top_n], indent=2))


if __name__ == "__main__":
    main()
