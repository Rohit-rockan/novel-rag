#!/usr/bin/env python3
"""
semantic_search.py — Stage 1b (semantic path): meaning-based retrieval
against the vector store built by embed_library.py.

This is the semantic counterpart to retrieve_candidates.py (which does
keyword-category retrieval against keyword-index.json). Use this when what
you need doesn't map cleanly to a fixed category — "a tense standoff
between former allies, mask slipping" isn't a keyword search, it's a
meaning search.

Both scripts can be combined: run this first for a semantically-ranked
list, then cross-reference against keyword-index.json's top_categories for
a hybrid filter (e.g. semantically similar AND tagged fight_scene).

Output of this script is a shortlist — same role as retrieve_candidates.py's
output. Only chapters that come back here should go on to the `tagger`
subagent for real structured tagging, per the project's cost guardrail.

Requires:
  VOYAGE_API_KEY environment variable set (only the query gets embedded live;
  chapter vectors are already precomputed in library/vector-index.db)

Usage:
  python semantic_search.py --library-path ./library \\
      --query "a tense standoff between former allies, mask slipping" \\
      --top-n 15
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def load_db(db_path: Path):
    if not db_path.exists():
        print(f"ERROR: {db_path} not found. Run embed_library.py first.", file=sys.stderr)
        sys.exit(1)
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def embed_query(query: str, model: str):
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    import voyageai
    client = voyageai.Client(api_key=api_key)
    result = client.embed([query], model=model, input_type="query")
    return result.embeddings[0]


def search(db, query_vec, top_n: int):
    import sqlite_vec
    # vec0's KNN planner needs an explicit `k = ?` on the raw vector scan —
    # a trailing LIMIT on the joined query isn't enough for it to recognize.
    rows = db.execute(
        """
        WITH matches AS (
            SELECT rowid, distance
            FROM chapter_vectors
            WHERE embedding MATCH ? AND k = ?
        )
        SELECT c.book, c.chapter, c.chunk_index, c.word_count, m.distance
        FROM matches m
        JOIN chapters c ON c.id = m.rowid
        ORDER BY m.distance
        """,
        (sqlite_vec.serialize_float32(query_vec), top_n),
    ).fetchall()
    return rows


def load_keyword_categories(library_path: Path, book: str, chapter: str):
    """Best-effort cross-reference against the free keyword index, for context only."""
    index_path = library_path / "keyword-index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index.get(book, []):
        if entry["chapter"] == chapter:
            return [c["category"] for c in entry.get("top_categories", [])]
    return []


def main():
    parser = argparse.ArgumentParser(description="Semantic (embedding-based) chapter search.")
    parser.add_argument("--library-path", type=Path, default=Path("library"))
    parser.add_argument("--query", required=True, help="Plain-language description of what the beat needs.")
    parser.add_argument("--model", default="voyage-4-lite")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    db_path = args.library_path / "vector-index.db"
    db = load_db(db_path)

    query_vec = embed_query(args.query, args.model)
    results = search(db, query_vec, args.top_n)

    output = []
    for book, chapter, chunk_index, word_count, distance in results:
        categories = load_keyword_categories(args.library_path, book, chapter)
        output.append({
            "book": book,
            "chapter": chapter,
            "chunk_index": chunk_index,  # 0 unless the chapter was oversized and split
            "file": str(args.library_path / "extracted" / book / f"{chapter}.md"),
            "word_count": word_count,
            "similarity_distance": round(distance, 4),
            "keyword_categories": categories,
        })

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
