#!/usr/bin/env python3
"""
build_lexical_index.py — Stage 1b (lexical path): adds a BM25/FTS5 index to
the existing library/vector-index.db, rowid-aligned with the `chapters` table
that embed_library.py already built.

What it does:
  SQLite's FTS5 extension gives real BM25 ranking over chapter text — exact
  lexical matches (character names, invented terms, proper nouns) that dense
  embeddings can blur past. This is the missing half of hybrid retrieval:
  hybrid_search.py fuses this lexical ranking with the existing semantic
  ranking via Reciprocal Rank Fusion.

  Entirely local, zero API calls, zero cost — same category as
  keyword_prefilter.py, not embed_library.py. Re-scans and indexes whatever
  is missing every run; no progress file needed.

  The FTS table is CONTENTLESS (content='') — it stores only the inverted
  index, not a second copy of 21,000+ chapters' raw text. Row text is read
  fresh from library/extracted/ at index time and discarded after indexing.

Requires:
  Nothing new — uses Python's built-in sqlite3 (FTS5 must be compiled in;
  verified present in this environment).

Usage:
  python build_lexical_index.py --library-path ./library              # index everything missing
  python build_lexical_index.py --library-path ./library --book "some-title"  # one book only (golden-test)
  python build_lexical_index.py --library-path ./library --limit 50   # first 50 unindexed chunks only
  python build_lexical_index.py --library-path ./library --rebuild    # drop + recreate (e.g. tokenizer change)
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_library import split_oversized, WORD_THRESHOLD  # noqa: E402

COMMIT_EVERY = 500


def create_fts_table(db: sqlite3.Connection, rebuild: bool):
    if rebuild:
        db.execute("DROP TABLE IF EXISTS chapter_fts")
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chapter_fts USING fts5(
            body,
            content='',
            tokenize='porter unicode61 remove_diacritics 2'
        )
    """)
    db.commit()


def find_unindexed(db: sqlite3.Connection, book_filter: str | None):
    query = """
        SELECT c.id, c.book, c.chapter, c.chunk_index
        FROM chapters c
        LEFT JOIN chapter_fts f ON f.rowid = c.id
        WHERE f.rowid IS NULL
    """
    params = ()
    if book_filter:
        query += " AND c.book LIKE ?"
        params = (f"%{book_filter}%",)
    query += " ORDER BY c.book, c.chapter, c.chunk_index"
    return db.execute(query, params).fetchall()


def chunk_text_for(library_path: Path, book: str, chapter: str, chunk_index: int) -> str | None:
    """Reproduce the exact text span embed_library.py embedded for this
    (book, chapter, chunk_index), so the lexical and dense indexes cover
    identical spans."""
    chapter_path = library_path / "extracted" / book / f"{chapter}.md"
    if not chapter_path.exists():
        return None
    text = chapter_path.read_text(encoding="utf-8", errors="ignore")
    if len(text.split()) <= WORD_THRESHOLD and chunk_index == 0:
        return text
    chunks = split_oversized(text)
    if chunk_index >= len(chunks):
        return None
    return chunks[chunk_index]


def main():
    parser = argparse.ArgumentParser(description="Build a BM25/FTS5 lexical index alongside the existing vector index.")
    parser.add_argument("--library-path", type=Path, default=Path("library"))
    parser.add_argument("--book", type=str, default=None, help="Only index chunks whose book matches this substring (golden-test).")
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N missing chunks this run.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate chapter_fts (e.g. after a tokenizer change).")
    args = parser.parse_args()

    db_path = args.library_path / "vector-index.db"
    if not db_path.exists():
        print(f"ERROR: {db_path} not found. Run embed_library.py first.", file=sys.stderr)
        sys.exit(1)

    db = sqlite3.connect(str(db_path))
    create_fts_table(db, args.rebuild)

    missing = find_unindexed(db, args.book)
    if args.limit:
        missing = missing[: args.limit]

    if not missing:
        print("Nothing to do — all matching chunks already lexically indexed.")
        db.close()
        return

    print(f"Indexing {len(missing)} chunk(s)...")
    indexed = 0
    skipped = 0
    for i, (row_id, book, chapter, chunk_index) in enumerate(missing):
        text = chunk_text_for(args.library_path, book, chapter, chunk_index)
        if text is None:
            skipped += 1
            continue
        db.execute("INSERT INTO chapter_fts (rowid, body) VALUES (?, ?)", (row_id, text))
        indexed += 1
        if (i + 1) % COMMIT_EVERY == 0:
            db.commit()
            print(f"  indexed {i + 1}/{len(missing)}...", end="\r")

    db.commit()
    db.close()

    print(f"\nDone. Indexed {indexed} chunk(s) this run" + (f", skipped {skipped} (source file missing/mismatched)." if skipped else "."))
    print("0 API calls, $0.")


if __name__ == "__main__":
    main()
