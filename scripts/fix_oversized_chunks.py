#!/usr/bin/env python3
"""
fix_oversized_chunks.py — one-time repair for chapters that exceeded Voyage's
32,000-token input cap and got silently truncated by embed_library.py's
default truncation=True behavior.

Background: extract_book.py splits epubs by their internal spine items. A
handful of source epubs (mostly single-file web-serial omnibus conversions)
have only ONE spine item for the entire book, so they landed in the library
as one giant chapter-NN.md instead of many small ones. embed_library.py
embedded these as single chunks, and Voyage truncated anything past ~24,000
words (~32K tokens) rather than erroring — so those chapters were only
partially represented in the vector store, with no visible failure.

This script finds any embedded chapter over WORD_THRESHOLD, re-splits it
into overlapping word-count chunks (paragraph-boundary-naive — good enough
for retrieval, not meant for display), deletes its old single truncated
vector row, and re-embeds each chunk as its own row (same book+chapter,
distinct chunk_index). Migrates the chapters table schema to support
multiple chunks per chapter (adds chunk_index, relaxes the old
UNIQUE(book, chapter) constraint to UNIQUE(book, chapter, chunk_index)).

Usage:
  python fix_oversized_chunks.py --library-path ./library --dry-run
  python fix_oversized_chunks.py --library-path ./library
"""
import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

WORD_THRESHOLD = 24000   # ~32K tokens at ~0.75 words/token — Voyage's actual cutoff
CHUNK_WORDS = 18000      # leaves headroom under the 24K/32K-token line
OVERLAP_WORDS = 200      # small overlap so a scene split across chunks isn't orphaned
MODEL = "voyage-4-lite"


def migrate_schema(db):
    cols = [r[1] for r in db.execute("PRAGMA table_info(chapters)").fetchall()]
    if "chunk_index" in cols:
        return  # already migrated
    print("Migrating chapters table to support multiple chunks per chapter...")
    db.execute("ALTER TABLE chapters RENAME TO chapters_old")
    db.execute("""
        CREATE TABLE chapters (
            id INTEGER PRIMARY KEY,
            book TEXT NOT NULL,
            chapter TEXT NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            word_count INTEGER,
            UNIQUE(book, chapter, chunk_index)
        )
    """)
    db.execute("""
        INSERT INTO chapters (id, book, chapter, chunk_index, word_count)
        SELECT id, book, chapter, 0, word_count FROM chapters_old
    """)
    db.execute("DROP TABLE chapters_old")
    db.commit()


def split_oversized(text: str, chunk_words: int, overlap_words: int):
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-path", type=Path, default=Path("library"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = args.library_path / "vector-index.db"
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    oversized = db.execute(
        "SELECT id, book, chapter, word_count FROM chapters WHERE word_count > ?",
        (WORD_THRESHOLD,),
    ).fetchall()

    print(f"Found {len(oversized)} oversized chapters (>{WORD_THRESHOLD:,} words).")
    for row_id, book, chapter, word_count in oversized:
        pct = round(CHUNK_WORDS / word_count * 100, 1) if word_count else 100
        print(f"  {book}/{chapter}: {word_count:,} words (was ~{pct}% embedded)")

    if args.dry_run or not oversized:
        print("\n--dry-run: no changes made." if args.dry_run else "Nothing to fix.")
        return

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    import voyageai
    client = voyageai.Client(api_key=api_key)

    migrate_schema(db)

    for row_id, book, chapter, word_count in oversized:
        chapter_path = args.library_path / "extracted" / book / f"{chapter}.md"
        text = chapter_path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_oversized(text, CHUNK_WORDS, OVERLAP_WORDS)
        print(f"\n{book}/{chapter}: splitting into {len(chunks)} chunks...")

        # Delete the old single (truncated) row + its vector.
        old_id = db.execute(
            "SELECT id FROM chapters WHERE book = ? AND chapter = ? AND chunk_index = 0",
            (book, chapter),
        ).fetchone()
        if old_id:
            db.execute("DELETE FROM chapter_vectors WHERE rowid = ?", (old_id[0],))
            db.execute("DELETE FROM chapters WHERE id = ?", (old_id[0],))

        result = client.embed(chunks, model=MODEL, input_type="document")
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, result.embeddings)):
            cw = len(chunk_text.split())
            cur = db.execute(
                "INSERT INTO chapters (book, chapter, chunk_index, word_count) VALUES (?, ?, ?, ?)",
                (book, chapter, idx, cw),
            )
            db.execute(
                "INSERT INTO chapter_vectors (rowid, embedding) VALUES (?, ?)",
                (cur.lastrowid, sqlite_vec.serialize_float32(embedding)),
            )
        db.commit()
        time.sleep(0.2)

    db.close()
    print(f"\nDone. Re-embedded {len(oversized)} chapters as properly-sized chunks.")


if __name__ == "__main__":
    main()
