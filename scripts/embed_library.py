#!/usr/bin/env python3
"""
embed_library.py — Stage 1d: semantic embedding pass over the library.

What it does:
  Unlike keyword_prefilter.py (free, regex-only), this makes real API calls
  to Voyage AI to compute one embedding vector per chapter, and stores them
  in a local sqlite-vec database for meaning-based similarity search.

  This is NOT the expensive step — Voyage's current-gen models (voyage-4*)
  include 200M free tokens/account, and even past that, embeddings run
  $0.02-0.12 per million tokens. At this library's ~80M estimated tokens,
  full-library embedding costs $0-10 total, one time. That's why this runs
  upfront across the WHOLE library, unlike the tagger subagent (which is
  genuinely expensive per-chapter and stays on-demand/shortlist-only).

Output:
  library/vector-index.db     — sqlite-vec database, one row per chapter
  library/embedding-progress.json — resume tracking, like extract_all.py

Requires:
  pip install voyageai sqlite-vec
  VOYAGE_API_KEY environment variable set

Usage:
  python embed_library.py --library-path ./library
  python embed_library.py --library-path ./library --model voyage-4-lite
  python embed_library.py --library-path ./library --dry-run   # count + cost estimate, no API calls
"""
import argparse
import json
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

DEFAULT_MODEL = "voyage-4-lite"
BATCH_SIZE = 100  # texts per Voyage API call (standard rate limits)
REDUCED_BATCH_SIZE = 1  # no payment method on file yet: capped at 3 RPM / 10K TPM
# Binding constraint is TPM, not RPM: a single ~5-6K-token chapter already uses
# half the 10K/minute budget, so two back-to-back requests can exceed it even
# under 3 RPM pacing. Space requests a full minute+ apart instead.
REDUCED_DELAY_SECONDS = 65
# Voyage's voyage-4* models cap input at 32,000 tokens/text and silently
# truncate past that (default truncation=True) rather than erroring. A small
# number of source epubs (single-spine-item web-serial omnibus conversions)
# produce one giant chapter file well past that limit. Anything over this
# word count gets pre-split into overlapping chunks instead of being sent as
# one oversized text — see split_oversized().
WORD_THRESHOLD = 24000
CHUNK_WORDS = 18000
OVERLAP_WORDS = 200
EMBED_DIM = {
    "voyage-4-lite": 1024,
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "voyage-3.5-lite": 512,
    "voyage-3.5": 1024,
    "voyage-3-lite": 512,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
}


def iter_chapters(library_path: Path):
    """Yield (book_title, chapter_stem, chapter_path, text) for every chapter,
    same directory convention as keyword_prefilter.py."""
    extracted_path = library_path / "extracted"
    for book_entry in sorted(extracted_path.iterdir()):
        if not book_entry.is_dir():
            continue
        book_title = book_entry.stem
        for chapter_file in sorted(book_entry.glob("*.md")):
            text = chapter_file.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield book_title, chapter_file.stem, chapter_file, text


def split_oversized(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS):
    """Split into overlapping word-count chunks. Word-boundary only (not
    paragraph-aware) — fine for retrieval, not meant for display."""
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


def load_progress(progress_path: Path):
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"model": None, "done": {}}  # done: {book_title: [chapter_stem, ...]}


def save_progress(progress_path: Path, progress: dict):
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def init_db(db_path: Path, dim: int):
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id INTEGER PRIMARY KEY,
            book TEXT NOT NULL,
            chapter TEXT NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            word_count INTEGER,
            UNIQUE(book, chapter, chunk_index)
        )
    """)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chapter_vectors USING vec0(
            embedding float[{dim}]
        )
    """)
    db.commit()
    return db


def already_embedded(progress: dict, book: str, chapter: str) -> bool:
    return chapter in progress["done"].get(book, [])


def mark_embedded(progress: dict, book: str, chapter: str):
    progress["done"].setdefault(book, []).append(chapter)


def estimate_cost(total_tokens: int, model: str) -> float:
    per_million = {
        "voyage-4-lite": 0.02, "voyage-4": 0.06, "voyage-4-large": 0.12,
        "voyage-3.5-lite": 0.02, "voyage-3.5": 0.06,
        "voyage-3-lite": 0.02, "voyage-3": 0.06, "voyage-3-large": 0.18,
    }.get(model, 0.06)
    return (total_tokens / 1_000_000) * per_million


def main():
    parser = argparse.ArgumentParser(description="Embed library chapters into a local vector store (Voyage AI + sqlite-vec).")
    parser.add_argument("--library-path", type=Path, default=Path("library"))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(EMBED_DIM.keys()))
    parser.add_argument("--dry-run", action="store_true", help="Count chapters/tokens and estimate cost. No API calls.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N unembedded chapters (for golden-testing).")
    parser.add_argument("--reduced-limits", action="store_true",
                         help="Use small batches + slow pacing for accounts without a payment "
                              "method on file (Voyage caps these at 3 RPM / 10K TPM).")
    args = parser.parse_args()

    library_path = args.library_path
    progress_path = library_path / "embedding-progress.json"
    db_path = library_path / "vector-index.db"

    progress = load_progress(progress_path)
    if progress["model"] and progress["model"] != args.model:
        print(f"ERROR: existing progress file was built with model '{progress['model']}', "
              f"but you asked for '{args.model}'. Vectors from different models aren't "
              f"comparable — delete {progress_path} and {db_path} to switch models, or "
              f"rerun with --model {progress['model']}.", file=sys.stderr)
        sys.exit(1)
    progress["model"] = args.model

    pending = []
    total_words = 0
    oversized_count = 0
    for book, chapter, path, text in iter_chapters(library_path):
        if already_embedded(progress, book, chapter):
            continue
        pending.append((book, chapter, path, text))
        total_words += len(text.split())
        if len(text.split()) > WORD_THRESHOLD:
            oversized_count += 1

    est_tokens = int(total_words / 0.75)  # ~0.75 words/token for English prose
    est_cost = estimate_cost(est_tokens, args.model)

    print(f"Model: {args.model}")
    print(f"Chapters already embedded (resumed): {sum(len(v) for v in progress['done'].values())}")
    print(f"Chapters pending: {len(pending)}")
    if oversized_count:
        print(f"  ({oversized_count} exceed {WORD_THRESHOLD:,} words and will be split into multiple chunks)")
    print(f"Estimated tokens for pending chapters: {est_tokens:,}")
    print(f"Estimated cost: ${est_cost:.2f} (before any free-tier allowance)")

    if args.dry_run:
        print("\n--dry-run: no API calls made.")
        return

    if not pending:
        print("Nothing to do — all chapters already embedded.")
        return

    if args.limit:
        pending = pending[: args.limit]
        print(f"\n--limit set: processing only {len(pending)} chapters this run (golden-test mode).")

    # Expand into flat (book, chapter, chunk_index, text, is_last_chunk) units —
    # most chapters produce exactly one unit; oversized ones produce several.
    work_units = []
    expected_chunks = {}
    for book, chapter, path, text in pending:
        chunks = split_oversized(text)
        expected_chunks[(book, chapter)] = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            work_units.append((book, chapter, idx, chunk_text, idx == len(chunks) - 1))

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        print("\nERROR: VOYAGE_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it, then rerun. Never pass the key on the command line.", file=sys.stderr)
        sys.exit(1)

    import voyageai
    client = voyageai.Client(api_key=api_key)
    dim = EMBED_DIM[args.model]
    db = init_db(db_path, dim)

    batch_size = REDUCED_BATCH_SIZE if args.reduced_limits else BATCH_SIZE
    delay_seconds = REDUCED_DELAY_SECONDS if args.reduced_limits else 0.1

    processed_chapters = 0
    seen_chunks = {}
    for i in range(0, len(work_units), batch_size):
        batch = work_units[i:i + batch_size]
        texts = [text for _, _, _, text, _ in batch]

        try:
            result = client.embed(texts, model=args.model, input_type="document")
        except Exception as e:
            print(f"\nAPI error on batch starting at index {i}: {e}", file=sys.stderr)
            print("Progress saved up to this point — rerun the same command to resume.", file=sys.stderr)
            save_progress(progress_path, progress)
            sys.exit(1)

        import sqlite_vec
        for (book, chapter, chunk_index, text, is_last_chunk), embedding in zip(batch, result.embeddings):
            word_count = len(text.split())
            cur = db.execute(
                "INSERT OR IGNORE INTO chapters (book, chapter, chunk_index, word_count) VALUES (?, ?, ?, ?)",
                (book, chapter, chunk_index, word_count),
            )
            row_id = cur.lastrowid or db.execute(
                "SELECT id FROM chapters WHERE book = ? AND chapter = ? AND chunk_index = ?",
                (book, chapter, chunk_index),
            ).fetchone()[0]
            db.execute(
                "INSERT OR REPLACE INTO chapter_vectors (rowid, embedding) VALUES (?, ?)",
                (row_id, sqlite_vec.serialize_float32(embedding)),
            )
            seen_chunks[(book, chapter)] = seen_chunks.get((book, chapter), 0) + 1
            if seen_chunks[(book, chapter)] >= expected_chunks[(book, chapter)]:
                mark_embedded(progress, book, chapter)
                processed_chapters += 1

        db.commit()
        save_progress(progress_path, progress)
        print(f"  embedded {processed_chapters}/{len(pending)} chapters...", end="\r")
        time.sleep(delay_seconds)

    db.close()
    print(f"\nDone. Embedded {processed_chapters} chapters this run.")
    print(f"Vector store: {db_path}")
    print(f"Progress file: {progress_path}")


if __name__ == "__main__":
    main()
