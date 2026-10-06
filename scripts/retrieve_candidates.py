#!/usr/bin/env python3
"""
retrieve_candidates.py — Stage 1b retrieval. Reads keyword-index.json
(built for $0 by keyword_prefilter.py) and shortlists candidate chapters
for a specific beat's needs. NO API calls happen here either — this is
still free. The output of this script is what actually gets sent to
Claude (the tagger subagent) for real tagging (only the shortlist, not
the whole library).

Tag cache layout note: this project stores tags as ONE combined array file
per book — library/tags/{book-slug}.json, a JSON array of chapter objects
(each with a "chapter": <int> field) — not one file per chapter. This
script checks that combined file to see whether a given chapter number is
already cached.

Usage examples:
  # Find fight-scene candidates, ranked
  python retrieve_candidates.py --library-path ./library --category fight_scene --top-n 15

  # Find chapters strong in BOTH magic and emotional tone (e.g. a tense magical loss scene)
  python retrieve_candidates.py --library-path ./library --category magic_system --category emotional_beat --top-n 10

  # Restrict to one book (e.g. you want references only from a specific series)
  python retrieve_candidates.py --library-path ./library --category fight_scene --book "Book Title"
"""

import argparse
import json
import re
from pathlib import Path

CHAPTER_NUM_RE = re.compile(r"(\d+)")


def load_index(library_path: Path):
    index_path = library_path / "keyword-index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found. Run keyword_prefilter.py first — it's free and takes seconds."
        )
    return json.loads(index_path.read_text(encoding="utf-8"))


def combined_score(chapter_entry: dict, categories: list[str]) -> float:
    """Sum the chapter's scores across the requested categories."""
    all_scores = chapter_entry.get("all_scores", {})
    return sum(all_scores.get(cat, 0.0) for cat in categories)


def chapter_number(chapter_heading: str) -> int | None:
    m = CHAPTER_NUM_RE.search(chapter_heading)
    return int(m.group(1)) if m else None


_tag_cache: dict[str, set] = {}


def already_tagged(library_path: Path, book_title: str, chapter_heading: str) -> bool:
    """Check if this chapter is already in the book's combined tags/{book}.json cache."""
    if book_title not in _tag_cache:
        tag_path = library_path / "tags" / f"{book_title}.json"
        if tag_path.exists():
            try:
                chapters = json.loads(tag_path.read_text(encoding="utf-8"))
                _tag_cache[book_title] = {c.get("chapter") for c in chapters}
            except (json.JSONDecodeError, AttributeError):
                _tag_cache[book_title] = set()
        else:
            _tag_cache[book_title] = set()

    n = chapter_number(chapter_heading)
    return n in _tag_cache[book_title]


def retrieve(library_path: Path, categories: list[str], top_n: int, book_filter: str | None):
    index = load_index(library_path)
    candidates = []

    for book_title, chapters in index.items():
        if book_filter and book_filter.lower() not in book_title.lower():
            continue
        for entry in chapters:
            score = combined_score(entry, categories)
            if score <= 0:
                continue
            candidates.append({
                "book": book_title,
                "chapter": entry["chapter"],
                "score": round(score, 3),
                "word_count": entry["word_count"],
                "already_tagged": already_tagged(library_path, book_title, entry["chapter"]),
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Retrieve candidate chapters from the free keyword index.")
    parser.add_argument("--library-path", type=str, default="./library")
    parser.add_argument(
        "--category", action="append", required=True,
        help="Category to match (repeatable — e.g. --category fight_scene --category emotional_beat). "
             "Valid: fight_scene, monster_encounter, magic_system, strategy_planning, "
             "dialogue_heavy, emotional_beat, world_engineering",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Max candidates to return")
    parser.add_argument("--book", type=str, default=None, help="Optional: restrict to books matching this substring")
    args = parser.parse_args()

    library_path = Path(args.library_path)
    results = retrieve(library_path, args.category, args.top_n, args.book)

    if not results:
        print("No candidates found for these categories. Try broadening the category list.")
        return

    new_count = sum(1 for r in results if not r["already_tagged"])
    cached_count = len(results) - new_count

    print(f"Top {len(results)} candidates for {args.category}:")
    print(f"  {cached_count} already tagged (free to reuse) | {new_count} new (will cost tokens if tagged)\n")

    for r in results:
        flag = "cached" if r["already_tagged"] else "NEW"
        print(f"  [{flag:>6}] {r['score']:>7} | {r['book']} — {r['chapter']} ({r['word_count']} words)")

    print(f"\nNext step: send only the rows marked NEW above to the tagger subagent.")
    print("Already-tagged chapters can be pulled straight from library/tags/ at zero additional cost.")


if __name__ == "__main__":
    main()
