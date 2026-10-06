#!/usr/bin/env python3
"""
generate_synthetic_queries.py — Stage 1b (evaluation): builds a "silver"
query set for evaluate_retrieval.py by sampling already-tagged chapters as
ground truth and generating paraphrase/lexically-masked queries + a hard
negative per sample.

Cost note (read before running for real): this is the ONE script in this
repo that calls the Anthropic API directly rather than going through a
Claude Code subagent — every other Claude-based step here (tagger, writer,
beat-classifier, ...) runs as a subagent instead. That's deliberate: this
is a bounded, non-interactive batch job (same shape as embed_library.py
calling Voyage directly), not an editorial-judgment task. It still costs
real tokens. ALWAYS run --dry-run first and get explicit go-ahead on the
sample size/cost before a real run — this script will refuse to make API
calls without --n explicitly set alongside a dry-run already having been
reviewed (there's no bypass flag; re-running with the same --n after a
dry-run is the explicit confirmation).

Requires:
  pip install anthropic
  ANTHROPIC_API_KEY environment variable set

Usage:
  python generate_synthetic_queries.py --library-path ./library --n 50 --dry-run
  python generate_synthetic_queries.py --library-path ./library --n 50 --seed 42
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

OUTPUT_PATH = Path("library/eval/synthetic-queries.json")
USAGE_LOG = Path("library/usage-log.md")

# Rough per-model cost table, mirrors the style of embed_library.py's estimate_cost().
# Sonnet pricing: $3/$15 per million input/output tokens (approximate, for estimation only).
COST_PER_MILLION = {
    "claude-sonnet-5": {"input": 3.0, "output": 15.0},
}

PROMPT_TEMPLATE = """You are generating a retrieval-evaluation query pair for a RAG system over a \
fiction library. Given the chapter excerpt below, produce JSON with exactly these fields:

- "paraphrase_query": a plain-language description of a specific scene/moment in this excerpt, \
written the way a user would phrase what they're looking for (not a summary of the whole chapter).
- "masked_query": the same query, but with distinctive proper nouns, invented terminology, and \
unusual vocabulary replaced with generic equivalents (so it tests whether semantic retrieval \
generalizes past exact string overlap, not whether it recognizes a name).

Chapter excerpt (book: {book}, chapter: {chapter}):
---
{excerpt}
---

Respond with ONLY the JSON object, no other text."""


def load_tagged_chapters(library_path: Path) -> list[dict]:
    """Ground truth requires structured content — only sample from chapters
    already in library/tags/*.json."""
    tags_dir = library_path / "tags"
    if not tags_dir.exists():
        return []
    samples = []
    for tag_file in sorted(tags_dir.glob("*.json")):
        book = tag_file.stem
        try:
            chapters = json.loads(tag_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for entry in chapters:
            samples.append({"book": book, "chapter": entry.get("chapter")})
    return samples


def load_categories_for(library_path: Path, book: str, chapter_num: int) -> list[str]:
    index_path = library_path / "keyword-index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index.get(book, []):
        heading = entry.get("chapter", "")
        if str(chapter_num) in heading:
            return [c["category"] for c in entry.get("top_categories", [])]
    return []


def pick_hard_negative(library_path: Path, book: str, chapter_num: int, categories: list[str], pool: list[dict]):
    """Same top_category, different chapter — a distractor with similar vocabulary."""
    if not categories:
        return None
    candidates = [
        s for s in pool
        if not (s["book"] == book and s["chapter"] == chapter_num)
        and set(load_categories_for(library_path, s["book"], s["chapter"])) & set(categories)
    ]
    return random.choice(candidates) if candidates else None


def estimate_cost(n: int, avg_words: int, model: str) -> tuple[int, float]:
    # Rough: ~1.3 tokens/word in, ~150 tokens out per call.
    input_tokens = n * int(avg_words * 1.3)
    output_tokens = n * 150
    rates = COST_PER_MILLION.get(model, COST_PER_MILLION["claude-sonnet-5"])
    cost = (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]
    return input_tokens + output_tokens, cost


def find_chapter_file(library_path: Path, book: str, chapter_num) -> Path | None:
    book_dir = library_path / "extracted" / book
    if not book_dir.exists():
        return None
    for f in book_dir.glob("*.md"):
        if str(chapter_num) in f.stem:
            return f
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic (silver) retrieval-evaluation query set.")
    parser.add_argument("--library-path", type=Path, default=Path("library"))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--book", type=str, default=None, help="Restrict sampling to books matching this substring.")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    pool = load_tagged_chapters(args.library_path)
    if args.book:
        pool = [s for s in pool if args.book.lower() in s["book"].lower()]

    if not pool:
        print("ERROR: no tagged chapters found (library/tags/*.json is empty or missing). "
              "Only already-tagged chapters give real ground truth — tag a shortlist first.", file=sys.stderr)
        sys.exit(1)

    sample = random.sample(pool, min(args.n, len(pool)))

    # Rough avg word count from a few sampled chapter files, for the cost estimate only.
    word_counts = []
    for s in sample[:10]:
        f = find_chapter_file(args.library_path, s["book"], s["chapter"])
        if f:
            word_counts.append(len(f.read_text(encoding="utf-8", errors="ignore").split()))
    avg_words = sum(word_counts) // len(word_counts) if word_counts else 2500

    total_tokens, est_cost = estimate_cost(len(sample), avg_words, args.model)
    print(f"Sampled {len(sample)} already-tagged chapters (pool size: {len(pool)}, seed: {args.seed}).")
    print(f"Model: {args.model}")
    print(f"Estimated tokens: ~{total_tokens:,} | Estimated cost: ~${est_cost:.4f}")

    if args.dry_run:
        print("\n--dry-run: no API calls made. Review the estimate above, then rerun without --dry-run to generate for real.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it, then rerun. Never pass the key on the command line.", file=sys.stderr)
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    results = []
    for i, s in enumerate(sample):
        chapter_file = find_chapter_file(args.library_path, s["book"], s["chapter"])
        if chapter_file is None:
            continue
        excerpt = chapter_file.read_text(encoding="utf-8", errors="ignore")[:6000]
        categories = load_categories_for(args.library_path, s["book"], s["chapter"])
        negative = pick_hard_negative(args.library_path, s["book"], s["chapter"], categories, pool)

        prompt = PROMPT_TEMPLATE.format(book=s["book"], chapter=s["chapter"], excerpt=excerpt)
        try:
            response = client.messages.create(
                model=args.model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = json.loads(response.content[0].text)
        except Exception as e:
            print(f"  [{i + 1}/{len(sample)}] skipped {s['book']}/{s['chapter']}: {e}", file=sys.stderr)
            continue

        results.append({
            "query_id": f"syn-{i + 1:04d}",
            "query_text": parsed.get("paraphrase_query"),
            "masked_query_text": parsed.get("masked_query"),
            "target": {"book": s["book"], "chapter": s["chapter"], "chunk_index": 0},
            "hard_negative": {"book": negative["book"], "chapter": negative["chapter"]} if negative else None,
            "category": categories[0] if categories else None,
            "generated_at": __import__("datetime").date.today().isoformat(),
            "model": args.model,
        })
        print(f"  [{i + 1}/{len(sample)}] generated for {s['book']}/{s['chapter']}", end="\r")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} synthetic queries to {OUTPUT_PATH}")

    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not USAGE_LOG.exists():
        USAGE_LOG.write_text(
            "# Library — Usage Log\n\n"
            "Library-wide run costs (lexical indexing, synthetic query generation, any future "
            "direct-API calls). Per-project generation runs are still logged in "
            "projects/{project}/usage-log.md — this file is only for library-wide work.\n\n"
            "| Date | Run | Tokens | Est. cost | Notes |\n|---|---|---|---|---|\n"
            f"| {__import__('datetime').date.today().isoformat()} | embed_library.py (historical, "
            "backfilled note) | ~80M est. | ~$0-10 (one-time, per CLAUDE.md) | Full-library "
            "Voyage embedding; not separately metered at the time, backfilled here for a non-empty "
            "starting record. |\n",
            encoding="utf-8",
        )
    with USAGE_LOG.open("a", encoding="utf-8") as f:
        f.write(f"| {__import__('datetime').date.today().isoformat()} | generate_synthetic_queries.py "
                f"(n={len(sample)}, model={args.model}) | ~{total_tokens:,} (est.) | ~${est_cost:.4f} (est.) | "
                f"{len(results)} queries written to {OUTPUT_PATH} |\n")


if __name__ == "__main__":
    main()
