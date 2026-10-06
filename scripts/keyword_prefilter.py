#!/usr/bin/env python3
"""
keyword_prefilter.py — ZERO-COST local pre-filter for the novel library.

What it does:
  Scans every chapter in your library and scores it against rough category
  signals (fight, dialogue, magic, monster, emotional, strategy, worldbuilding)
  using plain keyword/regex matching. No API calls, no tokens, no cost.

Output:
  library/keyword-index.json — a lightweight map of every chapter to its
  top candidate categories + a relevance score. Your generation pipeline
  uses THIS to shortlist a handful of candidate chapters, and only THOSE
  get sent to Claude for real tagging later (on-demand, not upfront).

Usage:
  python keyword_prefilter.py --library-path ./library

Expected input layout (either works):
  library/extracted/{book-title}/chapter-01.md, chapter-02.md, ...
  OR
  library/extracted/{book-title}.md   (combined file, auto-split on "Chapter N" headers)
"""
import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Category keyword banks. These are intentionally broad and rough — the goal
# is narrowing 10,000+ chapters down to a shortlist, not precise tagging.
# Precise tagging still happens later, via Claude, but only on the shortlist.
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "fight_scene": [
        r"\bsword\b", r"\bblade\b", r"\bstrike\b", r"\bparry\b", r"\bdodge\b",
        r"\bpunch(ed)?\b", r"\bkick(ed)?\b", r"\bblood\b", r"\bwound(ed)?\b",
        r"\bslash(ed)?\b", r"\battack(ed|ing)?\b", r"\bcombat\b", r"\bduel\b",
        r"\bclash(ed)?\b", r"\bstab(bed)?\b", r"\bshield\b", r"\barmor\b",
    ],
    "monster_encounter": [
        r"\bmonster\b", r"\bbeast\b", r"\bcreature\b", r"\bfang(s)?\b",
        r"\bclaw(s)?\b", r"\bgrowl(ed|ing)?\b", r"\bsnarl(ed|ing)?\b",
        r"\broar(ed|ing)?\b", r"\btentacle(s)?\b", r"\bwraith\b", r"\bdragon\b",
        r"\bdemon\b", r"\bhorde\b",
    ],
    "magic_system": [
        r"\bspell(s)?\b", r"\bmana\b", r"\bcast(ing|s)?\b", r"\benchant(ed|ment)?\b",
        r"\bmagic(al)?\b", r"\brune(s)?\b", r"\bincantation\b", r"\bsorcery\b",
        r"\britual\b", r"\barcane\b", r"\bward(s)?\b", r"\bconjure(d)?\b",
    ],
    "strategy_planning": [
        r"\bplan(ned|ning)?\b", r"\bstrateg(y|ic)\b", r"\btactic(s|al)?\b",
        r"\bambush\b", r"\bdecoy\b", r"\bformation\b", r"\bcounter-?attack\b",
        r"\bcalculat(ed|ing)\b", r"\bmaneuver\b",
    ],
    "dialogue_heavy": [
        # High density of quotation marks is the real signal here, handled
        # separately below — this list catches dialogue-adjacent verbs.
        r"\bsaid\b", r"\bwhispered\b", r"\bshouted\b", r"\bmurmured\b",
        r"\bretorted\b", r"\bsnapped\b", r"\breplied\b", r"\basked\b",
    ],
    "emotional_beat": [
        r"\bwept\b", r"\bcried\b", r"\btears\b", r"\bgrief\b", r"\bheartbreak\b",
        r"\btrembl(ed|ing)\b", r"\bsobbed\b", r"\bdespair\b", r"\brelief\b",
        r"\bjoy(ful)?\b", r"\bfear(ful)?\b", r"\bdread\b", r"\bnumb\b",
    ],
    "world_engineering": [
        r"\bmechanism\b", r"\bengineer(ed|ing)?\b", r"\bmachine(ry)?\b",
        r"\bcontraption\b", r"\bgear(s)?\b", r"\bblueprint\b", r"\bforge(d)?\b",
        r"\bsteam\b", r"\bcogwheel\b", r"\bschematic\b",
    ],
    "weapon_arsenal": [
        r"\bdagger\b", r"\bspear\b", r"\bbow\b", r"\barrow(s)?\b", r"\bhammer\b",
        r"\baxe\b", r"\bgauntlet(s)?\b", r"\bscabbard\b", r"\bhilt\b", r"\bblacksmith\b",
        r"\bartifact\b", r"\brelic\b", r"\benchanted\s+\w+\b", r"\bforged\s+\w+\b",
        r"\bgun\b", r"\brifle\b", r"\bpistol\b", r"\bcannon\b",
    ],
    "world_laws": [
        r"\blaw(s)?\s+of\b", r"\bforbidden\b", r"\btaboo\b", r"\bdecree(d)?\b",
        r"\bedict\b", r"\bcovenant\b", r"\bpact\b", r"\bsacred\s+rule\b",
        r"\bmust\s+not\b", r"\bpunishable\b", r"\bconsequence(s)?\s+of\s+breaking\b",
        r"\bancient\s+law\b",
    ],
    "science_technology": [
        r"\bexperiment(s|ed|ing)?\b", r"\bhypothesis\b", r"\bformula(s|e)?\b",
        r"\balgorithm\b", r"\bquantum\b", r"\bgenetic(s|ally)?\b", r"\bDNA\b",
        r"\bchemical(s)?\b", r"\breaction\b", r"\blaboratory\b", r"\bA\.?I\.?\b",
        r"\bdata\b", r"\bresearch(er)?\b", r"\bprototype\b",
    ],
}

CHAPTER_HEADER_RE = re.compile(r"(?:^|\n)\s*(Chapter\s+\d+|CHAPTER\s+[A-Z]+)[^\n]*", re.IGNORECASE)
QUOTE_RE = re.compile(r'["“”]')


def split_combined_file(text: str):
    """Split a single combined book file into chapters using header regex."""
    matches = list(CHAPTER_HEADER_RE.finditer(text))
    if not matches:
        return [("Chapter 1", text)]
    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(0).strip()
        chapters.append((heading, text[start:end].strip()))
    return chapters


def score_chapter(text: str):
    """Score one chapter's text against every category. Returns dict of scores."""
    word_count = max(len(text.split()), 1)
    scores = {}

    for category, patterns in CATEGORY_KEYWORDS.items():
        hits = 0
        for pattern in patterns:
            hits += len(re.findall(pattern, text, flags=re.IGNORECASE))
        # Normalize per 1000 words so long/short chapters are comparable.
        scores[category] = round((hits / word_count) * 1000, 3)

    # Dialogue density needs its own signal: quote-mark density, not just verbs.
    quote_hits = len(QUOTE_RE.findall(text))
    scores["dialogue_heavy"] = round(
        scores.get("dialogue_heavy", 0) + (quote_hits / word_count) * 500, 3
    )

    return scores, word_count


def top_categories(scores: dict, threshold: float = 0.5, max_categories: int = 3):
    """Return the top N categories above a minimal relevance threshold."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"category": cat, "score": score}
        for cat, score in ranked[:max_categories]
        if score >= threshold
    ]


def process_library(library_path: Path):
    extracted_path = library_path / "extracted"
    if not extracted_path.exists():
        print(f"ERROR: {extracted_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    index = defaultdict(list)
    book_count = 0
    chapter_count = 0

    for book_entry in sorted(extracted_path.iterdir()):
        book_title = book_entry.stem

        chapters = []
        if book_entry.is_dir():
            # Per-chapter files: chapter-01.md, chapter-02.md, ...
            for chapter_file in sorted(book_entry.glob("*.md")):
                text = chapter_file.read_text(encoding="utf-8", errors="ignore")
                chapters.append((chapter_file.stem, text))
        elif book_entry.suffix == ".md":
            # Combined single file per book
            text = book_entry.read_text(encoding="utf-8", errors="ignore")
            chapters = split_combined_file(text)
        else:
            continue

        if not chapters:
            continue

        book_count += 1
        for heading, text in chapters:
            scores, word_count = score_chapter(text)
            top = top_categories(scores)
            index[book_title].append({
                "chapter": heading,
                "word_count": word_count,
                "top_categories": top,
                "all_scores": scores,
            })
            chapter_count += 1

    output_path = library_path / "keyword-index.json"
    output_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"Done. Scanned {book_count} books, {chapter_count} chapters.")
    print(f"Zero API calls made — this cost $0.")
    print(f"Index written to: {output_path}")
    print(f"\nNext step: when generating a beat, pull candidates from this index")
    print(f"by category, then send ONLY that shortlist to Claude for real tagging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-cost keyword pre-filter for novel library.")
    parser.add_argument(
        "--library-path",
        type=str,
        default="./library",
        help="Path to the library folder (containing an 'extracted' subfolder). Default: ./library",
    )
    args = parser.parse_args()
    process_library(Path(args.library_path))
