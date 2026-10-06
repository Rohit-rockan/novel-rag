# novel-rag

Hybrid retrieval (BM25 + dense embeddings) over a personal library of fiction, built to
find reference chapters for a scene — "a tense standoff between former allies", "first
use of a new magic system" — while keeping LLM costs near zero.

No books, extracted text, indexes, or tags are included. Bring your own epubs.

## Pipeline

| Step | Script | Cost |
|---|---|---|
| 1. Extract epub → chapter text | `extract_book.py`, `extract_all.py` | free, local |
| 2. Keyword pre-filter (10 rough scene categories) | `keyword_prefilter.py` | free, local |
| 3. Dense embeddings into `sqlite-vec` | `embed_library.py` (+ `fix_oversized_chunks.py`) | Voyage AI, one pass |
| 4. BM25 index (SQLite FTS5, contentless, rowid-aligned) | `build_lexical_index.py` | free, local |
| 5. Search | `hybrid_search.py` (RRF fusion), `semantic_search.py`, `retrieve_candidates.py` | one query embed |
| 6. Evaluate Hit@K / Recall@K / MRR | `generate_synthetic_queries.py`, `evaluate_retrieval.py` | Anthropic API (dry-run first) |

Chapters over ~24k words are split into overlapping chunks before embedding, since Voyage
silently truncates input past 32k tokens.

## Setup

```bash
pip install -r scripts/requirements.txt
cp .env.example .env   # add your keys
mkdir sources          # put .epub files here, or set NOVEL_SOURCE_DIR
```

## Usage

```bash
python scripts/extract_book.py "Some Book.epub"
python scripts/keyword_prefilter.py --library-path ./library
python scripts/embed_library.py --library-path ./library --dry-run
python scripts/embed_library.py --library-path ./library
python scripts/build_lexical_index.py --library-path ./library
python scripts/hybrid_search.py --library-path ./library \
    --query "a tense standoff between former allies, mask slipping" \
    --top-n 15 --boost-category emotional_beat
```

`extract_all.py` expects a `library/fiction-manifest.json` (a JSON list of epub filenames).
Long-running jobs (`embed_library.py`) are resumable via a progress file in `library/`.

Some docstrings refer to Claude Code subagents (`tagger`, `beat-classifier`) from the larger
private pipeline this was extracted from; the retrieval scripts run standalone without them.

## License

MIT — see [LICENSE](LICENSE).
