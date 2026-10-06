"""
Extract every book in library/fiction-manifest.json (or craft-manifest.json) that hasn't
already been extracted. Idempotent - safe to re-run; skips books whose output folder
already exists and is non-empty.

Usage:
    python scripts/extract_all.py fiction
    python scripts/extract_all.py craft
"""
import json
import sys
from pathlib import Path

from extract_book import OUTPUT_ROOT, extract, slugify

ROOT = Path(__file__).resolve().parent.parent


def main(manifest_name: str):
    manifest_path = ROOT / "library" / f"{manifest_name}-manifest.json"
    books = json.loads(manifest_path.read_text())

    done, skipped, failed = 0, 0, []
    for filename in books:
        title = Path(filename).stem
        out_dir = OUTPUT_ROOT / slugify(title)
        if out_dir.exists() and any(out_dir.iterdir()):
            skipped += 1
            continue
        try:
            extract(filename)
            done += 1
        except Exception as exc:
            failed.append((filename, str(exc)))

    print(f"Extracted: {done}, already done: {skipped}, failed: {len(failed)}")
    for filename, err in failed:
        print(f"  FAILED: {filename} -> {err}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("fiction", "craft"):
        raise SystemExit("Usage: python scripts/extract_all.py [fiction|craft]")
    main(sys.argv[1])
