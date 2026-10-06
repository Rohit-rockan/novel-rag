"""
Extract plain-text chapters from one epub in the source library.

Usage:
    python scripts/extract_book.py "<book-filename>.epub"

Reads from the directory named by the NOVEL_SOURCE_DIR environment variable (default:
./sources — never committed to this repo) and writes library/extracted/{book-title}/chapter-NN.md
— one file per spine item, in reading order, with HTML markup stripped to plain text.

PDFs are not handled here: read them directly with Claude Code's Read tool (which supports
PDF, in <=20 page ranges) during tagging instead of pre-extracting them.

Stdlib only, no third-party dependencies.
"""
import html
import os
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SOURCE_LIBRARY = Path(
    os.environ.get("NOVEL_SOURCE_DIR", Path(__file__).resolve().parent.parent / "sources")
)
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "library" / "extracted"

NAMESPACES = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
}


def slugify(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()
    return slug or "untitled"


def strip_html(xhtml_bytes: bytes) -> str:
    text = xhtml_bytes.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style).*?</\1>", "", text)
    text = re.sub(r"(?i)<(p|div|br|h[1-6]|li)[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def spine_paths(zf: zipfile.ZipFile) -> list[str]:
    container = ET.fromstring(zf.read("META-INF/container.xml"))
    opf_path = container.find(".//container:rootfile", NAMESPACES).attrib["full-path"]
    opf_dir = str(Path(opf_path).parent)
    opf = ET.fromstring(zf.read(opf_path))

    manifest = {
        item.attrib["id"]: item.attrib["href"]
        for item in opf.findall(".//opf:manifest/opf:item", NAMESPACES)
    }
    spine_ids = [
        itemref.attrib["idref"]
        for itemref in opf.findall(".//opf:spine/opf:itemref", NAMESPACES)
    ]

    paths = []
    for item_id in spine_ids:
        href = manifest.get(item_id)
        if not href:
            continue
        path = href if opf_dir in (".", "") else f"{opf_dir}/{href}"
        paths.append(path)
    return paths


def extract(epub_filename: str) -> Path:
    source_path = SOURCE_LIBRARY / epub_filename
    if not source_path.exists():
        raise SystemExit(f"Not found in source library: {source_path}")

    title = source_path.stem
    out_dir = OUTPUT_ROOT / slugify(title)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_path) as zf:
        chapter_index = 0
        for path in spine_paths(zf):
            try:
                raw = zf.read(path)
            except KeyError:
                continue
            text = strip_html(raw)
            if len(text.split()) < 300:
                continue  # skip cover/toc/copyright/newsletter front-matter stubs
            chapter_index += 1
            out_file = out_dir / f"chapter-{chapter_index:02d}.md"
            out_file.write_text(text, encoding="utf-8")

    print(f"Wrote {chapter_index} chapter files to {out_dir}")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    extract(sys.argv[1])
