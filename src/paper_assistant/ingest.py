"""PDF -> clean text per page -> overlapping chunks that remember paper and page.

Design decision: a chunk NEVER crosses a page boundary. Every chunk therefore has exactly one
page number, so a citation like [2304.01543v1, p.4] is always correct. The price is that a
sentence running over a page break gets cut in two.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pymupdf

from paper_assistant import config

WORDS_PER_TOKEN = 0.75  # rule of thumb for English: 1 token ~ 0.75 words
MIN_CHUNK_WORDS = 20  # drop crumbs such as a page holding only a figure caption


@dataclass(frozen=True)
class Page:
    paper_id: str
    page: int  # 1-based: the number a human sees in a PDF viewer
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def tokens_to_words(tokens: int) -> int:
    return max(1, round(tokens * WORDS_PER_TOKEN))


def clean_text(text: str) -> str:
    """Normalise raw PDF text: ligatures, soft hyphens, hyphenated line breaks, whitespace."""
    text = unicodedata.normalize("NFKC", text)  # 'ﬁ' -> 'fi'
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)  # 'inter-\npretable' -> 'interpretable'
    return " ".join(text.split())  # newlines carry no meaning for us


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of at most `size` words; each repeats `overlap` words of the last."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    start = 0
    while True:
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):  # this window reached the end: stop (no overlap-only tail)
            break
        start += step
    return chunks


def load_pdf_pages(path: Path) -> list[Page]:
    """Extract cleaned text for every page. Page numbers start at 1."""
    paper_id = Path(path).stem
    pages = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            pages.append(Page(paper_id, number, clean_text(page.get_text())))
    return pages


_REF_HEADING = re.compile(r"\b(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def strip_references(pages: list[Page]) -> list[Page]:
    """Cut the bibliography. Heuristic: a 'References' heading in the later part of the paper
    followed by text containing many years. Bibliographies are walls of author names and titles
    that match almost any query without answering it, so removing them is an eval experiment."""
    if len(pages) < 3:
        return pages
    for i in range(len(pages) // 3, len(pages)):
        text = pages[i].text
        for match in _REF_HEADING.finditer(text):
            tail = text[match.end() :] + " " + " ".join(p.text for p in pages[i + 1 :])
            if len(_YEAR.findall(tail)) >= 5:
                kept = list(pages[:i])
                head = text[: match.start()].strip()
                if head:
                    kept.append(replace(pages[i], text=head))
                return kept
    return pages


def pages_to_chunks(
    pages: list[Page], size_words: int, overlap_words: int, min_words: int = MIN_CHUNK_WORDS
) -> list[Chunk]:
    chunks = []
    for page in pages:
        pieces = chunk_text(page.text, size_words, overlap_words)
        for n, piece in enumerate(pieces):
            if len(piece.split()) < min_words:
                continue
            chunk_id = f"{page.paper_id}:p{page.page}:c{n}"
            chunks.append(Chunk(chunk_id, page.paper_id, page.page, piece))
    return chunks


def build_chunks(
    pdf_dir: Path | None = None,
    chunk_tokens: int = config.CHUNK_TOKENS,
    overlap_tokens: int = config.OVERLAP_TOKENS,
    drop_references: bool = False,
) -> list[Chunk]:
    pdf_dir = Path(pdf_dir or config.PDF_DIR)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {pdf_dir}. Run: python -m paper_assistant fetch")
    size, overlap = tokens_to_words(chunk_tokens), tokens_to_words(overlap_tokens)
    chunks: list[Chunk] = []
    for pdf in pdfs:
        pages = load_pdf_pages(pdf)
        if drop_references:
            pages = strip_references(pages)
        chunks.extend(pages_to_chunks(pages, size, overlap))
    return chunks


def load_titles(manifest: Path | None = None) -> dict[str, str]:
    """paper_id -> title, from the manifest written by the download script."""
    manifest = Path(manifest or config.MANIFEST_PATH)
    if not manifest.exists():
        return {}
    with open(manifest, encoding="utf-8", newline="") as f:
        return {row["paper_id"]: row["title"] for row in csv.DictReader(f)}
