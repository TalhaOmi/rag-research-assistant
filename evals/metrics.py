"""Pure metric functions (no network, no models) so they can be unit tested."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from paper_assistant.ingest import Chunk


def normalize(text: str) -> str:
    """Lowercase, fold ligatures, and keep only letters/digits separated by single spaces, so a
    gold phrase still matches after line breaks, hyphenation or punctuation differences."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def chunk_matches_gold(chunk: Chunk, gold: dict) -> bool:
    """A chunk is relevant if it comes from the gold paper and contains the gold evidence phrase.

    Gold is defined by a verbatim phrase rather than a chunk id, so the same eval set stays
    valid when you change chunk size (chunk ids change, the evidence phrase does not).
    """
    return chunk.paper_id == gold["paper_id"] and normalize(gold["evidence"]) in normalize(
        chunk.text
    )


def first_relevant_rank(chunks: Sequence[Chunk], gold: dict) -> int | None:
    """1-based rank of the first relevant chunk, or None if none was retrieved."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_matches_gold(chunk, gold):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
