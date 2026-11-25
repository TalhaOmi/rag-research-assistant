"""Tiny text helpers shared by BM25 retrieval and the offline embedder."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric words. Single characters are dropped; they are mostly noise."""
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1]
