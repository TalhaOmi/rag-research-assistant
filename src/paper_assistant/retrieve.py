"""Hybrid retrieval: dense (meaning) + BM25 (exact words), merged with reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from paper_assistant import config
from paper_assistant.embedders import Embedder
from paper_assistant.index import load_chunks, open_collection
from paper_assistant.ingest import Chunk
from paper_assistant.retrieve_text import tokenize

MODES = ("dense", "bm25", "hybrid")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1 = best


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion.

    Each ranking is a list of ids, best first. An item scores sum(1 / (k + rank)) over the
    rankings it appears in (rank starts at 1). Only ranks are used, never raw scores, because
    cosine similarities and BM25 scores live on unrelated scales and cannot be added directly.
    Items ranked well by BOTH retrievers float to the top. Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        collection=None,
        *,
        candidates: int = config.CANDIDATES,
        rrf_k: int = config.RRF_K,
    ):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.collection = collection
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    @classmethod
    def from_index_dir(cls, index_dir: Path, embedder: Embedder, **kwargs) -> HybridRetriever:
        return cls(load_chunks(index_dir), embedder, open_collection(index_dir), **kwargs)

    def _dense_ranking(self, query: str) -> list[str]:
        if self.collection is None:
            raise RuntimeError("Dense search needs a vector collection")
        n = min(self.candidates, self.collection.count())
        if n == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)], n_results=n
        )
        return list(result["ids"][0])

    def _bm25_ranking(self, query: str) -> list[str]:
        tokens = tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i].chunk_id for i in order[: self.candidates] if scores[i] > 0]

    def search(self, query: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "dense":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self._dense_ranking(query))]
        elif mode == "bm25":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self._bm25_ranking(query))]
        else:
            ranked = rrf_fuse([self._dense_ranking(query), self._bm25_ranking(query)], self.rrf_k)
        return [
            Hit(self.by_id[chunk_id], score, rank)
            for rank, (chunk_id, score) in enumerate(ranked[:k], start=1)
        ]
