"""Turn text into vectors. Two implementations behind one small interface.

* SentenceTransformerEmbedder: the real model (BAAI/bge-small-en-v1.5). Use this for results.
* HashingEmbedder: a deterministic, dependency-free stand-in (hashed bag of words). It lets the
  tests and offline smoke runs exercise the dense-retrieval code path without downloading a
  model. Its numbers are NOT comparable to a real embedding model; never report them.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from paper_assistant import config
from paper_assistant.retrieve_text import tokenize


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = config.EMBED_MODEL, query_prefix: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -e '.[embed]'"
            ) from e
        self.name = "st"
        self.model_name = model_name
        self.query_prefix = config.QUERY_PREFIX if query_prefix is None else query_prefix
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0]
        return vector.tolist()


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.name = "hashing"
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embedder(name: str) -> Embedder:
    if name == "st":
        return SentenceTransformerEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedder {name!r}; choose 'st' or 'hashing'")
