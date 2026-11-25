"""The assistant: retrieve, generate, validate. One object shared by the API and the CLI."""

from __future__ import annotations

from pathlib import Path

from paper_assistant import config
from paper_assistant.embedders import Embedder, get_embedder
from paper_assistant.generate import Answer, answer_question
from paper_assistant.index import load_meta
from paper_assistant.ingest import load_titles
from paper_assistant.llm import DiskCache, get_client
from paper_assistant.retrieve import HybridRetriever


class Assistant:
    def __init__(
        self,
        index_dir: Path | None = None,
        *,
        embedder: Embedder | None = None,
        client=None,
        model: str = config.GENERATION_MODEL,
        cache: DiskCache | None = None,
    ):
        self.index_dir = Path(index_dir or config.INDEX_DIR)
        self._embedder = embedder
        self._client = client
        self.model = model
        self.cache = cache
        self.titles = load_titles()
        self._retriever: HybridRetriever | None = None

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            embedder = self._embedder or get_embedder(
                load_meta(self.index_dir).get("embedder", "st")
            )
            self._retriever = HybridRetriever.from_index_dir(self.index_dir, embedder)
        return self._retriever

    @property
    def client(self):
        if self._client is None:
            self._client = get_client()
        return self._client

    def ask(self, question: str, k: int = config.TOP_K, mode: str = "hybrid") -> Answer:
        hits = self.retriever.search(question, k=k, mode=mode)
        return answer_question(
            question, hits, self.client, model=self.model, titles=self.titles, cache=self.cache
        )
