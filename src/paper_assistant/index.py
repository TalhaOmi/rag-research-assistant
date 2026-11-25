"""Persist chunks: a JSONL file (text + metadata, used by BM25 and for display) and a Chroma
collection (vectors, used for dense search)."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from paper_assistant.embedders import Embedder
from paper_assistant.ingest import Chunk

COLLECTION = "chunks"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


def save_chunks(chunks: list[Chunk], index_dir: Path) -> None:
    with open(index_dir / CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(index_dir: Path) -> list[Chunk]:
    path = Path(index_dir) / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No index at {index_dir}. Run: python -m paper_assistant build-index"
        )
    with open(path, encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


def load_meta(index_dir: Path) -> dict:
    path = Path(index_dir) / META_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    index_dir: Path,
    *,
    batch_size: int = 64,
    meta: dict | None = None,
) -> None:
    """(Re)build the index from scratch in `index_dir`."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(chunks, index_dir)

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - collection did not exist yet
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embedder.embed_documents([c.text for c in batch]),
            documents=[c.text for c in batch],
            metadatas=[{"paper_id": c.paper_id, "page": c.page} for c in batch],
        )

    info = {"embedder": embedder.name, "n_chunks": len(chunks), **(meta or {})}
    (index_dir / META_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")


def open_collection(index_dir: Path):
    client = chromadb.PersistentClient(path=str(Path(index_dir) / "chroma"))
    return client.get_collection(COLLECTION)
