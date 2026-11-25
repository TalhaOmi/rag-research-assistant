"""Central settings. Anything tunable lives here or in an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PAPER_DATA_DIR", ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
MANIFEST_PATH = DATA_DIR / "papers.csv"
CACHE_DIR = Path(os.getenv("PAPER_CACHE_DIR", ROOT / ".cache"))

# Embedding model. bge models want this instruction prefix on *queries* (not on documents).
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Small, cheap model by default; override in .env to compare models.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "claude-haiku-5-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-haiku-5-5")

# Chunking is specified in tokens (the unit LLMs and embedders think in) and converted to words.
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 80

TOP_K = 5  # chunks sent to the LLM
CANDIDATES = 20  # how deep each retriever looks before fusion
RRF_K = 60  # reciprocal-rank-fusion constant (60 is the value from the original paper)
