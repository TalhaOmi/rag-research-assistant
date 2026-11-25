# Project: Research Paper Assistant (RAG)
Goal and metric: Answer questions about 50 arXiv explainable-AI papers with page-level citations; hit@5 >= 0.8 on the 20-question eval set, and unanswerable questions get "Not in the sources."
Stack: Python 3.10+, pytest, ruff, PyMuPDF, sentence-transformers (bge-small), rank_bm25, Chroma, Anthropic API, FastAPI, Streamlit
Rules: type hints everywhere; tests before features; no secrets in git; ask before adding a dependency; never commit data/pdfs or data/index; page numbers are 1-based everywhere; a chunk never crosses a page; never trust a model citation without checking it against the retrieved chunks
Commands: pytest | ruff check . | ruff format . | python -m paper_assistant build-index | python -m paper_assistant ask "..." --retrieval-only | python -m evals.run_eval
