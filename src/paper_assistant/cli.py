"""Command line entry points:

python -m paper_assistant fetch
python -m paper_assistant build-index [--embedder st|hashing] [--drop-references]
python -m paper_assistant ask "question" [--retrieval-only] [--mode hybrid|dense|bm25]
"""

from __future__ import annotations

import argparse
import sys

from paper_assistant import config


def _cmd_fetch(args: argparse.Namespace) -> int:
    from paper_assistant.corpus import fetch_corpus

    failed = fetch_corpus()
    if failed:
        print("Failed (re-run to retry):", ", ".join(failed))
        return 1
    print("All PDFs present.")
    return 0


def _cmd_build_index(args: argparse.Namespace) -> int:
    from paper_assistant.embedders import get_embedder
    from paper_assistant.index import build_index
    from paper_assistant.ingest import build_chunks

    chunks = build_chunks(
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        drop_references=args.drop_references,
    )
    papers = len({c.paper_id for c in chunks})
    print(f"{len(chunks)} chunks from {papers} papers. Embedding with '{args.embedder}'...")
    build_index(
        chunks,
        get_embedder(args.embedder),
        config.INDEX_DIR,
        meta={
            "chunk_tokens": args.chunk_tokens,
            "overlap_tokens": args.overlap_tokens,
            "drop_references": args.drop_references,
        },
    )
    print(f"Index written to {config.INDEX_DIR}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from paper_assistant.pipeline import Assistant

    assistant = Assistant()
    if args.retrieval_only:
        hits = assistant.retriever.search(args.question, k=args.k, mode=args.mode)
        for hit in hits:
            c = hit.chunk
            title = assistant.titles.get(c.paper_id, "")
            print(f"#{hit.rank}  [{c.paper_id}, p.{c.page}]  {title}")
            print(f"    {c.text[:300]}...\n")
        return 0

    answer = assistant.ask(args.question, k=args.k, mode=args.mode)
    print(answer.answer)
    for citation in answer.citations:
        print(f"  - {citation.paper_id}, p.{citation.page}")
    if answer.invalid_citations:
        print("Ignored citations that were not in the retrieved sources:", answer.invalid_citations)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="paper_assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the PDFs listed in data/papers.csv").set_defaults(
        func=_cmd_fetch
    )

    build = sub.add_parser("build-index", help="chunk, embed and index the PDFs")
    build.add_argument("--embedder", choices=["st", "hashing"], default="st")
    build.add_argument("--chunk-tokens", type=int, default=config.CHUNK_TOKENS)
    build.add_argument("--overlap-tokens", type=int, default=config.OVERLAP_TOKENS)
    build.add_argument("--drop-references", action="store_true")
    build.set_defaults(func=_cmd_build_index)

    ask = sub.add_parser("ask", help="ask a question (or just inspect retrieval)")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=config.TOP_K)
    ask.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    ask.add_argument("--retrieval-only", action="store_true", help="print chunks, no API call")
    ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
