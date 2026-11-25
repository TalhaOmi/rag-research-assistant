"""Evaluate retrieval and (optionally) generation quality on evals/dataset.jsonl.

    python -m evals.run_eval                                   # retrieval only: free, no API key
    python -m evals.run_eval --generate                        # + refusal, citation, faithfulness
    python -m evals.run_eval --configs baseline,small-chunks   # choose configurations
    python -m evals.run_eval --embedder hashing                # offline smoke test only

Each configuration changes ONE thing relative to `baseline`, so any difference in the table has
exactly one cause. Results are written to evals/results/ as JSON plus a Markdown table
(latest.md) you can paste into the README.

Metrics
    hit@5      was a relevant chunk in the top 5? (relevant = right paper + evidence phrase)
    MRR        mean of 1/rank of the first relevant chunk (0 if not retrieved in the top 10)
    refusal    answerable questions answered AND unanswerable questions refused
    cite_paper/cite_page   did the answer cite the gold paper / the gold page?
    faith      share of the answer's claims that the cited passages support (LLM judge)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from evals.metrics import first_relevant_rank, hit_at_k, mean, reciprocal_rank
from paper_assistant import config
from paper_assistant.embedders import get_embedder
from paper_assistant.generate import REFUSAL, answer_question
from paper_assistant.index import build_index
from paper_assistant.ingest import build_chunks, load_titles
from paper_assistant.judge import judge_faithfulness
from paper_assistant.llm import DiskCache, get_client
from paper_assistant.retrieve import HybridRetriever

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
README = EVAL_DIR.parent / "README.md"
RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"
TOP_K = 5
MRR_DEPTH = 10


@dataclass(frozen=True)
class EvalConfig:
    name: str
    chunk_tokens: int = config.CHUNK_TOKENS
    overlap_tokens: int = config.OVERLAP_TOKENS
    mode: str = "hybrid"
    drop_references: bool = False


CONFIGS = {
    c.name: c
    for c in [
        EvalConfig("baseline"),
        EvalConfig("small-chunks", chunk_tokens=250, overlap_tokens=40),
        EvalConfig("large-chunks", chunk_tokens=800, overlap_tokens=120),
        EvalConfig("dense-only", mode="dense"),
        EvalConfig("bm25-only", mode="bm25"),
        EvalConfig("no-references", drop_references=True),
    ]
}


def load_dataset(path: Path = DATASET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_for(cfg: EvalConfig, embedder_name: str, cache: dict) -> HybridRetriever:
    """Build (or reuse) one index per distinct chunking setup. Modes share an index."""
    key = (cfg.chunk_tokens, cfg.overlap_tokens, cfg.drop_references, embedder_name)
    if key not in cache:
        digest = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
        index_dir = config.CACHE_DIR / "eval-indexes" / digest
        embedder = get_embedder(embedder_name)
        chunks = build_chunks(
            chunk_tokens=cfg.chunk_tokens,
            overlap_tokens=cfg.overlap_tokens,
            drop_references=cfg.drop_references,
        )
        print(f"  [{cfg.name}] building index: {len(chunks)} chunks -> {index_dir.name}")
        build_index(chunks, embedder, index_dir)
        cache[key] = HybridRetriever.from_index_dir(index_dir, embedder)
    return cache[key]


def evaluate(cfg: EvalConfig, questions: list[dict], retriever, generate: bool, deps: dict) -> dict:
    ranks, hit5, rr = [], [], []
    refusal_ok, cite_paper, cite_page, faith = [], [], [], []
    rows = []

    for q in questions:
        hits = retriever.search(q["question"], k=MRR_DEPTH, mode=cfg.mode)
        row = {"id": q["id"], "answerable": q["answerable"]}

        if q["answerable"]:
            rank = first_relevant_rank([h.chunk for h in hits], q["gold"])
            ranks.append(rank)
            hit5.append(1.0 if hit_at_k(rank, TOP_K) else 0.0)
            rr.append(reciprocal_rank(rank))
            row["rank"] = rank

        if generate:
            answer = answer_question(
                q["question"],
                hits[:TOP_K],
                deps["client"],
                model=deps["model"],
                titles=deps["titles"],
                cache=deps["cache"],
            )
            refused = (not answer.answerable) or answer.answer == REFUSAL
            refusal_ok.append(1.0 if refused != q["answerable"] else 0.0)
            row["refused"] = refused
            if q["answerable"] and not refused:
                gold = q["gold"]
                cite_paper.append(
                    1.0 if any(c.paper_id == gold["paper_id"] for c in answer.citations) else 0.0
                )
                cite_page.append(
                    1.0
                    if any(
                        c.paper_id == gold["paper_id"] and c.page == gold["page"]
                        for c in answer.citations
                    )
                    else 0.0
                )
                verdict = judge_faithfulness(
                    answer, deps["client"], model=deps["judge_model"], cache=deps["cache"]
                )
                faith.append(verdict.faithfulness)
                row["faithfulness"] = verdict.faithfulness
            elif q["answerable"]:
                cite_paper.append(0.0)
                cite_page.append(0.0)
        rows.append(row)

    result = {
        "config": asdict(cfg),
        "n_answerable": sum(1 for q in questions if q["answerable"]),
        "n_unanswerable": sum(1 for q in questions if not q["answerable"]),
        "hit@5": mean(hit5),
        "mrr": mean(rr),
        "rows": rows,
    }
    if generate:
        result.update(
            {
                "refusal_accuracy": mean(refusal_ok),
                "citation_paper": mean(cite_paper),
                "citation_page": mean(cite_page),
                "faithfulness": mean(faith),
            }
        )
    return result


def describe(cfg: EvalConfig) -> str:
    bits = [f"{cfg.chunk_tokens}/{cfg.overlap_tokens} tok", cfg.mode]
    if cfg.drop_references:
        bits.append("no refs")
    return ", ".join(bits)


def to_markdown(results: list[dict], embedder: str, generate: bool) -> str:
    head = ["Config", "Setup", "hit@5", "MRR"]
    if generate:
        head += ["refusal acc", "cite paper", "cite page", "faithfulness"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        cfg = EvalConfig(**r["config"])
        cells = [cfg.name, describe(cfg), f"{r['hit@5']:.2f}", f"{r['mrr']:.2f}"]
        if generate:
            cells += [
                f"{r['refusal_accuracy']:.2f}",
                f"{r['citation_paper']:.2f}",
                f"{r['citation_page']:.2f}",
                f"{r['faithfulness']:.2f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    n_a, n_u = results[0]["n_answerable"], results[0]["n_unanswerable"]
    lines.append(f"\n{n_a} answerable + {n_u} unanswerable questions; embedder: `{embedder}`.")
    return "\n".join(lines)


def replace_results_block(readme_text: str, table: str) -> str:
    """Swap whatever sits between the RESULTS markers in the README for the new table."""
    pattern = re.compile(re.escape(RESULTS_START) + r".*?" + re.escape(RESULTS_END), re.DOTALL)
    if not pattern.search(readme_text):
        raise ValueError("README has no RESULTS:START / RESULTS:END markers")
    return pattern.sub(lambda _: f"{RESULTS_START}\n{table}\n{RESULTS_END}", readme_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="baseline,small-chunks")
    parser.add_argument("--embedder", choices=["st", "hashing"], default="st")
    parser.add_argument("--generate", action="store_true", help="also run Claude + the judge")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--update-readme", action="store_true", help="write the table into README.md"
    )
    args = parser.parse_args()

    questions = load_dataset(args.dataset)
    chosen = [CONFIGS[name.strip()] for name in args.configs.split(",")]
    deps: dict = {}
    if args.generate:
        deps = {
            "client": get_client(),
            "model": config.GENERATION_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "titles": load_titles(),
            "cache": DiskCache(),
        }

    index_cache: dict = {}
    results = []
    for cfg in chosen:
        print(f"Evaluating {cfg.name} ({describe(cfg)})")
        retriever = index_for(cfg, args.embedder, index_cache)
        results.append(evaluate(cfg, questions, retriever, args.generate, deps))

    table = to_markdown(results, args.embedder, args.generate)
    print("\n" + table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "generate" if args.generate else "retrieval"
    (RESULTS_DIR / f"{stamp}-{args.embedder}-{suffix}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "latest.md").write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_DIR}")
    if args.update_readme:
        README.write_text(
            replace_results_block(README.read_text(encoding="utf-8"), table), encoding="utf-8"
        )
        print(f"Updated {README.name}")


if __name__ == "__main__":
    main()
