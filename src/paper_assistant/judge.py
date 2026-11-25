"""Faithfulness: is every claim in the answer supported by the passages it cites?

An LLM judge splits the answer into factual claims and checks each against the cited chunks.
Faithfulness = supported claims / total claims. This catches the dangerous failure of RAG: a
fluent answer with a real-looking citation that does not actually say what the answer says.
Judge with a different (or cheaper) model than the generator when you can, and spot-check a
few judgements by hand: the judge is a measuring instrument and can be wrong too.
"""

from __future__ import annotations

from dataclasses import dataclass

from paper_assistant import config
from paper_assistant.generate import Answer
from paper_assistant.llm import DiskCache, call_tool

JUDGE_SYSTEM = """You are a strict fact-checker. You receive an ANSWER and the SOURCE passages \
it cites. Split the answer into its individual factual claims. For each claim decide whether the \
SOURCE passages directly support it. A claim that is plausible or true in general but not stated \
or clearly implied by the passages is NOT supported. Ignore citation markers when listing claims."""

JUDGE_TOOL = {
    "name": "submit_verdict",
    "description": "Report whether each claim in the answer is supported by the sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "supported": {"type": "boolean"},
                    },
                    "required": ["claim", "supported"],
                },
            }
        },
        "required": ["claims"],
    },
}


@dataclass
class Verdict:
    faithfulness: float  # supported / total, 1.0 when there are no claims
    claims: list[dict]


def cited_passages(answer: Answer) -> list[str]:
    cited = {(c.paper_id, c.page) for c in answer.citations}
    return [
        f"[{h.chunk.paper_id}, p.{h.chunk.page}] {h.chunk.text}"
        for h in answer.hits
        if (h.chunk.paper_id, h.chunk.page) in cited
    ]


def judge_faithfulness(
    answer: Answer,
    client,
    *,
    model: str = config.JUDGE_MODEL,
    cache: DiskCache | None = None,
) -> Verdict:
    passages = cited_passages(answer)
    user = (
        "SOURCES:\n\n" + "\n\n".join(passages) + f"\n\nANSWER:\n{answer.answer}"
        if passages
        else f"SOURCES: (none)\n\nANSWER:\n{answer.answer}"
    )
    payload = call_tool(
        client, model=model, system=JUDGE_SYSTEM, user=user, tool=JUDGE_TOOL, cache=cache
    )
    claims = [c for c in payload.get("claims", []) if isinstance(c, dict) and "supported" in c]
    if not claims:
        return Verdict(1.0, [])
    supported = sum(1 for c in claims if c["supported"])
    return Verdict(supported / len(claims), claims)
