from types import SimpleNamespace

import pytest

from evals.run_eval import EvalConfig, evaluate
from paper_assistant.embedders import HashingEmbedder
from paper_assistant.generate import REFUSAL, Citation
from paper_assistant.index import build_index
from paper_assistant.ingest import Chunk
from paper_assistant.pipeline import Assistant
from paper_assistant.retrieve import HybridRetriever

CHUNKS = [
    Chunk(
        "2001.00001v1:p3:c0",
        "2001.00001v1",
        3,
        "LIME explains single predictions using a local linear surrogate model",
    ),
    Chunk(
        "2001.00002v1:p5:c0",
        "2001.00002v1",
        5,
        "Counterfactual explanations show the smallest change that flips the decision",
    ),
    Chunk(
        "2001.00003v1:p2:c0",
        "2001.00003v1",
        2,
        "Saliency maps highlight the pixels that influenced the classifier",
    ),
]


class RoutingClient:
    """Fake Claude: answers questions that mention LIME, refuses everything else, and judges
    every claim as supported."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        tool = kwargs["tools"][0]["name"]
        user = kwargs["messages"][0]["content"]
        if tool == "submit_answer":
            if "LIME" in user.split("Question:")[-1]:
                payload = {
                    "answerable": True,
                    "answer": "LIME fits a local surrogate [2001.00001v1, p.3].",
                    "citations": [{"paper_id": "2001.00001v1", "page": 3}],
                }
            else:
                payload = {"answerable": False, "answer": REFUSAL, "citations": []}
        else:
            payload = {"claims": [{"claim": "LIME fits a local surrogate", "supported": True}]}
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=tool, input=payload)])


@pytest.fixture
def index_dir(tmp_path):
    build_index(CHUNKS, HashingEmbedder(), tmp_path / "idx")
    return tmp_path / "idx"


def test_assistant_answers_with_validated_citation(index_dir):
    assistant = Assistant(index_dir, embedder=HashingEmbedder(), client=RoutingClient())
    result = assistant.ask("What does LIME do?", k=3)
    assert result.answerable
    assert result.citations == [Citation("2001.00001v1", 3)]
    assert result.invalid_citations == []


def test_assistant_refuses_when_model_declines(index_dir):
    assistant = Assistant(index_dir, embedder=HashingEmbedder(), client=RoutingClient())
    result = assistant.ask("Who won the World Cup?", k=3)
    assert not result.answerable
    assert result.answer == REFUSAL


def test_missing_index_raises_clear_error(tmp_path):
    assistant = Assistant(tmp_path / "nope", embedder=HashingEmbedder(), client=RoutingClient())
    with pytest.raises(FileNotFoundError, match="build-index"):
        assistant.retriever  # noqa: B018


def test_eval_retrieval_and_generation_metrics(index_dir):
    retriever = HybridRetriever.from_index_dir(index_dir, HashingEmbedder())
    questions = [
        {
            "id": "a1",
            "question": "What does LIME do?",
            "answerable": True,
            "gold": {"paper_id": "2001.00001v1", "page": 3, "evidence": "local linear surrogate"},
        },
        {"id": "u1", "question": "Who won the World Cup?", "answerable": False, "gold": None},
    ]
    deps = {
        "client": RoutingClient(),
        "model": "m",
        "judge_model": "j",
        "titles": {},
        "cache": None,
    }
    result = evaluate(EvalConfig("t"), questions, retriever, True, deps)
    assert result["hit@5"] == 1.0
    assert result["mrr"] == 1.0
    assert result["refusal_accuracy"] == 1.0
    assert result["citation_paper"] == 1.0
    assert result["citation_page"] == 1.0
    assert result["faithfulness"] == 1.0
    assert result["n_answerable"] == 1 and result["n_unanswerable"] == 1


def test_eval_retrieval_only_needs_no_client(index_dir):
    retriever = HybridRetriever.from_index_dir(index_dir, HashingEmbedder())
    questions = [
        {
            "id": "a1",
            "question": "saliency maps pixels",
            "answerable": True,
            "gold": {"paper_id": "2001.00003v1", "page": 2, "evidence": "highlight the pixels"},
        }
    ]
    result = evaluate(EvalConfig("t", mode="bm25"), questions, retriever, False, {})
    assert result["hit@5"] == 1.0
    assert "refusal_accuracy" not in result
