from paper_assistant.generate import (
    REFUSAL,
    SYSTEM_PROMPT,
    Citation,
    answer_question,
    build_prompt,
)

from .conftest import FakeClient


def test_prompt_lists_sources_with_ids_and_pages(hits):
    prompt = build_prompt("What is SHAP?", hits, {"2304.01543v1": "A Review"})
    assert "[S1] paper_id: 2304.01543v1 | page: 4 | title: A Review" in prompt
    assert "Question: What is SHAP?" in prompt


def test_system_prompt_demands_refusal_and_citations():
    assert "Not in the sources." in SYSTEM_PROMPT
    assert "[paper_id, p.N]" in SYSTEM_PROMPT


def test_grounded_answer_keeps_valid_citations(hits):
    client = FakeClient(
        {
            "answerable": True,
            "answer": "SHAP explains predictions [2304.01543v1, p.4].",
            "citations": [{"paper_id": "2304.01543v1", "page": 4}],
        }
    )
    result = answer_question("What is SHAP?", hits, client)
    assert result.answerable
    assert result.citations == [Citation("2304.01543v1", 4)]
    assert result.invalid_citations == []


def test_model_declining_returns_refusal(hits):
    client = FakeClient({"answerable": False, "answer": "whatever", "citations": []})
    result = answer_question("Who won the 2022 World Cup?", hits, client)
    assert not result.answerable
    assert result.answer == REFUSAL
    assert result.citations == []


def test_hallucinated_citation_is_flagged_and_dropped(hits):
    client = FakeClient(
        {
            "answerable": True,
            "answer": "Fact [2304.01543v1, p.4] and another [2304.01543v1, p.19].",
            "citations": [{"paper_id": "2304.01543v1", "page": 4}],
        }
    )
    result = answer_question("q", hits, client)
    assert result.citations == [Citation("2304.01543v1", 4)]
    assert result.invalid_citations == [Citation("2304.01543v1", 19)]


def test_answer_with_only_invalid_citations_becomes_refusal(hits):
    client = FakeClient(
        {
            "answerable": True,
            "answer": "Confident claim [9999.99999v1, p.1].",
            "citations": [{"paper_id": "9999.99999v1", "page": 1}],
        }
    )
    result = answer_question("q", hits, client)
    assert not result.answerable
    assert result.answer == REFUSAL
    assert result.invalid_citations == [Citation("9999.99999v1", 1)]


def test_answer_without_any_citation_becomes_refusal(hits):
    client = FakeClient({"answerable": True, "answer": "Unsupported claim.", "citations": []})
    assert answer_question("q", hits, client).answer == REFUSAL


def test_malformed_citation_objects_are_ignored(hits):
    client = FakeClient(
        {
            "answerable": True,
            "answer": "Fact [2304.01543v1, p.4].",
            "citations": [{"paper_id": "2304.01543v1"}, "junk", {"paper_id": "x", "page": "abc"}],
        }
    )
    result = answer_question("q", hits, client)
    assert result.citations == [Citation("2304.01543v1", 4)]


def test_no_hits_means_refusal_without_calling_the_model():
    client = FakeClient({})
    result = answer_question("q", [], client)
    assert result.answer == REFUSAL
    assert client.calls == []


def test_forces_the_answer_tool(hits):
    client = FakeClient({"answerable": False, "answer": "", "citations": []})
    answer_question("q", hits, client)
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_answer"}
