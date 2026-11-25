from paper_assistant.generate import Answer, Citation
from paper_assistant.judge import cited_passages, judge_faithfulness

from .conftest import FakeClient, make_hit


def make_answer():
    hits = [make_hit("a", 1, "alpha text"), make_hit("b", 2, "beta text")]
    return Answer("q", "Claim one [a, p.1].", True, [Citation("a", 1)], [], hits)


def test_only_cited_passages_are_shown_to_the_judge():
    passages = cited_passages(make_answer())
    assert len(passages) == 1 and "alpha text" in passages[0]


def test_faithfulness_is_fraction_of_supported_claims():
    client = FakeClient(
        {
            "claims": [
                {"claim": "c1", "supported": True},
                {"claim": "c2", "supported": True},
                {"claim": "c3", "supported": False},
                {"claim": "c4", "supported": True},
            ]
        }
    )
    assert judge_faithfulness(make_answer(), client).faithfulness == 0.75


def test_no_claims_counts_as_fully_faithful():
    assert judge_faithfulness(make_answer(), FakeClient({"claims": []})).faithfulness == 1.0
