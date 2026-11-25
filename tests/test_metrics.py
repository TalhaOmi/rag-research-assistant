import pytest

from evals.metrics import (
    chunk_matches_gold,
    first_relevant_rank,
    hit_at_k,
    mean,
    normalize,
    reciprocal_rank,
)
from paper_assistant.ingest import Chunk

GOLD = {"paper_id": "p1", "page": 2, "evidence": "fidelity measures how well"}


def chunk(paper_id, text):
    return Chunk(f"{paper_id}:p1:c0", paper_id, 1, text)


def test_normalize_ignores_case_punctuation_and_whitespace():
    assert normalize("Fidelity,  measures\nHOW-well!") == "fidelity measures how well"


def test_match_needs_right_paper_and_phrase():
    assert chunk_matches_gold(chunk("p1", "Here Fidelity measures how well it works"), GOLD)
    assert not chunk_matches_gold(chunk("p2", "Fidelity measures how well"), GOLD)
    assert not chunk_matches_gold(chunk("p1", "nothing relevant"), GOLD)


def test_first_relevant_rank():
    chunks = [chunk("p1", "x"), chunk("p1", "y"), chunk("p1", "fidelity measures how well")]
    assert first_relevant_rank(chunks, GOLD) == 3
    assert first_relevant_rank(chunks[:2], GOLD) is None


def test_hit_at_k():
    assert hit_at_k(3, 5)
    assert not hit_at_k(6, 5)
    assert not hit_at_k(None, 5)


@pytest.mark.parametrize("rank,expected", [(1, 1.0), (2, 0.5), (4, 0.25), (None, 0.0)])
def test_reciprocal_rank(rank, expected):
    assert reciprocal_rank(rank) == expected


def test_mean():
    assert mean([1, 0, 1, 0]) == 0.5
    assert mean([]) == 0.0
