import pytest

from paper_assistant.embedders import HashingEmbedder
from paper_assistant.index import build_index, load_chunks
from paper_assistant.ingest import Chunk
from paper_assistant.retrieve import HybridRetriever

CHUNKS = [
    Chunk(
        "a:p1:c0", "a", 1, "LIME explains individual predictions with local linear surrogate models"
    ),
    Chunk(
        "b:p2:c0",
        "b",
        2,
        "Counterfactual explanations describe the smallest change that flips a decision",
    ),
    Chunk("c:p3:c0", "c", 3, "Saliency maps highlight pixels that influence the classifier output"),
    Chunk("d:p4:c0", "d", 4, "Fuzzy controllers for vehicles use interpretable linguistic rules"),
]


@pytest.fixture
def retriever(tmp_path):
    embedder = HashingEmbedder()
    build_index(CHUNKS, embedder, tmp_path / "idx")
    return HybridRetriever.from_index_dir(tmp_path / "idx", embedder)


def test_index_roundtrip(tmp_path):
    build_index(CHUNKS, HashingEmbedder(), tmp_path / "idx")
    assert load_chunks(tmp_path / "idx") == CHUNKS


def test_bm25_finds_exact_term(retriever):
    hits = retriever.search("LIME surrogate", k=2, mode="bm25")
    assert hits[0].chunk.chunk_id == "a:p1:c0"
    assert hits[0].rank == 1


def test_dense_returns_ranked_hits(retriever):
    hits = retriever.search("saliency maps for classifiers", k=3, mode="dense")
    assert hits[0].chunk.chunk_id == "c:p3:c0"
    assert [h.rank for h in hits] == [1, 2, 3]


def test_hybrid_contains_top_result_of_each(retriever):
    hits = retriever.search("counterfactual change decision", k=4, mode="hybrid")
    assert hits[0].chunk.chunk_id == "b:p2:c0"


def test_k_limits_results(retriever):
    assert len(retriever.search("explanations", k=2)) == 2


def test_unknown_words_return_nothing_for_bm25(retriever):
    assert retriever.search("zzzzqqq", k=3, mode="bm25") == []


def test_bad_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")
